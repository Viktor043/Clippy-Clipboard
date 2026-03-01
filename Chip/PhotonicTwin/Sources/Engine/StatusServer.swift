import Foundation
import Network

/// Lightweight HTTP server for LAN-accessible training status + live parameter control.
/// Serves a minimal HTML dashboard + JSON API on port 8080.
///
/// API endpoints:
///   GET  /api/status  — Training metrics (loss, accuracy, step, speed, histories)
///   GET  /api/config  — Current tunable parameters
///   POST /api/config  — Update tunable parameters (JSON body)
///   GET  /api/voxels?z=N           — Read voxel slice at depth z as JSON array
///   POST /api/voxels               — Write voxel data (JSON: {z, data})
///   GET  /api/logits?z=N&v=V       — Read logit distribution at voxel (z, voxel_idx)
///   GET  /api/stats                — Buffer statistics (deltaN range, gradient norms)
///   POST /api/logits               — Read/write raw logits at specific voxels
///   POST /api/reset_momentum       — Zero Adam momentum buffers (warm restart)
///   POST /api/inject_pattern       — Inject structured voxel patterns (lens, grating, etc.)
///   GET  /                         — HTML dashboard
final class StatusServer {
    let port: UInt16
    let monitor: TrainingMonitor
    var liveConfig: LiveConfig?

    /// Optional reference to MetalEngine for voxel read/write
    var engine: MetalEngine?

    private var listener: NWListener?

    init(port: UInt16 = 8080, monitor: TrainingMonitor) {
        self.port = port
        self.monitor = monitor
    }

    func start() {
        do {
            let params = NWParameters.tcp
            listener = try NWListener(using: params, on: NWEndpoint.Port(rawValue: port)!)
            listener?.newConnectionHandler = { [weak self] conn in
                self?.handleConnection(conn)
            }
            listener?.start(queue: .global(qos: .utility))
        } catch {
            print("StatusServer failed to start: \(error)")
        }
    }

    func stop() {
        listener?.cancel()
        listener = nil
    }

    /// Get all local IP addresses for display
    static func localIPAddresses() -> [String] {
        var addresses: [String] = []
        var ifaddr: UnsafeMutablePointer<ifaddrs>?
        guard getifaddrs(&ifaddr) == 0, let first = ifaddr else { return addresses }
        defer { freeifaddrs(first) }

        var ptr = first
        while true {
            let flags = Int32(ptr.pointee.ifa_flags)
            let addr = ptr.pointee.ifa_addr.pointee

            if (flags & (IFF_UP | IFF_RUNNING)) != 0 && addr.sa_family == UInt8(AF_INET) {
                var hostname = [CChar](repeating: 0, count: Int(NI_MAXHOST))
                if getnameinfo(ptr.pointee.ifa_addr, socklen_t(addr.sa_len), &hostname, socklen_t(hostname.count), nil, 0, NI_NUMERICHOST) == 0 {
                    let ip = String(cString: hostname)
                    if ip != "127.0.0.1" {
                        addresses.append(ip)
                    }
                }
            }

            guard let next = ptr.pointee.ifa_next else { break }
            ptr = next
        }
        return addresses
    }

    // MARK: - Connection handling

    private func handleConnection(_ conn: NWConnection) {
        conn.start(queue: .global(qos: .utility))
        conn.receive(minimumIncompleteLength: 1, maximumLength: 65536) { [weak self] data, _, _, _ in
            guard let self, let data else { conn.cancel(); return }
            let request = String(data: data, encoding: .utf8) ?? ""

            // Parse HTTP method and path
            let lines = request.split(separator: "\r\n", maxSplits: 1)
            let firstLine = String(lines.first ?? "")
            let parts = firstLine.split(separator: " ")
            let method = parts.count > 0 ? String(parts[0]) : "GET"
            let path = parts.count > 1 ? String(parts[1]) : "/"

            // Extract body for POST requests
            let body: String?
            if let bodyStart = request.range(of: "\r\n\r\n") {
                body = String(request[bodyStart.upperBound...])
            } else {
                body = nil
            }

            // Route requests
            if path.hasPrefix("/api/config") && method == "POST" {
                self.handleConfigUpdate(conn: conn, body: body)
            } else if path.hasPrefix("/api/config") {
                self.sendConfigJSON(conn: conn)
            } else if path.hasPrefix("/api/stats") {
                self.sendStatsJSON(conn: conn)
            } else if path.hasPrefix("/api/voxels") && method == "POST" {
                self.handleVoxelWrite(conn: conn, body: body)
            } else if path.hasPrefix("/api/voxels") {
                self.handleVoxelRead(conn: conn, path: path)
            } else if path.hasPrefix("/api/reset_momentum") && method == "POST" {
                self.handleResetMomentum(conn: conn)
            } else if path.hasPrefix("/api/restore_best") && method == "POST" {
                self.handleRestoreBest(conn: conn)
            } else if path.hasPrefix("/api/ema_swap") && method == "POST" {
                self.handleEMASwap(conn: conn, body: body)
            } else if path.hasPrefix("/api/inject_pattern") && method == "POST" {
                self.handleInjectPattern(conn: conn, body: body)
            } else if path.hasPrefix("/api/logits") && method == "POST" {
                self.handleLogitsReadWrite(conn: conn, body: body)
            } else if path.hasPrefix("/api/logits") {
                self.handleLogitsRead(conn: conn, path: path)
            } else if path.hasPrefix("/api/experiment") && method == "POST" {
                self.handleExperimentUpdate(conn: conn, body: body)
            } else if path.hasPrefix("/api/experiment") {
                self.sendExperimentJSON(conn: conn)
            } else if path.hasPrefix("/api/status") {
                let compact = path.contains("compact")
                self.sendJSON(conn: conn, compact: compact)
            } else {
                self.sendHTML(conn: conn)
            }
        }
    }

    // MARK: - Status API (existing)

    private func sendJSON(conn: NWConnection, compact: Bool = false) {
        func safe(_ v: Float) -> Double { v.isFinite ? Double(v) : 0.0 }
        func safeD(_ v: Double) -> Double { v.isFinite ? v : 0.0 }

        DispatchQueue.main.async { [weak self] in
            guard let self else { return }
            let m = self.monitor
            var json: [String: Any] = [
                "step": m.currentStep,
                "totalSteps": m.totalSteps,
                "loss": safe(m.currentLoss),
                "accuracy": safe(m.currentAccuracy),
                "smoothedLoss": safe(m.smoothedLoss),
                "smoothedAccuracy": safe(m.smoothedAccuracy),
                "peakAccuracy": safe(m.peakAccuracy),
                "peakAccuracyStep": m.peakAccuracyStep,
                "peakSmoothedAccuracy": safe(m.peakSmoothedAccuracy),
                "peakSmoothedAccuracyStep": m.peakSmoothedAccuracyStep,
                "bestSmoothedLoss": safe(m.bestSmoothedLoss),
                "temperature": safe(m.currentTemperature),
                "stepsPerSecond": safeD(m.stepsPerSecond),
                "eta_seconds": safeD(m.estimatedTimeRemaining),
                "isRunning": m.isRunning,
                "config": m.configSummary,
                "presetName": m.currentPresetName,
                "mode": m.isRegressionMode ? "regression" : "classification",
            ]
            if m.isRegressionMode {
                // Add regression-specific metrics
                let mse = Double(m.smoothedLoss)
                let psnr = mse > 1e-10 ? 10.0 * log10(1.0 / mse) : 0.0
                json["mse"] = safeD(mse)
                json["psnr"] = safeD(psnr)
                let bestMSE = Double(m.bestSmoothedLoss)
                let bestPSNR = bestMSE > 1e-10 ? 10.0 * log10(1.0 / bestMSE) : 0.0
                json["bestMSE"] = safeD(bestMSE)
                json["bestPSNR"] = safeD(bestPSNR)
            }
            if !compact {
                json["lossHistory"] = Array(m.lossHistory.suffix(500).map { safe($0) })
                json["accuracyHistory"] = Array(m.accuracyHistory.suffix(500).map { safe($0) })
                json["smoothedLossHistory"] = Array(m.smoothedLossHistory.suffix(500).map { safe($0) })
                json["smoothedAccuracyHistory"] = Array(m.smoothedAccuracyHistory.suffix(500).map { safe($0) })
            }
            self.sendJSONResponse(conn: conn, json: json)
        }
    }

    // MARK: - Config API

    private func sendConfigJSON(conn: NWConnection) {
        guard let lc = liveConfig else {
            sendErrorResponse(conn: conn, message: "No active training run")
            return
        }
        let json = lc.tunableParamsJSON()
        sendJSONResponse(conn: conn, json: json)
    }

    private func handleConfigUpdate(conn: NWConnection, body: String?) {
        guard let lc = liveConfig,
              let body,
              let data = body.data(using: .utf8),
              let dict = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            sendErrorResponse(conn: conn, message: "Invalid JSON or no active run")
            return
        }

        let changed = lc.applyJSON(dict)

        // Update monitor's totalSteps if nSteps changed
        if dict["nSteps"] != nil {
            let newSteps = lc.config.nSteps
            DispatchQueue.main.async { [weak self] in
                self?.monitor.totalSteps = newSteps
                self?.monitor.log("Live update: nSteps → \(newSteps)")
            }
        }

        // Log changes to monitor
        if !changed.isEmpty {
            let msg = "Live config: \(changed.joined(separator: ", "))"
            DispatchQueue.main.async { [weak self] in
                self?.monitor.log(msg)
            }
        }

        let response: [String: Any] = [
            "ok": true,
            "changed": changed,
            "config": lc.tunableParamsJSON()
        ]
        sendJSONResponse(conn: conn, json: response)
    }

    // MARK: - Voxel API

    private func handleVoxelRead(conn: NWConnection, path: String) {
        guard let eng = engine else {
            sendErrorResponse(conn: conn, message: "No engine available")
            return
        }
        guard let lc = liveConfig else {
            sendErrorResponse(conn: conn, message: "No active run")
            return
        }
        let cfg = lc.config
        // Parse ?z=N from path
        let z: Int
        if let range = path.range(of: "z="), let val = Int(path[range.upperBound...].prefix(while: { $0.isNumber })) {
            z = val
        } else {
            z = cfg.nz / 2  // default to middle slice
        }

        let fieldSize = cfg.nx * cfg.ny
        let offset = z * fieldSize
        let ptr = eng.deltaN.contents().bindMemory(to: Float.self, capacity: cfg.totalVoxels)
        var values: [Double] = []
        values.reserveCapacity(fieldSize)
        for i in 0..<fieldSize {
            let v = ptr[offset + i]
            values.append(v.isFinite ? Double(v) : 0.0)
        }

        let json: [String: Any] = [
            "z": z,
            "nx": cfg.nx,
            "ny": cfg.ny,
            "dnMax": cfg.dnMax,
            "data": values
        ]
        sendJSONResponse(conn: conn, json: json)
    }

    private func handleVoxelWrite(conn: NWConnection, body: String?) {
        guard let eng = engine,
              let lc = liveConfig,
              let body,
              let data = body.data(using: .utf8),
              let dict = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            sendErrorResponse(conn: conn, message: "Invalid request")
            return
        }

        let cfg = lc.config
        let z = dict["z"] as? Int ?? 0
        guard let values = dict["data"] as? [Double] else {
            sendErrorResponse(conn: conn, message: "Missing 'data' array")
            return
        }

        let fieldSize = cfg.nx * cfg.ny
        guard values.count == fieldSize else {
            sendErrorResponse(conn: conn, message: "Data size mismatch: expected \(fieldSize), got \(values.count)")
            return
        }

        let offset = z * fieldSize
        let ptr = eng.deltaN.contents().bindMemory(to: Float.self, capacity: cfg.totalVoxels)
        for i in 0..<fieldSize {
            ptr[offset + i] = Float(values[i])
        }

        let response: [String: Any] = [
            "ok": true,
            "z": z,
            "voxelsWritten": fieldSize
        ]
        sendJSONResponse(conn: conn, json: response)
    }

    // MARK: - Momentum Reset API

    private func handleResetMomentum(conn: NWConnection) {
        guard let eng = engine, let lc = liveConfig else {
            sendErrorResponse(conn: conn, message: "No engine available")
            return
        }
        let cfg = lc.config
        let nLogits = cfg.totalVoxels * cfg.nLevels

        // Zero out Adam momentum buffers
        let mPtr = eng.momentM.contents().bindMemory(to: Float.self, capacity: nLogits)
        let vPtr = eng.momentV.contents().bindMemory(to: Float.self, capacity: nLogits)
        for i in 0..<nLogits {
            mPtr[i] = 0
            vPtr[i] = 0
        }

        DispatchQueue.main.async { [weak self] in
            self?.monitor.log("Adam momentum reset (warm restart)")
        }

        sendJSONResponse(conn: conn, json: ["ok": true, "message": "Adam momentum buffers zeroed (\(nLogits) logits)"])
    }

    // MARK: - Restore Best Weights API

    private func handleRestoreBest(conn: NWConnection) {
        guard let eng = engine else {
            sendErrorResponse(conn: conn, message: "No engine available")
            return
        }
        eng.restoreBestWeights()
        // Also zero momentum for fresh start from best position
        let nLogits = eng.config.totalVoxels * eng.config.nLevels
        memset(eng.momentM.contents(), 0, nLogits * 4)
        memset(eng.momentV.contents(), 0, nLogits * 4)

        DispatchQueue.main.async { [weak self] in
            self?.monitor.log("Restored best weights + momentum reset")
        }

        sendJSONResponse(conn: conn, json: [
            "ok": true,
            "message": "Restored best accuracy weights to active (momentum zeroed)"
        ])
    }

    // MARK: - EMA Swap API

    private func handleEMASwap(conn: NWConnection, body: String?) {
        guard let eng = engine else {
            sendErrorResponse(conn: conn, message: "No engine available")
            return
        }

        // Parse optional direction: "ema_to_active" (default) or "active_to_ema" (reset EMA)
        var direction = "ema_to_active"
        var resetMomentum = true
        if let body = body,
           let data = body.data(using: .utf8),
           let dict = try? JSONSerialization.jsonObject(with: data) as? [String: Any] {
            direction = dict["direction"] as? String ?? "ema_to_active"
            resetMomentum = dict["resetMomentum"] as? Bool ?? true
        }

        if direction == "ema_to_active" {
            eng.swapEMAToActive()
            if resetMomentum {
                let nLogits = eng.config.totalVoxels * eng.config.nLevels
                memset(eng.momentM.contents(), 0, nLogits * 4)
                memset(eng.momentV.contents(), 0, nLogits * 4)
            }
            // Reset EMA to the new active position
            eng.resetEMAFromActive()

            DispatchQueue.main.async { [weak self] in
                self?.monitor.log("EMA→active swap (manual) \(resetMomentum ? "+ momentum reset" : "")")
            }
        } else {
            eng.resetEMAFromActive()
            DispatchQueue.main.async { [weak self] in
                self?.monitor.log("Active→EMA reset (manual)")
            }
        }

        sendJSONResponse(conn: conn, json: [
            "ok": true,
            "direction": direction,
            "resetMomentum": resetMomentum,
            "message": direction == "ema_to_active" ? "EMA weights copied to active, training continues from averaged position" : "EMA reset to current active weights"
        ])
    }

    // MARK: - Pattern Injection API

    private func handleInjectPattern(conn: NWConnection, body: String?) {
        guard let eng = engine, let lc = liveConfig,
              let body,
              let data = body.data(using: .utf8),
              let dict = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            sendErrorResponse(conn: conn, message: "Invalid request")
            return
        }

        let cfg = lc.config
        let pattern = dict["pattern"] as? String ?? "lens"
        let strength = Float(dict["strength"] as? Double ?? 1.0)
        let zStart = dict["zStart"] as? Int ?? 0
        let zEnd = dict["zEnd"] as? Int ?? cfg.nz

        let ptr = eng.deltaN.contents().bindMemory(to: Float.self, capacity: cfg.totalVoxels)
        let nx = cfg.nx, ny = cfg.ny
        let cx = Float(nx) / 2, cy = Float(ny) / 2
        var voxelsModified = 0

        for z in zStart..<min(zEnd, cfg.nz) {
            let offset = z * nx * ny
            for x in 0..<nx {
                for y in 0..<ny {
                    let idx = offset + x * ny + y
                    let fx = Float(x), fy = Float(y)
                    var delta: Float = 0

                    switch pattern {
                    case "lens":
                        // Parabolic phase profile: focusing lens
                        let r2 = (fx - cx) * (fx - cx) + (fy - cy) * (fy - cy)
                        let maxR2 = cx * cx + cy * cy
                        delta = strength * Float(cfg.dnMax) * (1.0 - 2.0 * r2 / maxR2)

                    case "grating":
                        // Periodic grating for diffraction routing
                        let period = dict["period"] as? Int ?? 8
                        delta = strength * Float(cfg.dnMax) * (x % period < period / 2 ? 1.0 : -1.0)

                    case "random":
                        // Random perturbation to escape local minima
                        delta = strength * Float(cfg.dnMax) * Float.random(in: -1...1)

                    case "zero":
                        // Reset to zero (clear glass)
                        delta = -ptr[idx]  // subtract current value to get zero

                    case "checkerboard":
                        // Checkerboard for structured diffraction
                        let period = dict["period"] as? Int ?? 4
                        delta = strength * Float(cfg.dnMax) * ((x / period + y / period) % 2 == 0 ? 1.0 : -1.0)

                    case "gradient_x":
                        // Linear gradient along x (beam steering)
                        delta = strength * Float(cfg.dnMax) * (2.0 * fx / Float(nx) - 1.0)

                    case "ring":
                        // Concentric rings (zone plate / Fresnel lens)
                        let r = sqrt((fx - cx) * (fx - cx) + (fy - cy) * (fy - cy))
                        let period = Float(dict["period"] as? Int ?? 8)
                        delta = strength * Float(cfg.dnMax) * sin(2.0 * Float.pi * r / period)

                    case "waveguide":
                        // Waveguide channels connecting each input spot to corresponding output detector.
                        // Creates high-index channels that guide light from input to output positions.
                        let nChannels = cfg.nOutputsA  // 4 channels for 4 classes
                        let inX = cfg.inputXFraction * Float(nx)
                        let outX = cfg.outputXFraction * Float(nx)
                        let margin: Float = 0.15
                        let zProgress = Float(z - zStart) / max(Float(zEnd - zStart - 1), 1.0)

                        var bestDist: Float = Float.infinity
                        for ch in 0..<nChannels {
                            let chFrac = margin + (1.0 - 2.0 * margin) * Float(ch) / Float(max(nChannels - 1, 1))
                            let channelY = chFrac * Float(ny)
                            // Channel center interpolates from input to output along z
                            let channelX = inX + (outX - inX) * zProgress
                            let dist = sqrt((fx - channelX) * (fx - channelX) + (fy - channelY) * (fy - channelY))
                            bestDist = min(bestDist, dist)
                        }
                        let sigma: Float = 3.0  // channel width
                        delta = strength * Float(cfg.dnMax) * exp(-bestDist * bestDist / (2.0 * sigma * sigma))

                    case "teacher_weights":
                        // Inject teacher weight matrix as refractive index modulation.
                        // Maps nInputs×nOutputs weight matrix to spatial routing pattern.
                        // Positive weights → high index (focus light), negative → low index (deflect).
                        let nIn = cfg.nInputsA
                        let nOut = cfg.nOutputsA
                        let margin: Float = 0.15
                        let zProgress = Float(z - zStart) / max(Float(zEnd - zStart - 1), 1.0)

                        // Determine which input-output pair this voxel is closest to
                        var maxWeight: Float = 0
                        for i in 0..<nIn {
                            let inY = margin + (1.0 - 2.0 * margin) * Float(i) / Float(max(nIn - 1, 1))
                            let inPos = SIMD2<Float>(cfg.inputXFraction * Float(nx), inY * Float(ny))
                            for o in 0..<nOut {
                                let outY = margin + (1.0 - 2.0 * margin) * Float(o) / Float(max(nOut - 1, 1))
                                let outPos = SIMD2<Float>(cfg.outputXFraction * Float(nx), outY * Float(ny))

                                // Point on the line from inPos to outPos at this z-depth
                                let lineX = inPos.x + (outPos.x - inPos.x) * zProgress
                                let lineY = inPos.y + (outPos.y - inPos.y) * zProgress
                                let dist = sqrt((fx - lineX) * (fx - lineX) + (fy - lineY) * (fy - lineY))
                                let sigma: Float = 4.0

                                if dist < sigma * 4.0 {
                                    // Weight from teacher matrix (provided in dict, or default identity)
                                    let weightIdx = i * nOut + o
                                    var w: Float = 0
                                    if let weights = dict["weights"] as? [Double], weightIdx < weights.count {
                                        w = Float(weights[weightIdx])
                                    } else {
                                        // Default: identity-like (stronger diagonal)
                                        w = (i == o) ? 1.0 : -0.3
                                    }
                                    maxWeight += w * exp(-dist * dist / (2.0 * sigma * sigma))
                                }
                            }
                        }
                        delta = strength * Float(cfg.dnMax) * maxWeight

                    case "multi_lens":
                        // D2NN-style: thin lens at each z-slice with different focal properties.
                        // Creates a "cascade of lenses" that routes light.
                        let r2 = (fx - cx) * (fx - cx) + (fy - cy) * (fy - cy)
                        let maxR2 = cx * cx + cy * cy
                        // Each z-slice gets a slightly different lens curvature (like a D2NN layer)
                        let zPhase = Float(z - zStart) * 2.0 * Float.pi / Float(max(zEnd - zStart, 1))
                        delta = strength * Float(cfg.dnMax) * cos(zPhase) * (1.0 - 2.0 * r2 / maxR2)

                    default:
                        break
                    }

                    ptr[idx] += delta
                    voxelsModified += 1
                }
            }
        }

        DispatchQueue.main.async { [weak self] in
            self?.monitor.log("Pattern injected: \(pattern) strength=\(strength) z=\(zStart)..\(zEnd)")
        }

        sendJSONResponse(conn: conn, json: [
            "ok": true,
            "pattern": pattern,
            "strength": Double(strength),
            "zRange": [zStart, zEnd],
            "voxelsModified": voxelsModified
        ])
    }

    // MARK: - Logits API

    private func handleLogitsRead(conn: NWConnection, path: String) {
        guard let eng = engine, let lc = liveConfig else {
            sendErrorResponse(conn: conn, message: "No engine available")
            return
        }
        let cfg = lc.config
        // Parse ?z=N&x=X&y=Y for single voxel logits
        var z = 0, x = -1, y = -1
        if let range = path.range(of: "z="), let val = Int(path[range.upperBound...].prefix(while: { $0.isNumber })) { z = val }
        if let range = path.range(of: "x="), let val = Int(path[range.upperBound...].prefix(while: { $0.isNumber })) { x = val }
        if let range = path.range(of: "y="), let val = Int(path[range.upperBound...].prefix(while: { $0.isNumber })) { y = val }

        let nLevels = cfg.nLevels
        let ptr = eng.logits.contents().bindMemory(to: Float.self, capacity: cfg.totalVoxels * nLevels)

        if x >= 0 && y >= 0 {
            // Single voxel: return full logit distribution
            let voxelIdx = z * cfg.fieldSize + x * cfg.ny + y
            let baseIdx = voxelIdx * nLevels
            var logits: [Double] = []
            for l in 0..<nLevels {
                let v = ptr[baseIdx + l]
                logits.append(v.isFinite ? Double(v) : 0.0)
            }
            sendJSONResponse(conn: conn, json: [
                "z": z, "x": x, "y": y,
                "nLevels": nLevels,
                "logits": logits
            ])
        } else {
            // Whole slice: return argmax refractive index for each voxel
            var argmaxSlice: [Int] = []
            argmaxSlice.reserveCapacity(cfg.fieldSize)
            let sliceOffset = z * cfg.fieldSize
            for v in 0..<cfg.fieldSize {
                let baseIdx = (sliceOffset + v) * nLevels
                var maxVal: Float = -.infinity
                var maxIdx = 0
                for l in 0..<nLevels {
                    if ptr[baseIdx + l] > maxVal {
                        maxVal = ptr[baseIdx + l]
                        maxIdx = l
                    }
                }
                argmaxSlice.append(maxIdx)
            }
            sendJSONResponse(conn: conn, json: [
                "z": z, "nx": cfg.nx, "ny": cfg.ny,
                "nLevels": nLevels,
                "argmax": argmaxSlice
            ])
        }
    }

    private func handleLogitsReadWrite(conn: NWConnection, body: String?) {
        guard let eng = engine, let lc = liveConfig,
              let body,
              let data = body.data(using: .utf8),
              let dict = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            sendErrorResponse(conn: conn, message: "Invalid request")
            return
        }

        let cfg = lc.config
        let z = dict["z"] as? Int ?? 0
        let x = dict["x"] as? Int ?? 0
        let y = dict["y"] as? Int ?? 0
        let nLevels = cfg.nLevels
        let ptr = eng.logits.contents().bindMemory(to: Float.self, capacity: cfg.totalVoxels * nLevels)

        if let logits = dict["logits"] as? [Double] {
            // Write logits for a single voxel
            let voxelIdx = z * cfg.fieldSize + x * cfg.ny + y
            let baseIdx = voxelIdx * nLevels
            let count = min(logits.count, nLevels)
            for l in 0..<count {
                ptr[baseIdx + l] = Float(logits[l])
            }
            sendJSONResponse(conn: conn, json: [
                "ok": true,
                "z": z, "x": x, "y": y,
                "logitsWritten": count
            ])
        } else {
            sendErrorResponse(conn: conn, message: "Missing 'logits' array")
        }
    }

    // MARK: - Stats API

    private func sendStatsJSON(conn: NWConnection) {
        guard let eng = engine, let lc = liveConfig else {
            sendErrorResponse(conn: conn, message: "No engine available")
            return
        }
        let cfg = lc.config
        let tv = cfg.totalVoxels

        // deltaN stats
        let dnPtr = eng.deltaN.contents().bindMemory(to: Float.self, capacity: tv)
        var dnMin: Float = .infinity, dnMax: Float = -.infinity, dnSum: Double = 0
        var dnNonZero = 0
        for i in 0..<tv {
            let v = dnPtr[i]
            if v.isFinite {
                dnMin = min(dnMin, v); dnMax = max(dnMax, v)
                dnSum += Double(abs(v))
                if abs(v) > 1e-8 { dnNonZero += 1 }
            }
        }

        // gradDeltaN stats
        let gdnPtr = eng.gradDeltaN.contents().bindMemory(to: Float.self, capacity: tv)
        var gdnMin: Float = .infinity, gdnMax: Float = -.infinity, gdnSum: Double = 0
        for i in 0..<tv {
            let v = gdnPtr[i]
            if v.isFinite {
                gdnMin = min(gdnMin, v); gdnMax = max(gdnMax, v)
                gdnSum += Double(abs(v))
            }
        }

        let json: [String: Any] = [
            "totalVoxels": tv,
            "deltaN": [
                "min": Double(dnMin), "max": Double(dnMax),
                "meanAbs": dnSum / Double(tv),
                "nonZeroCount": dnNonZero,
                "sparsity": 1.0 - Double(dnNonZero) / Double(tv)
            ],
            "gradDeltaN": [
                "min": Double(gdnMin), "max": Double(gdnMax),
                "meanAbs": gdnSum / Double(tv)
            ],
            "config": [
                "nx": cfg.nx, "ny": cfg.ny, "nz": cfg.nz,
                "dnMax": cfg.dnMax, "dz": cfg.dz,
                "batchSize": cfg.batchSize
            ]
        ]
        sendJSONResponse(conn: conn, json: json)
    }

    // MARK: - Experiment info API

    private func sendExperimentJSON(conn: NWConnection) {
        Task { @MainActor in
            let json: [String: Any] = [
                "title": self.monitor.experimentTitle,
                "description": self.monitor.experimentDescription,
                "method": self.monitor.experimentMethod,
                "findings": self.monitor.experimentFindings,
            ]
            self.sendJSONResponse(conn: conn, json: json)
        }
    }

    private func handleExperimentUpdate(conn: NWConnection, body: String?) {
        guard let body = body,
              let data = body.data(using: .utf8),
              let dict = try? JSONSerialization.jsonObject(with: data) as? [String: String] else {
            sendErrorResponse(conn: conn, message: "Invalid JSON")
            return
        }
        Task { @MainActor in
            if let v = dict["title"] { self.monitor.experimentTitle = v }
            if let v = dict["description"] { self.monitor.experimentDescription = v }
            if let v = dict["method"] { self.monitor.experimentMethod = v }
            if let v = dict["findings"] { self.monitor.experimentFindings = v }
            let json: [String: Any] = ["ok": true]
            self.sendJSONResponse(conn: conn, json: json)
        }
    }

    // MARK: - Response helpers

    private func sendJSONResponse(conn: NWConnection, json: [String: Any]) {
        if let data = try? JSONSerialization.data(withJSONObject: json),
           let body = String(data: data, encoding: .utf8) {
            let response = "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nAccess-Control-Allow-Origin: *\r\nAccess-Control-Allow-Methods: GET, POST, OPTIONS\r\nAccess-Control-Allow-Headers: Content-Type\r\nContent-Length: \(body.utf8.count)\r\n\r\n\(body)"
            conn.send(content: response.data(using: .utf8), completion: .contentProcessed { _ in conn.cancel() })
        }
    }

    private func sendErrorResponse(conn: NWConnection, message: String) {
        let json: [String: Any] = ["error": message]
        if let data = try? JSONSerialization.data(withJSONObject: json),
           let body = String(data: data, encoding: .utf8) {
            let response = "HTTP/1.1 400 Bad Request\r\nContent-Type: application/json\r\nAccess-Control-Allow-Origin: *\r\nContent-Length: \(body.utf8.count)\r\n\r\n\(body)"
            conn.send(content: response.data(using: .utf8), completion: .contentProcessed { _ in conn.cancel() })
        }
    }

    // MARK: - HTML Dashboard

    private func sendHTML(conn: NWConnection) {
        let html = Self.dashboardHTML(port: port)
        let response = "HTTP/1.1 200 OK\r\nContent-Type: text/html\r\nContent-Length: \(html.utf8.count)\r\n\r\n\(html)"
        conn.send(content: response.data(using: .utf8), completion: .contentProcessed { _ in conn.cancel() })
    }

    static func dashboardHTML(port: UInt16) -> String {
        return """
        <!DOCTYPE html>
        <html><head>
        <meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
        <title>Photonic Twin — Training Monitor</title>
        <style>
        *{margin:0;padding:0;box-sizing:border-box}
        body{font-family:-apple-system,system-ui,sans-serif;background:#0a0a0a;color:#e0e0e0;padding:16px}
        h1{font-size:18px;color:#60a5fa;margin-bottom:12px}
        .grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:8px;margin-bottom:16px}
        .card{background:#1a1a2e;border-radius:8px;padding:12px;text-align:center}
        .card .value{font-size:24px;font-weight:700;color:#60a5fa}
        .card .label{font-size:11px;color:#888;margin-top:4px}
        .card .sub{font-size:10px;color:#666;margin-top:2px}
        canvas{width:100%;height:200px;background:#111;border-radius:8px;margin-bottom:12px}
        .status{font-size:12px;color:#888;text-align:center;margin-top:8px}
        .tuner{background:#1a1a2e;border-radius:8px;padding:16px;margin-bottom:12px}
        .tuner h2{font-size:14px;color:#60a5fa;margin-bottom:8px}
        .tuner label{display:inline-block;width:120px;font-size:12px;color:#aaa}
        .tuner input[type=number]{width:100px;background:#111;color:#e0e0e0;border:1px solid #333;border-radius:4px;padding:4px;font-size:12px;font-family:monospace}
        .tuner button{background:#60a5fa;color:#000;border:none;border-radius:4px;padding:6px 16px;cursor:pointer;font-size:12px;margin-top:8px}
        .tuner button:hover{background:#93c5fd}
        .tuner .msg{font-size:10px;color:#888;margin-top:4px}
        </style></head><body>
        <h1>Photonic Neural Compute — Training Monitor</h1>
        <div class="grid" id="cards"></div>
        <canvas id="lossChart"></canvas>
        <canvas id="accChart"></canvas>
        <div class="tuner" id="tuner"><h2>Live Parameters</h2><div id="tunerFields"></div></div>
        <div class="status" id="status">Connecting...</div>
        <script>
        const api='/api/status',cfgApi='/api/config';
        function fmt(v,d=2){return typeof v==='number'?v.toFixed(d):v}
        function etaStr(s){if(s<=0)return'--';let h=Math.floor(s/3600),m=Math.floor(s%3600/60);return h>0?h+'h '+m+'m':m+'m'}
        function drawChart(id,data,smoothed,label,color,smoothColor){
          let c=document.getElementById(id),ctx=c.getContext('2d');
          c.width=c.clientWidth*2;c.height=c.clientHeight*2;ctx.scale(2,2);
          let w=c.clientWidth,h=c.clientHeight;
          ctx.clearRect(0,0,w,h);
          if(!data||data.length<2)return;
          let allVals=smoothed&&smoothed.length?[...data,...smoothed]:data;
          let mn=Math.min(...allVals),mx=Math.max(...allVals);
          if(mx===mn){mx+=1;mn-=1}
          function drawLine(d,col,lw){
            ctx.strokeStyle=col;ctx.lineWidth=lw;ctx.beginPath();
            for(let i=0;i<d.length;i++){
              let x=i/(d.length-1)*w,y=h-((d[i]-mn)/(mx-mn))*h*0.9-h*0.05;
              i===0?ctx.moveTo(x,y):ctx.lineTo(x,y);
            }
            ctx.stroke();
          }
          drawLine(data,color+'44',1);
          if(smoothed&&smoothed.length)drawLine(smoothed,smoothColor||color,2.5);
          ctx.fillStyle='#666';ctx.font='10px system-ui';
          let last=smoothed&&smoothed.length?smoothed[smoothed.length-1]:data[data.length-1];
          ctx.fillText(label+': '+fmt(last,4)+' (trend)',8,14);
          ctx.fillText('min:'+fmt(mn,4)+' max:'+fmt(mx,4),8,h-4);
        }
        let cfgLoaded=false;
        async function loadConfig(){
          try{
            let r=await fetch(cfgApi),d=await r.json();
            let fields='';
            const tunable=['lr','minLR','sparsityLambda','initialTemp','minTemp','annealRate','nSteps','adamBeta1','adamBeta2'];
            for(let k of tunable){
              if(d[k]!==undefined){
                fields+=`<div style="margin:4px 0"><label>${k}</label><input type="number" step="any" id="cfg_${k}" value="${d[k]}"></div>`;
              }
            }
            fields+=`<div style="margin:4px 0"><label>lrSchedule</label><select id="cfg_lrSchedule"><option ${d.lrSchedule==='cosine'?'selected':''}>cosine</option><option ${d.lrSchedule==='constant'?'selected':''}>constant</option></select></div>`;
            fields+=`<button onclick="applyConfig()">Apply Changes</button><span class="msg" id="cfgMsg"></span>`;
            fields+=`<div style="margin-top:8px;font-size:10px;color:#555">Grid: ${d.nx}×${d.ny}×${d.nz} | Batch: ${d.batchSize} | ${d.useDirect?'Direct':'Gumbel'} | ${d.useDifferentialDetection?'Differential':'Standard'}</div>`;
            document.getElementById('tunerFields').innerHTML=fields;
            cfgLoaded=true;
          }catch(e){}
        }
        async function applyConfig(){
          const tunable=['lr','minLR','sparsityLambda','initialTemp','minTemp','annealRate','adamBeta1','adamBeta2'];
          let update={};
          for(let k of tunable){
            let el=document.getElementById('cfg_'+k);
            if(el)update[k]=parseFloat(el.value);
          }
          let nStepsEl=document.getElementById('cfg_nSteps');
          if(nStepsEl)update.nSteps=parseInt(nStepsEl.value);
          let schedEl=document.getElementById('cfg_lrSchedule');
          if(schedEl)update.lrSchedule=schedEl.value;
          try{
            let r=await fetch(cfgApi,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(update)});
            let d=await r.json();
            document.getElementById('cfgMsg').textContent=d.ok?'Updated: '+d.changed.join(', '):'Error: '+d.error;
          }catch(e){document.getElementById('cfgMsg').textContent='Failed: '+e}
        }
        async function update(){
          try{
            let r=await fetch(api),d=await r.json();
            document.getElementById('cards').innerHTML=[
              ['Step',d.step+'/'+d.totalSteps,''],
              ['Loss',fmt(d.smoothedLoss,4),'raw: '+fmt(d.loss,4)],
              ...(d.mode==='regression' ? [
                ['MSE',fmt(d.mse,6),'best: '+fmt(d.bestMSE,6)],
                ['PSNR',fmt(d.psnr,1)+' dB','best: '+fmt(d.bestPSNR,1)+' dB'],
              ] : [
                ['Accuracy',(d.smoothedAccuracy*100).toFixed(1)+'%','raw: '+(d.accuracy*100).toFixed(1)+'%'],
                ['Peak Acc',(d.peakAccuracy*100).toFixed(1)+'%','trend: '+(d.peakSmoothedAccuracy*100).toFixed(1)+'%'],
              ]),
              ['Speed',fmt(d.stepsPerSecond,1)+'/s',''],
              ['ETA',etaStr(d.eta_seconds),d.presetName+' ('+d.mode+')']
            ].map(([l,v,s])=>'<div class="card"><div class="value">'+v+'</div><div class="label">'+l+'</div>'+(s?'<div class="sub">'+s+'</div>':'')+'</div>').join('');
            drawChart('lossChart',d.lossHistory,d.smoothedLossHistory,d.mode==='regression'?'MSE':'Loss','#f59e0b','#f59e0b');
            drawChart('accChart',d.accuracyHistory,d.smoothedAccuracyHistory,d.mode==='regression'?'PSNR (scaled)':'Accuracy','#10b981','#10b981');
            document.getElementById('status').textContent=d.isRunning?'Training...':'Idle — '+d.config;
          }catch(e){document.getElementById('status').textContent='Disconnected'}
        }
        update();setInterval(update,2000);
        loadConfig();setInterval(()=>{if(!cfgLoaded)loadConfig()},5000);
        </script></body></html>
        """
    }
}
