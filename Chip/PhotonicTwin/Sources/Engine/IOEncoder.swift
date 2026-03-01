import Foundation
import Metal

/// Computes I/O geometry (spot positions, detector masks) on CPU at init time.
/// The per-sample encoding runs on GPU via encode_input_batched kernel.
final class IOEncoder {
    let config: SimConfig
    let spotCentersData: [SIMD2<Float>]  // (cx, cy) for each input channel
    let detectorCenters: [SIMD2<Float>]  // (cx, cy) for each output channel
    let spotSigma: Float
    let detectorRadius: Int

    // Flattened detector mask pixel indices and offset arrays
    let maskPixels: [UInt32]       // pixel indices for all detectors
    let maskOffsets: [UInt32]      // prefix sums: [0, count0, count0+count1, ...]

    init(config: SimConfig) {
        self.config = config

        // Use only active input channels for layout (nInputsA, not nInputs)
        let nIn = config.nInputsA
        let nOut = config.nDetectors  // 4 standard or 8 differential
        // Spot size: use override if set, otherwise auto-scale
        let sigma: Float
        if config.spotSigmaOverride > 0 {
            sigma = config.spotSigmaOverride
        } else {
            sigma = max(Float(config.nx) / (4.0 * sqrt(Float(max(nIn, nOut)))), 3.0)
        }
        self.spotSigma = sigma
        // Detector radius: use override if set, otherwise 2*sigma
        if config.detectorRadiusOverride > 0 {
            self.detectorRadius = config.detectorRadiusOverride
        } else {
            self.detectorRadius = Int(sigma * 2.0)
        }

        // Layout input spots and output detectors.
        // D2NN-inspired: inputs on left (inputXFraction), outputs on right (outputXFraction).
        // Default xFraction=0.5 preserves co-located behavior for existing presets.
        self.spotCentersData = IOEncoder.layoutStrip(
            count: nIn, nx: config.nx, ny: config.ny, xFraction: config.inputXFraction
        )

        if config.useImageInput {
            // MNIST mode: spread detectors across full output plane in grid pattern.
            // Input is full-field image (z=0), output detectors at z=last.
            // No need to concentrate detectors at one edge.
            self.detectorCenters = IOEncoder.layoutGrid(
                count: nOut, nx: config.nx, ny: config.ny, margin: 0.15
            )
        } else {
            self.detectorCenters = IOEncoder.layoutStrip(
                count: nOut, nx: config.nx, ny: config.ny, xFraction: config.outputXFraction
            )
        }

        // Build detector masks: list of pixel indices within each detector circle
        var pixels: [UInt32] = []
        var offsets: [UInt32] = [0]

        for d in 0..<nOut {
            let cx = Int(detectorCenters[d].x)
            let cy = Int(detectorCenters[d].y)
            let r = detectorRadius

            for x in max(0, cx - r)..<min(config.nx, cx + r + 1) {
                for y in max(0, cy - r)..<min(config.ny, cy + r + 1) {
                    let dx = x - cx
                    let dy = y - cy
                    if dx * dx + dy * dy <= r * r {
                        pixels.append(UInt32(x * config.ny + y))
                    }
                }
            }
            offsets.append(UInt32(pixels.count))
        }

        self.maskPixels = pixels
        self.maskOffsets = offsets

        // Print detector layout for diagnostics
        let layout = config.useImageInput ? "grid" : "strip"
        print("[IO] \(nOut) detectors (\(layout)), radius=\(detectorRadius), \(pixels.count) total mask pixels (\(String(format: "%.1f", Float(pixels.count) / Float(config.nx * config.ny) * 100))% coverage)")
        for d in 0..<nOut {
            let pixCount = offsets[d + 1] - offsets[d]
            print("[IO]   det\(d): center=(\(Int(detectorCenters[d].x)),\(Int(detectorCenters[d].y))) \(pixCount) pixels")
        }
    }

    /// Upload I/O geometry to GPU buffers on the MetalEngine
    func uploadToEngine(_ engine: MetalEngine) {
        engine.spotSigma = spotSigma

        // Spot centers
        let centersSize = spotCentersData.count * MemoryLayout<SIMD2<Float>>.stride
        if let buf = engine.device.makeBuffer(bytes: spotCentersData, length: centersSize, options: .storageModeShared) {
            buf.label = "spotCenters"
            engine.spotCenters = buf
        }

        // Detector mask pixels
        let pixelSize = maskPixels.count * MemoryLayout<UInt32>.stride
        if let buf = engine.device.makeBuffer(bytes: maskPixels, length: max(pixelSize, 4), options: .storageModeShared) {
            buf.label = "detectorMaskData"
            engine.detectorMaskData = buf
        }

        // Detector offsets
        let offsetSize = maskOffsets.count * MemoryLayout<UInt32>.stride
        if let buf = engine.device.makeBuffer(bytes: maskOffsets, length: offsetSize, options: .storageModeShared) {
            buf.label = "detectorOffsets"
            engine.detectorOffsets = buf
        }
    }

    /// Arrange N channels vertically in a strip at the given horizontal position.
    /// xFraction: where the strip center is (0=left, 1=right)
    /// margin: vertical margin fraction
    static func layoutStrip(count: Int, nx: Int, ny: Int, xFraction: Float, margin: Float = 0.15) -> [SIMD2<Float>] {
        let cx = Float(nx) * xFraction
        var positions: [SIMD2<Float>] = []
        for idx in 0..<count {
            let cy = Float(ny) * (margin + (1.0 - 2.0 * margin) * Float(idx) / Float(max(count - 1, 1)))
            positions.append(SIMD2<Float>(cx, cy))
        }
        return positions
    }

    /// Arrange N detectors in a 2D grid across the full output plane.
    /// Used for MNIST/image input mode where input and output are on different z-planes.
    /// Detectors are evenly spaced in a roughly square grid with margin from edges.
    static func layoutGrid(count: Int, nx: Int, ny: Int, margin: Float = 0.15) -> [SIMD2<Float>] {
        if count <= 1 {
            return [SIMD2<Float>(Float(nx) * 0.5, Float(ny) * 0.5)]
        }
        // Find best grid dimensions (roughly square)
        let cols = Int(ceil(sqrt(Float(count))))
        let rows = Int(ceil(Float(count) / Float(cols)))

        var positions: [SIMD2<Float>] = []
        for idx in 0..<count {
            let row = idx / cols
            let col = idx % cols
            let fx = margin + (1.0 - 2.0 * margin) * Float(col) / Float(max(cols - 1, 1))
            let fy = margin + (1.0 - 2.0 * margin) * Float(row) / Float(max(rows - 1, 1))
            positions.append(SIMD2<Float>(Float(nx) * fx, Float(ny) * fy))
        }
        return positions
    }
}
