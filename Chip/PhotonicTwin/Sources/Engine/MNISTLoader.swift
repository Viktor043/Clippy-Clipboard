import Foundation

/// Loads MNIST dataset from IDX binary files.
/// Returns images as [Float] (0...1 normalized) and labels as [UInt8].
final class MNISTLoader {
    let images: [[Float]]  // [count][784] flattened 28×28
    let labels: [UInt8]
    let imageWidth = 28
    let imageHeight = 28

    /// Load MNIST from the given directory.
    /// Pass "train" or "test" for the split parameter.
    init(directory: String, split: String = "train") throws {
        let prefix = split == "train" ? "train" : "t10k"
        let imgPath = (directory as NSString).appendingPathComponent("\(prefix)-images-idx3-ubyte")
        let lblPath = (directory as NSString).appendingPathComponent("\(prefix)-labels-idx1-ubyte")

        guard let imgData = try? Data(contentsOf: URL(fileURLWithPath: imgPath)),
              let lblData = try? Data(contentsOf: URL(fileURLWithPath: lblPath)) else {
            throw NSError(domain: "MNISTLoader", code: 1,
                          userInfo: [NSLocalizedDescriptionKey: "Failed to load MNIST files from \(directory)"])
        }

        // Parse image file: [magic(4)] [count(4)] [rows(4)] [cols(4)] [pixels...]
        let count = Int(imgData[4...7].withUnsafeBytes { $0.load(as: UInt32.self).bigEndian })
        let rows = Int(imgData[8...11].withUnsafeBytes { $0.load(as: UInt32.self).bigEndian })
        let cols = Int(imgData[12...15].withUnsafeBytes { $0.load(as: UInt32.self).bigEndian })
        assert(rows == 28 && cols == 28, "Expected 28×28 images")

        var imgs: [[Float]] = []
        imgs.reserveCapacity(count)
        let pixelStart = 16
        for i in 0..<count {
            let offset = pixelStart + i * 784
            var img = [Float](repeating: 0, count: 784)
            for j in 0..<784 {
                img[j] = Float(imgData[offset + j]) / 255.0
            }
            imgs.append(img)
        }
        self.images = imgs

        // Parse label file: [magic(4)] [count(4)] [labels...]
        let lblCount = Int(lblData[4...7].withUnsafeBytes { $0.load(as: UInt32.self).bigEndian })
        assert(lblCount == count, "Image/label count mismatch")
        var lbls = [UInt8](repeating: 0, count: lblCount)
        for i in 0..<lblCount {
            lbls[i] = lblData[8 + i]
        }
        self.labels = lbls

        print("[MNIST] Loaded \(count) \(split) images (\(rows)×\(cols))")
    }

    /// Filter to specific digit classes and remap labels to 0..n-1.
    /// Returns (filtered_images, remapped_labels).
    func filtered(classes: [Int]) -> (images: [[Float]], labels: [UInt8]) {
        var filteredImages: [[Float]] = []
        var filteredLabels: [UInt8] = []
        let classSet = Set(classes)
        let remap = Dictionary(uniqueKeysWithValues: classes.enumerated().map { ($0.element, UInt8($0.offset)) })

        for i in 0..<labels.count {
            let lbl = Int(labels[i])
            if classSet.contains(lbl), let newLabel = remap[lbl] {
                filteredImages.append(images[i])
                filteredLabels.append(newLabel)
            }
        }
        print("[MNIST] Filtered to classes \(classes): \(filteredImages.count) images")
        return (filteredImages, filteredLabels)
    }

    /// Resize a 28×28 image to targetW × targetH using bilinear interpolation.
    /// Returns [targetH * targetW] flattened in row-major order.
    static func resize(image: [Float], targetW: Int, targetH: Int) -> [Float] {
        var result = [Float](repeating: 0, count: targetW * targetH)
        let srcW: Float = 28.0
        let srcH: Float = 28.0
        let scaleX = srcW / Float(targetW)
        let scaleY = srcH / Float(targetH)

        for y in 0..<targetH {
            for x in 0..<targetW {
                let srcX = (Float(x) + 0.5) * scaleX - 0.5
                let srcY = (Float(y) + 0.5) * scaleY - 0.5
                let x0 = Int(floor(srcX))
                let y0 = Int(floor(srcY))
                let x1 = min(x0 + 1, 27)
                let y1 = min(y0 + 1, 27)
                let fx = srcX - Float(x0)
                let fy = srcY - Float(y0)

                let x0c = max(x0, 0)
                let y0c = max(y0, 0)
                let v00 = image[y0c * 28 + x0c]
                let v10 = image[y0c * 28 + x1]
                let v01 = image[y1 * 28 + x0c]
                let v11 = image[y1 * 28 + x1]

                let v = v00 * (1 - fx) * (1 - fy) + v10 * fx * (1 - fy) +
                        v01 * (1 - fx) * fy + v11 * fx * fy
                result[y * targetW + x] = v
            }
        }
        return result
    }

    /// Pack a batch of images for GPU consumption.
    /// Returns flattened [batchSize * targetW * targetH] array of Float values (0...1).
    /// Images are resized and centered on a targetW × targetH grid.
    static func packBatch(
        images: [[Float]], labels: [UInt8],
        indices: [Int],
        targetW: Int, targetH: Int
    ) -> (imageData: [Float], batchLabels: [UInt32]) {
        let npixels = targetW * targetH
        var imageData = [Float](repeating: 0, count: indices.count * npixels)
        var batchLabels = [UInt32](repeating: 0, count: indices.count)

        for (b, idx) in indices.enumerated() {
            let resized = resize(image: images[idx], targetW: targetW, targetH: targetH)
            let offset = b * npixels
            for p in 0..<npixels {
                imageData[offset + p] = resized[p]
            }
            batchLabels[b] = UInt32(labels[idx])
        }
        return (imageData, batchLabels)
    }
}
