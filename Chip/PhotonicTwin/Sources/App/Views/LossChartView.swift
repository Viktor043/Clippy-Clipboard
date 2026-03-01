import SwiftUI
import Charts

struct LossChartView: View {
    let lossHistory: [Float]
    let accuracyHistory: [Float]
    let smoothedLossHistory: [Float]
    let smoothedAccuracyHistory: [Float]

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            // Loss chart
            VStack(alignment: .leading, spacing: 4) {
                Text("Loss").font(.caption).foregroundColor(.secondary)
                if lossHistory.count > 1 {
                    Chart {
                        // Raw loss (faded)
                        ForEach(Array(subsampledLoss.enumerated()), id: \.offset) { idx, val in
                            LineMark(
                                x: .value("Step", idx),
                                y: .value("Loss", val),
                                series: .value("Series", "Raw")
                            )
                            .foregroundStyle(.orange.opacity(0.25))
                            .lineStyle(StrokeStyle(lineWidth: 1))
                        }
                        // Smoothed trend line (bold)
                        ForEach(Array(subsampledSmoothedLoss.enumerated()), id: \.offset) { idx, val in
                            LineMark(
                                x: .value("Step", idx),
                                y: .value("Loss", val),
                                series: .value("Series", "Trend")
                            )
                            .foregroundStyle(.orange)
                            .lineStyle(StrokeStyle(lineWidth: 2.5))
                        }
                    }
                    .chartYScale(domain: lossRange)
                    .chartXAxis(.hidden)
                    .chartLegend(.hidden)
                    .chartYAxis {
                        AxisMarks(position: .leading) { value in
                            AxisGridLine(stroke: StrokeStyle(lineWidth: 0.5, dash: [2]))
                                .foregroundStyle(Color.gray.opacity(0.3))
                            AxisValueLabel()
                                .font(.system(size: 9, design: .monospaced))
                        }
                    }
                } else {
                    Text("Waiting for data...")
                        .font(.caption)
                        .foregroundColor(.secondary)
                        .frame(maxWidth: .infinity, minHeight: 60, alignment: .center)
                }
            }

            // Accuracy chart
            VStack(alignment: .leading, spacing: 4) {
                Text("Accuracy").font(.caption).foregroundColor(.secondary)
                if accuracyHistory.count > 1 {
                    Chart {
                        // Raw accuracy (faded)
                        ForEach(Array(subsampledAccuracy.enumerated()), id: \.offset) { idx, val in
                            LineMark(
                                x: .value("Step", idx),
                                y: .value("Accuracy", val * 100),
                                series: .value("Series", "Raw")
                            )
                            .foregroundStyle(.green.opacity(0.25))
                            .lineStyle(StrokeStyle(lineWidth: 1))
                        }
                        // Smoothed trend line (bold)
                        ForEach(Array(subsampledSmoothedAccuracy.enumerated()), id: \.offset) { idx, val in
                            LineMark(
                                x: .value("Step", idx),
                                y: .value("Accuracy", val * 100),
                                series: .value("Series", "Trend")
                            )
                            .foregroundStyle(.green)
                            .lineStyle(StrokeStyle(lineWidth: 2.5))
                        }

                        // Random baseline
                        RuleMark(y: .value("Random", 25))
                            .foregroundStyle(.red.opacity(0.5))
                            .lineStyle(StrokeStyle(lineWidth: 1, dash: [4]))
                    }
                    .chartYScale(domain: 0...100)
                    .chartXAxis(.hidden)
                    .chartLegend(.hidden)
                    .chartYAxis {
                        AxisMarks(values: [0, 25, 50, 75, 100]) { value in
                            AxisGridLine(stroke: StrokeStyle(lineWidth: 0.5, dash: [2]))
                                .foregroundStyle(Color.gray.opacity(0.3))
                            AxisValueLabel()
                                .font(.system(size: 9, design: .monospaced))
                        }
                    }
                } else {
                    Text("Waiting for data...")
                        .font(.caption)
                        .foregroundColor(.secondary)
                        .frame(maxWidth: .infinity, minHeight: 60, alignment: .center)
                }
            }
        }
    }

    // Subsample to max 500 points for chart performance
    private var subsampledLoss: [Double] {
        subsample(lossHistory.map { Double($0) }, maxPoints: 500)
    }

    private var subsampledAccuracy: [Double] {
        subsample(accuracyHistory.map { Double($0) }, maxPoints: 500)
    }

    private var subsampledSmoothedLoss: [Double] {
        subsample(smoothedLossHistory.map { Double($0) }, maxPoints: 500)
    }

    private var subsampledSmoothedAccuracy: [Double] {
        subsample(smoothedAccuracyHistory.map { Double($0) }, maxPoints: 500)
    }

    private var lossRange: ClosedRange<Double> {
        // Use the smoothed range for a more stable axis
        let allVals = subsampledSmoothedLoss.isEmpty ? subsampledLoss : subsampledSmoothedLoss
        guard let mn = allVals.min(), let mx = allVals.max(), mn < mx else {
            return 0...2
        }
        // Widen a bit to show raw data excursions
        let rawMin = subsampledLoss.min() ?? mn
        let rawMax = subsampledLoss.max() ?? mx
        return min(mn, rawMin) * 0.95 ... max(mx, rawMax) * 1.05
    }

    private func subsample(_ data: [Double], maxPoints: Int) -> [Double] {
        guard data.count > maxPoints else { return data }
        let step = Double(data.count) / Double(maxPoints)
        return (0..<maxPoints).map { i in
            data[min(Int(Double(i) * step), data.count - 1)]
        }
    }
}
