// OCRベンチマーク iOSハーネス(雛形)。
// 注意: この環境にはXcodeが無いためビルド未検証。onnxruntime-objc
// (https://onnxruntime.ai/docs/get-started/with-objc.html)のSwift APIに
// 準拠して書いているが、実機組み込み時にAPI差分の調整が必要な可能性がある。
// テンソルは mobile/dump_testdata.py が出力する .bin(リトルエンディアン生バイト列)
// + .json(shape/dtype) をバンドルに追加して使う。
import Foundation
import OnnxRuntime

struct TensorMeta: Decodable {
    let shape: [Int]
    let dtype: String
}

final class OCRBench {
    let env: ORTEnv

    init() throws {
        env = try ORTEnv(loggingLevel: .warning)
    }

    func run() {
        print("=== OCR Benchmark (iOS) ===\n")
        let footprintBefore = physFootprint()

        // (ファイル名, 表示名)
        let models: [(String, String)] = [
            ("PP-OCRv6_small_det_int8", "det"),
            ("PP-OCRv6_small_rec_int8", "rec"),
            ("PP-LCNet_x1_0_doc_ori_int8", "doc_ori"),
            ("PP-LCNet_x1_0_textline_ori_int8", "textline_ori"),
            ("model_int8_emb8", "labeler"),
        ]

        var sessions: [String: ORTSession] = [:]
        for (filename, name) in models {
            guard let path = Bundle.main.path(forResource: filename, ofType: "ort") else {
                print("Not found: \(filename).ort")
                continue
            }
            do {
                sessions[name] = try ORTSession(env: env, modelPath: path, sessionOptions: nil)
                print("Loaded: \(name)")
            } catch {
                print("Failed to load \(name): \(error)")
            }
        }

        let nIterations = 10
        for prefix in ["receipt", "receipt9"] {
            print("\n=== \(prefix) ===")
            bench(sessions["doc_ori"], inputs: ["x": "\(prefix)_doc_ori"], label: "doc_ori", n: nIterations)
            bench(sessions["det"], inputs: ["x": "\(prefix)_det"], label: "det", n: nIterations)
            var i = 0
            while tensorExists("\(prefix)_rec_\(i)") {
                bench(sessions["rec"], inputs: ["x": "\(prefix)_rec_\(i)"], label: "rec[\(i)]", n: nIterations)
                i += 1
            }
            bench(sessions["textline_ori"], inputs: ["x": "\(prefix)_textline"], label: "textline_ori", n: nIterations)
            bench(sessions["labeler"],
                  inputs: ["input_ids": "\(prefix)_labeler_ids",
                           "attention_mask": "\(prefix)_labeler_mask"],
                  label: "labeler", n: nIterations)
        }

        print("\n=== Memory ===")
        print("phys_footprint delta: \((physFootprint() - footprintBefore) / 1024 / 1024)MB")
    }

    private func bench(_ session: ORTSession?, inputs: [String: String], label: String, n: Int) {
        guard let session = session else { return }
        var feeds: [String: ORTValue] = [:]
        for (inputName, tensorName) in inputs {
            guard let v = loadTensor(name: tensorName) else {
                print("  \(label): tensor missing (\(tensorName))")
                return
            }
            feeds[inputName] = v
        }
        do {
            let outputNames = try session.outputNames()
            _ = try session.run(withInputs: feeds, outputNames: Set(outputNames), runOptions: nil) // ウォームアップ
            var times: [Double] = []
            for _ in 0..<n {
                let start = CFAbsoluteTimeGetCurrent()
                _ = try session.run(withInputs: feeds, outputNames: Set(outputNames), runOptions: nil)
                times.append((CFAbsoluteTimeGetCurrent() - start) * 1000)
            }
            let avg = times.reduce(0, +) / Double(times.count)
            print(String(format: "  %@: avg=%.1fms max=%.1fms", label, avg, times.max() ?? 0))
        } catch {
            print("  \(label): error - \(error)")
        }
    }

    private func tensorExists(_ name: String) -> Bool {
        Bundle.main.path(forResource: name, ofType: "bin") != nil
    }

    private func loadTensor(name: String) -> ORTValue? {
        guard let metaPath = Bundle.main.path(forResource: name, ofType: "json"),
              let binPath = Bundle.main.path(forResource: name, ofType: "bin"),
              let metaData = FileManager.default.contents(atPath: metaPath),
              let meta = try? JSONDecoder().decode(TensorMeta.self, from: metaData),
              let raw = FileManager.default.contents(atPath: binPath) else { return nil }
        let shape = meta.shape.map { NSNumber(value: $0) }
        let elementType: ORTTensorElementDataType =
            meta.dtype.contains("float32") ? .float : .int64
        let mutable = NSMutableData(data: raw)
        return try? ORTValue(tensorData: mutable, elementType: elementType, shape: shape)
    }

    private func physFootprint() -> Int {
        var info = task_vm_info_data_t()
        var count = mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<integer_t>.size)
        let result = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count)
            }
        }
        return result == KERN_SUCCESS ? Int(info.phys_footprint) : 0
    }
}

do {
    let bench = try OCRBench()
    bench.run()
} catch {
    print("init error: \(error)")
}
