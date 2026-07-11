// swift-tools-version:5.9
import PackageDescription

let package = Package(
    name: "OCRBench",
    platforms: [.iOS(.v16)],
    dependencies: [
        .package(url: "https://github.com/microsoft/onnxruntime-swift-cocoa.git", from: "1.16.3")
    ],
    targets: [
        .executableTarget(
            name: "OCRBench",
            dependencies: [
                .product(name: "OnnxRuntime", package: "onnxruntime-swift-cocoa")
            ],
            path: "OCRBench"
        )
    ]
)
