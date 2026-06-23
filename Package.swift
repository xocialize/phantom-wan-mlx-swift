// swift-tools-version: 6.2
// phantom-wan-mlx-swift — Swift/MLX port of Phantom-Wan-1.3B (subject-to-video; the lead
// consumer image-grounding model — multi-subject ≤4 reference images). A wan-core consumer:
// the DiT IS the stock wan-core `WanModel` UNCHANGED (826 keys = the WanModel set exactly), and
// the conditioning is pure INPUT ASSEMBLY — K single-frame VAE-encoded refs concatenated to the
// target latent's temporal tail (F+K grid), with a 3-forward S2V sampler. No new DiT architecture,
// no net-new weights. Python-MLX oracle: /Volumes/DEV_ARCHIVE/phantom-wan-mlx.

import PackageDescription

let package = Package(
    name: "PhantomWan",
    platforms: [
        .macOS(.v26)
    ],
    products: [
        .library(name: "PhantomWan", targets: ["PhantomWan"]),
        // The MLXEngine wrapper: a conformant `ModelPackage` over the core pipeline.
        .library(name: "MLXPhantom", targets: ["MLXPhantom"]),
    ],
    dependencies: [
        .package(url: "https://github.com/ml-explore/mlx-swift.git", from: "0.30.0"),
        // MLXEngine contract (MLXToolKit) for the wrapper target.
        .package(url: "https://github.com/xocialize/mlx-engine-swift", from: "0.9.1"),
        // The neutral Wan substrate (WanModel + 16-ch VAE + umT5 + RoPE + schedulers + loader).
        .package(path: "../wan-core-mlx-swift"),
    ],
    targets: [
        .target(
            name: "PhantomWan",
            dependencies: [
                .product(name: "WanCore", package: "wan-core-mlx-swift"),
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXFast", package: "mlx-swift"),
                .product(name: "MLXRandom", package: "mlx-swift"),
            ],
            path: "Sources/PhantomWan"
        ),
        .target(
            name: "MLXPhantom",
            dependencies: [
                "PhantomWan",
                .product(name: "WanCore", package: "wan-core-mlx-swift"),
                .product(name: "MLXToolKit", package: "mlx-engine-swift"),
            ],
            path: "Sources/MLXPhantom"
        ),
        .testTarget(
            name: "PhantomWanTests",
            dependencies: [
                "PhantomWan",
                .product(name: "WanCore", package: "wan-core-mlx-swift"),
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
            ],
            path: "Tests/PhantomWanTests"
        ),
    ]
)
