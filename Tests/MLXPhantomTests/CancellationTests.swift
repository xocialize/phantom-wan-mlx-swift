// CancellationTests.swift — Phantom-Wan through the engine's CAN gate (offline, no MLX
// kernels). CAN-1/2 drive the real run() pre-cancelled (the entry checkpoint fires before
// the notLoaded guard or weights); CAN-3 is the document of record for the checkpoint
// cadence: the S2V sampler threads a throwing `onStep` closure — `try
// Task.checkCancellation()` once per denoising step (MLXPhantomPackage.runS2V →
// PhantomSampling `try onStep?(i, steps, latent)`) — and the wan-core streaming VAE
// decode bails per temporal chunk (`Task.isCancelled` in decodeStreaming), with the
// wrapper's post-core `try Task.checkCancellation()` discarding a truncated result.

import Foundation
import MLXServeConformance
import MLXToolKit
import XCTest
@testable import MLXPhantom

final class CancellationTests: XCTestCase {

    // MARK: - CAN-1 / CAN-2 — pre-cancelled run() propagation + classification

    func testCANGatePreCancelledRun() async {
        // Stub config; construction is cheap (C13) and the entry checkpoint throws before
        // validation or weights are touched, so this is offline-safe.
        let package = MLXPhantomPackage(configuration: PhantomConfiguration())
        let report = await CancellationConformance.checkRun(
            package: package,
            request: T2VRequest(prompt: "probe"))
        XCTAssertTrue(report.passed, report.summary)
    }

    // MARK: - CAN-3 — checkpoint-cadence declaration (the document of record)

    func testCANCadenceDeclaration() {
        // textToVideo is a long-run capability — the sub-second exemption is not available.
        XCTAssertTrue(CancellationConformance.longRunImplied(by: MLXPhantomPackage.manifest))

        let report = CancellationConformance.checkCadence(
            manifest: MLXPhantomPackage.manifest,
            posture: .cadence([
                // Per denoising step: the throwing onStep closure threaded into the
                // 3-forward S2V sampler (PhantomSampling.swift).
                .init(phase: .denoise, unit: .step),
                // Per VAE-decode temporal chunk: wan-core decodeStreaming bails on
                // Task.isCancelled; the wrapper's post-core checkpoint rethrows.
                .init(phase: .decode, unit: .chunk),
            ]))
        XCTAssertTrue(report.passed, report.summary)
    }
}
