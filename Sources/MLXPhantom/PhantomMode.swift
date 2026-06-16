import WanCore
import MLXToolKit

extension Mode {
    /// Fewer denoising steps — the quicker path (Phantom denoises with FlowUniPC; `.fast` trims steps).
    public static let fast: Mode = "fast"
    /// The reference quality path (config-default steps); the package default.
    public static let quality: Mode = "quality"
}

/// Resolve `mode` (+ any explicit `steps`) to a denoise step count. Explicit `steps` wins; `.fast` → 20.
func resolveSteps(mode: Mode?, steps: Int?) -> Int? {
    if let steps { return steps }
    switch mode {
    case .fast: return 20
    default: return nil  // nil / .quality / unknown → config-default steps
    }
}
