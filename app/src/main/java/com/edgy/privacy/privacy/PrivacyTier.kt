package com.edgy.privacy.privacy

/**
 * DDF (Dual-phase Disentangled Filter) privacy tiers.
 *
 * | Tier     | What leaves the device          | What is stripped                          |
 * |----------|--------------------------------|------------------------------------------|
 * | LOW      | Raw audio (passthrough)        | Nothing                                  |
 * | MODERATE | VQ codes + speaker embedding   | Emotion, accent, health                  |
 * | HIGH     | VQ codes only                  | Everything including speaker identity    |
 */
enum class PrivacyTier(val level: Int, val description: String) {
    LOW(0, "Passthrough — raw audio, no filtering"),
    MODERATE(1, "VQ codes + speaker embedding — strips emotion, accent, health"),
    HIGH(2, "VQ codes only — strips all paralinguistic information");

    companion object {
        fun fromLevel(level: Int): PrivacyTier {
            return entries.first { it.level == level }
        }
    }
}
