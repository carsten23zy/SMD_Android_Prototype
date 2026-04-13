package com.edgy.privacy.privacy

import com.edgy.privacy.audio.MelSpectrogramExtractor
import com.edgy.privacy.ml.EdgyEncoder
import com.edgy.privacy.vocoder.GriffinLimVocoder

/**
 * Orchestrates audio processing based on the selected privacy tier.
 *
 * - LOW:      Passthrough — returns raw audio unchanged
 * - MODERATE: Mel → ONNX encode → VQ embedding + speaker embedding
 * - HIGH:     Mel → ONNX encode → VQ embedding only (no speaker identity)
 */
class PrivacyPipeline(
    private val melExtractor: MelSpectrogramExtractor,
    private val encoder: EdgyEncoder,
    private val speakerEmbeddings: Array<FloatArray>,
    private var speakerIndex: Int = 0,
    private val vocoder: GriffinLimVocoder? = null
) {

    private var currentTier: PrivacyTier = PrivacyTier.LOW

    /**
     * Set the active privacy tier.
     */
    fun setTier(tier: PrivacyTier) {
        currentTier = tier
    }

    /**
     * Get the current privacy tier.
     */
    fun getTier(): PrivacyTier = currentTier

    /**
     * Set the speaker index for MODERATE tier.
     */
    fun setSpeakerIndex(index: Int) {
        require(index in speakerEmbeddings.indices) {
            "Speaker index $index out of range [0, ${speakerEmbeddings.size})"
        }
        speakerIndex = index
    }

    /**
     * Process a PCM audio chunk through the privacy pipeline.
     *
     * @param pcm PCM float samples in [-1, 1] range
     * @return PrivacyOutput containing the filtered data appropriate for the tier
     */
    fun processChunk(pcm: FloatArray): PrivacyOutput {
        val startTime = System.nanoTime()

        val output = when (currentTier) {
            PrivacyTier.LOW -> {
                PrivacyOutput(
                    rawAudio = pcm,
                    vqEmbedding = null,
                    speakerEmbedding = null,
                    codebookIndices = null,
                    tier = PrivacyTier.LOW,
                    processingTimeMs = 0
                )
            }
            PrivacyTier.MODERATE -> {
                val mel = melExtractor.extract(pcm)
                val encoderOutput = encoder.encode(mel)
                val speakerEmb = if (speakerEmbeddings.isNotEmpty()) {
                    speakerEmbeddings[speakerIndex]
                } else {
                    null
                }
                val reconstructed = vocoder?.synthesize(mel)
                PrivacyOutput(
                    rawAudio = null,
                    reconstructedAudio = reconstructed,
                    vqEmbedding = encoderOutput.vqEmbedding,
                    speakerEmbedding = speakerEmb,
                    codebookIndices = encoderOutput.codebookIndices,
                    tier = PrivacyTier.MODERATE,
                    processingTimeMs = encoderOutput.inferenceTimeMs
                )
            }
            PrivacyTier.HIGH -> {
                val mel = melExtractor.extract(pcm)
                val encoderOutput = encoder.encode(mel)
                val reconstructed = vocoder?.synthesize(mel)
                PrivacyOutput(
                    rawAudio = null,
                    reconstructedAudio = reconstructed,
                    vqEmbedding = encoderOutput.vqEmbedding,
                    speakerEmbedding = null,
                    codebookIndices = encoderOutput.codebookIndices,
                    tier = PrivacyTier.HIGH,
                    processingTimeMs = encoderOutput.inferenceTimeMs
                )
            }
        }

        val totalTimeMs = (System.nanoTime() - startTime) / 1_000_000
        return output.copy(processingTimeMs = totalTimeMs)
    }

    /**
     * Whether vocoder synthesis is available.
     */
    fun isVocoderAvailable(): Boolean = vocoder?.isAvailable == true

    /**
     * Process a PCM chunk in streaming mode (maintains overlap buffers).
     */
    fun processChunkStreaming(pcmChunk: FloatArray): PrivacyOutput {
        val startTime = System.nanoTime()

        val output = when (currentTier) {
            PrivacyTier.LOW -> {
                PrivacyOutput(
                    rawAudio = pcmChunk,
                    tier = PrivacyTier.LOW,
                    processingTimeMs = 0
                )
            }
            PrivacyTier.MODERATE -> {
                val mel = melExtractor.extractStreaming(pcmChunk)
                if (mel[0].isEmpty()) {
                    return PrivacyOutput(tier = PrivacyTier.MODERATE, processingTimeMs = 0)
                }
                val encoderOutput = encoder.encode(mel)
                val speakerEmb = if (speakerEmbeddings.isNotEmpty()) {
                    speakerEmbeddings[speakerIndex]
                } else {
                    null
                }
                val reconstructed = vocoder?.synthesize(mel)
                PrivacyOutput(
                    reconstructedAudio = reconstructed,
                    vqEmbedding = encoderOutput.vqEmbedding,
                    speakerEmbedding = speakerEmb,
                    codebookIndices = encoderOutput.codebookIndices,
                    tier = PrivacyTier.MODERATE,
                    processingTimeMs = encoderOutput.inferenceTimeMs
                )
            }
            PrivacyTier.HIGH -> {
                val mel = melExtractor.extractStreaming(pcmChunk)
                if (mel[0].isEmpty()) {
                    return PrivacyOutput(tier = PrivacyTier.HIGH, processingTimeMs = 0)
                }
                val encoderOutput = encoder.encode(mel)
                val reconstructed = vocoder?.synthesize(mel)
                PrivacyOutput(
                    reconstructedAudio = reconstructed,
                    vqEmbedding = encoderOutput.vqEmbedding,
                    codebookIndices = encoderOutput.codebookIndices,
                    tier = PrivacyTier.HIGH,
                    processingTimeMs = encoderOutput.inferenceTimeMs
                )
            }
        }

        val totalTimeMs = (System.nanoTime() - startTime) / 1_000_000
        return output.copy(processingTimeMs = totalTimeMs)
    }
}

/**
 * Output of the privacy pipeline for a single chunk.
 */
data class PrivacyOutput(
    val rawAudio: FloatArray? = null,
    val reconstructedAudio: FloatArray? = null,
    val vqEmbedding: Array<FloatArray>? = null,
    val speakerEmbedding: FloatArray? = null,
    val codebookIndices: IntArray? = null,
    val tier: PrivacyTier,
    val processingTimeMs: Long
) {
    /**
     * Serialize embedding data for file output (MODERATE/HIGH tiers).
     * Returns a map suitable for saving as multiple .npy files.
     */
    fun toSerializableMap(): Map<String, Any> {
        val map = mutableMapOf<String, Any>()
        map["tier"] = tier.name
        rawAudio?.let { map["raw_audio"] = it }
        reconstructedAudio?.let { map["reconstructed_audio"] = it }
        vqEmbedding?.let { map["vq_embedding"] = it }
        speakerEmbedding?.let { map["speaker_embedding"] = it }
        codebookIndices?.let { map["codebook_indices"] = it }
        return map
    }

    /**
     * Summary string for display.
     */
    fun summary(): String = buildString {
        append("Tier: ${tier.name}\n")
        append("Processing: ${processingTimeMs}ms\n")
        when (tier) {
            PrivacyTier.LOW -> {
                append("Output: raw audio (${rawAudio?.size ?: 0} samples)\n")
            }
            PrivacyTier.MODERATE -> {
                val vqShape = vqEmbedding?.let { "[${it.size}, ${it.firstOrNull()?.size ?: 0}]" } ?: "null"
                val spkShape = speakerEmbedding?.let { "[${it.size}]" } ?: "null"
                append("VQ embedding: $vqShape\n")
                append("Speaker embedding: $spkShape\n")
                append("Codebook indices: ${codebookIndices?.size ?: 0} codes\n")
                append("Reconstructed audio: ${reconstructedAudio?.size ?: 0} samples\n")
            }
            PrivacyTier.HIGH -> {
                val vqShape = vqEmbedding?.let { "[${it.size}, ${it.firstOrNull()?.size ?: 0}]" } ?: "null"
                append("VQ embedding: $vqShape\n")
                append("Speaker embedding: null (stripped)\n")
                append("Codebook indices: ${codebookIndices?.size ?: 0} codes\n")
                append("Reconstructed audio: ${reconstructedAudio?.size ?: 0} samples\n")
            }
        }
    }
}
