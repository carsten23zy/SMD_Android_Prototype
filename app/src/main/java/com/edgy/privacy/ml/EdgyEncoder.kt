package com.edgy.privacy.ml

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import java.nio.FloatBuffer
import java.nio.LongBuffer

/**
 * Runs ONNX inference on mel spectrogram input.
 *
 * Input:  "mel_spectrogram" float32 [batch=1, 80, T]
 * Output: "vq_embedding"    float32 [batch, T/2, 64]
 *         "codebook_indices" int64  [batch * T/2]
 */
class EdgyEncoder(
    private val env: OrtEnvironment,
    private val session: OrtSession,
    private val config: ModelConfig
) {

    data class EncoderOutput(
        val vqEmbedding: Array<FloatArray>,   // [T', 64]
        val codebookIndices: IntArray,         // [T']
        val inferenceTimeMs: Long
    )

    /**
     * Run encoder inference on a mel spectrogram.
     *
     * @param mel Mel spectrogram [nMels][T] — output of MelSpectrogramExtractor
     * @return EncoderOutput with VQ embeddings and codebook indices
     */
    fun encode(mel: Array<FloatArray>): EncoderOutput {
        val nMels = mel.size
        val timeSteps = if (nMels > 0) mel[0].size else 0
        require(nMels == config.preprocessing.nMels) {
            "Expected ${config.preprocessing.nMels} mel bins, got $nMels"
        }
        require(timeSteps > 0) { "Mel spectrogram has no time steps" }

        // Reshape [nMels, T] → flat [1, nMels, T] for ONNX
        val flatMel = FloatArray(nMels * timeSteps)
        for (m in 0 until nMels) {
            System.arraycopy(mel[m], 0, flatMel, m * timeSteps, timeSteps)
        }

        val shape = longArrayOf(1, nMels.toLong(), timeSteps.toLong())
        /** val env = session.environment **/

        val startTime = System.nanoTime()

        val inputTensor = OnnxTensor.createTensor(
            env,
            FloatBuffer.wrap(flatMel),
            shape
        )

        val results = session.run(mapOf("mel_spectrogram" to inputTensor))
        val inferenceTimeMs = (System.nanoTime() - startTime) / 1_000_000

        // Parse vq_embedding output
        val vqTensor = results.get("vq_embedding")
        val vqValue = vqTensor.get().value
        val vqEmbedding = parseVqEmbedding(vqValue)

        // Parse codebook_indices output
        val indicesTensor = results.get("codebook_indices")
        val indicesValue = indicesTensor.get().value
        val codebookIndices = parseIndices(indicesValue)

        // Clean up
        inputTensor.close()
        results.close()

        return EncoderOutput(
            vqEmbedding = vqEmbedding,
            codebookIndices = codebookIndices,
            inferenceTimeMs = inferenceTimeMs
        )
    }

    /**
     * Parse the VQ embedding output tensor.
     * Expected shape: [1, T', 64] → extract as [T', 64]
     */
    private fun parseVqEmbedding(value: Any): Array<FloatArray> {
        return when (value) {
            is Array<*> -> {
                // [batch][T'][64]
                @Suppress("UNCHECKED_CAST")
                val batch = value as Array<Array<FloatArray>>
                batch[0]
            }
            is FloatArray -> {
                // Flat array, reshape to [T', embeddingDim]
                val embDim = config.encoder.embeddingDim
                val tPrime = value.size / embDim
                Array(tPrime) { t ->
                    FloatArray(embDim) { d -> value[t * embDim + d] }
                }
            }
            else -> throw IllegalArgumentException(
                "Unexpected vq_embedding type: ${value::class.java}"
            )
        }
    }

    /**
     * Parse codebook indices output tensor.
     * Expected shape: [batch * T'] as int64 → convert to IntArray
     */
    private fun parseIndices(value: Any): IntArray {
        return when (value) {
            is LongArray -> IntArray(value.size) { value[it].toInt() }
            is IntArray -> value
            is Array<*> -> {
                // Nested array: flatten
                @Suppress("UNCHECKED_CAST")
                val longArrays = value as Array<LongArray>
                longArrays.flatMap { it.toList() }.map { it.toInt() }.toIntArray()
            }
            else -> throw IllegalArgumentException(
                "Unexpected codebook_indices type: ${value::class.java}"
            )
        }
    }

    /**
     * Close the ONNX session.
     */
    fun close() {
        session.close()
    }
}
