package com.edgy.privacy.audio

import com.edgy.privacy.privacy.PrivacyOutput
import com.edgy.privacy.privacy.PrivacyTier
import com.edgy.privacy.util.WavWriter
import java.io.DataOutputStream
import java.io.File
import java.io.FileOutputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * Routes privacy pipeline output to files.
 * Supports WAV output (LOW tier) and embedding serialization (MODERATE/HIGH tiers).
 */
class AudioOutputService {

    /**
     * Write raw audio to a WAV file (for LOW tier).
     */
    fun writeToWavFile(path: String, pcm: FloatArray, sampleRate: Int = 16000) {
        WavWriter.writeWav(path, pcm, sampleRate)
    }

    /**
     * Write privacy pipeline output to files based on tier.
     *
     * - LOW: saves as .wav
     * - MODERATE: saves vq_embedding.bin, speaker_embedding.bin, codebook_indices.bin
     * - HIGH: saves vq_embedding.bin, codebook_indices.bin
     *
     * @param directory Output directory
     * @param output Privacy pipeline output
     * @param prefix File name prefix (e.g., "utterance_001")
     * @param sampleRate Sample rate for WAV output
     */
    fun writeOutput(
        directory: String,
        output: PrivacyOutput,
        prefix: String = "output",
        sampleRate: Int = 16000
    ): List<String> {
        val dir = File(directory)
        if (!dir.exists()) dir.mkdirs()

        val savedFiles = mutableListOf<String>()

        when (output.tier) {
            PrivacyTier.LOW -> {
                output.rawAudio?.let { pcm ->
                    val path = File(dir, "${prefix}.wav").absolutePath
                    writeToWavFile(path, pcm, sampleRate)
                    savedFiles.add(path)
                }
            }
            PrivacyTier.MODERATE -> {
                // Save reconstructed audio if vocoder produced it
                output.reconstructedAudio?.let { pcm ->
                    if (pcm.isNotEmpty()) {
                        val path = File(dir, "${prefix}_reconstructed.wav").absolutePath
                        writeToWavFile(path, pcm, sampleRate)
                        savedFiles.add(path)
                    }
                }
                output.vqEmbedding?.let { vq ->
                    val path = File(dir, "${prefix}_vq_embedding.bin").absolutePath
                    writeFloatMatrix(path, vq)
                    savedFiles.add(path)
                }
                output.speakerEmbedding?.let { spk ->
                    val path = File(dir, "${prefix}_speaker_embedding.bin").absolutePath
                    writeFloatArray(path, spk)
                    savedFiles.add(path)
                }
                output.codebookIndices?.let { idx ->
                    val path = File(dir, "${prefix}_codebook_indices.bin").absolutePath
                    writeIntArray(path, idx)
                    savedFiles.add(path)
                }
                // Write metadata
                val metaPath = File(dir, "${prefix}_meta.txt").absolutePath
                File(metaPath).writeText(output.summary())
                savedFiles.add(metaPath)
            }
            PrivacyTier.HIGH -> {
                // Save reconstructed audio if vocoder produced it
                output.reconstructedAudio?.let { pcm ->
                    if (pcm.isNotEmpty()) {
                        val path = File(dir, "${prefix}_reconstructed.wav").absolutePath
                        writeToWavFile(path, pcm, sampleRate)
                        savedFiles.add(path)
                    }
                }
                output.vqEmbedding?.let { vq ->
                    val path = File(dir, "${prefix}_vq_embedding.bin").absolutePath
                    writeFloatMatrix(path, vq)
                    savedFiles.add(path)
                }
                output.codebookIndices?.let { idx ->
                    val path = File(dir, "${prefix}_codebook_indices.bin").absolutePath
                    writeIntArray(path, idx)
                    savedFiles.add(path)
                }
                val metaPath = File(dir, "${prefix}_meta.txt").absolutePath
                File(metaPath).writeText(output.summary())
                savedFiles.add(metaPath)
            }
        }

        return savedFiles
    }

    /**
     * Write a 2D float array [rows][cols] to binary file.
     * Format: [rows: int32][cols: int32][data: float32...]
     */
    private fun writeFloatMatrix(path: String, data: Array<FloatArray>) {
        DataOutputStream(FileOutputStream(path)).use { dos ->
            val rows = data.size
            val cols = if (rows > 0) data[0].size else 0
            val buffer = ByteBuffer.allocate(8 + rows * cols * 4).order(ByteOrder.LITTLE_ENDIAN)
            buffer.putInt(rows)
            buffer.putInt(cols)
            for (row in data) {
                for (value in row) {
                    buffer.putFloat(value)
                }
            }
            dos.write(buffer.array())
        }
    }

    /**
     * Write a 1D float array to binary file.
     * Format: [length: int32][data: float32...]
     */
    private fun writeFloatArray(path: String, data: FloatArray) {
        DataOutputStream(FileOutputStream(path)).use { dos ->
            val buffer = ByteBuffer.allocate(4 + data.size * 4).order(ByteOrder.LITTLE_ENDIAN)
            buffer.putInt(data.size)
            for (value in data) {
                buffer.putFloat(value)
            }
            dos.write(buffer.array())
        }
    }

    /**
     * Write an int array to binary file.
     * Format: [length: int32][data: int32...]
     */
    private fun writeIntArray(path: String, data: IntArray) {
        DataOutputStream(FileOutputStream(path)).use { dos ->
            val buffer = ByteBuffer.allocate(4 + data.size * 4).order(ByteOrder.LITTLE_ENDIAN)
            buffer.putInt(data.size)
            for (value in data) {
                buffer.putInt(value)
            }
            dos.write(buffer.array())
        }
    }
}
