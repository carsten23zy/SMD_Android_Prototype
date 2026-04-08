package com.edgy.privacy.util

import java.io.File
import java.io.FileOutputStream
import java.io.InputStream
import java.io.RandomAccessFile
import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * Minimal WAV file reader/writer for 16kHz mono PCM audio.
 */
object WavWriter {

    /**
     * Write PCM float samples to a WAV file.
     * @param path Output file path
     * @param pcm Audio samples in [-1.0, 1.0] range
     * @param sampleRate Sample rate in Hz (default 16000)
     * @param bitsPerSample Bits per sample (default 16)
     */
    fun writeWav(path: String, pcm: FloatArray, sampleRate: Int = 16000, bitsPerSample: Int = 16) {
        val numChannels = 1
        val byteRate = sampleRate * numChannels * bitsPerSample / 8
        val blockAlign = numChannels * bitsPerSample / 8
        val dataSize = pcm.size * bitsPerSample / 8
        val fileSize = 36 + dataSize

        FileOutputStream(File(path)).use { fos ->
            val buffer = ByteBuffer.allocate(44).order(ByteOrder.LITTLE_ENDIAN)

            // RIFF header
            buffer.put("RIFF".toByteArray())
            buffer.putInt(fileSize)
            buffer.put("WAVE".toByteArray())

            // fmt chunk
            buffer.put("fmt ".toByteArray())
            buffer.putInt(16) // chunk size
            buffer.putShort(1) // PCM format
            buffer.putShort(numChannels.toShort())
            buffer.putInt(sampleRate)
            buffer.putInt(byteRate)
            buffer.putShort(blockAlign.toShort())
            buffer.putShort(bitsPerSample.toShort())

            // data chunk
            buffer.put("data".toByteArray())
            buffer.putInt(dataSize)

            fos.write(buffer.array())

            // Write PCM data
            val dataBuffer = ByteBuffer.allocate(pcm.size * 2).order(ByteOrder.LITTLE_ENDIAN)
            for (sample in pcm) {
                val clamped = sample.coerceIn(-1f, 1f)
                val scaled = (clamped * 32768f).toInt().coerceIn(-32768, 32767)
                dataBuffer.putShort(scaled.toShort())
            }
            fos.write(dataBuffer.array())
        }
    }

    /**
     * Read a WAV file and return PCM float samples in [-1.0, 1.0] range.
     * Supports 16-bit PCM WAV files.
     * @param inputStream Input stream of the WAV file
     * @return Pair of (sampleRate, pcmFloat)
     */
    fun readWav(inputStream: InputStream): Pair<Int, FloatArray> {
        val bytes = inputStream.readBytes()
        val buffer = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN)

        // RIFF header
        val riff = ByteArray(4)
        buffer.get(riff)
        require(String(riff) == "RIFF") { "Not a RIFF file" }

        buffer.int // file size
        val wave = ByteArray(4)
        buffer.get(wave)
        require(String(wave) == "WAVE") { "Not a WAVE file" }

        var sampleRate = 16000
        var bitsPerSample = 16
        var numChannels = 1
        var pcmData: FloatArray? = null

        // Read chunks
        while (buffer.hasRemaining()) {
            val chunkId = ByteArray(4)
            if (buffer.remaining() < 8) break
            buffer.get(chunkId)
            val chunkSize = buffer.int
            val chunkName = String(chunkId)

            when (chunkName) {
                "fmt " -> {
                    val audioFormat = buffer.short.toInt()
                    numChannels = buffer.short.toInt()
                    sampleRate = buffer.int
                    buffer.int // byteRate
                    buffer.short // blockAlign
                    bitsPerSample = buffer.short.toInt()
                    // Skip any extra fmt bytes
                    val extraBytes = chunkSize - 16
                    if (extraBytes > 0) {
                        buffer.position(buffer.position() + extraBytes)
                    }
                    require(audioFormat == 1) { "Only PCM format supported, got $audioFormat" }
                }
                "data" -> {
                    val numSamples = chunkSize / (bitsPerSample / 8) / numChannels
                    pcmData = FloatArray(numSamples)
                    for (i in 0 until numSamples) {
                        when (bitsPerSample) {
                            16 -> {
                                var sample = 0f
                                for (ch in 0 until numChannels) {
                                    sample += buffer.short.toFloat() / 32768f
                                }
                                pcmData[i] = sample / numChannels
                            }
                            else -> throw IllegalArgumentException(
                                "Unsupported bits per sample: $bitsPerSample"
                            )
                        }
                    }
                }
                else -> {
                    // Skip unknown chunks
                    if (buffer.remaining() >= chunkSize) {
                        buffer.position(buffer.position() + chunkSize)
                    } else {
                        break
                    }
                }
            }
        }

        require(pcmData != null) { "No data chunk found in WAV file" }
        return Pair(sampleRate, pcmData)
    }

    /**
     * Read a WAV file from a file path.
     */
    fun readWav(path: String): Pair<Int, FloatArray> {
        return File(path).inputStream().use { readWav(it) }
    }
}
