package com.edgy.privacy.sdk

import android.os.ParcelFileDescriptor
import android.util.Log
import java.io.FileOutputStream
import java.io.IOException
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.concurrent.atomic.AtomicBoolean

/**
 * Wraps a ParcelFileDescriptor pipe for streaming PCM audio to AIDL clients.
 *
 * Creates a pipe pair: the write-end stays in this process, the read-end
 * is handed to the client via AIDL. Audio samples are written as 16-bit
 * PCM (little-endian, mono, 16kHz).
 *
 * Thread-safe: write() can be called from the processing thread while
 * the client reads from the other end.
 */
class AudioStreamPipe {

    companion object {
        private const val TAG = "AudioStreamPipe"
        private const val WRITE_BUFFER_SIZE = 4096 // bytes
    }

    private var writeFd: ParcelFileDescriptor? = null
    private var readFd: ParcelFileDescriptor? = null
    private var outputStream: FileOutputStream? = null
    private val isOpen = AtomicBoolean(false)

    /**
     * Create the pipe and return the read-end for the AIDL client.
     *
     * @return Read-end ParcelFileDescriptor to hand to the client
     * @throws IOException if pipe creation fails
     */
    fun open(): ParcelFileDescriptor {
        if (isOpen.get()) {
            close()
        }

        val pipe = ParcelFileDescriptor.createPipe()
        readFd = pipe[0]
        writeFd = pipe[1]
        outputStream = FileOutputStream(writeFd!!.fileDescriptor)
        isOpen.set(true)

        Log.i(TAG, "Audio pipe opened")
        return readFd!!
    }

    /**
     * Write PCM float samples to the pipe as 16-bit PCM.
     *
     * @param pcm Float samples in [-1, 1] range
     * @return true if write succeeded, false if pipe is closed/broken
     */
    fun write(pcm: FloatArray): Boolean {
        if (!isOpen.get()) return false

        val stream = outputStream ?: return false
        val byteBuffer = ByteBuffer.allocate(pcm.size * 2).order(ByteOrder.LITTLE_ENDIAN)

        for (sample in pcm) {
            val clamped = sample.coerceIn(-1f, 1f)
            val shortVal = (clamped * 32767f).toInt().toShort()
            byteBuffer.putShort(shortVal)
        }

        return try {
            stream.write(byteBuffer.array())
            stream.flush()
            true
        } catch (e: IOException) {
            Log.w(TAG, "Pipe write failed (client may have disconnected): ${e.message}")
            close()
            false
        }
    }

    /**
     * Whether the pipe is currently open.
     */
    fun isOpen(): Boolean = isOpen.get()

    /**
     * Close both ends of the pipe.
     */
    fun close() {
        isOpen.set(false)

        try { outputStream?.close() } catch (_: Exception) {}
        try { writeFd?.close() } catch (_: Exception) {}
        // Note: readFd is owned by the client after handoff; don't close it here
        // unless it was never handed off
        outputStream = null
        writeFd = null
        readFd = null

        Log.i(TAG, "Audio pipe closed")
    }
}
