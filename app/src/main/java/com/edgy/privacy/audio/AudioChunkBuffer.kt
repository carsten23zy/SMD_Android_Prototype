package com.edgy.privacy.audio

import java.util.concurrent.ArrayBlockingQueue

/**
 * Lock-free ring buffer connecting audio capture and processing threads.
 * Uses ArrayBlockingQueue for thread safety.
 *
 * @param capacity Number of chunks the buffer can hold (default: 16)
 */
class AudioChunkBuffer(private val capacity: Int = 16) {

    private val queue = ArrayBlockingQueue<FloatArray>(capacity)

    /**
     * Push a chunk into the buffer.
     * @return true if the chunk was added, false if the buffer is full (chunk dropped)
     */
    fun push(chunk: FloatArray): Boolean {
        return queue.offer(chunk)
    }

    /**
     * Pull a chunk from the buffer.
     * @return The next chunk, or null if the buffer is empty
     */
    fun pull(): FloatArray? {
        return queue.poll()
    }

    /**
     * Number of chunks currently in the buffer.
     */
    fun size(): Int = queue.size

    /**
     * Whether the buffer is empty.
     */
    fun isEmpty(): Boolean = queue.isEmpty()

    /**
     * Whether the buffer is full.
     */
    fun isFull(): Boolean = queue.size >= capacity

    /**
     * Clear all chunks from the buffer.
     */
    fun clear() {
        queue.clear()
    }
}
