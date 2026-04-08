package com.edgy.privacy

import com.edgy.privacy.audio.AudioChunkBuffer
import org.junit.Assert.*
import org.junit.Test

/**
 * Unit tests for AudioChunkBuffer.
 */
class AudioChunkBufferTest {

    @Test
    fun testPushAndPull() {
        val buffer = AudioChunkBuffer(capacity = 4)

        val chunk1 = floatArrayOf(1f, 2f, 3f)
        val chunk2 = floatArrayOf(4f, 5f, 6f)

        assertTrue(buffer.push(chunk1))
        assertTrue(buffer.push(chunk2))
        assertEquals(2, buffer.size())

        val pulled1 = buffer.pull()
        assertNotNull(pulled1)
        assertArrayEquals(chunk1, pulled1!!, 0f)

        val pulled2 = buffer.pull()
        assertNotNull(pulled2)
        assertArrayEquals(chunk2, pulled2!!, 0f)

        assertNull(buffer.pull())
    }

    @Test
    fun testCapacityLimit() {
        val buffer = AudioChunkBuffer(capacity = 2)

        assertTrue(buffer.push(floatArrayOf(1f)))
        assertTrue(buffer.push(floatArrayOf(2f)))
        assertFalse(buffer.push(floatArrayOf(3f))) // full
        assertTrue(buffer.isFull())
    }

    @Test
    fun testEmpty() {
        val buffer = AudioChunkBuffer()
        assertTrue(buffer.isEmpty())
        assertNull(buffer.pull())
    }

    @Test
    fun testClear() {
        val buffer = AudioChunkBuffer()
        buffer.push(floatArrayOf(1f))
        buffer.push(floatArrayOf(2f))
        assertEquals(2, buffer.size())

        buffer.clear()
        assertTrue(buffer.isEmpty())
        assertEquals(0, buffer.size())
    }

    @Test
    fun testFIFOOrder() {
        val buffer = AudioChunkBuffer(capacity = 8)
        for (i in 0 until 5) {
            buffer.push(floatArrayOf(i.toFloat()))
        }

        for (i in 0 until 5) {
            val chunk = buffer.pull()
            assertNotNull(chunk)
            assertEquals(i.toFloat(), chunk!![0], 0f)
        }
    }
}