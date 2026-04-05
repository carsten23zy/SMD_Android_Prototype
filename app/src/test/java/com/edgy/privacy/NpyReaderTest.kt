package com.edgy.privacy

import com.edgy.privacy.util.NpyReader
import org.junit.Assert.*
import org.junit.Test
import java.io.ByteArrayOutputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * Unit tests for NpyReader.
 */
class NpyReaderTest {

    /**
     * Create a synthetic .npy file in memory for testing.
     */
    private fun createNpyBytes(
        shape: IntArray,
        dtype: String = "<f4",
        data: FloatArray
    ): ByteArray {
        val header = "{'descr': '$dtype', 'fortran_order': False, 'shape': (${shape.joinToString(", ")}), }"
        // Pad header to multiple of 64 bytes (including magic + version + header length)
        val preambleLen = 10 // magic(6) + version(2) + headerLen(2)
        val paddedLen = ((preambleLen + header.length + 1 + 63) / 64) * 64
        val headerPadded = header.padEnd(paddedLen - preambleLen - 1) + "\n"

        val baos = ByteArrayOutputStream()
        // Magic
        baos.write(byteArrayOf(0x93.toByte(), 'N'.code.toByte(), 'U'.code.toByte(), 'M'.code.toByte(), 'P'.code.toByte(), 'Y'.code.toByte()))
        // Version 1.0
        baos.write(byteArrayOf(1, 0))
        // Header length (little-endian short)
        val headerLenBytes = ByteBuffer.allocate(2).order(ByteOrder.LITTLE_ENDIAN)
            .putShort(headerPadded.length.toShort()).array()
        baos.write(headerLenBytes)
        // Header
        baos.write(headerPadded.toByteArray())
        // Data
        val dataBuffer = ByteBuffer.allocate(data.size * 4).order(ByteOrder.LITTLE_ENDIAN)
        for (v in data) dataBuffer.putFloat(v)
        baos.write(dataBuffer.array())

        return baos.toByteArray()
    }

    @Test
    fun testRead1DFloat() {
        val data = floatArrayOf(1f, 2f, 3f, 4f, 5f)
        val npyBytes = createNpyBytes(intArrayOf(5), "<f4", data)
        val result = NpyReader.read(npyBytes.inputStream())

        assertTrue(result.isFloat)
        assertArrayEquals(intArrayOf(5), result.shape)
        assertArrayEquals(data, result.floatData!!, 1e-6f)
    }

    @Test
    fun testRead2DFloat() {
        val data = floatArrayOf(1f, 2f, 3f, 4f, 5f, 6f)
        val npyBytes = createNpyBytes(intArrayOf(2, 3), "<f4", data)
        val result = NpyReader.read(npyBytes.inputStream())

        assertTrue(result.isFloat)
        assertArrayEquals(intArrayOf(2, 3), result.shape)

        val matrix = result.toFloatMatrix()
        assertEquals(2, matrix.size)
        assertEquals(3, matrix[0].size)
        assertArrayEquals(floatArrayOf(1f, 2f, 3f), matrix[0], 1e-6f)
        assertArrayEquals(floatArrayOf(4f, 5f, 6f), matrix[1], 1e-6f)
    }

    @Test
    fun testReadFloatMatrix() {
        val data = floatArrayOf(0.1f, 0.2f, 0.3f, 0.4f)
        val npyBytes = createNpyBytes(intArrayOf(2, 2), "<f4", data)
        val (shape, flatData) = NpyReader.readFloatMatrix(npyBytes.inputStream())

        assertArrayEquals(intArrayOf(2, 2), shape)
        assertArrayEquals(data, flatData, 1e-6f)
    }
}
