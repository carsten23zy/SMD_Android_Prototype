package com.edgy.privacy.util

import java.io.InputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * Minimal .npy file parser for reading NumPy arrays exported from the training notebook.
 * Supports float32 and int64 dtypes in C-contiguous (row-major) order.
 */
object NpyReader {

    data class NpyArray(
        val shape: IntArray,
        val floatData: FloatArray?,
        val longData: LongArray?,
        val dtype: String
    ) {
        val isFloat: Boolean get() = floatData != null
        val isLong: Boolean get() = longData != null

        fun toFloatMatrix(): Array<FloatArray> {
            require(isFloat && shape.size == 2) {
                "Expected 2D float array, got shape=${shape.contentToString()}, dtype=$dtype"
            }
            val rows = shape[0]
            val cols = shape[1]
            return Array(rows) { r ->
                FloatArray(cols) { c -> floatData!![r * cols + c] }
            }
        }

        fun toFloat2D(): Array<FloatArray> = toFloatMatrix()

        fun toIntArray(): IntArray {
            require(isLong) { "Expected int/long array, got dtype=$dtype" }
            return IntArray(longData!!.size) { longData[it].toInt() }
        }
    }

    /**
     * Parse a .npy file from an InputStream.
     * Returns shape and data as NpyArray.
     */
    fun read(inputStream: InputStream): NpyArray {
        val bytes = inputStream.readBytes()
        val buffer = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN)

        // Magic: \x93NUMPY
        val magic = ByteArray(6)
        buffer.get(magic)
        require(magic[0] == 0x93.toByte() && String(magic, 1, 5) == "NUMPY") {
            "Not a valid .npy file"
        }

        val majorVersion = buffer.get().toInt() and 0xFF
        val minorVersion = buffer.get().toInt() and 0xFF

        // Header length
        val headerLen = if (majorVersion >= 2) {
            buffer.int
        } else {
            (buffer.short.toInt() and 0xFFFF)
        }

        val headerBytes = ByteArray(headerLen)
        buffer.get(headerBytes)
        val header = String(headerBytes).trim()

        // Parse dtype
        val dtype = parseDtype(header)
        val shape = parseShape(header)
        val isFortranOrder = header.contains("'fortran_order': True") ||
                header.contains("'fortran_order':True")

        require(!isFortranOrder) { "Fortran-order arrays not supported" }

        // Calculate total elements
        val totalElements = if (shape.isEmpty()) 1 else shape.fold(1) { acc, v -> acc * v }

        return when {
            dtype.contains("f4") || dtype.contains("float32") -> {
                val data = FloatArray(totalElements)
                for (i in 0 until totalElements) {
                    data[i] = buffer.float
                }
                NpyArray(shape, floatData = data, longData = null, dtype = "float32")
            }
            dtype.contains("f8") || dtype.contains("float64") -> {
                val data = FloatArray(totalElements)
                for (i in 0 until totalElements) {
                    data[i] = buffer.double.toFloat()
                }
                NpyArray(shape, floatData = data, longData = null, dtype = "float64")
            }
            dtype.contains("i8") || dtype.contains("int64") -> {
                val data = LongArray(totalElements)
                for (i in 0 until totalElements) {
                    data[i] = buffer.long
                }
                NpyArray(shape, floatData = null, longData = data, dtype = "int64")
            }
            dtype.contains("i4") || dtype.contains("int32") -> {
                val data = LongArray(totalElements)
                for (i in 0 until totalElements) {
                    data[i] = buffer.int.toLong()
                }
                NpyArray(shape, floatData = null, longData = data, dtype = "int32")
            }
            else -> throw IllegalArgumentException("Unsupported dtype: $dtype")
        }
    }

    /**
     * Convenience: read a float matrix [rows, cols] from .npy stream.
     */
    fun readFloatMatrix(inputStream: InputStream): Pair<IntArray, FloatArray> {
        val npy = read(inputStream)
        require(npy.isFloat) { "Expected float array, got dtype=${npy.dtype}" }
        return Pair(npy.shape, npy.floatData!!)
    }

    private fun parseDtype(header: String): String {
        // Match 'descr': '<f4' or similar
        val regex = Regex("'descr'\\s*:\\s*'([^']+)'")
        val match = regex.find(header)
            ?: throw IllegalArgumentException("Could not parse dtype from header: $header")
        return match.groupValues[1]
    }

    private fun parseShape(header: String): IntArray {
        // Match 'shape': (80, 1025) or (512,) or ()
        val regex = Regex("'shape'\\s*:\\s*\\(([^)]*)\\)")
        val match = regex.find(header)
            ?: throw IllegalArgumentException("Could not parse shape from header: $header")
        val shapeStr = match.groupValues[1].trim()
        if (shapeStr.isEmpty()) return intArrayOf()
        return shapeStr.split(",")
            .map { it.trim() }
            .filter { it.isNotEmpty() }
            .map { it.toInt() }
            .toIntArray()
    }
}
