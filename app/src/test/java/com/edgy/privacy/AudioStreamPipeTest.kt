package com.edgy.privacy

import com.edgy.privacy.sdk.AudioStreamPipe
import org.junit.Assert.*
import org.junit.Test

/**
 * Unit tests for AudioStreamPipe.
 * Note: Full pipe testing requires Android instrumented tests since
 * ParcelFileDescriptor is an Android framework class. These tests
 * verify the basic state management.
 */
class AudioStreamPipeTest {

    @Test
    fun `pipe starts closed`() {
        val pipe = AudioStreamPipe()
        assertFalse(pipe.isOpen())
    }

    @Test
    fun `write returns false when pipe is closed`() {
        val pipe = AudioStreamPipe()
        val result = pipe.write(FloatArray(100) { 0.5f })
        assertFalse(result)
    }

    @Test
    fun `close on already closed pipe does not throw`() {
        val pipe = AudioStreamPipe()
        pipe.close() // should not throw
        pipe.close() // double close should not throw
    }
}
