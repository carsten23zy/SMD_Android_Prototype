package com.edgy.privacy

import android.app.Activity
import android.content.Intent
import android.net.Uri
import android.os.Bundle
import android.view.View
import android.widget.Button
import android.widget.ProgressBar
import android.widget.RadioGroup
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import com.edgy.privacy.audio.AudioOutputService
import com.edgy.privacy.audio.MelSpectrogramExtractor
import com.edgy.privacy.ml.EdgyEncoder
import com.edgy.privacy.ml.ModelManager
import com.edgy.privacy.privacy.PrivacyOutput
import com.edgy.privacy.privacy.PrivacyPipeline
import com.edgy.privacy.privacy.PrivacyTier
import com.edgy.privacy.util.WavWriter
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

/**
 * Main activity for EDGY Privacy Filter.
 *
 * Stage 1: Process bundled test WAV files through the offline pipeline.
 * Stage 2: Pick WAV files, process at selected privacy tier, save outputs.
 */
class MainActivity : AppCompatActivity() {

    companion object {
        private const val PICK_WAV_REQUEST = 1001
    }

    private lateinit var tvModelInfo: TextView
    private lateinit var tvStatus: TextView
    private lateinit var tvResults: TextView
    private lateinit var rgTier: RadioGroup
    private lateinit var btnProcessTest: Button
    private lateinit var btnPickFile: Button
    private lateinit var btnProcessFile: Button
    private lateinit var progressBar: ProgressBar

    private var modelManager: ModelManager? = null
    private var melExtractor: MelSpectrogramExtractor? = null
    private var encoder: EdgyEncoder? = null
    private var pipeline: PrivacyPipeline? = null
    private var audioOutputService = AudioOutputService()

    private var selectedFileUri: Uri? = null
    private var processingJob: Job? = null

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        tvModelInfo = findViewById(R.id.tvModelInfo)
        tvStatus = findViewById(R.id.tvStatus)
        tvResults = findViewById(R.id.tvResults)
        rgTier = findViewById(R.id.rgTier)
        btnProcessTest = findViewById(R.id.btnProcessTest)
        btnPickFile = findViewById(R.id.btnPickFile)
        btnProcessFile = findViewById(R.id.btnProcessFile)
        progressBar = findViewById(R.id.progressBar)

        rgTier.setOnCheckedChangeListener { _, checkedId ->
            val tier = when (checkedId) {
                R.id.rbLow -> PrivacyTier.LOW
                R.id.rbModerate -> PrivacyTier.MODERATE
                R.id.rbHigh -> PrivacyTier.HIGH
                else -> PrivacyTier.LOW
            }
            pipeline?.setTier(tier)
            updateStatus("Tier set to ${tier.name}")
        }

        btnProcessTest.setOnClickListener { processTestWav() }
        btnPickFile.setOnClickListener { pickWavFile() }
        btnProcessFile.setOnClickListener { processSelectedFile() }

        // Initialize model on background thread
        initializeModel()
    }

    override fun onDestroy() {
        super.onDestroy()
        processingJob?.cancel()
        modelManager?.close()
    }

    @Suppress("DEPRECATION")
    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)
        if (requestCode == PICK_WAV_REQUEST && resultCode == Activity.RESULT_OK) {
            selectedFileUri = data?.data
            if (selectedFileUri != null) {
                btnProcessFile.isEnabled = true
                updateStatus("File selected: ${selectedFileUri?.lastPathSegment}")
            }
        }
    }

    private fun initializeModel() {
        updateStatus("Loading model...")
        setProcessingEnabled(false)

        CoroutineScope(Dispatchers.IO).launch {
            try {
                val manager = ModelManager(this@MainActivity)
                val config = manager.loadConfig()

                val extractor = MelSpectrogramExtractor(config)

                // Try to load encoder — may fail if ONNX model not bundled yet
                var enc: EdgyEncoder? = null
                var speakerEmbeddings = emptyArray<FloatArray>()
                try {
                    enc = manager.loadEncoder(preferInt8 = false)
                } catch (e: Exception) {
                    withContext(Dispatchers.Main) {
                        appendResult("⚠ ONNX encoder not available: ${e.message}")
                        appendResult("  Mel spectrogram extraction will still work.")
                        appendResult("  Place ONNX model in assets/models/ to enable encoding.\n")
                    }
                }

                try {
                    speakerEmbeddings = manager.loadSpeakerEmbeddings()
                } catch (e: Exception) {
                    withContext(Dispatchers.Main) {
                        appendResult("⚠ Speaker embeddings not found: ${e.message}\n")
                    }
                }

                modelManager = manager
                melExtractor = extractor

                if (enc != null) {
                    encoder = enc
                    pipeline = PrivacyPipeline(extractor, enc, speakerEmbeddings)
                    // Set initial tier from radio buttons
                    val tier = getSelectedTier()
                    pipeline?.setTier(tier)
                }

                withContext(Dispatchers.Main) {
                    tvModelInfo.text = manager.getModelInfo()
                    updateStatus("Model loaded — ready")
                    setProcessingEnabled(true)
                }
            } catch (e: Exception) {
                withContext(Dispatchers.Main) {
                    updateStatus("Error loading model: ${e.message}")
                    appendResult("ERROR: ${e.stackTraceToString()}")
                }
            }
        }
    }

    /**
     * Stage 1: Process a bundled test WAV and show mel + encoder results.
     */
    private fun processTestWav() {
        setProcessingEnabled(false)
        progressBar.visibility = View.VISIBLE
        tvResults.text = ""
        updateStatus("Processing test WAV...")

        processingJob = CoroutineScope(Dispatchers.IO).launch {
            try {
                val results = StringBuilder()

                // Try to load test WAV from assets
                val wavAssets = assets.list("models")?.filter { it.endsWith(".wav") } ?: emptyList()

                if (wavAssets.isEmpty()) {
                    // Generate a synthetic test signal (440Hz sine wave, 1 second)
                    results.append("No test WAV found in assets. Using synthetic 440Hz tone.\n\n")
                    val sampleRate = modelManager?.loadConfig()?.preprocessing?.sampleRate ?: 16000
                    val duration = 1.0f
                    val numSamples = (sampleRate * duration).toInt()
                    val pcm = FloatArray(numSamples) { i ->
                        (0.5 * kotlin.math.sin(2.0 * Math.PI * 440.0 * i / sampleRate)).toFloat()
                    }
                    processAndReport(pcm, "synthetic_440hz", results)
                } else {
                    for (wavFile in wavAssets) {
                        val (sampleRate, pcm) = assets.open("models/$wavFile").use {
                            WavWriter.readWav(it)
                        }
                        results.append("=== $wavFile (${pcm.size} samples, ${sampleRate}Hz) ===\n")
                        processAndReport(pcm, wavFile.removeSuffix(".wav"), results)
                        results.append("\n")
                    }
                }

                withContext(Dispatchers.Main) {
                    tvResults.text = results.toString()
                    updateStatus("Processing complete")
                    progressBar.visibility = View.GONE
                    setProcessingEnabled(true)
                }
            } catch (e: Exception) {
                withContext(Dispatchers.Main) {
                    appendResult("ERROR: ${e.stackTraceToString()}")
                    updateStatus("Error during processing")
                    progressBar.visibility = View.GONE
                    setProcessingEnabled(true)
                }
            }
        }
    }

    /**
     * Stage 2: Pick a WAV file from device storage.
     */
    @Suppress("DEPRECATION")
    private fun pickWavFile() {
        val intent = Intent(Intent.ACTION_GET_CONTENT).apply {
            type = "audio/wav"
            addCategory(Intent.CATEGORY_OPENABLE)
        }
        try {
            startActivityForResult(Intent.createChooser(intent, "Select WAV file"), PICK_WAV_REQUEST)
        } catch (e: Exception) {
            Toast.makeText(this, "No file picker available", Toast.LENGTH_SHORT).show()
        }
    }

    /**
     * Stage 2: Process the selected WAV file at the chosen privacy tier.
     */
    private fun processSelectedFile() {
        val uri = selectedFileUri ?: return
        setProcessingEnabled(false)
        progressBar.visibility = View.VISIBLE
        tvResults.text = ""

        val tier = getSelectedTier()
        updateStatus("Processing at ${tier.name} tier...")

        processingJob = CoroutineScope(Dispatchers.IO).launch {
            try {
                val results = StringBuilder()
                val (sampleRate, pcm) = contentResolver.openInputStream(uri)!!.use {
                    WavWriter.readWav(it)
                }
                val fileName = uri.lastPathSegment ?: "selected_file"
                results.append("=== $fileName ===\n")
                results.append("Samples: ${pcm.size}, Rate: ${sampleRate}Hz\n")
                results.append("Duration: ${"%.2f".format(pcm.size.toFloat() / sampleRate)}s\n")
                results.append("Tier: ${tier.name}\n\n")

                pipeline?.setTier(tier)

                if (pipeline != null) {
                    val output = pipeline!!.processChunk(pcm)
                    results.append(output.summary())

                    // Save output to cache dir
                    val outputDir = "${cacheDir.absolutePath}/edgy_output"
                    val prefix = "output_${System.currentTimeMillis()}"
                    val savedFiles = audioOutputService.writeOutput(outputDir, output, prefix, sampleRate)
                    results.append("\nSaved files:\n")
                    savedFiles.forEach { results.append("  $it\n") }

                    // Stage 2 verification
                    results.append("\n--- Stage 2 Verification ---\n")
                    verifyTierOutput(output, tier, results)
                } else {
                    // Encoder not available, just compute mel
                    val mel = melExtractor!!.extract(pcm)
                    results.append("Mel spectrogram: [${mel.size}, ${mel[0].size}]\n")
                    results.append("(Encoder not loaded — ONNX model needed for full pipeline)\n")
                }

                withContext(Dispatchers.Main) {
                    tvResults.text = results.toString()
                    updateStatus("Processing complete")
                    progressBar.visibility = View.GONE
                    setProcessingEnabled(true)
                }
            } catch (e: Exception) {
                withContext(Dispatchers.Main) {
                    appendResult("ERROR: ${e.stackTraceToString()}")
                    updateStatus("Error during processing")
                    progressBar.visibility = View.GONE
                    setProcessingEnabled(true)
                }
            }
        }
    }

    /**
     * Process audio and report mel + encoder results.
     */
    private fun processAndReport(pcm: FloatArray, name: String, results: StringBuilder) {
        // Compute mel spectrogram
        val melStartTime = System.nanoTime()
        val mel = melExtractor!!.extract(pcm)
        val melTimeMs = (System.nanoTime() - melStartTime) / 1_000_000

        results.append("Mel spectrogram: [${mel.size}, ${mel[0].size}] (${melTimeMs}ms)\n")

        // Check mel value range
        var melMin = Float.MAX_VALUE
        var melMax = Float.MIN_VALUE
        for (row in mel) {
            for (v in row) {
                if (v < melMin) melMin = v
                if (v > melMax) melMax = v
            }
        }
        results.append("Mel range: [${"%.4f".format(melMin)}, ${"%.4f".format(melMax)}]\n")

        // Run encoder if available
        if (encoder != null) {
            try {
                val encoderOutput = encoder!!.encode(mel)
                results.append("VQ embedding: [${encoderOutput.vqEmbedding.size}, ${encoderOutput.vqEmbedding[0].size}] (${encoderOutput.inferenceTimeMs}ms)\n")
                results.append("Codebook indices: ${encoderOutput.codebookIndices.size} codes\n")

                // Check unique codes used
                val uniqueCodes = encoderOutput.codebookIndices.toSet().size
                results.append("Unique codes: $uniqueCodes / 512\n")

                // Process through pipeline at each tier for verification
                results.append("\n--- Pipeline Verification ---\n")
                for (tier in PrivacyTier.entries) {
                    pipeline?.setTier(tier)
                    val output = pipeline!!.processChunk(pcm)
                    results.append("[${tier.name}] ${output.summary()}")
                }
            } catch (e: Exception) {
                results.append("Encoder error: ${e.message}\n")
            }
        } else {
            results.append("(Encoder not loaded)\n")
        }

        // Compare with test fixtures if available
        try {
            val refMelStream = assets.open("models/test_fixtures/${name}_mel.npy")
            val refNpy = com.edgy.privacy.util.NpyReader.read(refMelStream)
            refMelStream.close()
            val refMel = refNpy.toFloatMatrix()

            // Compute max absolute difference
            var maxDiff = 0f
            val rows = minOf(mel.size, refMel.size)
            val cols = minOf(
                if (mel.isNotEmpty()) mel[0].size else 0,
                if (refMel.isNotEmpty()) refMel[0].size else 0
            )
            for (m in 0 until rows) {
                for (t in 0 until cols) {
                    val diff = kotlin.math.abs(mel[m][t] - refMel[m][t])
                    if (diff > maxDiff) maxDiff = diff
                }
            }
            results.append("\n--- Parity Test ---\n")
            results.append("Ref mel shape: [${refMel.size}, ${refMel[0].size}]\n")
            results.append("Max abs diff: ${"%.6f".format(maxDiff)}\n")
            results.append("PASS (< 1e-3): ${maxDiff < 1e-3}\n")
        } catch (e: Exception) {
            // No test fixtures available - that's okay
        }
    }

    /**
     * Stage 2: Verify tier output meets expected criteria.
     */
    private fun verifyTierOutput(output: PrivacyOutput, tier: PrivacyTier, results: StringBuilder) {
        when (tier) {
            PrivacyTier.LOW -> {
                val hasAudio = output.rawAudio != null && output.rawAudio.isNotEmpty()
                results.append("✓ Raw audio present: $hasAudio\n")
                results.append("✓ VQ embedding null: ${output.vqEmbedding == null}\n")
                results.append("✓ Speaker embedding null: ${output.speakerEmbedding == null}\n")
            }
            PrivacyTier.MODERATE -> {
                val hasVq = output.vqEmbedding != null && output.vqEmbedding.isNotEmpty()
                val vqDim = output.vqEmbedding?.firstOrNull()?.size ?: 0
                val hasSpk = output.speakerEmbedding != null
                val spkDim = output.speakerEmbedding?.size ?: 0
                results.append("✓ VQ embedding present (shape [T', 64]): $hasVq (dim=$vqDim)\n")
                results.append("✓ Speaker embedding present (shape [64]): $hasSpk (dim=$spkDim)\n")
                results.append("✓ Raw audio null: ${output.rawAudio == null}\n")
            }
            PrivacyTier.HIGH -> {
                val hasVq = output.vqEmbedding != null && output.vqEmbedding.isNotEmpty()
                val vqDim = output.vqEmbedding?.firstOrNull()?.size ?: 0
                results.append("✓ VQ embedding present (shape [T', 64]): $hasVq (dim=$vqDim)\n")
                results.append("✓ Speaker embedding null (stripped): ${output.speakerEmbedding == null}\n")
                results.append("✓ Raw audio null: ${output.rawAudio == null}\n")
            }
        }
    }

    private fun getSelectedTier(): PrivacyTier {
        return when (rgTier.checkedRadioButtonId) {
            R.id.rbLow -> PrivacyTier.LOW
            R.id.rbModerate -> PrivacyTier.MODERATE
            R.id.rbHigh -> PrivacyTier.HIGH
            else -> PrivacyTier.LOW
        }
    }

    private fun updateStatus(message: String) {
        tvStatus.text = "Status: $message"
    }

    private fun appendResult(message: String) {
        tvResults.append("$message\n")
    }

    private fun setProcessingEnabled(enabled: Boolean) {
        btnProcessTest.isEnabled = enabled
        btnPickFile.isEnabled = enabled
        btnProcessFile.isEnabled = enabled && selectedFileUri != null
    }
}
