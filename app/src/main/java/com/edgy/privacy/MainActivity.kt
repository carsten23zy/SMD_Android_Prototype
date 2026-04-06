package com.edgy.privacy

import android.Manifest
import android.app.Activity
import android.app.AlertDialog
import android.content.ComponentName
import android.content.Context
import android.content.Intent
import android.content.ServiceConnection
import android.content.pm.PackageManager
import android.net.Uri
import android.os.Bundle
import android.os.Handler
import android.os.IBinder
import android.os.Looper
import android.view.View
import android.widget.Button
import android.widget.ProgressBar
import android.widget.RadioGroup
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import com.edgy.privacy.audio.AudioCaptureService
import com.edgy.privacy.audio.AudioOutputService
import com.edgy.privacy.audio.MelSpectrogramExtractor
import com.edgy.privacy.audio.RealtimePipelineManager
import com.edgy.privacy.ml.EdgyEncoder
import com.edgy.privacy.ml.InferenceStats
import com.edgy.privacy.ml.ModelManager
import com.edgy.privacy.privacy.PrivacyOutput
import com.edgy.privacy.privacy.PrivacyPipeline
import com.edgy.privacy.privacy.PrivacyTier
import com.edgy.privacy.util.WavWriter
import com.edgy.privacy.vocoder.GriffinLimVocoder
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
 * Stage 3: Real-time mic capture with three-thread pipeline and live latency display.
 * Stage 4: Model hot-swap, external model scanning, inference stats.
 * Stage 5: Griffin-Lim vocoder for audio reconstruction from VQ embeddings.
 * Stage 6: AIDL Audio Provider SDK for inter-app audio streaming.
 */
class MainActivity : AppCompatActivity() {

    companion object {
        private const val PICK_WAV_REQUEST = 1001
        private const val PERMISSION_REQUEST_CODE = 2001
        private const val STATS_UPDATE_INTERVAL_MS = 500L
    }

    // UI elements
    private lateinit var tvModelInfo: TextView
    private lateinit var tvStatus: TextView
    private lateinit var tvResults: TextView
    private lateinit var tvLatency: TextView
    private lateinit var tvCaptureStats: TextView
    private lateinit var tvInferenceStats: TextView
    private lateinit var rgTier: RadioGroup
    private lateinit var btnProcessTest: Button
    private lateinit var btnPickFile: Button
    private lateinit var btnProcessFile: Button
    private lateinit var btnStartCapture: Button
    private lateinit var btnStopCapture: Button
    private lateinit var btnReloadModel: Button
    private lateinit var btnScanModels: Button
    private lateinit var progressBar: ProgressBar

    // Core components
    private var modelManager: ModelManager? = null
    private var melExtractor: MelSpectrogramExtractor? = null
    private var encoder: EdgyEncoder? = null
    private var vocoder: GriffinLimVocoder? = null
    private var pipeline: PrivacyPipeline? = null
    private var audioOutputService = AudioOutputService()
    private val inferenceStats = InferenceStats()

    // Stage 3: Real-time capture
    private var captureService: AudioCaptureService? = null
    private var isBound = false
    private var pipelineManager: RealtimePipelineManager? = null
    private val statsHandler = Handler(Looper.getMainLooper())
    private var statsUpdateRunnable: Runnable? = null

    // State
    private var selectedFileUri: Uri? = null
    private var processingJob: Job? = null

    private val serviceConnection = object : ServiceConnection {
        override fun onServiceConnected(name: ComponentName?, service: IBinder?) {
            val binder = service as AudioCaptureService.LocalBinder
            captureService = binder.getService()
            isBound = true

            // Wire capture service output to pipeline manager
            captureService?.setOnChunkReadyListener { chunk ->
                pipelineManager?.pushCapturedChunk(chunk)
            }
        }

        override fun onServiceDisconnected(name: ComponentName?) {
            captureService = null
            isBound = false
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        bindViews()
        setupListeners()
        initializeModel()
    }

    override fun onDestroy() {
        super.onDestroy()
        stopStatsUpdates()
        stopCapture()
        if (isBound) {
            unbindService(serviceConnection)
            isBound = false
        }
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

    override fun onRequestPermissionsResult(
        requestCode: Int, permissions: Array<out String>, grantResults: IntArray
    ) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode == PERMISSION_REQUEST_CODE) {
            if (grantResults.isNotEmpty() && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                startCapture()
            } else {
                Toast.makeText(this, "Microphone permission required for capture", Toast.LENGTH_SHORT).show()
            }
        }
    }

    // ─── View Binding ───

    private fun bindViews() {
        tvModelInfo = findViewById(R.id.tvModelInfo)
        tvStatus = findViewById(R.id.tvStatus)
        tvResults = findViewById(R.id.tvResults)
        tvLatency = findViewById(R.id.tvLatency)
        tvCaptureStats = findViewById(R.id.tvCaptureStats)
        tvInferenceStats = findViewById(R.id.tvInferenceStats)
        rgTier = findViewById(R.id.rgTier)
        btnProcessTest = findViewById(R.id.btnProcessTest)
        btnPickFile = findViewById(R.id.btnPickFile)
        btnProcessFile = findViewById(R.id.btnProcessFile)
        btnStartCapture = findViewById(R.id.btnStartCapture)
        btnStopCapture = findViewById(R.id.btnStopCapture)
        btnReloadModel = findViewById(R.id.btnReloadModel)
        btnScanModels = findViewById(R.id.btnScanModels)
        progressBar = findViewById(R.id.progressBar)
    }

    private fun setupListeners() {
        rgTier.setOnCheckedChangeListener { _, checkedId ->
            val tier = when (checkedId) {
                R.id.rbLow -> PrivacyTier.LOW
                R.id.rbModerate -> PrivacyTier.MODERATE
                R.id.rbHigh -> PrivacyTier.HIGH
                else -> PrivacyTier.LOW
            }
            pipeline?.setTier(tier)
            captureService?.updateNotification("Tier: ${tier.name}")
            updateStatus("Tier set to ${tier.name}")
        }

        // Stage 1-2
        btnProcessTest.setOnClickListener { processTestWav() }
        btnPickFile.setOnClickListener { pickWavFile() }
        btnProcessFile.setOnClickListener { processSelectedFile() }

        // Stage 3
        btnStartCapture.setOnClickListener { requestPermissionsAndStartCapture() }
        btnStopCapture.setOnClickListener { stopCapture() }

        // Stage 4
        btnReloadModel.setOnClickListener { reloadModel() }
        btnScanModels.setOnClickListener { scanExternalModels() }
        tvModelInfo.setOnClickListener { showDetailedModelInfo() }
    }

    // ─── Model Initialization ───

    private fun initializeModel() {
        updateStatus("Loading model...")
        setProcessingEnabled(false)

        CoroutineScope(Dispatchers.IO).launch {
            try {
                val manager = ModelManager(this@MainActivity)
                val config = manager.loadConfig()
                val extractor = MelSpectrogramExtractor(config)

                var enc: EdgyEncoder? = null
                var speakerEmbeddings = emptyArray<FloatArray>()

                try {
                    enc = manager.loadEncoder(preferInt8 = false)
                } catch (e: Exception) {
                    withContext(Dispatchers.Main) {
                        appendResult("ONNX encoder not available: ${e.message}")
                        appendResult("  Place ONNX model in assets/models/ to enable encoding.\n")
                    }
                }

                try {
                    speakerEmbeddings = manager.loadSpeakerEmbeddings()
                } catch (e: Exception) {
                    withContext(Dispatchers.Main) {
                        appendResult("Speaker embeddings not found: ${e.message}\n")
                    }
                }

                // Stage 5: Load projection matrix and create vocoder
                val projMatrix = manager.loadProjectionMatrix()
                val codebook = try { manager.loadCodebook() } catch (e: Exception) { null }
                val voc = GriffinLimVocoder(config, projMatrix, codebook)
                vocoder = voc

                modelManager = manager
                melExtractor = extractor

                if (enc != null) {
                    encoder = enc
                    val pl = PrivacyPipeline(extractor, enc, speakerEmbeddings, vocoder = voc)
                    pl.setTier(getSelectedTier())
                    pipeline = pl
                    pipelineManager = RealtimePipelineManager(pl, inferenceStats)
                }

                withContext(Dispatchers.Main) {
                    val vocoderStatus = if (voc.isAvailable) "vocoder: ON" else "vocoder: OFF (no projection matrix)"
                    tvModelInfo.text = "${manager.getModelInfo()}\n$vocoderStatus"
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

    // ─── Stage 3: Real-Time Capture ───

    private fun requestPermissionsAndStartCapture() {
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO)
            != PackageManager.PERMISSION_GRANTED
        ) {
            ActivityCompat.requestPermissions(
                this,
                arrayOf(Manifest.permission.RECORD_AUDIO),
                PERMISSION_REQUEST_CODE
            )
        } else {
            startCapture()
        }
    }

    private fun startCapture() {
        if (pipeline == null) {
            Toast.makeText(this, "Model not loaded — cannot start capture", Toast.LENGTH_SHORT).show()
            return
        }

        // Bind to capture service
        val serviceIntent = Intent(this, AudioCaptureService::class.java).apply {
            action = AudioCaptureService.ACTION_START
        }
        ContextCompat.startForegroundService(this, serviceIntent)
        bindService(serviceIntent, serviceConnection, Context.BIND_AUTO_CREATE)

        // Start pipeline processing threads
        pipelineManager?.apply {
            setFileOutput(
                audioOutputService,
                "${cacheDir.absolutePath}/edgy_realtime"
            )
            start()
        }

        // Start stats UI updates
        startStatsUpdates()

        btnStartCapture.isEnabled = false
        btnStopCapture.isEnabled = true
        updateStatus("Capturing at ${getSelectedTier().name} tier...")
        tvResults.text = "Real-time capture active...\n"
    }

    private fun stopCapture() {
        // Stop pipeline
        pipelineManager?.stop()

        // Stop capture service
        if (isBound) {
            captureService?.stopCapture()
        }
        val stopIntent = Intent(this, AudioCaptureService::class.java).apply {
            action = AudioCaptureService.ACTION_STOP
        }
        try { startService(stopIntent) } catch (_: Exception) {}

        stopStatsUpdates()

        btnStartCapture.isEnabled = pipeline != null
        btnStopCapture.isEnabled = false

        // Show final stats
        val captureStats = captureService?.getStats()
        val results = StringBuilder()
        results.append("=== Capture Session Complete ===\n\n")
        captureStats?.let { results.append("Capture:\n${it.summary()}\n") }
        results.append("Inference:\n${inferenceStats.summary()}\n")

        // Stage 3 verification
        captureStats?.let { stats ->
            results.append("--- Stage 3 Verification ---\n")
            results.append("Dropped chunks: ${stats.droppedChunks} (${if (stats.droppedChunks == 0L) "PASS" else "WARN"})\n")

            val totalStats = inferenceStats.getTotalStats()
            results.append("p50 latency: ${totalStats.p50}ms (target < 30ms): ${if (totalStats.p50 < 30) "PASS" else "WARN"}\n")
            results.append("p95 latency: ${totalStats.p95}ms (target < 50ms): ${if (totalStats.p95 < 50) "PASS" else "WARN"}\n")
        }

        tvResults.text = results.toString()
        updateStatus("Capture stopped")
    }

    private fun startStatsUpdates() {
        statsUpdateRunnable = object : Runnable {
            override fun run() {
                updateLiveStats()
                statsHandler.postDelayed(this, STATS_UPDATE_INTERVAL_MS)
            }
        }
        statsHandler.post(statsUpdateRunnable!!)
    }

    private fun stopStatsUpdates() {
        statsUpdateRunnable?.let { statsHandler.removeCallbacks(it) }
        statsUpdateRunnable = null
    }

    private fun updateLiveStats() {
        // Latency line
        tvLatency.text = "Latency: ${inferenceStats.compactSummary()}"

        // Capture stats
        val captureStats = captureService?.getStats()
        tvCaptureStats.text = if (captureStats != null) {
            "Capture: ${"%.1f".format(captureStats.elapsedSeconds)}s | ${captureStats.totalChunks} chunks | ${captureStats.droppedChunks} dropped"
        } else {
            "Capture: idle"
        }

        // Inference stats
        tvInferenceStats.text = "Inference: ${inferenceStats.compactSummary()}"
    }

    // ─── Stage 4: Model Hot-Swap ───

    private fun reloadModel() {
        updateStatus("Reloading model...")
        setProcessingEnabled(false)
        progressBar.visibility = View.VISIBLE

        CoroutineScope(Dispatchers.IO).launch {
            try {
                val manager = modelManager ?: return@launch
                val result = manager.reloadBundledModel()

                withContext(Dispatchers.Main) {
                    if (result.success && result.encoder != null && result.melExtractor != null) {
                        encoder = result.encoder
                        melExtractor = result.melExtractor
                        val spk = result.speakerEmbeddings ?: emptyArray()
                        val voc = GriffinLimVocoder(
                            result.config!!, result.projectionMatrix,
                            try { manager.loadCodebook() } catch (e: Exception) { null }
                        )
                        vocoder = voc
                        val pl = PrivacyPipeline(result.melExtractor, result.encoder, spk, vocoder = voc)
                        pl.setTier(getSelectedTier())
                        pipeline = pl
                        pipelineManager = RealtimePipelineManager(pl, inferenceStats)
                        inferenceStats.reset()
                        val vocoderStatus = if (voc.isAvailable) "vocoder: ON" else "vocoder: OFF"
                        tvModelInfo.text = "${manager.getModelInfo()}\n$vocoderStatus"
                        updateStatus("Model reloaded successfully")
                        appendResult("Model reloaded from bundled assets\n")
                    } else {
                        updateStatus("Reload failed: ${result.error}")
                        appendResult("Reload error: ${result.error}\n")
                    }
                    progressBar.visibility = View.GONE
                    setProcessingEnabled(true)
                }
            } catch (e: Exception) {
                withContext(Dispatchers.Main) {
                    updateStatus("Reload error: ${e.message}")
                    progressBar.visibility = View.GONE
                    setProcessingEnabled(true)
                }
            }
        }
    }

    private fun scanExternalModels() {
        updateStatus("Scanning for external models...")

        CoroutineScope(Dispatchers.IO).launch {
            val manager = modelManager ?: return@launch
            val models = manager.scanExternalModels()

            withContext(Dispatchers.Main) {
                if (models.isEmpty()) {
                    updateStatus("No external models found")
                    appendResult("No models found at /sdcard/edgy_models/\n")
                    appendResult("Push models via: adb push exported_models/ /sdcard/edgy_models/\n")
                    return@withContext
                }

                // Show model selection dialog
                val validModels = models.filter { it.isValid }
                if (validModels.isEmpty()) {
                    updateStatus("Found ${models.size} model(s), none valid")
                    val sb = StringBuilder("External models (all invalid):\n")
                    models.forEach { sb.append(it.summary()).append("\n") }
                    appendResult(sb.toString())
                    return@withContext
                }

                val names = validModels.map { it.name }.toTypedArray()
                AlertDialog.Builder(this@MainActivity)
                    .setTitle("Select External Model")
                    .setItems(names) { _, which ->
                        loadExternalModel(validModels[which])
                    }
                    .setNegativeButton("Cancel", null)
                    .show()

                updateStatus("Found ${validModels.size} valid model(s)")
            }
        }
    }

    private fun loadExternalModel(modelInfo: ModelManager.ExternalModelInfo) {
        updateStatus("Loading ${modelInfo.name}...")
        setProcessingEnabled(false)
        progressBar.visibility = View.VISIBLE

        CoroutineScope(Dispatchers.IO).launch {
            val manager = modelManager ?: return@launch
            val result = manager.reloadModel(modelInfo.path)

            withContext(Dispatchers.Main) {
                if (result.success && result.encoder != null && result.melExtractor != null) {
                    encoder = result.encoder
                    melExtractor = result.melExtractor
                    val spk = result.speakerEmbeddings ?: emptyArray()
                    val voc = GriffinLimVocoder(
                        result.config!!, result.projectionMatrix,
                        try { manager.loadCodebook() } catch (e: Exception) { null }
                    )
                    vocoder = voc
                    val pl = PrivacyPipeline(result.melExtractor, result.encoder, spk, vocoder = voc)
                    pl.setTier(getSelectedTier())
                    pipeline = pl
                    pipelineManager = RealtimePipelineManager(pl, inferenceStats)
                    inferenceStats.reset()
                    val vocoderStatus = if (voc.isAvailable) "vocoder: ON" else "vocoder: OFF"
                    tvModelInfo.text = "${manager.getModelInfo()}\n$vocoderStatus"
                    updateStatus("Loaded: ${modelInfo.name}")
                    appendResult("External model loaded: ${modelInfo.name}\n")
                    appendResult("Size: ${"%.1f".format(result.modelSizeBytes / 1024.0 / 1024.0)} MB\n")
                } else {
                    updateStatus("Failed to load ${modelInfo.name}: ${result.error}")
                    appendResult("Load error: ${result.error}\n")
                    appendResult("Previous model kept.\n")
                }
                progressBar.visibility = View.GONE
                setProcessingEnabled(true)
            }
        }
    }

    private fun showDetailedModelInfo() {
        val manager = modelManager ?: return
        val info = manager.getDetailedModelInfo()

        AlertDialog.Builder(this)
            .setTitle("Model Details")
            .setMessage(info.summary())
            .setPositiveButton("OK", null)
            .show()
    }

    // ─── Stage 1: Process Test WAV ───

    private fun processTestWav() {
        setProcessingEnabled(false)
        progressBar.visibility = View.VISIBLE
        tvResults.text = ""
        updateStatus("Processing test WAV...")

        processingJob = CoroutineScope(Dispatchers.IO).launch {
            try {
                val results = StringBuilder()
                val wavAssets = assets.list("models")?.filter { it.endsWith(".wav") } ?: emptyList()

                if (wavAssets.isEmpty()) {
                    results.append("No test WAV found in assets. Using synthetic 440Hz tone.\n\n")
                    val sampleRate = modelManager?.loadConfig()?.preprocessing?.sampleRate ?: 16000
                    val numSamples = sampleRate
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

    // ─── Stage 2: File Processing ───

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

                    val outputDir = "${cacheDir.absolutePath}/edgy_output"
                    val prefix = "output_${System.currentTimeMillis()}"
                    val savedFiles = audioOutputService.writeOutput(outputDir, output, prefix, sampleRate)
                    results.append("\nSaved files:\n")
                    savedFiles.forEach { results.append("  $it\n") }

                    results.append("\n--- Stage 2 Verification ---\n")
                    verifyTierOutput(output, tier, results)
                } else {
                    val mel = melExtractor!!.extract(pcm)
                    results.append("Mel spectrogram: [${mel.size}, ${mel[0].size}]\n")
                    results.append("(Encoder not loaded)\n")
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

    // ─── Processing Helpers ───

    private fun processAndReport(pcm: FloatArray, name: String, results: StringBuilder) {
        val melStartTime = System.nanoTime()
        val mel = melExtractor!!.extract(pcm)
        val melTimeMs = (System.nanoTime() - melStartTime) / 1_000_000

        results.append("Mel spectrogram: [${mel.size}, ${mel[0].size}] (${melTimeMs}ms)\n")

        var melMin = Float.MAX_VALUE
        var melMax = Float.MIN_VALUE
        for (row in mel) {
            for (v in row) {
                if (v < melMin) melMin = v
                if (v > melMax) melMax = v
            }
        }
        results.append("Mel range: [${"%.4f".format(melMin)}, ${"%.4f".format(melMax)}]\n")

        if (encoder != null) {
            try {
                val encoderOutput = encoder!!.encode(mel)
                results.append("VQ embedding: [${encoderOutput.vqEmbedding.size}, ${encoderOutput.vqEmbedding[0].size}] (${encoderOutput.inferenceTimeMs}ms)\n")
                results.append("Codebook indices: ${encoderOutput.codebookIndices.size} codes\n")

                val uniqueCodes = encoderOutput.codebookIndices.toSet().size
                results.append("Unique codes: $uniqueCodes / 512\n")

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
        } catch (_: Exception) { }
    }

    private fun verifyTierOutput(output: PrivacyOutput, tier: PrivacyTier, results: StringBuilder) {
        when (tier) {
            PrivacyTier.LOW -> {
                val hasAudio = output.rawAudio != null && output.rawAudio.isNotEmpty()
                results.append("Raw audio present: $hasAudio\n")
                results.append("VQ embedding null: ${output.vqEmbedding == null}\n")
                results.append("Speaker embedding null: ${output.speakerEmbedding == null}\n")
            }
            PrivacyTier.MODERATE -> {
                val hasVq = output.vqEmbedding != null && output.vqEmbedding.isNotEmpty()
                val vqDim = output.vqEmbedding?.firstOrNull()?.size ?: 0
                val hasSpk = output.speakerEmbedding != null
                val spkDim = output.speakerEmbedding?.size ?: 0
                results.append("VQ embedding present [T', 64]: $hasVq (dim=$vqDim)\n")
                results.append("Speaker embedding present [64]: $hasSpk (dim=$spkDim)\n")
                results.append("Raw audio null: ${output.rawAudio == null}\n")
                results.append("Reconstructed audio: ${output.reconstructedAudio?.size ?: 0} samples\n")
                results.append("Vocoder active: ${pipeline?.isVocoderAvailable() == true}\n")
            }
            PrivacyTier.HIGH -> {
                val hasVq = output.vqEmbedding != null && output.vqEmbedding.isNotEmpty()
                val vqDim = output.vqEmbedding?.firstOrNull()?.size ?: 0
                results.append("VQ embedding present [T', 64]: $hasVq (dim=$vqDim)\n")
                results.append("Speaker embedding null (stripped): ${output.speakerEmbedding == null}\n")
                results.append("Raw audio null: ${output.rawAudio == null}\n")
                results.append("Reconstructed audio: ${output.reconstructedAudio?.size ?: 0} samples\n")
                results.append("Vocoder active: ${pipeline?.isVocoderAvailable() == true}\n")
            }
        }
    }

    // ─── UI Helpers ───

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
        btnStartCapture.isEnabled = enabled && pipeline != null && captureService?.isCapturing() != true
        btnReloadModel.isEnabled = enabled
        btnScanModels.isEnabled = enabled
    }
}
