package com.edgy.privacy.ml

import com.google.gson.annotations.SerializedName

/**
 * Data classes for parsing model_config.json.
 * The app reads all preprocessing parameters from this config — never hardcode values.
 */
data class ModelConfig(
    @SerializedName("preprocessing") val preprocessing: PreprocessingConfig,
    @SerializedName("encoder") val encoder: EncoderConfig,
    @SerializedName("files") val files: FilesConfig,
    @SerializedName("android_onnx_settings") val onnxSettings: OnnxSettings
)

data class PreprocessingConfig(
    @SerializedName("sample_rate") val sampleRate: Int,
    @SerializedName("n_fft") val nFft: Int,
    @SerializedName("n_mels") val nMels: Int,
    @SerializedName("hop_length") val hopLength: Int,
    @SerializedName("win_length") val winLength: Int,
    @SerializedName("fmin") val fmin: Int,
    @SerializedName("preemph") val preemph: Float,
    @SerializedName("top_db") val topDb: Int
)

data class EncoderConfig(
    @SerializedName("n_embeddings") val nEmbeddings: Int,
    @SerializedName("embedding_dim") val embeddingDim: Int,
    @SerializedName("downsampling_factor") val downsamplingFactor: Int
)

data class FilesConfig(
    @SerializedName("encoder_fp32") val encoderFp32: String,
    @SerializedName("encoder_int8") val encoderInt8: String,
    @SerializedName("codebook") val codebook: String,
    @SerializedName("speaker_embeddings") val speakerEmbeddings: String,
    @SerializedName("projection_matrix") val projectionMatrix: String = "mel_pseudo_inverse.npy"
)

data class OnnxSettings(
    @SerializedName("intra_op_num_threads") val intraOpNumThreads: Int,
    @SerializedName("inter_op_num_threads") val interOpNumThreads: Int,
    @SerializedName("execution_mode") val executionMode: String,
    @SerializedName("optimization_level") val optimizationLevel: String
)
