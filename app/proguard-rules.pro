# EDGY Privacy - ProGuard rules

# Keep ONNX Runtime
-keep class ai.onnxruntime.** { *; }

# Keep JTransforms
-keep class org.jtransforms.** { *; }

# Keep Gson serialization
-keepattributes Signature
-keepattributes *Annotation*
-keep class com.edgy.privacy.ml.** { *; }
