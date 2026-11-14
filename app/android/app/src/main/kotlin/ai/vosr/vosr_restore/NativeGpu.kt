package ai.vosr.vosr_restore

internal object NativeGpu {
    init { System.loadLibrary("vosr_gpu") }
    @JvmStatic external fun probe(): String
    @JvmStatic external fun create(models: String): Long
    @JvmStatic external fun run(handle: Long,rgb: FloatArray,dino: FloatArray,noise: FloatArray): FloatArray
    @JvmStatic external fun stats(handle: Long): String
    @JvmStatic external fun resetCancel(handle: Long)
    @JvmStatic external fun profile(handle: Long,enabled: Boolean)
    @JvmStatic external fun cancel(handle: Long)
    @JvmStatic external fun release(handle: Long)
}
