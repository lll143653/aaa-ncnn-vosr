package ai.vosr.vosr_restore

import android.app.Activity
import android.app.Instrumentation
import android.os.Bundle
import org.json.JSONObject
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder

/** One real JNI pass with public reference tensors. Does not access photos or the screen. */
class ModelChecks : Instrumentation() {
    private var selected="npu"
    override fun onCreate(arguments: Bundle?) {
        super.onCreate(arguments)
        selected=arguments?.getString("backend") ?: "npu"
        start()
    }
    override fun onStart() {
        val folder=File(targetContext.getExternalFilesDir(null),"model-checks-1024")
        try {
            require(selected=="npu" || selected=="gpu")
            fun read(name: String): FloatArray {
                val bytes=File(folder,"$name.raw").readBytes()
                require(bytes.size%4==0)
                return FloatArray(bytes.size/4).also { ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer().get(it) }
            }
            val models=File(targetContext.getExternalFilesDir(null),if(selected=="npu")"models-1024" else "models-gpu-1024")
            val lease=ResidentEngine.acquire(targetContext,models,selected)
            val rgb=read("rgb"); val dino=read("dino_input"); val noise=read("noise")
            require(rgb.size==3*1024*1024 && noise.size==4*128*128)
            val pixels=if(selected=="npu") NativeQnn.run(lease.handle,rgb,dino,noise) else NativeGpu.run(lease.handle,rgb,dino,noise)
            check(pixels.size==3*1024*1024 && pixels.all { it.isFinite() })
            val data=ByteBuffer.allocate(pixels.size*4).order(ByteOrder.LITTLE_ENDIAN)
            data.asFloatBuffer().put(pixels)
            File(folder,"$selected-output.raw").writeBytes(data.array())
            val stats=if(selected=="npu")NativeQnn.stats(lease.handle) else NativeGpu.stats(lease.handle)
            File(folder,"$selected-result.json").writeText(JSONObject().put("passed",true).put("stats",JSONObject(stats)).put("initializationMs",lease.initializationMs).toString(2))
            finish(Activity.RESULT_OK,Bundle().apply { putString("passed","1024 JNI $selected") })
        } catch(error: Throwable) {
            folder.mkdirs()
            File(folder,"$selected-result.json").writeText(JSONObject().put("error",error.stackTraceToString()).toString(2))
            finish(Activity.RESULT_CANCELED,Bundle().apply { putString("error",error.stackTraceToString()) })
        } finally { ResidentEngine.close() }
    }
}
