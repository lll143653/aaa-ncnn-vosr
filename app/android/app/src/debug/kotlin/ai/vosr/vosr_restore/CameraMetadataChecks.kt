package ai.vosr.vosr_restore

import android.app.Activity
import android.app.Instrumentation
import android.graphics.BitmapFactory
import android.content.Intent
import android.provider.MediaStore
import android.os.Bundle
import androidx.exifinterface.media.ExifInterface
import org.json.JSONObject
import java.io.File
import java.security.MessageDigest

/** Exports a metadata whitelist only. Original photo bytes remain on the device. */
class CameraMetadataChecks : Instrumentation() {
    private var arguments=Bundle()
    override fun onCreate(arguments: Bundle?) { super.onCreate(arguments); this.arguments=arguments ?: Bundle(); start() }
    override fun onStart() {
        try {
            val imported=arguments.getString("importFile")
            val name=imported ?: arguments.getString("captureFile") ?: error("Missing photo filename")
            require(if(imported==null) name.matches(Regex("capture_[a-zA-Z0-9_]+\\.jpg")) else name.matches(Regex("input_[0-9]+\\.img")))
            val file=File(targetContext.cacheDir,if(imported==null) "captures/$name" else name)
            require(file.isFile)
            val exif=ExifInterface(file)
            val legacy=android.media.ExifInterface(file.absolutePath)
            val dimensions=BitmapFactory.Options().apply { inJustDecodeBounds=true }
            BitmapFactory.decodeFile(file.absolutePath,dimensions)
            val fields=listOf(ExifInterface.TAG_MAKE,ExifInterface.TAG_MODEL,ExifInterface.TAG_SOFTWARE,
                ExifInterface.TAG_ORIENTATION,ExifInterface.TAG_EXIF_VERSION,ExifInterface.TAG_DIGITAL_ZOOM_RATIO,
                ExifInterface.TAG_FOCAL_LENGTH,ExifInterface.TAG_FOCAL_LENGTH_IN_35MM_FILM,
                ExifInterface.TAG_LENS_MODEL,ExifInterface.TAG_IMAGE_WIDTH,ExifInterface.TAG_IMAGE_LENGTH,
                ExifInterface.TAG_PIXEL_X_DIMENSION,ExifInterface.TAG_PIXEL_Y_DIMENSION)
            val selected=JSONObject(); val platform=JSONObject()
            for(tag in fields) { selected.put(tag,exif.getAttribute(tag) ?: JSONObject.NULL); platform.put(tag,legacy.getAttribute(tag) ?: JSONObject.NULL) }
            val digest=MessageDigest.getInstance("SHA-256")
            file.inputStream().use { stream ->
                val buffer=ByteArray(65536)
                while(true) { val count=stream.read(buffer); if(count<0)break; digest.update(buffer,0,count) }
            }
            val metadata=JSONObject().apply {
                val handler=Intent(MediaStore.ACTION_IMAGE_CAPTURE).resolveActivity(targetContext.packageManager)
                put("resolvedCaptureHandler",handler?.flattenToShortString() ?: JSONObject.NULL)
                put("source",if(imported==null) "capture-intent-output" else "document-picker-byte-copy")
                put("file",name); put("bytes",file.length()); put("width",dimensions.outWidth); put("height",dimensions.outHeight)
                put("mimeType",dimensions.outMimeType)
                put("sha256",digest.digest().joinToString(""){"%02x".format(it)})
                put("androidx",selected); put("platform",platform)
                put("androidxDigitalZoomDouble",exif.getAttributeDouble(ExifInterface.TAG_DIGITAL_ZOOM_RATIO,-1.0))
                put("gateResult",runCatching { CameraPhoto.zoom(file); "accepted" }.getOrElse { it.message ?: "rejected" })
            }
            val report=File(targetContext.getExternalFilesDir(null),"camera-metadata-summary.json")
            report.writeText(metadata.toString(2))
            finish(Activity.RESULT_OK,Bundle().apply { putString("metadata",metadata.toString()); putString("report",report.absolutePath) })
        } catch(error: Throwable) { finish(Activity.RESULT_CANCELED,Bundle().apply { putString("error",error.stackTraceToString()) }) }
    }
}
