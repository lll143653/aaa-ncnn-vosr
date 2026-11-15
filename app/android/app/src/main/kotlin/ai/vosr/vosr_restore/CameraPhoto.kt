package ai.vosr.vosr_restore

import androidx.exifinterface.media.ExifInterface
import java.io.File
import kotlin.math.max
import kotlin.math.roundToInt

internal object CameraPhoto {
    fun zoom(file: File): Double {
        val exif=try { ExifInterface(file) }
            catch(error: Exception) { throw IllegalArgumentException("无法读取照片 EXIF，未进行超分。",error) }
        val raw=exif.getAttribute(ExifInterface.TAG_DIGITAL_ZOOM_RATIO)
            ?: throw IllegalArgumentException("照片缺少 EXIF 变焦倍数字段，无法严格确认 ≥10×，未进行超分。这不代表照片没有其它 EXIF。")
        val parts=raw.trim().split('/')
        val ratio=when(parts.size) {
            1 -> parts[0].toDoubleOrNull() ?: Double.NaN
            2 -> (parts[0].toDoubleOrNull() ?: Double.NaN)/(parts[1].toDoubleOrNull() ?: Double.NaN)
            else -> Double.NaN
        }
        require(ratio.isFinite() && ratio>=10.0) {
            if(!ratio.isFinite() || ratio<=0) "照片 EXIF 变焦倍数值无效，无法严格确认 ≥10×，未进行超分。"
            else "EXIF 数字变焦记录为 ${"%.2f".format(java.util.Locale.ROOT,ratio)}×，无法确认 ≥10×，不进行复原。"
        }
        return ratio
    }
    fun metadata(file: File): Map<String,Any> {
        val ratio=zoom(file)
        return NativeImage(file).use { mapOf("path" to file.absolutePath,"width" to it.width,"height" to it.height,
            "mode" to "camera","zoomRatio" to ratio,"exifOrientation" to it.orientation) }
    }
    fun workingSize(width: Int,height: Int): Pair<Int,Int> {
        val factor=832.0/max(width,height)
        return Pair(max(1,(width*factor).roundToInt()),max(1,(height*factor).roundToInt()))
    }
}
