package ai.vosr.vosr_restore

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.BitmapRegionDecoder
import android.graphics.Rect
import android.media.ExifInterface
import java.io.Closeable
import java.io.File
import kotlin.math.*

/** Original encoded image; only small regions are decoded, at inSampleSize=1. */
@Suppress("DEPRECATION")
class NativeImage(file: File) : Closeable {
    private val decoder = BitmapRegionDecoder.newInstance(file.absolutePath, false)
        ?: error("当前格式不支持按块读取，请使用 JPG、PNG 或 WebP")
    private val encodedWidth = decoder.width
    private val encodedHeight = decoder.height
    val orientation = try {
        ExifInterface(file.absolutePath).getAttributeInt(ExifInterface.TAG_ORIENTATION, 1).coerceIn(1,8)
    } catch (_: Exception) { 1 }
    val width = if (orientation >= 5) encodedHeight else encodedWidth
    val height = if (orientation >= 5) encodedWidth else encodedHeight

    private fun encodedX(x: Int,y: Int): Int = when (orientation) {
        2,3 -> encodedWidth-1-x
        5,6 -> y
        7,8 -> encodedWidth-1-y
        else -> x
    }
    private fun encodedY(x: Int,y: Int): Int = when (orientation) {
        3,4 -> encodedHeight-1-y
        5,8 -> x
        6,7 -> encodedHeight-1-x
        else -> y
    }
    private fun encoded(x: Int,y: Int) = Pair(encodedX(x,y),encodedY(x,y))

    fun patch(left: Int, top: Int, tile: Int, scale: Int): FloatArray {
        // Two native pixels of interpolation halo keep patch boundaries consistent.
        val x0=(floor((left+.5)/scale-.5).toInt()-1).coerceIn(0,width-1)
        val y0=(floor((top+.5)/scale-.5).toInt()-1).coerceIn(0,height-1)
        val x1=(floor((left+tile-.5)/scale-.5).toInt()+2).coerceIn(0,width-1)
        val y1=(floor((top+tile-.5)/scale-.5).toInt()+2).coerceIn(0,height-1)
        val corners=listOf(encoded(x0,y0),encoded(x1,y0),encoded(x0,y1),encoded(x1,y1))
        val region=Rect(corners.minOf{it.first},corners.minOf{it.second},
                        corners.maxOf{it.first}+1,corners.maxOf{it.second}+1)
        val bitmap=decoder.decodeRegion(region,BitmapFactory.Options().apply {
            inSampleSize=1; inPreferredConfig=Bitmap.Config.ARGB_8888
        }) ?: error("无法读取图像 patch")
        val pixels=IntArray(bitmap.width*bitmap.height)
        bitmap.getPixels(pixels,0,bitmap.width,0,0,bitmap.width,bitmap.height)
        val pitch=bitmap.width
        bitmap.recycle()
        val plane=tile*tile
        val result=FloatArray(plane*3)
        for (y in 0 until tile) for (x in 0 until tile) {
            val sy=(top+y+.5)/scale-.5; val sx=(left+x+.5)/scale-.5
            val iy=floor(sy).toInt(); val ix=floor(sx).toInt()
            var red=0.0; var green=0.0; var blue=0.0
            for (dy in -1..2) for (dx in -1..2) {
                val xx=(ix+dx).coerceIn(0,width-1); val yy=(iy+dy).coerceIn(0,height-1)
                val color=pixels[(encodedY(xx,yy)-region.top)*pitch+encodedX(xx,yy)-region.left]
                val weight=cubic(sy-iy-dy,-.5)*cubic(sx-ix-dx,-.5)
                red+=((color shr 16) and 255)*weight/255.0
                green+=((color shr 8) and 255)*weight/255.0
                blue+=(color and 255)*weight/255.0
            }
            val index=y*tile+x
            result[index]=red.toFloat().coerceIn(0f,1f)
            result[plane+index]=green.toFloat().coerceIn(0f,1f)
            result[plane*2+index]=blue.toFloat().coerceIn(0f,1f)
        }
        return result
    }

    override fun close() { decoder.recycle() }

    companion object {
        fun cubic(x: Double, a: Double): Double {
            val d=abs(x)
            return if (d<=1) (a+2)*d*d*d-(a+3)*d*d+1
                else if (d<2) a*d*d*d-5*a*d*d+8*a*d-4*a else 0.0
        }
    }
}
