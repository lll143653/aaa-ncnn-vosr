package ai.vosr.vosr_restore

import android.graphics.Bitmap

internal object PhotoNative {
    init { System.loadLibrary("vosr_photo") }
    @JvmStatic external fun resample(raw: Bitmap,output: Bitmap,regionLeft: Int,regionTop: Int,
        regionWidth: Int,regionHeight: Int,encodedWidth: Int,encodedHeight: Int,orientation: Int,
        left: Int,top: Int,fullWidth: Int,fullHeight: Int)
    @JvmStatic external fun fuse(original: Bitmap,restored: Bitmap,halo: Int,width: Int,height: Int): ByteArray
}
