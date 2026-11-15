package ai.vosr.vosr_restore

/** All patch geometry here is in the resized LR/input coordinate system. */
internal object PhotoPlan {
    const val INPUT_TILE=512
    const val INPUT_OVERLAP=96
    const val INPUT_STRIDE=INPUT_TILE-INPUT_OVERLAP
    const val SCALE=2
    const val OUTPUT_TILE=INPUT_TILE*SCALE
    const val OUTPUT_OVERLAP=INPUT_OVERLAP*SCALE
    fun count(length: Int)=if(length<=INPUT_TILE)1 else (length-INPUT_TILE+INPUT_STRIDE-1)/INPUT_STRIDE+1
}
