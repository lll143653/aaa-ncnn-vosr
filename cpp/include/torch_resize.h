// torch_resize.h - separable bicubic resize on float HWC data, matching
// torch.nn.functional.interpolate(mode='bicubic', antialias=False,
// align_corners=False): cubic kernel with a=-0.75, fixed support=2 (no
// antialias widening on downscale), edge indices clamped, weights NOT
// renormalized after clamping.
#pragma once
#include <vector>
#include <cmath>

namespace torchresize {

static inline double cubic_filter(double x)
{
    const double a = -0.75;
    if (x < 0.0) x = -x;
    if (x < 1.0) return ((a + 2.0) * x - (a + 3.0)) * x * x + 1.0;
    if (x < 2.0) return (((x - 5.0) * x + 8.0) * x - 4.0) * a;
    return 0.0;
}

struct Axis
{
    int taps = 4;
    std::vector<int> idx;    // [out*taps] clamped source indices
    std::vector<float> w;    // [out*taps] weights
};

static inline Axis precompute(int inSize, int outSize)
{
    Axis ax;
    ax.idx.resize(outSize * ax.taps);
    ax.w.resize(outSize * ax.taps);
    const double scale = (double)inSize / (double)outSize;
    for (int i = 0; i < outSize; i++)
    {
        const double pos = (i + 0.5) * scale - 0.5;
        const int base = (int)std::floor(pos);
        for (int k = 0; k < ax.taps; k++)
        {
            const int src = base + k - 1;
            double weight = cubic_filter(std::fabs(src - pos));
            int c = src;
            if (c < 0) c = 0;
            if (c > inSize - 1) c = inSize - 1;
            ax.idx[i * ax.taps + k] = c;
            ax.w[i * ax.taps + k] = (float)weight;
        }
    }
    return ax;
}

// img: HWC float, interleaved. out must be outH x outW x C.
static inline void resize_hwc(const float* img, int inW, int inH, int channels,
                              float* out, int outW, int outH)
{
    Axis ay = precompute(inH, outH);
    Axis ax = precompute(inW, outW);

    std::vector<float> tmp((size_t)outH * inW * channels);
    // vertical
    for (int oy = 0; oy < outH; oy++)
    {
        const int* idx = &ay.idx[oy * ay.taps];
        const float* w = &ay.w[oy * ay.taps];
        for (int x = 0; x < inW; x++)
        {
            for (int c = 0; c < channels; c++)
            {
                float s = 0.f;
                for (int k = 0; k < ay.taps; k++)
                    s += w[k] * img[((size_t)idx[k] * inW + x) * channels + c];
                tmp[((size_t)oy * inW + x) * channels + c] = s;
            }
        }
    }
    // horizontal
    for (int y = 0; y < outH; y++)
    {
        const float* row = &tmp[(size_t)y * inW * channels];
        for (int ox = 0; ox < outW; ox++)
        {
            const int* idx = &ax.idx[ox * ax.taps];
            const float* w = &ax.w[ox * ax.taps];
            for (int c = 0; c < channels; c++)
            {
                float s = 0.f;
                for (int k = 0; k < ax.taps; k++)
                    s += w[k] * row[(size_t)idx[k] * channels + c];
                out[((size_t)y * outW + ox) * channels + c] = s;
            }
        }
    }
}

} // namespace torchresize
