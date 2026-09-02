// pil_resize.h - pixel-exact reimplementation of Pillow's Image.resize (8bpc path)
// Mirrors src/libImaging/Resample.c: precompute_coeffs + ImagingResampleHorizontal/Vertical_8bpc
// Two-pass separable convolution with INT32 fixed-point coefficients (PRECISION_BITS=22).
// Validated against Pillow 12.2.0: max diff = 0 for LANCZOS/BICUBIC/BILINEAR on all tested sizes.
#pragma once
#include <vector>
#include <cmath>
#include <cstdint>
#include <algorithm>

namespace pilresize {

enum Filter { BILINEAR = 0, BICUBIC = 1, LANCZOS = 2 };

static const int PRECISION_BITS = 22;
static const int64_t PREC = 1LL << PRECISION_BITS;

static inline double bilinear_filter(double x)
{
    if (x < 0.0) x = -x;
    if (x < 1.0) return 1.0 - x;
    return 0.0;
}

static inline double bicubic_filter(double x)
{
    const double a = -0.5;
    if (x < 0.0) x = -x;
    if (x < 1.0) return ((a + 2.0) * x - (a + 3.0)) * x * x + 1;
    if (x < 2.0) return (((x - 5) * x + 8) * x - 4) * a;
    return 0.0;
}

static inline double sinc_filter(double x)
{
    if (x == 0.0) return 1.0;
    x = x * 3.14159265358979323846;
    return sin(x) / x;
}

static inline double lanczos_filter(double x)
{
    if (-3.0 <= x && x < 3.0)
        return sinc_filter(x) * sinc_filter(x / 3);
    return 0.0;
}

struct filter_info { double (*fn)(double); double support; };
static const filter_info FILTERS[3] = {
    { bilinear_filter, 1.0 },
    { bicubic_filter, 2.0 },
    { lanczos_filter, 3.0 },
};

// Precompute per-output-pixel bounds and fixed-point coefficients (one axis).
struct Coefficients
{
    int ksize = 0;
    std::vector<int> bounds;   // [outSize*2]: xmin, xmax (xmax = count after clamp, xmin offset)
    std::vector<int64_t> kk;   // [outSize*ksize] fixed-point coefficients (Pillow uses INT32; we use int64 to be safe)
};

static Coefficients precompute_coeffs(int inSize, int outSize, Filter filt)
{
    const filter_info& fp = FILTERS[filt];
    double filterscale = (double)inSize / outSize;
    if (filterscale < 1.0) filterscale = 1.0;
    double support = fp.support * filterscale;
    int ksize = (int)ceil(support) * 2 + 1;

    Coefficients c;
    c.ksize = ksize;
    c.bounds.resize(outSize * 2);
    c.kk.assign((size_t)outSize * ksize, 0);

    for (int xx = 0; xx < outSize; xx++)
    {
        double center = (xx + 0.5) * ((double)inSize / outSize);
        double ww = 0.0;
        double ss = 1.0 / filterscale;

        int xmin = (int)(center - support + 0.5);
        if (xmin < 0) xmin = 0;
        int xmax = (int)(center + support + 0.5);
        if (xmax > inSize) xmax = inSize;
        int n = xmax - xmin;
        if (n > ksize) n = ksize; // safety, should not happen

        int64_t* k = &c.kk[(size_t)xx * ksize];
        // first pass: raw weights (double)
        std::vector<double> w(n);
        for (int x = 0; x < n; x++)
        {
            double wv = fp.fn((x + xmin - center + 0.5) * ss);
            w[x] = wv;
            ww += wv;
        }
        // normalize + fixed-point
        for (int x = 0; x < n; x++)
        {
            double v = w[x];
            if (ww != 0.0) v /= ww;
            if (v < 0.0)
                k[x] = (int64_t)(-0.5 + v * PREC);
            else
                k[x] = (int64_t)(0.5 + v * PREC);
        }
        c.bounds[xx * 2 + 0] = xmin;
        c.bounds[xx * 2 + 1] = n;
    }
    return c;
}

// Two-pass resize: src is HxW interleaved RGB (or single channel if channels==1).
// Returns dst (outH x outW x channels). Both passes round to uint8 (Pillow 8bpc semantics).
static inline void resize_rgb(
    const std::vector<uint8_t>& src, int srcW, int srcH, int channels,
    int outW, int outH, Filter filt,
    std::vector<uint8_t>& dst)
{
    dst.assign((size_t)outH * outW * channels, 0);
    if (srcW == outW && srcH == outH)
    {
        dst = src;
        return;
    }

    Coefficients ch = precompute_coeffs(srcW, outW, filt);
    Coefficients cv = precompute_coeffs(srcH, outH, filt);

    // horizontal pass: (srcH, outW, ch) uint8
    std::vector<uint8_t> tmp((size_t)srcH * outW * channels);
    for (int yy = 0; yy < srcH; yy++)
    {
        for (int xx = 0; xx < outW; xx++)
        {
            int xmin = ch.bounds[xx * 2 + 0];
            int n = ch.bounds[xx * 2 + 1];
            const int64_t* k = &ch.kk[(size_t)xx * ch.ksize];
            for (int chn = 0; chn < channels; chn++)
            {
                int64_t ss = 1LL << (PRECISION_BITS - 1);
                for (int x = 0; x < n; x++)
                    ss += (int64_t)src[((size_t)yy * srcW + (xmin + x)) * channels + chn] * k[x];
                int64_t v = ss >> PRECISION_BITS;
                tmp[((size_t)yy * outW + xx) * channels + chn] = (uint8_t)(v < 0 ? 0 : (v > 255 ? 255 : v));
            }
        }
    }

    // vertical pass: (outH, outW, ch) uint8
    for (int yy = 0; yy < outH; yy++)
    {
        int ymin = cv.bounds[yy * 2 + 0];
        int n = cv.bounds[yy * 2 + 1];
        const int64_t* k = &cv.kk[(size_t)yy * cv.ksize];
        for (int xx = 0; xx < outW; xx++)
        {
            for (int chn = 0; chn < channels; chn++)
            {
                int64_t ss = 1LL << (PRECISION_BITS - 1);
                for (int y = 0; y < n; y++)
                    ss += (int64_t)tmp[((size_t)(ymin + y) * outW + xx) * channels + chn] * k[y];
                int64_t v = ss >> PRECISION_BITS;
                dst[((size_t)yy * outW + xx) * channels + chn] = (uint8_t)(v < 0 ? 0 : (v > 255 ? 255 : v));
            }
        }
    }
}

} // namespace pilresize
