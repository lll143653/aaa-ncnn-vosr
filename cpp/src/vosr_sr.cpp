// vosr_sr.cpp - VOSR2 (one-step 1.4B DiT) super-resolution on ncnn.
//
// Pipeline (mirrors scripts/run_inference.py and inference_vosr_onestep.py):
//   LR image --PIL bicubic--> 512x512 [-1,1]
//     -> vae_encoder.ncnn   -> lq_latent (16,64,64)
//     -> torch-bicubic 512->448 + ImageNet norm -> dinov2l_feat.ncnn -> z_raw
//     -> vosr_dit_a.ncnn(lq_latent, noise, z_raw) -> x_tokens (1024,1536)
//     -> vosr_dit_b.ncnn(x_tokens, z_raw) -> tokens (1024,64)
//     -> host unpatchify -> sr_latent = noise - u   (one-step Euler, t: 1 -> 0)
//     -> vae_decoder.ncnn   -> pixels (3,512,512) [-1,1]
//     -> wavelet color fix (sigma=5 low/high split against the upscaled input)
//
// Usage:
//   vosr_sr <model_dir> <input> <output> [-u 4] [--cpu] [--fp16] [--seed N]
//           [--noise noise.bin] [--dump-out out.bin] [--threads N]
//
// --noise: fp32 CHW 16*64*64 noise dump, to reproduce a torch run exactly.
// --dump-out: dump the decoded float pixels (3*512*512 fp32, [-1,1]).
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <random>
#include <string>
#include <vector>
#include <chrono>
#ifdef _WIN32
#include <process.h>  // _exit
#endif

#include "net.h"
#include "mat.h"

#include <opencv2/opencv.hpp>

#include "pil_resize.h"
#include "torch_resize.h"

static const int RES = 512;    // fixed pixel tile
static const int LAT = 64;     // latent side
static const int GRID = 32;    // token grid side
static const int PATCH = 2;
static const int ZC = 16;      // latent channels
static const int DINO = 448;
static const int NTOK = 1024;  // (448/14)^2
static const int ENC_DIM = 1024;

static double now_ms()
{
    return std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

// chw float -> ncnn Mat (w, h, c), copy per channel
static ncnn::Mat chw_to_mat(const float* chw, int w, int h, int c)
{
    ncnn::Mat m(w, h, c);
    for (int i = 0; i < c; i++)
        memcpy(m.channel(i), chw + (size_t)i * w * h, (size_t)w * h * sizeof(float));
    return m;
}

static ncnn::Mat hw_to_mat(const float* hw, int w, int h)
{
    ncnn::Mat m(w, h);
    memcpy(m, hw, (size_t)w * h * sizeof(float));
    return m;
}

static int run_net(ncnn::Net& net, const std::vector<ncnn::Mat>& inputs, ncnn::Mat& out)
{
    ncnn::Extractor ex = net.create_extractor();
    for (size_t i = 0; i < inputs.size(); i++)
    {
        char name[8];
        snprintf(name, sizeof(name), "in%d", (int)i);
        if (ex.input(name, inputs[i]) != 0) return -1;
    }
    return ex.extract("out0", out);
}

static void load_net(ncnn::Net& net, const std::string& dir, const std::string& name,
                     bool fp16, int use_gpu, int threads)
{
    net.opt.num_threads = threads;
    net.opt.use_vulkan_compute = use_gpu;
    if (use_gpu)
    {
        // Vulkan fp16 breaks RMSNorm/LayerNorm precision - force fp32 compute
        net.opt.use_fp16_packed = false;
        net.opt.use_fp16_storage = false;
        net.opt.use_fp16_arithmetic = false;
    }
    std::string stem = dir + "/" + name + (fp16 ? "_fp16" : "");
    if (net.load_param((stem + ".ncnn.param").c_str()) != 0)
    {
        fprintf(stderr, "load_param failed: %s\n", (stem + ".ncnn.param").c_str());
        exit(1);
    }
    if (net.load_model((stem + ".ncnn.bin").c_str()) != 0)
    {
        fprintf(stderr, "load_model failed: %s\n", (stem + ".ncnn.bin").c_str());
        exit(1);
    }
}

int main(int argc, char** argv)
{
    if (argc < 4)
    {
        fprintf(stderr, "usage: vosr_sr <model_dir> <input> <output> [-u 4] [--cpu] [--fp16] "
                        "[--seed N] [--noise noise.bin] [--dump-out out.bin] [--threads N]\n");
        return 1;
    }
    std::string model_dir = argv[1];
    std::string input_path = argv[2];
    std::string output_path = argv[3];
    int upscale = 4;
    int use_gpu = 1;
    int fp16 = 0;
    int seed = 42;
    int threads = 8;
    std::string noise_path, dump_path;
    for (int i = 4; i < argc; i++)
    {
        if (!strcmp(argv[i], "--cpu")) use_gpu = 0;
        else if (!strcmp(argv[i], "--fp16")) fp16 = 1;
        else if (!strcmp(argv[i], "-u") && i + 1 < argc) upscale = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--seed") && i + 1 < argc) seed = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--threads") && i + 1 < argc) threads = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--noise") && i + 1 < argc) noise_path = argv[++i];
        else if (!strcmp(argv[i], "--dump-out") && i + 1 < argc) dump_path = argv[++i];
        else { fprintf(stderr, "unknown arg: %s\n", argv[i]); return 1; }
    }

    if (use_gpu && ncnn::create_gpu_instance() != 0)
    {
        fprintf(stderr, "[warn] no Vulkan device, falling back to CPU\n");
        use_gpu = 0;
    }
    fprintf(stderr, "backend: %s, weights: %s\n", use_gpu ? "vulkan" : "cpu",
            fp16 ? "fp16" : "fp32");

    // ---- load + upscale input ----
    cv::Mat bgr = cv::imread(input_path, cv::IMREAD_COLOR);
    if (bgr.empty()) { fprintf(stderr, "cannot read %s\n", input_path.c_str()); return 1; }
    if (bgr.cols * upscale != RES || bgr.rows * upscale != RES)
    {
        fprintf(stderr, "unsupported geometry: %dx%d x%d -> %dx%d; ncnn graphs are fixed "
                "at %dx%d\n", bgr.cols, bgr.rows, upscale, bgr.cols * upscale,
                bgr.rows * upscale, RES, RES);
        return 1;
    }
    cv::Mat rgb;
    cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);

    // PIL 8bit bicubic upscale to 512x512
    std::vector<uint8_t> src(rgb.data, rgb.data + (size_t)RES / upscale * RES / upscale * 3);
    std::vector<uint8_t> up8;
    pilresize::resize_rgb(src, RES / upscale, RES / upscale, 3, RES, RES,
                          pilresize::BICUBIC, up8);

    std::vector<float> up01((size_t)RES * RES * 3);
    for (size_t i = 0; i < up01.size(); i++) up01[i] = up8[i] / 255.0f;

    std::vector<float> lq_chw((size_t)3 * RES * RES);
    for (int c = 0; c < 3; c++)
        for (int y = 0; y < RES; y++)
            for (int x = 0; x < RES; x++)
                lq_chw[((size_t)c * RES + y) * RES + x] =
                    up01[((size_t)y * RES + x) * 3 + c] * 2.0f - 1.0f;

    // ---- load models ----
    // The 1.4B DiT is split into two chained halves (pnnx crashes on the full
    // 5.5GB fp32 graph): vosr_dit_a = embed + blocks 0..17, vosr_dit_b =
    // blocks 18..35 + final layer.
    ncnn::Net venc_net, dino_net, dit_a_net, dit_b_net, vdec_net;
    load_net(venc_net, model_dir, "vae_encoder", fp16, use_gpu, threads);
    load_net(dino_net, model_dir, "dinov2l_feat", fp16, use_gpu, threads);
    load_net(dit_a_net, model_dir, "vosr_dit_a", fp16, use_gpu, threads);
    load_net(dit_b_net, model_dir, "vosr_dit_b", fp16, use_gpu, threads);
    load_net(vdec_net, model_dir, "vae_decoder", fp16, use_gpu, threads);

    // ---- vae encoder ----
    double t0 = now_ms();
    ncnn::Mat lq_latent;
    {
        ncnn::Mat in = chw_to_mat(lq_chw.data(), RES, RES, 3);
        if (run_net(venc_net, {in}, lq_latent) != 0) { fprintf(stderr, "vae_encoder failed\n"); return 1; }
    }
    double t1 = now_ms();

    // ---- dinov2 features ----
    std::vector<float> dino01((size_t)DINO * DINO * 3);
    torchresize::resize_hwc(up01.data(), RES, RES, 3, dino01.data(), DINO, DINO);
    std::vector<float> dino_chw((size_t)3 * DINO * DINO);
    const float mean[3] = {0.485f, 0.456f, 0.406f};
    const float istd[3] = {1.f / 0.229f, 1.f / 0.224f, 1.f / 0.225f};
    for (int c = 0; c < 3; c++)
        for (int y = 0; y < DINO; y++)
            for (int x = 0; x < DINO; x++)
            {
                float v = dino01[((size_t)y * DINO + x) * 3 + c];
                if (v < 0.f) v = 0.f;
                if (v > 1.f) v = 1.f;
                dino_chw[((size_t)c * DINO + y) * DINO + x] = (v - mean[c]) * istd[c];
            }
    ncnn::Mat z_raw;
    {
        ncnn::Mat in = chw_to_mat(dino_chw.data(), DINO, DINO, 3);
        if (run_net(dino_net, {in}, z_raw) != 0) { fprintf(stderr, "dinov2l_feat failed\n"); return 1; }
        if (z_raw.w != ENC_DIM || z_raw.h != NTOK)
        { fprintf(stderr, "bad dino output %dx%d\n", z_raw.w, z_raw.h); return 1; }
    }
    double t2 = now_ms();

    // ---- noise ----
    std::vector<float> noise((size_t)ZC * LAT * LAT);
    if (!noise_path.empty())
    {
        FILE* f = fopen(noise_path.c_str(), "rb");
        if (!f) { fprintf(stderr, "cannot open %s\n", noise_path.c_str()); return 1; }
        if (fread(noise.data(), sizeof(float), noise.size(), f) != noise.size())
        { fprintf(stderr, "bad noise file size\n"); fclose(f); return 1; }
        fclose(f);
    }
    else
    {
        std::mt19937 rng(seed);
        std::normal_distribution<float> gauss(0.f, 1.f);
        for (size_t i = 0; i < noise.size(); i++) noise[i] = gauss(rng);
    }

    // ---- DiT one step ----
    ncnn::Mat tokens;
    {
        ncnn::Mat in1 = chw_to_mat(noise.data(), LAT, LAT, ZC);
        ncnn::Mat x_tokens;
        if (run_net(dit_a_net, {lq_latent, in1, z_raw}, x_tokens) != 0)
        { fprintf(stderr, "vosr_dit_a failed\n"); return 1; }
        if (run_net(dit_b_net, {x_tokens, z_raw}, tokens) != 0)
        { fprintf(stderr, "vosr_dit_b failed\n"); return 1; }
        if (tokens.w != PATCH * PATCH * ZC || tokens.h != GRID * GRID)
        { fprintf(stderr, "bad dit output %dx%d\n", tokens.w, tokens.h); return 1; }
    }
    double t3 = now_ms();

    // host unpatchify: token (ty,tx) [(py*P+px)*ZC + c] -> (c, ty*P+py, tx*P+px)
    // sr_latent = noise - u  (one-step Euler from t=1)
    std::vector<float> sr_latent((size_t)ZC * LAT * LAT);
    for (int ty = 0; ty < GRID; ty++)
        for (int tx = 0; tx < GRID; tx++)
        {
            const float* tok = tokens.row(ty * GRID + tx);
            for (int py = 0; py < PATCH; py++)
                for (int px = 0; px < PATCH; px++)
                    for (int c = 0; c < ZC; c++)
                    {
                        size_t idx = ((size_t)c * LAT + ty * PATCH + py) * LAT + tx * PATCH + px;
                        sr_latent[idx] = noise[idx] - tok[(py * PATCH + px) * ZC + c];
                    }
        }

    // ---- vae decoder ----
    ncnn::Mat pix;
    {
        ncnn::Mat in = chw_to_mat(sr_latent.data(), LAT, LAT, ZC);
        if (run_net(vdec_net, {in}, pix) != 0) { fprintf(stderr, "vae_decoder failed\n"); return 1; }
        if (pix.w != RES || pix.h != RES || pix.c != 3)
        { fprintf(stderr, "bad vae_decoder output %dx%dx%d\n", pix.w, pix.h, pix.c); return 1; }
    }
    double t4 = now_ms();

    if (!dump_path.empty())
    {
        FILE* f = fopen(dump_path.c_str(), "wb");
        std::vector<float> chw((size_t)3 * RES * RES);
        for (int c = 0; c < 3; c++)
            memcpy(chw.data() + (size_t)c * RES * RES, pix.channel(c),
                   (size_t)RES * RES * sizeof(float));
        fwrite(chw.data(), sizeof(float), chw.size(), f);
        fclose(f);
        fprintf(stderr, "dumped %s\n", dump_path.c_str());
    }

    // ---- to hwc [0,1] + wavelet color fix (sigma=5 low/high split) ----
    cv::Mat out01(RES, RES, CV_32FC3);
    for (int y = 0; y < RES; y++)
        for (int x = 0; x < RES; x++)
            for (int c = 0; c < 3; c++)
                out01.at<cv::Vec3f>(y, x)[c] =
                    (((const float*)pix.channel(c))[(size_t)y * RES + x] + 1.f) * 0.5f;

    cv::Mat src01(RES, RES, CV_32FC3, up01.data());
    cv::Mat src_low, tgt_low, fixed;
    cv::GaussianBlur(src01, src_low, cv::Size(0, 0), 5.0);
    cv::GaussianBlur(out01, tgt_low, cv::Size(0, 0), 5.0);
    cv::add(src_low, out01, fixed);
    cv::subtract(fixed, tgt_low, fixed);
    cv::min(fixed, 1.0, fixed);
    cv::max(fixed, 0.0, fixed);

    cv::Mat out8, out8bgr;
    fixed.convertTo(out8, CV_8UC3, 255.0);
    cv::cvtColor(out8, out8bgr, cv::COLOR_RGB2BGR);
    if (!cv::imwrite(output_path, out8bgr))
    { fprintf(stderr, "cannot write %s\n", output_path.c_str()); return 1; }

    fprintf(stderr, "vae_encoder %.0f ms | dinov2 %.0f ms | dit %.0f ms | vae_decoder %.0f ms\n",
            t1 - t0, t2 - t1, t3 - t2, t4 - t3);
    fprintf(stderr, "saved %s\n", output_path.c_str());

    // Skip ncnn::destroy_gpu_instance()/static teardown: Vulkan shutdown order
    // segfaults on some driver/ncnn combinations. Hard-exit instead.
    fflush(stdout);
    fflush(stderr);
    _exit(0);
}
