#include <jni.h>
#include <android/bitmap.h>
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace {
struct Bitmap {
    JNIEnv* env; jobject object; AndroidBitmapInfo info{}; void* pixels=nullptr;
    Bitmap(JNIEnv* e,jobject b):env(e),object(b) {
        if(AndroidBitmap_getInfo(env,b,&info)!=0 || info.format!=ANDROID_BITMAP_FORMAT_RGBA_8888 ||
           AndroidBitmap_lockPixels(env,b,&pixels)!=0) throw std::runtime_error("Cannot access RGBA bitmap");
    }
    ~Bitmap() { if(pixels) AndroidBitmap_unlockPixels(env,object); }
    unsigned char* row(int y) { return static_cast<unsigned char*>(pixels)+y*info.stride; }
};
double cubic(double x) {
    x=std::abs(x);
    return x<=1?1.5*x*x*x-2.5*x*x+1:x<2?-.5*x*x*x+2.5*x*x-4*x+2:0;
}
void fail(JNIEnv* env,const std::exception& e) { env->ThrowNew(env->FindClass("java/lang/IllegalStateException"),e.what()); }
std::vector<float> reduce(const std::vector<float>& in,int w,int h) {
    const int nw=(w+1)/2,nh=(h+1)/2; constexpr float k[5]={1/16.f,4/16.f,6/16.f,4/16.f,1/16.f};
    std::vector<float> temp(nw*h*3),out(nw*nh*3);
    for(int y=0;y<h;y++) for(int x=0;x<nw;x++) for(int c=0;c<3;c++) {
        float v=0; for(int i=-2;i<=2;i++) v+=k[i+2]*in[(y*w+std::clamp(x*2+i,0,w-1))*3+c];
        temp[(y*nw+x)*3+c]=v;
    }
    for(int y=0;y<nh;y++) for(int x=0;x<nw;x++) for(int c=0;c<3;c++) {
        float v=0; for(int i=-2;i<=2;i++) v+=k[i+2]*temp[(std::clamp(y*2+i,0,h-1)*nw+x)*3+c];
        out[(y*nw+x)*3+c]=v;
    }
    return out;
}
std::vector<float> expand(const std::vector<float>& in,int w,int h,int ow,int oh) {
    std::vector<float> out(ow*oh*3);
    for(int y=0;y<oh;y++) for(int x=0;x<ow;x++) for(int c=0;c<3;c++) {
        int x0=std::min(x/2,w-1),x1=std::min(x0+1,w-1),y0=std::min(y/2,h-1),y1=std::min(y0+1,h-1);
        float fx=(x%2)*.5f,fy=(y%2)*.5f;
        out[(y*ow+x)*3+c]=(1-fy)*((1-fx)*in[(y0*w+x0)*3+c]+fx*in[(y0*w+x1)*3+c])+
            fy*((1-fx)*in[(y1*w+x0)*3+c]+fx*in[(y1*w+x1)*3+c]);
    }
    return out;
}
}

extern "C" JNIEXPORT void JNICALL Java_ai_vosr_vosr_1restore_PhotoNative_resample(
    JNIEnv* env,jclass,jobject raw,jobject output,jint regionLeft,jint regionTop,jint regionWidth,jint regionHeight,
    jint encodedWidth,jint encodedHeight,jint orientation,jint left,jint top,jint fullWidth,jint fullHeight) {
    try {
        Bitmap src(env,raw),dst(env,output);
        int logicalW=orientation>=5?encodedHeight:encodedWidth,logicalH=orientation>=5?encodedWidth:encodedHeight;
        if(fullWidth==logicalW && fullHeight==logicalH && int(src.info.width)==regionWidth && int(src.info.height)==regionHeight) {
            // The full-resolution LR is already on the output grid. Copy rather
            // than evaluating a cubic kernel whose non-central weights are zero.
            for(unsigned y=0;y<dst.info.height;y++) for(unsigned x=0;x<dst.info.width;x++) {
                int lx=std::clamp(left+int(x),0,logicalW-1),ly=std::clamp(top+int(y),0,logicalH-1),ex=lx,ey=ly;
                switch(orientation) {
                    case 2: ex=encodedWidth-1-lx; break;
                    case 3: ex=encodedWidth-1-lx; ey=encodedHeight-1-ly; break;
                    case 4: ey=encodedHeight-1-ly; break;
                    case 5: ex=ly; ey=lx; break;
                    case 6: ex=ly; ey=encodedHeight-1-lx; break;
                    case 7: ex=encodedWidth-1-ly; ey=encodedHeight-1-lx; break;
                    case 8: ex=encodedWidth-1-ly; ey=lx; break;
                }
                auto* a=src.row(ey-regionTop)+4*(ex-regionLeft); auto* b=dst.row(y)+4*x;
                for(int c=0;c<3;c++) b[c]=a[c]; b[3]=255;
            }
            return;
        }
        if(orientation==1) {
            struct Sample { int index[4]; double weight[4]; };
            std::vector<Sample> xs(dst.info.width),ys(dst.info.height);
            auto samples=[](std::vector<Sample>& table,int start,int full,int original,int offset,int extent,int decoded) {
                for(size_t i=0;i<table.size();i++) {
                    double p=std::clamp((start+int(i)+.5)*original/full-.5,0.,double(original-1));
                    double s=(p-offset+.5)*decoded/extent-.5; int base=int(std::floor(s));
                    for(int k=0;k<4;k++) { table[i].index[k]=std::clamp(base+k-1,0,decoded-1); table[i].weight[k]=cubic(s-base-k+1); }
                }
            };
            samples(xs,left,fullWidth,encodedWidth,regionLeft,regionWidth,src.info.width);
            samples(ys,top,fullHeight,encodedHeight,regionTop,regionHeight,src.info.height);
            int w=dst.info.width;
            std::vector<double> horizontal(size_t(w)*src.info.height*3);
            for(unsigned y=0;y<src.info.height;y++) for(int x=0;x<w;x++) for(int c=0;c<3;c++) {
                double v=0; for(int k=0;k<4;k++) v+=src.row(y)[4*xs[x].index[k]+c]*xs[x].weight[k];
                horizontal[(y*w+x)*3+c]=v;
            }
            for(unsigned y=0;y<dst.info.height;y++) for(int x=0;x<w;x++) {
                auto* pixel=dst.row(y)+x*4;
                for(int c=0;c<3;c++) {
                    double v=0; for(int k=0;k<4;k++) v+=horizontal[(ys[y].index[k]*w+x)*3+c]*ys[y].weight[k];
                    pixel[c]=std::clamp(int(std::lround(v)),0,255);
                }
                pixel[3]=255;
            }
            return;
        }
        for(unsigned y=0;y<dst.info.height;y++) for(unsigned x=0;x<dst.info.width;x++) {
            double lx=std::clamp((left+int(x)+.5)*logicalW/fullWidth-.5,0.,double(logicalW-1));
            double ly=std::clamp((top+int(y)+.5)*logicalH/fullHeight-.5,0.,double(logicalH-1));
            double ex=lx,ey=ly;
            switch(orientation) {
                case 2: ex=encodedWidth-1-lx; break;
                case 3: ex=encodedWidth-1-lx; ey=encodedHeight-1-ly; break;
                case 4: ey=encodedHeight-1-ly; break;
                case 5: ex=ly; ey=lx; break;
                case 6: ex=ly; ey=encodedHeight-1-lx; break;
                case 7: ex=encodedWidth-1-ly; ey=encodedHeight-1-lx; break;
                case 8: ex=encodedWidth-1-ly; ey=lx; break;
            }
            double sx=(ex-regionLeft+.5)*src.info.width/regionWidth-.5,sy=(ey-regionTop+.5)*src.info.height/regionHeight-.5;
            int ix=int(std::floor(sx)),iy=int(std::floor(sy)); double channels[3]={0,0,0};
            for(int dy=-1;dy<=2;dy++) for(int dx=-1;dx<=2;dx++) {
                double weight=cubic(sx-ix-dx)*cubic(sy-iy-dy);
                auto* pixel=src.row(std::clamp(iy+dy,0,int(src.info.height)-1))+4*std::clamp(ix+dx,0,int(src.info.width)-1);
                for(int c=0;c<3;c++) channels[c]+=pixel[c]*weight;
            }
            auto* pixel=dst.row(y)+x*4;
            for(int c=0;c<3;c++) pixel[c]=std::clamp(int(std::lround(channels[c])),0,255);
            pixel[3]=255;
        }
    } catch(const std::exception& e) { fail(env,e); }
}

extern "C" JNIEXPORT jbyteArray JNICALL Java_ai_vosr_vosr_1restore_PhotoNative_fuse(
    JNIEnv* env,jclass,jobject original,jobject restored,jint halo,jint width,jint height) {
    try {
        Bitmap lr(env,original),sr(env,restored);
        int w=lr.info.width,h=lr.info.height;
        if(w!=int(sr.info.width)||h!=int(sr.info.height)||halo<0||width<=0||height<=0||
           width+halo>w||height+halo>h) throw std::runtime_error("Invalid fusion region");
        std::vector<float> detail(w*h*3);
        for(int y=0;y<h;y++) for(int x=0;x<w;x++) for(int c=0;c<3;c++)
            detail[(y*w+x)*3+c]=float(sr.row(y)[x*4+c])-lr.row(y)[x*4+c];
        int ws[6]={w},hs[6]={h};
        for(int level=0;level<5;level++) {
            detail=reduce(detail,ws[level],hs[level]); ws[level+1]=(ws[level]+1)/2; hs[level+1]=(hs[level]+1)/2;
        }
        for(int level=4;level>=0;level--) detail=expand(detail,ws[level+1],hs[level+1],ws[level],hs[level]);
        std::vector<jbyte> out(width*height*3);
        for(int y=0;y<height;y++) for(int x=0;x<width;x++) for(int c=0;c<3;c++)
            out[(y*width+x)*3+c]=std::clamp(int(std::lround(sr.row(y+halo)[(x+halo)*4+c]-detail[((y+halo)*w+x+halo)*3+c])),0,255);
        auto result=env->NewByteArray(out.size());
        if(result) env->SetByteArrayRegion(result,0,out.size(),out.data()); return result;
    } catch(const std::exception& e) { fail(env,e); return nullptr; }
}
