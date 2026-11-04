#!/usr/bin/env python3
# а1
from pathlib import Path
import re
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parent
CPP = ROOT / "tier.cpp"
COMPAT = ROOT / "gpu_compat.h"

def extract_function(source: str, token: str) -> str:
    start = source.find(token)
    if start < 0:
        raise RuntimeError(f"source function not found: {token}")
    brace = source.find("{", start)
    if brace < 0:
        raise RuntimeError(f"body not found: {token}")
    depth = 0
    for i in range(brace, len(source)):
        if source[i] == "{": depth += 1
        elif source[i] == "}":
            depth -= 1
            if depth == 0:
                return source[start:i + 1].replace("__global__", "")
    raise RuntimeError(f"unterminated function: {token}")

def extract_fp16_struct(source: str) -> str:
    start = source.index("struct tier_fp16 {")
    end_token = 'static_assert(sizeof(tier_fp16)==2, "FP16 storage must occupy two bytes");'
    end = source.index(end_token, start) + len(end_token)
    return source[start:end].replace("__host__", "").replace("__device__", "")


def main() -> int:
    code = CPP.read_text()
    compat = COMPAT.read_text()
    fp16_struct = extract_fp16_struct(compat)
    kernels = "\n\n".join([
        extract_function(code, "__global__ void fp16_to_f32_kernel("),
        extract_function(code, "__global__ void f32_to_fp16_kernel("),
        extract_function(code, "__global__ void fp16_to_bf16_kernel("),
        extract_function(code, "__global__ void split3_k("),
        extract_function(code, "__global__ void split3_fp_k("),
        extract_function(code, "__global__ void adam_step_kernel("),
        extract_function(code, "__global__ void lora_adam_k("),
    ])

    harness = r'''#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <random>
#include <vector>
#include <string>
#include <type_traits>
#include <cstdlib>
using std::size_t;
struct CpuDim { size_t x = 0; };
static CpuDim blockIdx, blockDim, threadIdx;
''' + fp16_struct + r'''
struct hip_bfloat16 {
    uint16_t bits=0;
    hip_bfloat16() = default;
    hip_bfloat16(float x) {
        uint32_t u; std::memcpy(&u,&x,sizeof u);
        if ((u & 0x7fffffffU) > 0x7f800000U) bits=(uint16_t)((u>>16)|0x0040U);
        else { u += 0x7fffU + ((u>>16)&1U); bits=(uint16_t)(u>>16); }
    }
    operator float() const { uint32_t u=(uint32_t)bits<<16; float x; std::memcpy(&x,&u,sizeof x); return x; }
};
''' + kernels + r'''

static uint16_t native_half(float x) {
    _Float16 h = (_Float16)x; uint16_t bits; std::memcpy(&bits, &h, sizeof bits); return bits;
}
static void launch_f32_to_fp16(const float* in, tier_fp16* out, size_t n) {
    blockIdx.x=0; blockDim.x=1;
    for(size_t i=0;i<n;++i){ threadIdx.x=i; f32_to_fp16_kernel(in,out,n); }
}
static void launch_fp16_to_f32(const tier_fp16* in, float* out, size_t n) {
    blockIdx.x=0; blockDim.x=1;
    for(size_t i=0;i<n;++i){ threadIdx.x=i; fp16_to_f32_kernel(in,out,n); }
}
static void launch_split3(const float* x, tier_fp16* h, tier_fp16* m, tier_fp16* l, size_t n) {
    blockIdx.x=0; blockDim.x=1;
    for(size_t i=0;i<n;++i){ threadIdx.x=i; split3_fp_k(x,h,m,l,n); }
}
static void launch_fp16_to_bf16(const tier_fp16* in, hip_bfloat16* out, size_t n) {
    blockIdx.x=0; blockDim.x=1;
    for(size_t i=0;i<n;++i){ threadIdx.x=i; fp16_to_bf16_kernel(in,out,n); }
}
static void launch_split3_bf16(const float* x, hip_bfloat16* h, hip_bfloat16* m, hip_bfloat16* l, size_t n) {
    blockIdx.x=0; blockDim.x=1;
    for(size_t i=0;i<n;++i){ threadIdx.x=i; split3_k(x,h,m,l,n); }
}
static void launch_adam(const float* g, float* w, float* m, float* v, size_t n,
                        float lr, float b1, float b2, float bc1, float bc2, float eps) {
    blockIdx.x=0; blockDim.x=1;
    for(size_t i=0;i<n;++i){ threadIdx.x=i; adam_step_kernel(g,w,m,v,n,lr,b1,b2,bc1,bc2,eps); }
}
static void launch_lora_adam(float* w, const float* g, float* m, float* v, size_t n,
                             float lr, float bc1, float bc2) {
    blockIdx.x=0; blockDim.x=1;
    for(size_t i=0;i<n;++i){ threadIdx.x=i; lora_adam_k(w,g,m,v,n,lr,bc1,bc2); }
}
static void fail(const char* s){ std::cerr << "FAIL: " << s << "\n"; std::exit(1); }
int main(){
    // Exhaustively test all binary16 encodings: finite values and signed zero must round-trip.
    for(uint32_t u=0;u<65536;++u){
        uint16_t h=(uint16_t)u; float f=tier_fp16::to_float(h); uint16_t rt=tier_fp16::from_float(f);
        const unsigned exp=(h>>10)&31u, mant=h&1023u;
        if(exp==31u && mant!=0u){ if(!std::isnan(f)) fail("FP16 NaN decode"); }
        else if(rt!=h) fail("exhaustive FP16 encoding round-trip");
    }
    // Compare the actual conversion function with the platform IEEE _Float16 conversion.
    std::mt19937 rng(0x71e2);
    for(int i=0;i<250000;++i){
        uint32_t bits=rng(); float x; std::memcpy(&x,&bits,4);
        if(!std::isfinite(x)) continue;
        uint16_t got=tier_fp16::from_float(x), expected=native_half(x);
        if(got!=expected){ std::cerr<<"FP16 mismatch x="<<x<<" got="<<got<<" expected="<<expected<<"\n"; fail("FP16 rounding compared to native half"); }
    }

    // Execute actual conversion kernels on CPU-emulated GPU threads.
    std::vector<float> src;
    src.reserve(200000);
    std::uniform_real_distribution<float> dist(-60000.0f,60000.0f);
    for(int i=0;i<100000;++i) src.push_back(dist(rng));
    for(int i=0;i<10000;++i){ src.push_back(std::ldexp(dist(rng)/60000.0f, (i%50)-40)); }
    std::vector<tier_fp16> half(src.size()); std::vector<float> back(src.size());
    launch_f32_to_fp16(src.data(),half.data(),src.size());
    launch_fp16_to_f32(half.data(),back.data(),src.size());
    for(size_t i=0;i<src.size();++i){
        if(std::isfinite(src[i]) && tier_fp16::from_float(src[i])!=half[i].bits) fail("f32_to_fp16 kernel disagrees with conversion");
        if(back[i]!=tier_fp16::to_float(half[i].bits)) fail("fp16_to_f32 kernel disagreement");
    }

    // Execute the actual FP16x3 split kernel; it should recover FP32 inputs much more closely than one FP16.
    std::vector<tier_fp16> hi(src.size()),mid(src.size()),lo(src.size());
    launch_split3(src.data(),hi.data(),mid.data(),lo.data(),src.size());
    long double single_sq=0, triple_sq=0, ref_sq=0; double max_abs=0;
    for(size_t i=0;i<src.size();++i){
        double s=(float)half[i], t=(float)lo[i]+(float)mid[i]+(float)hi[i], x=src[i];
        single_sq+=(s-x)*(s-x); triple_sq+=(t-x)*(t-x); ref_sq+=x*x;
        max_abs=std::max(max_abs,std::abs(t-x));
    }
    double single_rel=std::sqrt((double)(single_sq/ref_sq));
    double triple_rel=std::sqrt((double)(triple_sq/ref_sq));
    if(!(triple_rel < single_rel * 0.02)) fail("FP16x3 does not significantly improve input reconstruction");

    // Small matrix-vector calculation using the actual split kernel output and FP16 frozen weights.
    // Compare three-pass FP32 accumulation to FP64 reference based on the same FP16-stored weights.
    const int R=19, K=67; std::vector<float> wsrc(R*K), x(K);
    std::uniform_real_distribution<float> wd(-1.0f,1.0f), xd(-100.0f,100.0f);
    for(auto& q:wsrc) q=wd(rng); for(auto& q:x) q=xd(rng);
    std::vector<tier_fp16> wh(R*K),xh(K),xm(K),xl(K);
    launch_f32_to_fp16(wsrc.data(),wh.data(),wh.size());
    launch_split3(x.data(),xh.data(),xm.data(),xl.data(),K);
    double worst_rel=0, worst_abs=0, direct_sq=0, triple_sq2=0, ref_sq2=0;
    for(int r=0;r<R;++r){
        double ref=0; float y3=0, y1=0;
        for(int k=0;k<K;++k){
            float w=(float)wh[r*K+k];
            ref+=(double)w*(double)x[k];
            y1 += w*(float)tier_fp16(x[k]);
        }
        // Source fp3gemm accumulates low, middle, then high with beta=0,1,1.
        for(int k=0;k<K;++k) y3 += (float)wh[r*K+k]*(float)xl[k];
        for(int k=0;k<K;++k) y3 += (float)wh[r*K+k]*(float)xm[k];
        for(int k=0;k<K;++k) y3 += (float)wh[r*K+k]*(float)xh[k];
        const double err=std::abs((double)y3-ref); worst_abs=std::max(worst_abs,err);
        worst_rel=std::max(worst_rel,err/(std::abs(ref)+1e-9));
        direct_sq+=((double)y1-ref)*((double)y1-ref); triple_sq2+=err*err; ref_sq2+=ref*ref;
    }
    double direct_rel=std::sqrt(direct_sq/ref_sq2), triple_mat_rel=std::sqrt(triple_sq2/ref_sq2);
    if(!(triple_mat_rel < direct_rel * 0.15)) fail("FP16x3 matrix-vector result insufficiently accurate vs FP16 single-pass input");

    // Execute actual FP16->BF16 conversion and BF16 split kernels used by the restored fallback.
    std::vector<hip_bfloat16> bf_from_half(wh.size());
    launch_fp16_to_bf16(wh.data(),bf_from_half.data(),wh.size());
    for(size_t i=0;i<wh.size();++i){
        hip_bfloat16 expected((float)wh[i]);
        if(bf_from_half[i].bits!=expected.bits) fail("fp16_to_bf16 kernel mismatch");
    }
    std::vector<hip_bfloat16> bh(src.size()),bm(src.size()),bl(src.size());
    launch_split3_bf16(src.data(),bh.data(),bm.data(),bl.data(),src.size());
    long double bf_single_sq=0,bf_triple_sq=0,bf_ref_sq=0;
    for(size_t i=0;i<src.size();++i){
        hip_bfloat16 one(src[i]);
        double ref=src[i], single=(float)one;
        double triple=(float)bh[i]+(float)bm[i]+(float)bl[i];
        bf_single_sq+=(single-ref)*(single-ref); bf_triple_sq+=(triple-ref)*(triple-ref); bf_ref_sq+=ref*ref;
    }
    double bf_single_rel=std::sqrt((double)(bf_single_sq/bf_ref_sq));
    double bf_triple_rel=std::sqrt((double)(bf_triple_sq/bf_ref_sq));
    if(!(bf_triple_rel < bf_single_rel * 0.02)) fail("BF16x3 does not significantly improve reconstruction vs single BF16");

    // Run the actual Adam and LoRA Adam kernels on CPU-emulated threads and verify FP32 moments.
    const size_t n=6;
    float g[n]={1e-30f, 1e-20f, 1e-8f, -1e-4f, 0.1f, -0.75f};
    float w[n]={1.0f,1.0f,1.0f,1.0f,1.0f,1.0f};
    float m[n]={0.0f,0.0f,0.0f,0.0f,1e-5f,-0.2f};
    float v[n]={0.0f,0.0f,0.0f,0.0f,1e-8f,0.5f};
    float ew[n],em[n],ev[n]; std::copy(w,w+n,ew); std::copy(m,m+n,em); std::copy(v,v+n,ev);
    const float lr=1e-3f,b1=0.9f,b2=0.999f,bc1=0.1f,bc2=0.001f,eps=1e-8f;
    for(size_t i=0;i<n;++i){ em[i]=b1*em[i]+(1.0f-b1)*g[i]; ev[i]=b2*ev[i]+(1.0f-b2)*g[i]*g[i]; ew[i]-=lr*(em[i]/bc1)/(std::sqrt(ev[i]/bc2)+eps); }
    launch_adam(g,w,m,v,n,lr,b1,b2,bc1,bc2,eps);
    for(size_t i=0;i<n;++i){
        if(std::abs(m[i]-em[i])>1e-7f*std::max(1.0f,std::abs(em[i]))) fail("Adam first moment mismatch");
        if(std::abs(v[i]-ev[i])>1e-7f*std::max(1.0f,std::abs(ev[i]))) fail("Adam second moment mismatch");
        if(std::abs(w[i]-ew[i])>1e-7f*std::max(1.0f,std::abs(ew[i]))) fail("Adam weight update mismatch");
    }
    if(m[0]==0.0f || !std::isfinite(m[0])) fail("very small FP32 Adam first moment was lost");
    // Verify actual LoRA Adam kernel with one regular gradient and one tiny gradient.
    float lw[2]={0.5f,-0.5f}, lg[2]={1e-6f,-1e-5f}, lm[2]={0,0}, lv[2]={0,0};
    float lre=0.01f, lbc1=0.1f, lbc2=0.001f;
    float lmw[2]={0.1f*lg[0],0.1f*lg[1]}, lvw[2]={0.001f*lg[0]*lg[0],0.001f*lg[1]*lg[1]}, lww[2];
    for(int i=0;i<2;++i) lww[i]=lw[i]-lre*(lmw[i]/lbc1)/(std::sqrt(lvw[i]/lbc2)+1e-8f);
    launch_lora_adam(lw,lg,lm,lv,2,lre,lbc1,lbc2);
    for(int i=0;i<2;++i) if(std::abs(lw[i]-lww[i])>1e-7f) fail("LoRA Adam mismatch");
    if(sizeof(float)!=4 || sizeof(tier_fp16)!=2) fail("unexpected storage widths");

    std::cout << "PASS exhaustive fp16 encodings: 65536\n";
    std::cout << "PASS native FP16 rounding comparisons: 250000 random float bit patterns (finite values)\n";
    std::cout << "PASS actual fp32<->fp16 kernels: " << src.size() << " values\n";
    std::cout << "PASS actual split3_fp_k: single FP16 rel-RMS=" << single_rel
              << ", FP16x3 rel-RMS=" << triple_rel << ", max abs=" << max_abs << "\n";
    std::cout << "PASS simulated FP16x3 GEMV: direct rel-RMS=" << direct_rel
              << ", triple rel-RMS=" << triple_mat_rel << ", worst abs=" << worst_abs
              << ", worst rel=" << worst_rel << "\n";
    std::cout << "PASS actual FP16->BF16 conversion kernel: " << wh.size() << " values\n";
    std::cout << "PASS actual split3_k BF16 kernel: single BF16 rel-RMS=" << bf_single_rel
              << ", BF16x3 rel-RMS=" << bf_triple_rel << "\n";
    std::cout << "PASS actual Adam kernel: FP32 m/v and updates match host reference; tiny m=" << m[0] << "\n";
    std::cout << "PASS actual LoRA Adam kernel\n";
    return 0;
}
'''

    with tempfile.TemporaryDirectory(prefix="tier_cpu_gpu_sim_") as td:
        cpp = Path(td) / "cpu_gpu_sim.cpp"
        exe = Path(td) / "cpu_gpu_sim"
        cpp.write_text(harness)
        compile_cmd = ["g++", "-O2", "-std=c++17", "-Wall", "-Wextra", str(cpp), "-o", str(exe)]
        comp = subprocess.run(compile_cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if comp.returncode:
            sys.stderr.write(comp.stdout + comp.stderr)
            return comp.returncode
        run = subprocess.run([str(exe)], text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        sys.stdout.write(run.stdout)
        if run.returncode:
            sys.stderr.write(run.stderr)
        if run.returncode:
            return run.returncode

    # г1
    checks = {
        "Adam state stores float32 m/v": "std::vector<float> hm(slot.n,0.f), hv(slot.n,0.f);" in code,
        "Adam state file allocates two float32 arrays": "g_weight_layout_total_floats * 2 * sizeof(float)" in code,
        "frozen FP16 storage independent of GEMM probe": "g_fp16_storage=true; // формат хранения всегда FP16" in code,
        "FP16 probe selects 16F input type": "static bool fp16_probe(){ return lt_probe(HIP_R_16F); }" in code,
        "BF16 probe remains available for fallback": "static bool bf3_probe(){ return lt_probe(HIP_R_16BF); }" in code,
        "BF16x3 fallback converts FP16 weights through a reusable shadow": "fp16_to_bf16_temp((const tier_fp16*)W" in code and "g_use_bf16_x3" in code,
        "runtime selects FP16x3 before BF16x3": "else if(g_bf3 && fp16_x3_available) g_use_fp16_x3=true;" in code and "else if(g_bf3 && bf16_x3_available) g_use_bf16_x3=true;" in code,
    }
    for name, ok in checks.items():
        print(("PASS" if ok else "FAIL") + " static: " + name)
    if not all(checks.values()):
        return 1
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
