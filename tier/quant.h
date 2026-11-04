// а1
#pragma once
#include <cstdint>
#include <cstring>
#include <cmath>
#include <algorithm>
#if defined(__HIPCC__) || defined(__CUDACC__) || defined(__CUDA__) || defined(__HIP__)
#  define QHD __host__ __device__
#  define QTAB_DECL(T,n,N,...) static __constant__ T n##_d[N] = __VA_ARGS__; static const T n##_h[N] = __VA_ARGS__;
#else
#  define QHD
#  define QTAB_DECL(T,n,N,...) static const T n##_h[N] = __VA_ARGS__;
#endif
#if defined(__HIP_DEVICE_COMPILE__) || defined(__CUDA_ARCH__)
#  define QT(n) n##_d
#else
#  define QT(n) n##_h
#endif
#include "quant_tables.h"

enum { QT_F32=0, QT_F16=1, QT_Q4_0=2, QT_Q4_1=3, QT_Q5_0=6, QT_Q5_1=7, QT_Q8_0=8, QT_Q2_K=10, QT_Q3_K=11, QT_Q4_K=12, QT_Q5_K=13, QT_Q6_K=14,
       QT_IQ2_XXS=16, QT_IQ2_XS=17, QT_IQ3_XXS=18, QT_IQ1_S=19, QT_IQ4_NL=20, QT_IQ3_S=21, QT_IQ2_S=22, QT_IQ4_XS=23, QT_IQ1_M=29, QT_BF16=30,
       QT_TQ1_0=34, QT_TQ2_0=35, QT_MXFP4=39,
       QT_NF4=200, QT_FP4=201 };   // б1
struct QInfo { int type; const char* name; int blk, bytes; };
static const QInfo Q_TYPES[]={ {QT_F32,"F32",1,4},{QT_F16,"F16",1,2},{QT_BF16,"BF16",1,2},{QT_Q4_0,"Q4_0",32,18},{QT_Q4_1,"Q4_1",32,20},{QT_Q5_0,"Q5_0",32,22},
    {QT_Q5_1,"Q5_1",32,24},{QT_Q8_0,"Q8_0",32,34},{QT_Q2_K,"Q2_K",256,84},{QT_Q3_K,"Q3_K",256,110},{QT_Q4_K,"Q4_K",256,144},{QT_Q5_K,"Q5_K",256,176},
    {QT_Q6_K,"Q6_K",256,210},{QT_IQ2_XXS,"IQ2_XXS",256,66},{QT_IQ2_XS,"IQ2_XS",256,74},{QT_IQ2_S,"IQ2_S",256,82},{QT_IQ3_XXS,"IQ3_XXS",256,98},
    {QT_IQ3_S,"IQ3_S",256,110},{QT_IQ1_S,"IQ1_S",256,50},{QT_IQ1_M,"IQ1_M",256,56},{QT_IQ4_NL,"IQ4_NL",32,18},{QT_IQ4_XS,"IQ4_XS",256,136},
    {QT_TQ1_0,"TQ1_0",256,54},{QT_TQ2_0,"TQ2_0",256,66},{QT_MXFP4,"MXFP4",32,17},{QT_NF4,"NF4",64,36},{QT_FP4,"FP4",64,36} };
static inline const QInfo* q_info(int t){ for(auto& q:Q_TYPES) if(q.type==t) return &q; return nullptr; }
static inline const QInfo* q_info(const char* s){ for(auto& q:Q_TYPES) if(!strcmp(q.name,s)) return &q; return nullptr; }
static inline size_t q_row_bytes(int t, int cols){ const QInfo* q=q_info(t); return q ? (size_t)cols/q->blk*q->bytes : 0; }
static inline bool q_cols_ok(int t, int cols){ const QInfo* q=q_info(t); return q && cols%q->blk==0; }

QHD inline float q_u2f(uint32_t u){ union { uint32_t u; float f; } c; c.u=u; return c.f; }   // и1
QHD inline float q_h2f(uint16_t h){            // к1
    uint32_t s=(uint32_t)(h&0x8000)<<16, e=(h>>10)&0x1f, m=h&0x3ff, u;
    if(e==0){ if(!m) u=s; else { int k=-1; do{ m<<=1; ++k; } while(!(m&0x400)); u=s|((uint32_t)(112-k)<<23)|((m&0x3ff)<<13); } }
    else if(e==31) u=s|0x7f800000u|(m<<13);
    else u=s|((e+112)<<23)|(m<<13);
    return q_u2f(u); }
QHD inline float q_b2f(uint16_t b){ return q_u2f((uint32_t)b<<16); }
QHD inline uint16_t q_u16(const uint8_t* p){ return (uint16_t)(p[0]|(p[1]<<8)); }
QHD inline uint32_t q_u32(const uint8_t* p){ return (uint32_t)p[0]|((uint32_t)p[1]<<8)|((uint32_t)p[2]<<16)|((uint32_t)p[3]<<24); }
QHD inline float q_d(const uint8_t* p){ return q_h2f(q_u16(p)); }
QHD inline void q4k_scale_min(int j, const uint8_t* q, int& d, int& m){
    if(j<4){ d=q[j]&63; m=q[j+4]&63; } else { d=(q[j+4]&0xF)|((q[j-4]>>6)<<4); m=(q[j+4]>>4)|((q[j]>>6)<<4); } }
QHD inline float q_sign(const uint8_t s, int b){ return (s>>b)&1 ? -1.f : 1.f; }
QHD inline float q_nf4(int v){                 // с1
    const float t[16]={-1.0f,-0.6961928009986877f,-0.5250730514526367f,-0.39491748809814453f,-0.28444138169288635f,-0.18477343022823334f,
        -0.09105003625154495f,0.0f,0.07958029955625534f,0.16093020141124725f,0.24611230194568634f,0.33791524171829224f,0.44070982933044434f,
        0.5626170039176941f,0.7229568362236023f,1.0f}; return t[v]; }
QHD inline float q_fp4(int v){                 // т1
    const float t[16]={0.0f,0.0052083333f,0.6666667f,1.0f,0.33333334f,0.5f,0.16666667f,0.25f,-0.0f,-0.0052083333f,-0.6666667f,-1.0f,-0.33333334f,-0.5f,-0.16666667f,-0.25f};
    return t[v]; }

// у1
QHD inline float q_val(const uint8_t* row, int type, int col){
    switch(type){
    case QT_F32: return q_u2f(q_u32(row+(size_t)col*4));
    case QT_F16: return q_h2f(q_u16(row+(size_t)col*2));
    case QT_BF16: return q_b2f(q_u16(row+(size_t)col*2));
    case QT_Q4_0: { const uint8_t* b=row+(size_t)(col>>5)*18; int i=col&31; int v=(b[2+(i&15)]>>(4*(i>>4)))&15; return q_d(b)*(float)(v-8); }
    case QT_Q4_1: { const uint8_t* b=row+(size_t)(col>>5)*20; int i=col&31; int v=(b[4+(i&15)]>>(4*(i>>4)))&15; return q_d(b)*(float)v+q_d(b+2); }
    case QT_Q5_0: { const uint8_t* b=row+(size_t)(col>>5)*22; int i=col&31; int v=((b[6+(i&15)]>>(4*(i>>4)))&15)|(((q_u32(b+2)>>i)&1)<<4); return q_d(b)*(float)(v-16); }
    case QT_Q5_1: { const uint8_t* b=row+(size_t)(col>>5)*24; int i=col&31; int v=((b[8+(i&15)]>>(4*(i>>4)))&15)|(((q_u32(b+4)>>i)&1)<<4); return q_d(b)*(float)v+q_d(b+2); }
    case QT_Q8_0: { const uint8_t* b=row+(size_t)(col>>5)*34; return q_d(b)*(float)(int8_t)b[2+(col&31)]; }
    case QT_IQ4_NL: { const uint8_t* b=row+(size_t)(col>>5)*18; int i=col&31; int v=(b[2+(i&15)]>>(4*(i>>4)))&15; return q_d(b)*(float)QT(qt_kv_iq4)[v]; }
    case QT_MXFP4: { const uint8_t* b=row+(size_t)(col>>5)*17; int i=col&31; int v=(b[1+(i&15)]>>(4*(i>>4)))&15;
        uint32_t e=b[0], u= e<2 ? (0x00200000u<<e) : ((e-1)<<23); return q_u2f(u)*(float)QT(qt_kv_mxfp4)[v]; }
    case QT_NF4: case QT_FP4: { const uint8_t* b=row+(size_t)(col>>6)*36; int i=col&63; float a=q_u2f(q_u32(b)); uint8_t y=b[4+(i>>1)];
        int v= (i&1) ? (y&15) : (y>>4); return (type==QT_NF4 ? q_nf4(v) : q_fp4(v))*a; }   // у2
    }
    const int i=col&255;
    switch(type){
    case QT_Q2_K: { const uint8_t* b=row+(size_t)(col>>8)*84; int h=i>>7, s=(i&127)>>5, l=i&31; int q=(b[16+h*32+l]>>(2*s))&3; uint8_t sc=b[i>>4];
        return (q_d(b+80)*(float)(sc&15))*(float)q - q_d(b+82)*(float)(sc>>4); }
    case QT_Q3_K: { const uint8_t* b=row+(size_t)(col>>8)*110; const uint8_t* s=b+96; int j=i>>4;
        int ls= j<8 ? (s[j]&15) : (s[j-8]>>4), hs=(s[8+(j&3)]>>(2*(j>>2)))&3; int sc=(int)(int8_t)(uint8_t)(ls|(hs<<4))-32;
        int h=i>>7, sh=(i&127)>>5, l=i&31; int ql=(b[32+h*32+l]>>(2*sh))&3, qh=((b[l]>>(i>>5))&1)^1;
        return (q_d(b+108)*(float)sc)*(float)(ql-(qh<<2)); }
    case QT_Q4_K: { const uint8_t* b=row+(size_t)(col>>8)*144; int j=i>>6, l=i&63, sc,mn; q4k_scale_min(2*j+(l>=32),b+4,sc,mn);
        uint8_t y=b[16+j*32+(l&31)]; int v= l<32 ? (y&15) : (y>>4); return (q_d(b)*(float)sc)*(float)v - q_d(b+2)*(float)mn; }
    case QT_Q5_K: { const uint8_t* b=row+(size_t)(col>>8)*176; int j=i>>6, l=i&63, sc,mn; q4k_scale_min(2*j+(l>=32),b+4,sc,mn);
        uint8_t y=b[48+j*32+(l&31)]; int v=(l<32 ? (y&15) : (y>>4)) | (((b[16+(i&31)]>>(i>>5))&1)<<4);
        return (q_d(b)*(float)sc)*(float)v - q_d(b+2)*(float)mn; }
    case QT_Q6_K: { const uint8_t* b=row+(size_t)(col>>8)*210; int a=i>>7, c=i&63, half=(i&127)>>6, s=(i&127)>>5, l=i&31;
        int ql=(b[a*64+c]>>(4*half))&15, qh=(b[128+a*32+l]>>(2*s))&3; int q=(int)(int8_t)(uint8_t)(ql|(qh<<4))-32;
        return (q_d(b+208)*(float)(int8_t)b[192+(i>>4)])*(float)q; }
    case QT_IQ4_XS: { const uint8_t* b=row+(size_t)(col>>8)*136; int j=i>>5, half=(i&31)>>4, l=i&15;
        int ls=(b[4+(j>>1)]>>(4*(j&1)))&15, hs=(q_u16(b+2)>>(2*j))&3; int sc=(int)(int8_t)(uint8_t)(ls|(hs<<4))-32;
        int v=(b[8+j*16+l]>>(4*half))&15; return (q_d(b)*(float)sc)*(float)QT(qt_kv_iq4)[v]; }
    case QT_IQ2_XXS: { const uint8_t* b=row+(size_t)(col>>8)*66; int g=i>>5, k=(i&31)>>3, e=i&7; uint32_t a0=q_u32(b+2+g*8), a1=q_u32(b+6+g*8);
        float db=q_d(b)*(0.5f+(float)(a1>>28))*0.25f; int gi=(a0>>(8*k))&255; uint8_t sg=QT(qt_ksigns)[(a1>>(7*k))&127];
        return (db*(float)QT(qt_iq2xxs_grid)[gi*8+e])*q_sign(sg,e); }
    case QT_IQ2_XS: { const uint8_t* b=row+(size_t)(col>>8)*74; int k=i>>3, e=i&7, s=i>>4; uint16_t q=q_u16(b+2+k*2);
        float db=q_d(b)*(0.5f+(float)((b[66+(s>>1)]>>(4*(s&1)))&15))*0.25f; uint8_t sg=QT(qt_ksigns)[q>>9];
        return (db*(float)QT(qt_iq2xs_grid)[(q&511)*8+e])*q_sign(sg,e); }
    case QT_IQ2_S: { const uint8_t* b=row+(size_t)(col>>8)*82; int k=i>>3, e=i&7, s=i>>4;
        float db=q_d(b)*(0.5f+(float)((b[74+(s>>1)]>>(4*(s&1)))&15))*0.25f; int gi=b[2+k]|(((b[66+(k>>2)]>>(2*(k&3)))&3)<<8);
        return (db*(float)QT(qt_iq2s_grid)[gi*8+e])*q_sign(b[34+k],e); }
    case QT_IQ3_XXS: { const uint8_t* b=row+(size_t)(col>>8)*98; int g=i>>5, k=(i&31)>>3, e=i&7; uint32_t s=q_u32(b+66+g*4);
        float db=q_d(b)*(0.5f+(float)(s>>28))*0.5f; uint8_t sg=QT(qt_ksigns)[(s>>(7*k))&127];
        return (db*(float)QT(qt_iq3xxs_grid)[b[2+(i>>2)]*4+(i&3)])*q_sign(sg,e); }
    case QT_IQ3_S: { const uint8_t* b=row+(size_t)(col>>8)*110; int g=i>>5, k=i>>2, e=i&7;
        float db=q_d(b)*(float)(1+2*((b[106+(g>>1)]>>(4*(g&1)))&15)); int gi=b[2+k]|(((b[66+(k>>3)]>>(k&7))&1)<<8);
        return (db*(float)QT(qt_iq3s_grid)[gi*4+(i&3)])*q_sign(b[74+(i>>3)],e); }
    case QT_IQ1_S: { const uint8_t* b=row+(size_t)(col>>8)*50; int g=i>>5, k=i>>3, e=i&7; uint16_t qh=q_u16(b+34+g*2);
        float dl=q_d(b)*(float)(2*((qh>>12)&7)+1), dt=(qh&0x8000) ? -0.125f : 0.125f; int gi=b[2+k]|(((qh>>(3*(k&3)))&7)<<8);
        return dl*((float)QT(qt_iq1s_grid)[gi*8+e]+dt); }
    case QT_IQ1_M: { const uint8_t* b=row+(size_t)(col>>8)*56; const uint8_t* sc=b+48; int k=i>>3, e=i&7, s=i>>4;
        uint16_t s0=q_u16(sc),s1=q_u16(sc+2),s2=q_u16(sc+4),s3=q_u16(sc+6); uint16_t dh=(uint16_t)(((s0&0xF000)>>12)|((s1&0xF000)>>8)|((s2&0xF000)>>4)|(s3&0xF000));
        float dl=q_h2f(dh)*(float)(2*((q_u16(sc+2*(s>>2))>>(3*(s&3)))&7)+1); int qh=(b[32+(k>>1)]>>(4*(k&1)))&15;
        int gi=b[k]|((qh&7)<<8); float dt=(qh&8) ? -0.125f : 0.125f; return dl*((float)QT(qt_iq1s_grid)[gi*8+e]+dt); }
    case QT_TQ2_0: { const uint8_t* b=row+(size_t)(col>>8)*66; int h=i>>7, s=(i&127)>>5, l=i&31; return q_d(b+64)*(float)(((b[h*32+l]>>(2*s))&3)-1); }
    case QT_TQ1_0: { const uint8_t* b=row+(size_t)(col>>8)*54; const uint8_t p3[5]={1,3,9,27,81}; uint8_t x;
        if(i<160) x=(uint8_t)(b[i&31]*p3[i>>5]); else if(i<240){ int e=i-160; x=(uint8_t)(b[32+(e&15)]*p3[e>>4]); } else { int e=i-240; x=(uint8_t)(b[48+(e&3)]*p3[e>>2]); }
        return q_d(b+52)*(float)((int)(((uint16_t)x*3)>>8)-1); }
    }
    return 0.f; }

// ф1
static inline uint16_t q_f2h(float f){         // ф2
    uint32_t u; memcpy(&u,&f,4); uint32_t s=(u>>16)&0x8000; int e=(int)((u>>23)&0xff)-127+15; uint32_t m=u&0x7fffff;
    if(((u>>23)&0xff)==0xff) return (uint16_t)(s|0x7c00|(m?0x200:0));
    if(e>=31) return (uint16_t)(s|0x7c00);
    if(e<=0){ if(e<-10) return (uint16_t)s; m|=0x800000; int sh=14-e; uint32_t hm=m>>sh, rem=m&((1u<<sh)-1), half=1u<<(sh-1);
        if(rem>half || (rem==half && (hm&1))) ++hm; return (uint16_t)(s|hm); }
    uint32_t hm=m>>13, rem=m&0x1fff; uint32_t h=s|((uint32_t)e<<10)|hm;
    if(rem>0x1000 || (rem==0x1000 && (hm&1))) ++h; return (uint16_t)h; }
static inline void q_put16(uint8_t* p, uint16_t v){ p[0]=(uint8_t)v; p[1]=(uint8_t)(v>>8); }
static inline int q_nearest(float f){ float v=f+12582912.f; int i; memcpy(&i,&v,4); return (i&0x007fffff)-0x00400000; }   // ц1
static inline void q8_0_row(const float* x, uint8_t* o, int k){
    for(int b=0;b<k/32;++b, o+=34){ const float* v=x+b*32; float amax=0; for(int j=0;j<32;++j) amax=std::max(amax,std::fabs(v[j]));
        float d=amax/127.f, id=d? 1.f/d : 0.f; q_put16(o,q_f2h(d)); for(int j=0;j<32;++j) o[2+j]=(uint8_t)(int8_t)std::roundf(v[j]*id); } }
// ш1
static float q_qkx2(int n, int nmax, const float* x, const float* w, uint8_t* L, float* the_min, uint8_t* Laux, float rmin, float rdelta, int nstep){
    float mn=x[0], mx=x[0], sw=w[0], sx=sw*x[0];
    for(int i=1;i<n;++i){ if(x[i]<mn) mn=x[i]; if(x[i]>mx) mx=x[i]; sw+=w[i]; sx+=w[i]*x[i]; }
    if(mn>0) mn=0;
    if(mx==mn){ for(int i=0;i<n;++i) L[i]=0; *the_min=-mn; return 0.f; }
    float iscale=nmax/(mx-mn), scale=1/iscale, best=0;
    for(int i=0;i<n;++i){ int l=q_nearest(iscale*(x[i]-mn)); L[i]=(uint8_t)std::max(0,std::min(nmax,l)); float df=scale*L[i]+mn-x[i]; best+=w[i]*df*df; }
    for(int is=0;is<=nstep;++is){ iscale=(rmin+rdelta*is+nmax)/(mx-mn); float sl=0,sl2=0,sxl=0;
        for(int i=0;i<n;++i){ int l=q_nearest(iscale*(x[i]-mn)); l=std::max(0,std::min(nmax,l)); Laux[i]=(uint8_t)l; sl+=w[i]*l; sl2+=w[i]*l*l; sxl+=w[i]*l*x[i]; }
        float D=sw*sl2-sl*sl;
        if(D>0){ float ts=(sw*sxl-sx*sl)/D, tm=(sl2*sx-sl*sxl)/D;
            if(tm>0){ tm=0; ts=sxl/sl2; }
            float mad=0; for(int i=0;i<n;++i){ float df=ts*Laux[i]+tm-x[i]; mad+=w[i]*df*df; }
            if(mad<best){ for(int i=0;i<n;++i) L[i]=Laux[i]; best=mad; scale=ts; mn=tm; } } }
    *the_min=-mn; return scale; }
// э1
static inline void qk_minmax_row(const float* x, uint8_t* o, int k, int nmax){
    const bool q5=nmax==31; const int BB= q5?176:144; uint8_t L[256], La[32]; float w[32], mins[8], scales[8];
    for(int b=0;b<k/256;++b, o+=BB){ const float* v=x+(size_t)b*256; memset(o,0,BB);
        float s2=0; for(int l=0;l<256;++l) s2+=v[l]*v[l]; float av=std::sqrt(s2/256);
        float ms=0, mm=0;
        for(int j=0;j<8;++j){ for(int l=0;l<32;++l) w[l]=av+std::fabs(v[32*j+l]);
            scales[j]=q_qkx2(32,nmax,v+32*j,w,L+32*j,&mins[j],La,-0.5f*(q5?1:2),0.1f,q5?15:20);   // э2
            ms=std::max(ms,scales[j]); mm=std::max(mm,mins[j]); }
        float is=ms>0?63.f/ms:0.f, im=mm>0?63.f/mm:0.f; uint8_t* sc=o+4;
        for(int j=0;j<8;++j){ int ls=std::min(63,q_nearest(is*scales[j])), lm=std::min(63,q_nearest(im*mins[j]));
            if(j<4){ sc[j]=(uint8_t)ls; sc[j+4]=(uint8_t)lm; }
            else { sc[j+4]=(uint8_t)((ls&0xF)|((lm&0xF)<<4)); sc[j-4]|=(uint8_t)((ls>>4)<<6); sc[j]|=(uint8_t)((lm>>4)<<6); } }
        uint16_t hd=q_f2h(ms/63.f), hm=q_f2h(mm/63.f); q_put16(o,hd); q_put16(o+2,hm);
        for(int j=0;j<8;++j){ int s,m; q4k_scale_min(j,sc,s,m); float d=q_h2f(hd)*s; if(!d) continue; float dm=q_h2f(hm)*m;
            for(int ii=0;ii<32;++ii){ int l=q_nearest((v[32*j+ii]+dm)/d); L[32*j+ii]=(uint8_t)std::max(0,std::min(nmax,l)); } }
        if(!q5){ for(int j=0;j<256;j+=64) for(int l=0;l<32;++l) o[16+j/2+l]=(uint8_t)((L[j+l]&15)|((L[j+l+32]&15)<<4)); }
        else { uint8_t* qh=o+16; uint8_t* ql=o+48; int m1=1, m2=2;
            for(int n=0;n<256;n+=64){ for(int j=0;j<32;++j){ int l1=L[n+j], l2=L[n+j+32];
                    if(l1>15){ l1-=16; qh[j]|=m1; } if(l2>15){ l2-=16; qh[j]|=m2; } ql[j]=(uint8_t)(l1|(l2<<4)); }
                m1<<=2; m2<<=2; ql+=32; } } } }
// ю1
static float q_qx(int n, int nmax, const float* x, int8_t* L){
    float amax=0, mx=0; for(int i=0;i<n;++i){ float a=std::fabs(x[i]); if(a>amax){ amax=a; mx=x[i]; } }
    if(amax<1e-15f){ for(int i=0;i<n;++i) L[i]=0; return 0.f; }
    float iscale=-nmax/mx, sumlx=0, suml2=0;
    for(int i=0;i<n;++i){ int l=q_nearest(iscale*x[i]); l=std::max(-nmax,std::min(nmax-1,l)); L[i]=(int8_t)(l+nmax); float w=x[i]*x[i]; sumlx+=w*x[i]*l; suml2+=w*l*l; }
    float scale= suml2>0 ? sumlx/suml2 : 0.f, best=scale*sumlx;
    for(int is=-9;is<=9;++is){ if(!is) continue; iscale=-(nmax+0.1f*is)/mx; sumlx=suml2=0;
        for(int i=0;i<n;++i){ int l=q_nearest(iscale*x[i]); l=std::max(-nmax,std::min(nmax-1,l)); float w=x[i]*x[i]; sumlx+=w*x[i]*l; suml2+=w*l*l; }
        if(suml2>0 && sumlx*sumlx>best*suml2){ for(int i=0;i<n;++i){ int l=q_nearest(iscale*x[i]); L[i]=(int8_t)(nmax+std::max(-nmax,std::min(nmax-1,l))); }
            scale=sumlx/suml2; best=scale*sumlx; } }
    return scale; }
static inline void q6_k_row(const float* x, uint8_t* o, int k){
    int8_t L[256]; float scales[16];
    for(int b=0;b<k/256;++b, o+=210){ const float* v=x+(size_t)b*256; memset(o,0,210); float ma=0, ms=0;
        for(int j=0;j<16;++j){ scales[j]=q_qx(16,32,v+16*j,L+16*j); float a=std::fabs(scales[j]); if(a>ma){ ma=a; ms=scales[j]; } }
        if(ma<1e-15f) continue;
        float is=-128.f/ms; uint16_t hd=q_f2h(1/is); q_put16(o+208,hd); int8_t* sc=(int8_t*)(o+192);
        for(int j=0;j<16;++j) sc[j]=(int8_t)std::min(127,q_nearest(is*scales[j]));
        for(int j=0;j<16;++j){ float d=q_h2f(hd)*sc[j]; if(!d) continue;
            for(int ii=0;ii<16;++ii){ int l=q_nearest(v[16*j+ii]/d); L[16*j+ii]=(int8_t)(std::max(-32,std::min(31,l))+32); } }
        uint8_t* ql=o; uint8_t* qh=o+128;
        for(int j=0;j<256;j+=128){ for(int l=0;l<32;++l){ int q1=L[j+l]&15, q2=L[j+l+32]&15, q3=L[j+l+64]&15, q4=L[j+l+96]&15;
                ql[l]=(uint8_t)(q1|(q3<<4)); ql[l+32]=(uint8_t)(q2|(q4<<4));
                qh[l]=(uint8_t)((L[j+l]>>4)|((L[j+l+32]>>4)<<2)|((L[j+l+64]>>4)<<4)|((L[j+l+96]>>4)<<6)); }
            ql+=64; qh+=32; } } }
static const int Q_MAKE[]={QT_Q8_0,QT_Q6_K,QT_Q5_K,QT_Q4_K};   // аа1
static inline bool q_row(int t, const float* x, uint8_t* o, int k){
    switch(t){ case QT_Q8_0: q8_0_row(x,o,k); return true; case QT_Q4_K: qk_minmax_row(x,o,k,15); return true;
               case QT_Q5_K: qk_minmax_row(x,o,k,31); return true; case QT_Q6_K: q6_k_row(x,o,k); return true; }
    return false; }
