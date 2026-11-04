// а1
#pragma once
#if !defined(TIER_CUDA)
namespace tw {
typedef __attribute__((ext_vector_type(8))) float f8;
typedef unsigned u4 __attribute__((ext_vector_type(4)));
#if defined(__gfx1200__) || defined(__gfx1201__)
#define TW_GFX12 1
#endif
template<typename T> struct Op;
template<> struct Op<_Float16>{ typedef __attribute__((ext_vector_type(8))) _Float16 V;
#ifdef TW_GFX12
    static __device__ f8 mma(V a, V b, f8 c){ return __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a,b,c); }
#endif
};
template<> struct Op<__bf16>{ typedef __attribute__((ext_vector_type(8))) __bf16 V;
#ifdef TW_GFX12
    static __device__ f8 mma(V a, V b, f8 c){ return __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a,b,c); }
#endif
};

// б1
template<typename T, int P, int MR, int NR, int KT, bool TR=false>
__global__ __launch_bounds__(256) void x3k(const T* __restrict__ X, const T* __restrict__ Wt, float* __restrict__ C, int n, int m, int k, int ldc, int acc1){
#ifdef TW_GFX12
    constexpr int MW=4, NW=2, NWV=8, BM=MW*MR*16, BN=NW*NR*16, MT=BM/16, NTL=BN/16, TA=P*MT*KT/NWV, TB=NTL*KT/NWV;
    static_assert(P*MT*KT%NWV==0 && NTL*KT%NWV==0, "плитки должны делиться на волны");
    __shared__ __attribute__((aligned(16))) char sm[(P*BM+BN)*KT*16*2];
    const int L=threadIdx.x&31, w=threadIdx.x>>5, wm=w/NW, wn=w%NW, r=L&15, kh=(L>>4)*8;
    const int gm=gridDim.y, gn=gridDim.x, pid=blockIdx.y*gn+blockIdx.x, G=8, grp=pid/(G*gn), fm=grp*G, sz=min(gm-fm,G);
    const int bm=(fm+(pid%(G*gn))%sz)*BM, bn=((pid%(G*gn))/sz)*BN;
    u4 ga[TA], gb[TB];
#define TW_AT(mt,kt) (sm + ((mt)*KT+(kt))*512)
#define TW_BT(nt,kt) (sm + P*BM*KT*32 + ((nt)*KT+(kt))*512)
    // б2
#define TW_GLOAD(k0) { _Pragma("unroll") for(int i=0;i<TA;++i){ int q=w*TA+i, mt=q/KT, kt=q%KT; int row=min(bm+(mt%MT)*16+r,n-1); \
                       ga[i]=*(const u4*)(X+(size_t)(mt/MT)*n*k+(size_t)row*k+(k0)+kt*16+kh); } \
                     _Pragma("unroll") for(int i=0;i<TB;++i){ int q=w*TB+i, nt=q/KT, kt=q%KT; \
                       if(TR) gb[i]=*(const u4*)(Wt+(size_t)((k0)+kt*16+(L>>1))*m+bn+nt*16+(L&1)*8); \
                       else gb[i]=*(const u4*)(Wt+(size_t)(bn+nt*16+r)*k+(k0)+kt*16+kh); } }
#define TW_SSTORE() { _Pragma("unroll") for(int i=0;i<TA;++i){ int q=w*TA+i, mt=q/KT, kt=q%KT; *(u4*)(TW_AT(mt,kt)+L*16)=ga[i]; } \
                      _Pragma("unroll") for(int i=0;i<TB;++i){ int q=w*TB+i, nt=q/KT, kt=q%KT; \
                        if(TR){ const unsigned short* hv=(const unsigned short*)&gb[i]; int c=L>>1, r0=(L&1)*8;   /* б3 */ \
                          _Pragma("unroll") for(int e=0;e<8;++e) *(unsigned short*)(TW_BT(nt,kt)+((c>>3)*16+r0+e)*16+(c&7)*2)=hv[e]; } \
                        else *(u4*)(TW_BT(nt,kt)+L*16)=gb[i]; } }
#define TW_BAR() asm volatile("s_wait_dscnt 0x0\n\ts_barrier_signal -1\n\ts_barrier_wait -1" ::: "memory")
    typedef typename Op<T>::V V;
    f8 acc[MR][NR];
    _Pragma("unroll") for(int i=0;i<MR;++i) _Pragma("unroll") for(int j=0;j<NR;++j) acc[i][j]=f8{};
#define TW_COMPUTE() { _Pragma("unroll") for(int kt=0;kt<KT;++kt){ V b[NR]; \
        _Pragma("unroll") for(int j=0;j<NR;++j) b[j]=*(const V*)(TW_BT(wn+NW*j,kt)+L*16); \
        _Pragma("unroll") for(int pp=0;pp<P;++pp) _Pragma("unroll") for(int i=0;i<MR;++i){ V a=*(const V*)(TW_AT(pp*MT+wm+MW*i,kt)+L*16); \
            _Pragma("unroll") for(int j=0;j<NR;++j) acc[i][j]=Op<T>::mma(a,b[j],acc[i][j]); } } }
    const int nk=k/(KT*16);
    TW_GLOAD(0); TW_SSTORE();
    for(int t=0;t<nk-1;++t){
        TW_GLOAD((t+1)*KT*16);
        TW_BAR(); TW_COMPUTE(); TW_BAR(); TW_SSTORE();
        _Pragma("unroll") for(int i=0;i<TA+TB;++i) __builtin_amdgcn_sched_group_barrier(0x020,1,0);
        _Pragma("unroll") for(int kt=0;kt<KT;++kt){ _Pragma("unroll") for(int j=0;j<NR;++j) __builtin_amdgcn_sched_group_barrier(0x100,1,0);
            _Pragma("unroll") for(int i=0;i<P*MR;++i){ __builtin_amdgcn_sched_group_barrier(0x100,1,0); _Pragma("unroll") for(int j=0;j<NR;++j) __builtin_amdgcn_sched_group_barrier(0x008,1,0); } }
        _Pragma("unroll") for(int i=0;i<TA+TB;++i) __builtin_amdgcn_sched_group_barrier(0x200,1,0);
    }
    TW_BAR(); TW_COMPUTE();
#undef TW_AT
#undef TW_BT
#undef TW_GLOAD
#undef TW_SSTORE
#undef TW_BAR
#undef TW_COMPUTE
    // б4
    _Pragma("unroll") for(int i=0;i<MR;++i) _Pragma("unroll") for(int j=0;j<NR;++j){ int col=bn+(wn+NW*j)*16+(L&15);
        _Pragma("unroll") for(int e=0;e<8;++e){ int row=bm+(wm+MW*i)*16+(L>>4)*8+e; if(row<n){ float* p=C+(size_t)row*ldc+col; *p = acc1 ? *p+acc[i][j][e] : acc[i][j][e]; } } }
#endif
}

// б5
struct Var { int mr, nr, kt; };
static const Var VARS[] = { {1,6,4}, {1,8,4}, {2,4,4}, {1,4,4}, {2,6,2} };   // б6
template<typename T, bool TR=false> static bool launch(int v, const T* X, const T* Wt, float* C, int n, int m, int k, int ldc, bool acc){
    const Var& q=VARS[v]; int BM=64*q.mr, BN=32*q.nr; if(m%BN || k%(16*q.kt) || k<16*q.kt) return false;
    dim3 g(m/BN,(n+BM-1)/BM), b(256); int a=acc?1:0;
    switch(v){
        case 0: x3k<T,3,1,6,4,TR><<<g,b>>>(X,Wt,C,n,m,k,ldc,a); break;
        case 1: x3k<T,3,1,8,4,TR><<<g,b>>>(X,Wt,C,n,m,k,ldc,a); break;
        case 2: x3k<T,3,2,4,4,TR><<<g,b>>>(X,Wt,C,n,m,k,ldc,a); break;
        case 3: x3k<T,3,1,4,4,TR><<<g,b>>>(X,Wt,C,n,m,k,ldc,a); break;
        case 4: x3k<T,3,2,6,2,TR><<<g,b>>>(X,Wt,C,n,m,k,ldc,a); break;
        default: return false; }
    return true; }
static const int NVAR = sizeof(VARS)/sizeof(VARS[0]);
} // б7
#endif
