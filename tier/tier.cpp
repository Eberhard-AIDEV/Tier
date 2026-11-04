#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include "gpu_compat.h"
#include "tier_wmma.h"
#include <map>
#if !defined(TIER_CUDA)
// а1
static std::map<void*,std::pair<size_t,int>> g_dmap;
static hipError_t trk_malloc(void** p, size_t n, int line){ hipError_t e=(hipMalloc)(p,n); if(e==hipSuccess) g_dmap[*p]={n,line}; return e; }
static hipError_t trk_free(void* p){ g_dmap.erase(p); return (hipFree)(p); }
#define hipMalloc(p,n) trk_malloc((void**)(p),(n),__LINE__)
#define hipFree(p) trk_free((void*)(p))
#endif
#include "quant.h"
#include <sstream>
#include <array>
#include <sys/sysmacros.h>   // в1
#include <dirent.h>
// в2

#include <tuple>
#include <fcntl.h>
#include <cerrno>
#include <sys/stat.h>
#include <sys/sysinfo.h>
#include <unistd.h>
#include <thread>
#include <functional>
#include <mutex>
#include <condition_variable>
#include <atomic>
#if defined(__x86_64__) || defined(__i386__)
  #if defined(__CUDACC__) && !defined(__clang__)
    // в3
    #define _AMXTILEINTRIN_H_INCLUDED
    #define _AMXINT8INTRIN_H_INCLUDED
    #define _AMXBF16INTRIN_H_INCLUDED
    #define _AMXFP16INTRIN_H_INCLUDED
    #define _AMXCOMPLEXINTRIN_H_INCLUDED
  #endif
  #include <immintrin.h>
  #define TIER_X86 1
#else
  #define TIER_X86 0
  static inline void _mm_pause(){}   // г1
#endif
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <string>
#include <map>
#include <unordered_map>
#include <set>
#include <glob.h>
#include <vector>
#include <cmath>
#include <algorithm>
#include <random>
#include <chrono>

#define HIP_CHECK(x) do { hipError_t e=(x); if(e!=hipSuccess){fprintf(stderr,"HIP err %d at %d\n",e,__LINE__);exit(1);} } while(0)
#define HIPBLAS_CHECK(x) do { hipblasStatus_t e=(x); if(e!=HIPBLAS_STATUS_SUCCESS){fprintf(stderr,"hipBLAS err %d at %d\n",e,__LINE__);exit(1);} } while(0)
// ж1
struct LiveBlk{ size_t n; int line; int step; };
static std::multimap<size_t,void*> g_pool; static std::map<void*,LiveBlk> g_live; static int g_cur_step=0;
// ж2
static bool g_tty = isatty(2);   // ж3
// з1
static bool g_to_file = []{ struct stat st; return fstat(2,&st)==0 && S_ISREG(st.st_mode); }();
static bool g_use_arena=false, g_arena_ok=false, g_in_layer=false; static char* g_ar=nullptr; static size_t g_ar_cap=0, g_ar_off=0, g_lb=0, g_need=0, g_ar_miss=0;
static bool vram_relief(); // з2
static void conf_grow_oom(size_t short_b); // з3
static size_t vram_slack(); // з4
static hipError_t cmalloc_(void** p, size_t n, int line){ size_t a=(n+255)/256*256; if(g_in_layer) g_lb+=a;
    if(g_in_layer && g_ar && g_ar_off+a<=g_ar_cap){ *p=g_ar+g_ar_off; g_ar_off+=a; g_live[*p]={n,line,g_cur_step}; return hipSuccess; }
    if(g_in_layer && g_ar) ++g_ar_miss;
    // и1
    auto fits=[&]{ size_t fr=0,tt=0; if(hipMemGetInfo(&fr,&tt)!=hipSuccess) return true; return fr>=a+vram_slack(); };
    hipError_t e = fits() ? hipMalloc(p,n) : hipErrorOutOfMemory;
    if(e==hipErrorOutOfMemory){ (void)hipGetLastError(); // и2
        if(vram_relief()) e = fits() ? hipMalloc(p,n) : hipErrorOutOfMemory;
        // и3
        const char* wv=getenv("TIER_VRAM_WAIT"); int wmax = wv ? atoi(wv) : 60;
        for(int w=0; e==hipErrorOutOfMemory && w<wmax; ++w){ (void)hipGetLastError();
            if(w%15==0){ size_t fr=0,tt=0; hipMemGetInfo(&fr,&tt); fprintf(stderr,"Видеопамять занята: требуется %.0f МиБ, свободно %.0f МиБ. Ожидание (%d с из %d).\n",n/1048576.0,fr/1048576.0,w,wmax); }
            sleep(1); e = fits() ? hipMalloc(p,n) : hipErrorOutOfMemory; }
        if(e==hipErrorOutOfMemory){ size_t fr=0,tt=0; hipMemGetInfo(&fr,&tt);
            fprintf(stderr,"Недостаточно видеопамяти: требуется %.0f МиБ, свободно %.0f из %.0f МиБ. Рекомендуется уменьшить упаковку (--pack) "
                           "или запас в ~/.config/tier.conf (headroom_mib), либо закрыть программы, занимающие карту.\n",
                    n/1048576.0,fr/1048576.0,tt/1048576.0);
            size_t need=a+vram_slack(); conf_grow_oom(need>fr ? need-fr : a); }
        if(e==hipSuccess) fprintf(stderr,"Видеопамять освободилась. Продолжение.\n"); }
    if(e==hipSuccess) g_live[*p]={n,line,g_cur_step}; return e; }
static bool g_eval=false; // и4
static bool g_bf3=false; static bool g_fp16_storage=false; static bool g_use_fp16_x3=false, g_use_bf16_x3=false; static hip_bfloat16* g_fp16_bf16_shadow=nullptr; static size_t g_fp16_bf16_shadow_n=0; static std::set<const void*> g_bf16w; static std::set<const void*> g_fp16w; static std::map<std::string,void*> g_rbf; static std::set<const void*> g_rbf_ptr; // и5
static int g_cpu_thr = 0;                                     // и6
static std::set<const float*> g_cpu_w;                        // и7
static std::map<const float*, size_t> g_cpu_n;                // и8
struct RowSplit { const uint16_t* host; int rows, R, K; };    // к1
static std::map<const float*, RowSplit> g_split;              // к2
static std::map<std::string,int> g_pf_rows;   // к3
static std::set<void*> g_rbf_inuse; static std::vector<void*> g_rbf_zombie;   // к4
static hipError_t cfree(void* p){ if(!p) return hipSuccess; if(g_rbf_ptr.count(p)) return hipSuccess;
    g_split.erase((const float*)p);   // л1
    g_live.erase(p); g_bf16w.erase(p); g_fp16w.erase(p); if(g_ar && (char*)p>=g_ar && (char*)p<g_ar+g_ar_cap) return hipSuccess; return hipFree(p); }
static int weight_type(const void* p){ return g_fp16w.count(p)?2:(g_bf16w.count(p)?1:0); } // м1
static bool is_norm_tensor(const std::string& name){ return name.size()>=11 && name.compare(name.size()-11,11,"norm.weight")==0; }
static void layer_begin(){ g_in_layer=true; g_lb=0; g_ar_off=0;
    if(g_arena_ok && !g_ar && g_need){ size_t c=(size_t)(g_need*1.03); /* о1 */ c=(c+(2u<<20)-1)/(2u<<20)*(2u<<20);
        size_t fr=0,tt=0; hipMemGetInfo(&fr,&tt); const size_t HEAD=(640ull<<20)+vram_slack(); // о2
        if(fr<c+HEAD){ g_use_arena=g_arena_ok=false; fprintf(stderr,"Область видеопамяти: требуется %.0f МиБ плюс запас %.0f МиБ, свободно %.0f МиБ. Работа без области видеопамяти.\n",c/1048576.0,HEAD/1048576.0,fr/1048576.0); return; }
        if(hipMalloc((void**)&g_ar,c)==hipSuccess){ g_ar_cap=c; fprintf(stderr,"Область видеопамяти: %.0f МиБ (замер первого шага: %.0f МиБ).\n",c/1048576.0,g_need/1048576.0); }
        else { g_ar=nullptr; g_use_arena=g_arena_ok=false; fprintf(stderr,"Область видеопамяти: не хватило видеопамяти на %.0f МиБ. Работа без области видеопамяти.\n",c/1048576.0); } } }
static void layer_end(){ g_in_layer=false; if(g_lb>g_need) g_need=g_lb; }
#define cmalloc(p,n) cmalloc_((void**)(p),(n),__LINE__)
// р1
static hipError_t dmalloc_(void** p, size_t n){ void* q=nullptr; bool il=g_in_layer; g_in_layer=false;   // р2
    hipError_t e=cmalloc_(&q,n,__LINE__); g_in_layer=il; if(e==hipSuccess){ g_live.erase(q); *p=q; } return e; }
#define dmalloc(p,n) dmalloc_((void**)(p),(n))

// р3
static int HIDDEN=0, INTER=0, NUM_HEADS=0, VOCAB=0;
// с1
#define BLOCK_MUST_BE(N) do{ if(blockDim.x!=(N)){ if(threadIdx.x==0 && blockIdx.x==0) \
    printf("Ядро запущено с %d потоками вместо %d: порядок сложения разошёлся бы.\n", (int)blockDim.x, (int)(N)); \
    TIER_TRAP(); } }while(0)
static const int NB_ROW  = 256;   // с2
static const int NB_HEAD = 64;    // с3
static float RMS_EPS=0.f, SOFTCAP=0.f;   // с4
__device__ float g_eps_d=1e-6f, g_w1p=0.f;   // с5

static int NUM_KV_HEADS, HEAD_DIM, Q_DIM, KV_DIM, WINDOW, ROTARY_DIM;
static float ROPE_THETA; static const float* ROPE_INVF=nullptr;
static bool HAS_V_PROJ;
static int N_LAYERS = 0; // с6
static int g_model_layers = 0;   // с7

// т1
struct LayerSpec { bool full=false, has_v=false; int hd=0, nkv=0, window=0, rot=0; float theta=0.f; std::vector<float> invf; float* d_invf=nullptr; };
struct Arch {
    std::string family, pre, lp;
    bool norm1p=false;          // у1
    bool post_attn=false, post_ffn=false;   // у2
    bool qk_norm=false, v_norm=false, layer_scalar=false, qkv_bias=false, silu=false, tied=false;
    float emb_scale=-1.f, attn_scale=1.0f, attn_cap=0.f;   // у3
    bool rope_rot_pairs=false;   // у4
    float rope_msc=1.f;          // у5
    bool qk_full=false;          // у6
    float resid_mul=1.f, logit_div=1.f;   // у7
    std::vector<int> eos;
    std::map<std::string,std::string> nm;   // у8
    std::vector<LayerSpec> L;
    std::string head_name;                  // у9
};
static Arch A;
static const char* SLOT7[7]={"q","k","v","o","gate","up","down"};   // ф1
static inline int rope_half(int hd,int rot){ return A.rope_rot_pairs ? rot/2 : hd/2; }
static std::string arch_tn(const std::string& slot){ auto it=A.nm.find(slot); return it==A.nm.end()?"":it->second; }
static bool arch_has(const std::string& slot){ std::string t=arch_tn(slot); return !t.empty() && t!="-"; }
static std::string LPX(int l){ return A.lp+std::to_string(l)+"."; }
static bool is_eos(int t){ for(int e : A.eos) if(e==t) return true; return false; }
// ю1
[[noreturn]] static void arch_fail(const std::string& f, const std::string& why){
    fprintf(stderr,"%s: %s\nДвижку требуется описание модели tier_arch.txt (создаётся скриптом arch.py по config.json и заголовку весов). "
                   "Для создания: python3 arch.py <папка модели> (в программе этот шаг выполняется автоматически при подготовке модели), "
                   "либо указать готовый файл через TIER_MODEL_DESC.\n",f.c_str(),why.c_str()); exit(1); }
static void arch_check_tensors(const std::string& f);   // ю2
// я1
static void arch_load(const std::string& model_path){
    std::string f; if(const char* e=getenv("TIER_MODEL_DESC")) f=e; else { auto p=model_path.rfind('/'); f=(p==std::string::npos?std::string("."):model_path.substr(0,p))+"/tier_arch.txt"; }
    FILE* fp=fopen(f.c_str(),"r");
    if(!fp) arch_fail(f,"Описание модели отсутствует ("+std::string(strerror(errno))+").");
    static const char* NEED_KEYS[]={"family","prefix","layer_prefix","hidden","inter","heads","vocab","layers","eps","softcap","emb_scale","attn_scale","norm1p","post_attn","post_ffn",
        "qk_norm","v_norm","layer_scalar","qkv_bias","act","tied","eos"};
    static const char* NEED_SLOTS[]={"iln","q","k","v","o","gate","up","down","paln","pfln","pofln","qn","kn","ls","qb","kb","vb","final","embed"};
    std::set<std::string> seen; std::set<int> seen_l;
    char buf[1<<16]; int n_l=-1; std::vector<LayerSpec> LL;
    while(fgets(buf,sizeof buf,fp)){ std::istringstream is(buf); std::string k; if(!(is>>k) || k[0]=='#') continue;
        if(k=="family") is>>A.family; else if(k=="prefix") is>>A.pre; else if(k=="layer_prefix") is>>A.lp;
        else if(k=="hidden") is>>HIDDEN; else if(k=="inter") is>>INTER; else if(k=="heads") is>>NUM_HEADS; else if(k=="vocab") is>>VOCAB;
        else if(k=="layers"){ is>>n_l; if(n_l>0&&n_l<=4096) LL.assign(n_l,LayerSpec{}); }
        else if(k=="eps") is>>RMS_EPS; else if(k=="softcap") is>>SOFTCAP; else if(k=="emb_scale"){ std::string v; is>>v; A.emb_scale = v=="sqrt" ? -1.f : (float)atof(v.c_str()); } else if(k=="attn_scale") is>>A.attn_scale;
        else if(k=="norm1p") is>>A.norm1p; else if(k=="post_attn") is>>A.post_attn; else if(k=="post_ffn") is>>A.post_ffn;
        else if(k=="qk_norm") is>>A.qk_norm; else if(k=="v_norm") is>>A.v_norm; else if(k=="layer_scalar") is>>A.layer_scalar;
        else if(k=="qkv_bias") is>>A.qkv_bias; else if(k=="tied") is>>A.tied; else if(k=="head") is>>A.head_name;
        else if(k=="act"){ std::string a; is>>a; A.silu=(a=="silu"); }
        else if(k=="rope_pairs"){ std::string v; is>>v; A.rope_rot_pairs=(v=="rot"); } else if(k=="rope_mscale") is>>A.rope_msc;
        else if(k=="resid_mul") is>>A.resid_mul; else if(k=="qk_norm_full") is>>A.qk_full; else if(k=="logit_div") is>>A.logit_div; else if(k=="attn_softcap") is>>A.attn_cap;
        else if(k=="eos"){ A.eos.clear(); int t; while(is>>t) A.eos.push_back(t); is.clear(); }
        else if(k=="name"){ std::string a,b; is>>a>>b; if(b.empty()) arch_fail(f,"Строка «name "+a+"» без имени тензора."); A.nm[a]=b; }
        else if(k=="layer"){ int i=-1; LayerSpec s; int fu=0,hv=0; is>>i>>fu>>s.hd>>s.nkv>>hv>>s.window>>s.theta>>s.rot; s.full=fu; s.has_v=hv;
            if(!is.fail() && i>=0&&i<(int)LL.size()){ LL[i]=s; seen_l.insert(i); } else arch_fail(f,"Строка «layer» не разобрана или номер "+std::to_string(i)+" вне диапазона 0.."+std::to_string(n_l-1)+" (строка «layers» должна идти раньше)."); }
        else if(k=="freq"){ int i=-1; is>>i; std::vector<float> v; float x; while(is>>x) v.push_back(x); is.clear(); if(i>=0&&i<(int)LL.size()) LL[i].invf=v; }
        else if(k=="requires"){ static const std::set<std::string> KN={"rope_pairs","rope_mscale","qk_norm_full","resid_mul","logit_div","attn_softcap"};
            std::string w; while(is>>w) if(!KN.count(w)) arch_fail(f,"Описание требует ключ «"+w+"», а этот движок его не знает. Нужна пересборка движка (build.sh)."); is.clear(); }
        else if(k=="unsupported"){ std::string why; std::getline(is,why); fprintf(stderr,"Эта модель пока не поддерживается:%s\n",why.c_str()); exit(1); }
        else continue;   // я2
        if(is.fail() && k!="eos" && k!="freq" && k!="head") arch_fail(f,"Строка «"+k+"» не разобрана: "+std::string(buf).substr(0,80));
        seen.insert(k); }
    fclose(fp);
    { std::string miss; for(const char* nk : NEED_KEYS) if(!seen.count(nk)) miss+=std::string(miss.empty()?"":", ")+nk;
      for(const char* ns : NEED_SLOTS) if(!A.nm.count(ns)) miss+=std::string(miss.empty()?"":", ")+"name "+ns;
      if(n_l>0) for(int i=0;i<n_l;++i) if(!seen_l.count(i)){ miss+=std::string(miss.empty()?"":", ")+"layer "+std::to_string(i); break; }
      if(!miss.empty()) arch_fail(f,"В описании не хватает: "+miss); }
    if(n_l<=0||n_l>4096||HIDDEN<=0||INTER<=0||NUM_HEADS<=0||VOCAB<=0) arch_fail(f,"Описание повреждено: слоёв "+std::to_string(n_l)+", ширина "+std::to_string(HIDDEN)+", MLP "+std::to_string(INTER)+", голов "+std::to_string(NUM_HEADS)+", словарь "+std::to_string(VOCAB)+".");
    for(int i=0;i<n_l;++i){ const LayerSpec& s=LL[i]; if(s.hd<=0||s.nkv<=0||s.rot<0||s.rot>s.hd||s.theta<=0.f) arch_fail(f,"Слой "+std::to_string(i)+": неверные размеры внимания (голова "+std::to_string(s.hd)+", KV-голов "+std::to_string(s.nkv)+", RoPE "+std::to_string(s.rot)+")."); }
    if(A.eos.empty()) arch_fail(f,"В описании нет концов ответа (eos).");
    N_LAYERS=n_l; A.L=LL;
    g_model_layers=N_LAYERS;
    if(A.head_name.empty() && !A.tied) arch_fail(f,"LM head не связан с эмбеддингами, но его имя не указано.");
    arch_check_tensors(f);   // я3
    fprintf(stderr,"Архитектура %s: слоёв %d, ширина %d, MLP %d, голов %d, словарь %d (%s).\n",A.family.c_str(),N_LAYERS,HIDDEN,INTER,NUM_HEADS,VOCAB,f.c_str());
}
__global__ void set_eps_k(float e, float w1){ g_eps_d=e; g_w1p=w1; }

static void arch_device(){   // аб1
    set_eps_k<<<1,1>>>(RMS_EPS, A.norm1p?1.f:0.f);
    for(auto& s : A.L){ if(s.d_invf || s.invf.empty()) continue;
        HIP_CHECK(hipMalloc(&s.d_invf,s.invf.size()*4)); HIP_CHECK(hipMemcpy(s.d_invf,s.invf.data(),s.invf.size()*4,hipMemcpyHostToDevice)); }
    HIP_CHECK(hipDeviceSynchronize()); }
// ав1
static bool layer_set(int l){ const LayerSpec& s=A.L[l];
    HEAD_DIM=s.hd; NUM_KV_HEADS=s.nkv; HAS_V_PROJ=s.has_v; WINDOW=s.window; ROPE_THETA=s.theta; ROTARY_DIM=s.rot; ROPE_INVF=s.d_invf;
    Q_DIM=NUM_HEADS*HEAD_DIM; KV_DIM=NUM_KV_HEADS*HEAD_DIM; return s.full; }

// аг1
struct TensorRef { uint64_t offset, size; int64_t cols=0; };   // аг2
static std::map<std::string, TensorRef> g_tensors;
static uint64_t g_data_base; static std::string g_path;

static void arch_check_tensors(const std::string& f){
    // ад1
    for(int l=0;l<N_LAYERS;++l) for(auto& [slot,nmv] : A.nm){ auto b=nmv.find('['); if(b==std::string::npos) continue;
        std::string base=LPX(l)+nmv.substr(0,b), key=LPX(l)+nmv; long r0=0,r1=0;
        if(sscanf(nmv.c_str()+b,"[%ld:%ld]",&r0,&r1)!=2) arch_fail(f,"Окно «"+nmv+"» не разобрано.");
        auto it=g_tensors.find(base); if(it==g_tensors.end()) arch_fail(f,"В весах нет тензора «"+base+"».");
        TensorRef t=it->second; uint64_t row=(uint64_t)t.cols*2;   // ад2
        if(!row || r0<0 || r1<=r0 || (uint64_t)r1*row>t.size) arch_fail(f,"Окно «"+nmv+"» выходит за пределы тензора «"+base+"».");
        t.offset+=(uint64_t)r0*row; t.size=(uint64_t)(r1-r0)*row; g_tensors[key]=t; }
    auto need=[&](const std::string& t){ if(!g_tensors.count(t)) arch_fail(f,"В весах нет тензора «"+t+"»: возможно, описание относится к другой модели. Для пересоздания: python3 arch.py <папка модели>."); };
    need(A.pre+arch_tn("embed")); need(A.pre+arch_tn("final")); if(!A.tied) need(A.head_name);
    for(int l=0;l<N_LAYERS;++l){ for(const char* sl : {"iln","q","k","o","gate","up","down"}) if(arch_has(sl)) need(LPX(l)+arch_tn(sl)); if(A.L[l].has_v) need(LPX(l)+arch_tn("v")); } }
static size_t skip_ws(const std::string& s, size_t i){ while(i<s.size()&&(s[i]==' '||s[i]=='\n'||s[i]=='\t'||s[i]=='\r'))++i; return i; }
static std::string parse_string(const std::string& s, size_t& i){ ++i; std::string o; while(s[i]!='"'){if(s[i]=='\\'){++i;o+=s[i];++i;}else{o+=s[i];++i;}} ++i; return o; }
static std::vector<int64_t> parse_int_array(const std::string& s, size_t& i){ std::vector<int64_t> o; ++i; i=skip_ws(s,i); while(s[i]!=']'){size_t st=i; while(s[i]!=','&&s[i]!=']')++i; o.push_back(strtoll(s.substr(st,i-st).c_str(),nullptr,10)); if(s[i]==','){++i;i=skip_ws(s,i);}} ++i; return o; }

static void parse_header(const char* path){
    g_path=path;
    int fd=open(path,O_RDONLY); uint64_t hl; pread(fd,&hl,8,0);
    std::string h(hl,'\0'); pread(fd,h.data(),hl,8); close(fd);
    g_data_base=8+hl;
    size_t i=skip_ws(h,0); ++i; i=skip_ws(h,i);
    while(h[i]!='}'){
        std::string key=parse_string(h,i); i=skip_ws(h,i); ++i; i=skip_ws(h,i);
        if(key=="__metadata__"){ ++i; int d=1; while(d>0){if(h[i]=='{')++d; else if(h[i]=='}')--d; ++i;} }
        else{
            TensorRef t; ++i; i=skip_ws(h,i);
            while(h[i]!='}'){
                std::string f=parse_string(h,i); i=skip_ws(h,i); ++i; i=skip_ws(h,i);
                if(f=="shape"){ auto sh=parse_int_array(h,i); if(!sh.empty()) t.cols=sh.back(); }
                else if(f=="dtype") parse_string(h,i);
                else if(f=="data_offsets"){ auto off=parse_int_array(h,i); t.offset=g_data_base+off[0]; t.size=off[1]-off[0]; }
                i=skip_ws(h,i); if(h[i]==','){++i;i=skip_ws(h,i);}
            }
            ++i; g_tensors[key]=t;
        }
        i=skip_ws(h,i); if(h[i]==','){++i;i=skip_ws(h,i);}
    }
}

// ам1
struct QRef { int type, rows, cols; uint64_t offset, size; size_t rb; };
static std::map<std::string,QRef> g_tq; static std::string g_tq_path;
static bool tq_load(const std::string& path){
    int fd=open(path.c_str(),O_RDONLY); if(fd<0){ perror(path.c_str()); return false; }
    char first[128]={0}; if(pread(fd,first,sizeof first-1,0)<8){ close(fd); return false; }
    size_t hb=0,n=0;
    if(sscanf(first,"TQB1 %zu %zu",&hb,&n)!=2 || hb<64 || hb>(64ull<<20)){ fprintf(stderr,"%s: файл не является квантованной базой TierLM.\n",path.c_str()); close(fd); return false; }
    std::string h(hb,'\0'); size_t t=0; while(t<hb){ ssize_t r=pread(fd,h.data()+t,hb-t,t); if(r<=0) break; t+=r; } close(fd);
    std::istringstream is(h.substr(0,h.find('\0'))); std::string line; std::getline(is,line); std::map<std::string,int> cnt;
    while(std::getline(is,line)){ if(line.empty()||line[0]=='#') continue;
        char nm[256],ty[32]; int rows,cols; unsigned long long off,sz;
        if(sscanf(line.c_str(),"%255s %31s %d %d %llu %llu",nm,ty,&rows,&cols,&off,&sz)!=6) continue;
        const QInfo* qi=q_info(ty);
        if(!qi || !q_cols_ok(qi->type,cols) || q_row_bytes(qi->type,cols)*(size_t)rows!=sz){ fprintf(stderr,"%s: матрица %s типа %s не разобрана.\n",path.c_str(),nm,ty); return false; }
        g_tq[nm]={qi->type,rows,cols,off,sz,q_row_bytes(qi->type,cols)}; ++cnt[ty]; }
    g_tq_path=path; size_t tot=0; for(auto& [k,q] : g_tq) tot+=q.size;
    std::string mix; for(auto& [k,c] : cnt) mix+=(mix.empty()?"":", ")+k+" ×"+std::to_string(c);
    fprintf(stderr,"Квантованная база: матриц %zu, %.2f ГиБ (%s).\n",g_tq.size(),tot/1073741824.0,mix.c_str()); return !g_tq.empty(); }
static const QRef* tq_of(const std::string& nm){ auto it=g_tq.find(nm); return it==g_tq.end()?nullptr:&it->second; }
// ап1
template<typename T> __global__ void dq_k(const uint8_t* W, int type, size_t rb, int cols, size_t r0, size_t n, T* out){
    size_t i=blockIdx.x*(size_t)blockDim.x+threadIdx.x; if(i>=n) return; size_t r=r0+i/cols; int c=(int)(i%cols);
    out[i]=T(q_val(W+r*rb,type,c)); }
template<typename T> static void dq_launch(const void* W, const QRef& Q, size_t r0, size_t nrows, T* out, hipStream_t s=0){
    size_t n=nrows*Q.cols; dq_k<T><<<(unsigned)((n+255)/256),256,0,s>>>((const uint8_t*)W,Q.type,Q.rb,Q.cols,r0,n,out); }
// ас1
__global__ void gemv_q(const uint8_t* W, int type, size_t rb, const float* x, float* y, int N, int K){
    int wpb=blockDim.x/warpSize, row=blockIdx.x*wpb+threadIdx.x/warpSize, lane=threadIdx.x%warpSize;
    if(row>=N) return;
    const uint8_t* w=W+(size_t)row*rb; float acc=0;
    for(int k=lane;k<K;k+=warpSize) acc+=q_val(w,type,k)*x[k];
    for(int o=warpSize/2;o>0;o>>=1) acc+=__shfl_down(acc,o);
    if(lane==0) y[row]=acc; }

__global__ void bf16_to_f32_kernel(const hip_bfloat16* in, float* out, size_t n){ /* ат1 */
    size_t i=(blockIdx.x*(size_t)blockDim.x+threadIdx.x)*4;
    if(i+3<n){ const ushort4 v=*(const ushort4*)(in+i); float4 o;
        o.x=__uint_as_float((unsigned)v.x<<16); o.y=__uint_as_float((unsigned)v.y<<16);
        o.z=__uint_as_float((unsigned)v.z<<16); o.w=__uint_as_float((unsigned)v.w<<16); *(float4*)(out+i)=o; }
    else for(size_t k=i;k<n && k<i+4;++k) out[k]=(float)in[k]; }

__global__ void fp16_to_f32_kernel(const tier_fp16* in, float* out, size_t n){
    size_t i=blockIdx.x*(size_t)blockDim.x+threadIdx.x; if(i<n) out[i]=(float)in[i]; }
__global__ void bf16_to_fp16_kernel(const hip_bfloat16* in, tier_fp16* out, size_t n){
    size_t i=blockIdx.x*(size_t)blockDim.x+threadIdx.x; if(i<n) out[i]=tier_fp16((float)in[i]); }
__global__ void f32_to_fp16_kernel(const float* in, tier_fp16* out, size_t n){
    size_t i=blockIdx.x*(size_t)blockDim.x+threadIdx.x; if(i<n) out[i]=tier_fp16(in[i]); }
__global__ void fp16_to_bf16_kernel(const tier_fp16* in, hip_bfloat16* out, size_t n){
    size_t i=blockIdx.x*(size_t)blockDim.x+threadIdx.x; if(i<n) out[i]=hip_bfloat16((float)in[i]); }
static hip_bfloat16* fp16_to_bf16_temp(const tier_fp16* in, size_t n, hipStream_t stream=0){
    if(n>g_fp16_bf16_shadow_n){
        hip_bfloat16* next=nullptr;
        if(hipMalloc(&next,n*sizeof(hip_bfloat16))!=hipSuccess){ (void)hipGetLastError(); return nullptr; }
        if(g_fp16_bf16_shadow) (void)hipFree(g_fp16_bf16_shadow);
        g_fp16_bf16_shadow=next; g_fp16_bf16_shadow_n=n;
    }
    const unsigned th=256, bl=(unsigned)((n+th-1)/th);
    fp16_to_bf16_kernel<<<bl,th,0,stream>>>(in,g_fp16_bf16_shadow,n);
    if(hipStreamSynchronize(stream)!=hipSuccess){ (void)hipGetLastError(); return nullptr; }
    return g_fp16_bf16_shadow;
}
static void low16_to_f32(const void* in, float* out, size_t n){
    int th=256; unsigned bl=(unsigned)((n+th-1)/th);
    if(g_fp16w.count(in)) fp16_to_f32_kernel<<<bl,th>>>((const tier_fp16*)in,out,n);
    else bf16_to_f32_kernel<<<(unsigned)((n/4+th)/th),th>>>((const hip_bfloat16*)in,out,n);
}
// аш1
static size_t g_vram_peak=0, g_vram_total=0, g_vram_peak_need=0, g_rbf_bytes=0;   // аш2
static size_t g_free_ok=0; static size_t g_free_max=0, g_run_len=0, g_pf_lost=0;   // аш3
 static int g_probe_pack=0;   // аш4
 // аш5
// аэ1
static size_t step_transient(size_t used_now, size_t freed=0){ if(!g_vram_peak||g_vram_peak<=used_now+freed) return 0;   // аэ2
    return (size_t)((g_vram_peak-used_now-freed)*1.1)+(128ull<<20); }   // аэ3
// аэ4
static std::string g_hwmon;
// аэ5
static int g_dev=0; static char g_pci[32]="";
static void gpu_pick(){
    int n=0; hipError_t e=hipGetDeviceCount(&n);
    if(e!=hipSuccess || n<=0){ fprintf(stderr,"Видеокарта не найдена (%s): отсутствует драйвер, нет прав на устройство или карта не поддерживается этой сборкой.\n",
                                         e!=hipSuccess?hipGetErrorString(e):"0 устройств"); exit(1); }
    if(const char* s=getenv("TIER_GPU_INDEX")) g_dev=std::max(0,std::min(n-1,atoi(s)));
    else { size_t best=0; for(int i=0;i<n;++i){ hipDeviceProp_t p; if(hipGetDeviceProperties(&p,i)!=hipSuccess) continue;
               size_t m=p.integrated?p.totalGlobalMem/4:p.totalGlobalMem; if(m>best){ best=m; g_dev=i; } } }
    if(hipSetDevice(g_dev)!=hipSuccess){ fprintf(stderr,"Не удалось выбрать карту %d.\n",g_dev); exit(1); }
    hipDeviceProp_t p; if(hipGetDeviceProperties(&p,g_dev)==hipSuccess){
        snprintf(g_pci,sizeof g_pci,"%04x:%02x:%02x.0",(unsigned)p.pciDomainID,(unsigned)p.pciBusID,(unsigned)p.pciDeviceID);
        if(n>1) fprintf(stderr,"Карт: %d, расчёт на карте №%d — %s, %.1f ГиБ.\n",n,g_dev,p.name,p.totalGlobalMem/1073741824.0); }
}
static int hw_read(const char* f){ if(g_hwmon.empty()) return -1; FILE* h=fopen((g_hwmon+f).c_str(),"r"); if(!h) return -1; int v=-1; if(fscanf(h,"%d",&v)!=1) v=-1; fclose(h); return v; }
static bool g_thermal_stop=false; static double tclk(){ return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
static int g_t_edge=-1, g_t_junc=-1, g_t_mem=-1, g_fan=-1;
static std::string g_tf_edge="temp1_input", g_tf_junc="temp2_input", g_tf_mem="temp3_input";   // ба1
static void hwmon_labels(){ for(int i=1;i<=8;++i){ std::string f=g_hwmon+"temp"+std::to_string(i)+"_label"; FILE* h=fopen(f.c_str(),"r"); if(!h) continue; char n[32]={0}; if(fscanf(h,"%31s",n)==1){ std::string L=n, in="temp"+std::to_string(i)+"_input";
        if(L=="edge") g_tf_edge=in; else if(L=="junction"||L=="hotspot") g_tf_junc=in; else if(L=="mem") g_tf_mem=in; } fclose(h); } }
// бв1
struct TSens { std::string in; int crit; const char* what; };
static std::vector<TSens> g_tsens; static bool g_tguard_done=false;
static int rd_int(const std::string& f){ FILE* h=fopen(f.c_str(),"r"); if(!h) return -1; int v=-1; if(fscanf(h,"%d",&v)!=1) v=-1; fclose(h); return v; }
static void tguard_init(){ if(g_tguard_done) return; g_tguard_done=true; if(getenv("TIER_TEMP_GUARD") && !atoi(getenv("TIER_TEMP_GUARD"))) return;
    glob_t g; if(glob("/sys/class/hwmon/hwmon*/name",0,nullptr,&g)!=0) return;
    for(size_t i=0;i<g.gl_pathc;++i){ std::string d=g.gl_pathv[i]; d.resize(d.size()-4); char n[32]={0}; FILE* h=fopen(g.gl_pathv[i],"r"); if(!h) continue; if(fscanf(h,"%31s",n)!=1) n[0]=0; fclose(h);
        std::string N=n; char r1[PATH_MAX]={0}, r2[PATH_MAX]={0}; bool same = g_hwmon.empty() || (realpath(d.c_str(),r1) && realpath(g_hwmon.c_str(),r2) && !strcmp(r1,r2));
        bool gpu = N=="amdgpu" && same, cpu = N=="k10temp"||N=="coretemp"||N=="zenpower";
        if(!gpu && !cpu) continue;
        for(int k=1;k<=16;++k){ std::string in=d+"temp"+std::to_string(k)+"_input"; if(rd_int(in)<0) continue;
            int c=rd_int(d+"temp"+std::to_string(k)+"_crit"); if(c<=0 && N=="k10temp") c=95000; if(c<=0) continue;
            if(const char* e=getenv(cpu?"TIER_CPU_TMAX":"TIER_GPU_TMAX")) c=atoi(e)*1000;
            g_tsens.push_back({in,c,gpu?"карта":"процессор"}); } }
    globfree(&g);
    if(!g_tsens.empty()){ std::string l; for(auto& t:g_tsens) l+=std::string(l.empty()?"":", ")+t.what+" "+std::to_string(t.crit/1000); fprintf(stderr,"Сторож температуры: пределы %s °C (пауза за 5 градусов до предела).\n",l.c_str()); } }
static void tguard(){ if(g_tsens.empty()) return; static double last=0; double t=tclk(); if(t-last<1.0) return; last=t;
    for(auto& s : g_tsens){ int v=rd_int(s.in); if(v<0 || v < s.crit-5000) continue;
        HIP_CHECK(hipDeviceSynchronize()); fprintf(stderr,"Перегрев: %s %d °C, предел %d °C. Счёт на паузе до остывания.\n",s.what,v/1000,s.crit/1000);
        for(int w=0; w<300; ++w){ sleep(1); v=rd_int(s.in); if(v>=0 && v < s.crit-15000){ fprintf(stderr,"Остыло: %d °C. Продолжение.\n",v/1000); break; } }
        if(v >= s.crit-15000){ fprintf(stderr,"За 5 минут не остыло (%d °C): работа остановлена, чтобы не повредить оборудование.\n",v/1000); g_thermal_stop=true; } } }
static void hwmon_init(){
    glob_t g;
    if(g_pci[0]){ std::string pat=std::string("/sys/bus/pci/devices/")+g_pci+"/hwmon/hwmon*/name";   // бж1
        if(glob(pat.c_str(),0,nullptr,&g)==0){ for(size_t i=0;i<g.gl_pathc;++i){ char n[64]={0}; FILE* h=fopen(g.gl_pathv[i],"r"); if(!h) continue;
                if(fscanf(h,"%63s",n)==1 && std::string(n)=="amdgpu"){ g_hwmon=g.gl_pathv[i]; g_hwmon.resize(g_hwmon.size()-4); } fclose(h); if(!g_hwmon.empty()) break; }
            globfree(&g); if(!g_hwmon.empty()){ hwmon_labels(); return; } } }
    if(glob("/sys/class/drm/card*/device/hwmon/hwmon*/name",0,nullptr,&g)!=0) return;
    for(size_t i=0;i<g.gl_pathc;++i){ char n[64]={0}; FILE* h=fopen(g.gl_pathv[i],"r"); if(!h) continue; fscanf(h,"%63s",n); fclose(h);
        if(std::string(n)=="amdgpu"){ g_hwmon=g.gl_pathv[i]; g_hwmon.resize(g_hwmon.size()-4); break; } }
    globfree(&g); if(!g_hwmon.empty()) hwmon_labels();
}
static std::map<int,size_t> g_peak_by_line, g_dpeak; static size_t g_peak_live=0; // бж2
static void vram_sample(){
    size_t f,t; if(hipMemGetInfo(&f,&t)==hipSuccess){ g_vram_total=t;
        if(f < vram_slack()) vram_relief();   // бз1
        { size_t u=t-f, nd = u>g_rbf_bytes ? u-g_rbf_bytes : 0; if(nd>g_vram_peak_need) g_vram_peak_need=nd; }
        if(t-f>g_vram_peak){ g_vram_peak=t-f; g_peak_by_line.clear(); g_peak_live=0;
            for(auto& kv : g_live){ g_peak_by_line[kv.second.line]+=kv.second.n; g_peak_live+=kv.second.n; }
#if !defined(TIER_CUDA)
            g_dpeak.clear(); for(auto& [q,v] : g_dmap) g_dpeak[v.second]+=v.first;
#endif
        } }
    static int tick=0; if(++tick%10) return;   /* бз2 */
    g_t_edge=std::max(g_t_edge,hw_read(g_tf_edge.c_str())); g_t_junc=std::max(g_t_junc,hw_read(g_tf_junc.c_str()));
    tguard();
    g_t_mem=std::max(g_t_mem,hw_read(g_tf_mem.c_str())); g_fan=std::max(g_fan,hw_read("fan1_input"));
}
// бз3
static std::map<std::string,void*> g_ram;
// бз4
static bool g_use_pf=false, g_pf_on=false, g_pf_ready=false; static hipStream_t g_pf=nullptr; static hipEvent_t g_pf_ev=nullptr, g_conv_ev=nullptr;
static char* g_stg=nullptr; static size_t g_stg_cap=0; static std::map<std::string,void*> g_pfm; static size_t g_pf_hit=0, g_pf_miss=0;
static hip_bfloat16* g_dbf=nullptr; static size_t g_dbf_n=0;
static size_t g_ram_bytes=0;
static size_t g_h2d=0;   // бз5
// бз6
static int g_cap_pct=98; static size_t g_snap_bytes=0;   // бз7
struct HostMem { size_t total=0, avail=0; };
static HostMem host_mem(){ HostMem m; FILE* f=fopen("/proc/meminfo","r"); if(!f) return m; char ln[160]; unsigned long long v;
    while(fgets(ln,sizeof ln,f)){ if(sscanf(ln,"MemTotal: %llu",&v)==1) m.total=(size_t)v<<10; else if(sscanf(ln,"MemAvailable: %llu",&v)==1) m.avail=(size_t)v<<10; }
    fclose(f); return m; }
static size_t cap_slack(size_t total){ return total/100*(size_t)(100-std::min(99,std::max(50,g_cap_pct))); }   // бл1
// бл2
static size_t g_others0=0; static bool g_others0_set=false;
// бм1
static size_t sys_vram_used(){ if(!g_pci[0]) return 0; unsigned long long v=0; FILE* f=fopen((std::string("/sys/bus/pci/devices/")+g_pci+"/mem_info_vram_used").c_str(),"r");
    if(f){ if(fscanf(f,"%llu",&v)!=1) v=0; fclose(f); } return (size_t)v; }   // бм2
static size_t display_reserve(){ static size_t r=0; static bool done=false; if(done) return r; done=true;
    size_t scan=0; int outs=0; glob_t g;
    std::string pat = g_pci[0] ? std::string("/sys/bus/pci/devices/")+g_pci+"/drm/card*/card*-*/enabled" : std::string("/sys/class/drm/card*-*/enabled");
    if(glob(pat.c_str(),0,nullptr,&g)==0){ for(size_t i=0;i<g.gl_pathc;++i){ char st[16]={0}; FILE* f=fopen(g.gl_pathv[i],"r"); if(!f) continue; if(fscanf(f,"%15s",st)!=1) st[0]=0; fclose(f);
            if(strcmp(st,"enabled")) continue; std::string d=g.gl_pathv[i]; d.resize(d.size()-7); int w=0,h=0; FILE* m=fopen((d+"modes").c_str(),"r");
            if(m){ if(fscanf(m,"%dx%d",&w,&h)!=2) w=h=0; fclose(m); } if(w>0&&h>0){ scan+=(size_t)w*h*4*3; ++outs; } } globfree(&g); }
    r = scan + 2*g_others0 + (128ull<<20);
    if(outs) fprintf(stderr,"Запас под экран: %.0f МиБ (выходов %d, кадровые буферы %.0f МиБ, рабочий стол и другие программы %.0f МиБ).\n",r/1048576.0,outs,scan/1048576.0,g_others0/1048576.0);
    return r; }
static size_t vram_slack(){ size_t fr=0,tt=0; if(hipMemGetInfo(&fr,&tt)!=hipSuccess) return 0; return std::max(cap_slack(tt),display_reserve()); }
static char* g_arena=nullptr; static size_t g_arena_cap=0, g_arena_used=0;
static void arena_init(){   // бп1
    if(g_arena) return;
    for(auto& [k,r] : g_tensors) if(k.find(".layers.")!=std::string::npos){ const QRef* q=tq_of(k); g_arena_cap+=((q?q->size:r.size)+255)/256*256; }   // бп2
    // бп3
    { struct sysinfo si;
      if(sysinfo(&si)==0){ size_t ram=(size_t)si.totalram*si.mem_unit, FLOOR=g_snap_bytes + std::max<size_t>(cap_slack(ram),1ull<<30);
          if(const char* e=getenv("TIER_RAM_FLOOR_GIB")) FLOOR=(size_t)(atof(e)*1073741824.0);
          if(ram>FLOOR && g_arena_cap>ram-FLOOR){
              fprintf(stderr,"Кэш весов: %.1f ГиБ не помещается; системе остаётся %.1f ГиБ, под кэш выделяется %.1f ГиБ, остальное читается с диска на каждом проходе.\n",
                      g_arena_cap/1073741824.0, FLOOR/1073741824.0, (ram-FLOOR)/1073741824.0);
              g_arena_cap=ram-FLOOR; }
          else if(ram<=FLOOR) g_arena_cap=0; } }
    { HostMem m=host_mem(); if(m.total){   // бп4
          size_t other=1ull<<30; for(auto& [k,r] : g_tensors) if(k.find(".layers.")==std::string::npos) other+=r.size;   // бп5
          if(const char* e=getenv("TIER_RAM_OTHER_GIB")) other=(size_t)(atof(e)*1073741824.0);
          size_t slack=cap_slack(m.total), room = m.avail>other+slack ? m.avail-other-slack : 0;
          if(g_arena_cap>room){
              fprintf(stderr,"Кэш весов: потолок памяти %d%%, свободно сейчас %.1f ГиБ, системе остаётся %.1f, процессу требуется ещё %.1f; под кэш выделяется %.1f из %.1f ГиБ, остальное читается с диска.\n",
                      g_cap_pct,m.avail/1073741824.0,slack/1073741824.0,other/1073741824.0,room/1073741824.0,g_arena_cap/1073741824.0);
              g_arena_cap=room; } } }
    if(hipHostMalloc((void**)&g_arena,g_arena_cap)!=hipSuccess){ g_arena=(char*)malloc(g_arena_cap);
        fprintf(stderr,"Кэш весов: не удалось закрепить %.1f ГиБ, используется обычная память.\n",g_arena_cap/1073741824.0); }
}
// бр1
static float* load_tensor_f32(const std::string& name, void* stage, size_t cap, bool cache=false){
    // бр2
    const QRef* Q = cache ? tq_of(name) : nullptr;
    const TensorRef ref = Q ? TensorRef{Q->offset,Q->size} : g_tensors.at(name);
    const size_t n = Q ? (size_t)Q->rows*Q->cols : ref.size/2;
    const bool proj = name.size()>12 && name.compare(name.size()-12,12,"_proj.weight")==0;
    const bool low_proj = cache && proj && (g_fp16_storage || g_bf3);
    const bool low_norm = cache && is_norm_tensor(name) && g_fp16_storage;
    const bool low_weight = low_proj || low_norm;
    if(low_proj){ if(auto rp=g_rbf.find(name); rp!=g_rbf.end()){   // бр3
        if(!Q){ g_rbf_inuse.insert(rp->second); return (float*)rp->second; }   // бр4
        if(g_fp16_storage){ tier_fp16* wb; HIP_CHECK(cmalloc(&wb,n*2)); dq_launch(rp->second,*Q,0,Q->rows,wb); HIP_CHECK(hipStreamSynchronize(0)); g_bf16w.insert(wb); g_fp16w.insert(wb); return (float*)wb; }
        hip_bfloat16* wb; HIP_CHECK(cmalloc(&wb,n*2)); dq_launch(rp->second,*Q,0,Q->rows,wb); HIP_CHECK(hipStreamSynchronize(0)); g_bf16w.insert(wb); return (float*)wb; } }
    void* src; bool disk=false;   // бр5
    if(auto it=g_ram.find(name); it!=g_ram.end()) src=it->second;
    else {
        src=stage;
        if(cache){
            arena_init();
            if(g_arena && g_arena_used+ref.size<=g_arena_cap){ src=g_arena+g_arena_used; g_arena_used+=(ref.size+255)/256*256; g_ram[name]=src; g_ram_bytes+=ref.size; }
            else { // бр6
                static size_t streamed=0; if(!streamed++) fprintf(stderr,"Недостаточно ОЗУ для кэша весов: часть матриц читается с диска при каждом проходе.\n");
                disk=true; } }
        if((!cache || disk) && ref.size>cap){fprintf(stderr,"Буфер подкачки слишком мал: требуется %lu байт, имеется %zu байт.\n",ref.size,cap);exit(1);}
        static int fd0=-1, fdq=-1; int& fd = Q ? fdq : fd0;   /* бр7 */
        if(fd<0){ fd=open((Q?g_tq_path:g_path).c_str(),O_RDONLY); if(fd>=0) posix_fadvise(fd,0,0,POSIX_FADV_SEQUENTIAL); }
        uint64_t t=0;
        while(t<ref.size){ ssize_t r=pread(fd,(char*)src+t,ref.size-t,ref.offset+t); if(r<=0){perror("Чтение (pread)");exit(1);} t+=r; }
        // бр8
        static uint64_t drop_from=0;
        if(cache && !disk && !Q){
            if(ref.offset<drop_from) drop_from=ref.offset;
            else if(ref.offset-drop_from>=(256ull<<20)){
                posix_fadvise(fd, drop_from, ref.offset-drop_from, POSIX_FADV_DONTNEED); drop_from=ref.offset; } }
    }
    const hip_bfloat16* dbf;
    if(auto p=g_pfm.find(name); p!=g_pfm.end()){ HIP_CHECK(hipStreamWaitEvent(0,g_pf_ev,0)); dbf=(const hip_bfloat16*)p->second; g_pfm.erase(p); ++g_pf_hit; } // бр9
    else { if(g_pf_on && cache) ++g_pf_miss; // бр10
        if(ref.size>g_dbf_n){ if(g_dbf) HIP_CHECK(hipFree(g_dbf)); HIP_CHECK(hipMalloc(&g_dbf,ref.size)); g_dbf_n=ref.size; }
        HIP_CHECK(hipMemcpy(g_dbf,src,ref.size,hipMemcpyHostToDevice)); g_h2d+=ref.size; dbf=g_dbf; }
    if(Q){   // бр11
        if(low_proj){
            if(g_fp16_storage){ tier_fp16* wb; HIP_CHECK(cmalloc(&wb,n*2)); dq_launch(dbf,*Q,0,Q->rows,wb); HIP_CHECK(hipStreamSynchronize(0)); g_bf16w.insert(wb); g_fp16w.insert(wb); return (float*)wb; }
            hip_bfloat16* wb; HIP_CHECK(cmalloc(&wb,n*2)); dq_launch(dbf,*Q,0,Q->rows,wb); HIP_CHECK(hipStreamSynchronize(0)); g_bf16w.insert(wb); return (float*)wb;
        }
        float* df; HIP_CHECK(cmalloc(&df,n*sizeof(float))); dq_launch(dbf,*Q,0,Q->rows,df); HIP_CHECK(hipStreamSynchronize(0)); return df; }
    if(low_weight){
        if(g_fp16_storage){ tier_fp16* wb; HIP_CHECK(cmalloc(&wb,n*2)); bf16_to_fp16_kernel<<<(unsigned)((n+255)/256),256>>>(dbf,wb,n); HIP_CHECK(hipStreamSynchronize(0)); g_bf16w.insert(wb); g_fp16w.insert(wb); return (float*)wb; }
        hip_bfloat16* wb; HIP_CHECK(cmalloc(&wb,n*2)); HIP_CHECK(hipMemcpyAsync(wb,dbf,n*2,hipMemcpyDeviceToDevice,0)); HIP_CHECK(hipStreamSynchronize(0));
        g_bf16w.insert(wb); return (float*)wb; } // бр12
    float* df; HIP_CHECK(cmalloc(&df,n*sizeof(float)));
    int th=256,bl=(int)((n/4+th)/th); bf16_to_f32_kernel<<<bl,th>>>(dbf,df,n); HIP_CHECK(hipStreamSynchronize(0));
    return df;
}

__device__ inline float norm_weight_at(const void* w, int ty, int i){
    if(ty==2) return (float)((const tier_fp16*)w)[i];
    if(ty==1) return (float)((const hip_bfloat16*)w)[i];
    return ((const float*)w)[i];
}
// бт1
__global__ void rmsnorm_fwd(const float* x, const void* w, float* y, float* r_out, float* n_out, int dim, int wtype){
    BLOCK_MUST_BE(NB_ROW);
    int row=blockIdx.x; const float* xr=x+(size_t)row*dim; float* yr=y+(size_t)row*dim;
    __shared__ float warp_sums[32]; int lane=threadIdx.x%warpSize, wid=threadIdx.x/warpSize;
    int nw=(NB_ROW+warpSize-1)/warpSize;
    float local=0; for(int i=threadIdx.x;i<dim;i+=NB_ROW) local+=xr[i]*xr[i];
    for(int o=warpSize/2;o>0;o>>=1) local+=__shfl_down(local,o);
    if(lane==0) warp_sums[wid]=local; __syncthreads();
    __shared__ float total; if(threadIdx.x==0){float s=0;for(int i=0;i<nw;++i)s+=warp_sums[i]; total=s;} __syncthreads();
    float r=rsqrtf(total/dim+g_eps_d);
    if(threadIdx.x==0) r_out[row]=r;
    for(int i=threadIdx.x;i<dim;i+=NB_ROW){ float n=xr[i]*r; n_out[(size_t)row*dim+i]=n; yr[i]=n*(norm_weight_at(w,wtype,i)+g_w1p); }
}
__global__ void rmsnorm_bwd(const float* dy, const float* n, const void* w, const float* r,
                             float* dx, float* dw_accum, int dim, int wtype){
    int row=blockIdx.x; const float* dyr=dy+(size_t)row*dim; const float* nr=n+(size_t)row*dim;
    float* dxr=dx+(size_t)row*dim;
    __shared__ float warp_sums[32]; int lane=threadIdx.x%warpSize, wid=threadIdx.x/warpSize;
    int nw=(blockDim.x+warpSize-1)/warpSize;
    float local=0;
    for(int i=threadIdx.x;i<dim;i+=blockDim.x){ float dn=dyr[i]*(norm_weight_at(w,wtype,i)+g_w1p); local+=dn*nr[i]; atomicAdd(&dw_accum[i], dyr[i]*nr[i]); }
    for(int o=warpSize/2;o>0;o>>=1) local+=__shfl_down(local,o);
    if(lane==0) warp_sums[wid]=local; __syncthreads();
    __shared__ float mean_dn_n; if(threadIdx.x==0){float s=0;for(int i=0;i<nw;++i)s+=warp_sums[i]; mean_dn_n=s/dim;} __syncthreads();
    float rr=r[row];
    for(int i=threadIdx.x;i<dim;i+=blockDim.x){ float dn=dyr[i]*(norm_weight_at(w,wtype,i)+g_w1p); dxr[i]=rr*(dn - nr[i]*mean_dn_n); }
}
__global__ void head_rmsnorm_fwd(const float* x, const void* w, float* y, float* r_out, float* n_out, int n_heads, int head_dim, int wtype){
    BLOCK_MUST_BE(NB_HEAD);
    int tok=blockIdx.x, head=blockIdx.y;
    const float* v=x+((size_t)tok*n_heads+head)*head_dim;
    float* yv=y+((size_t)tok*n_heads+head)*head_dim; float* nv=n_out+((size_t)tok*n_heads+head)*head_dim;
    __shared__ float rr;
    __shared__ float red_[NB_HEAD]; { float a_=0; for(int i=threadIdx.x;i<head_dim;i+=NB_HEAD) a_+=v[i]*v[i]; red_[threadIdx.x]=a_; __syncthreads();
      for(int k=NB_HEAD/2;k>0;k>>=1){ if(threadIdx.x<k) red_[threadIdx.x]+=red_[threadIdx.x+k]; __syncthreads(); } }
    if(threadIdx.x==0){ rr=rsqrtf(red_[0]/head_dim+g_eps_d); r_out[tok*n_heads+head]=rr; }
    __syncthreads();
    for(int i=threadIdx.x;i<head_dim;i+=NB_HEAD){ float n=v[i]*rr; nv[i]=n; yv[i]=n*(norm_weight_at(w,wtype,i)+g_w1p); }
}
__global__ void head_rmsnorm_bwd(const float* dy, const float* n, const void* w, const float* r,
                                  float* dx, float* dw_accum, int n_heads, int head_dim, int wtype){
    int tok=blockIdx.x, head=blockIdx.y; size_t o=((size_t)tok*n_heads+head)*head_dim;
    const float* dyv=dy+o; const float* nv=n+o; float* dxv=dx+o;
    __shared__ float red[64]; float a=0;                                   // бх1
    for(int i=threadIdx.x;i<head_dim;i+=blockDim.x) a+=dyv[i]*(norm_weight_at(w,wtype,i)+g_w1p)*nv[i];
    red[threadIdx.x]=a; __syncthreads();
    for(int k=blockDim.x/2;k>0;k>>=1){ if(threadIdx.x<k) red[threadIdx.x]+=red[threadIdx.x+k]; __syncthreads(); }
    float mean_dn_n=red[0]/head_dim, rr=r[tok*n_heads+head];
    for(int i=threadIdx.x;i<head_dim;i+=blockDim.x){ float dn=dyv[i]*(norm_weight_at(w,wtype,i)+g_w1p); dxv[i]=rr*(dn-nv[i]*mean_dn_n); }
}
// бц1
__global__ void head_norm_dw(const float* dy, const float* n, float* dw, int rows, int head_dim){
    for(int i=threadIdx.x;i<head_dim;i+=blockDim.x){ float a=0;
        for(int r=blockIdx.x;r<rows;r+=gridDim.x) a+=dy[(size_t)r*head_dim+i]*n[(size_t)r*head_dim+i];
        atomicAdd(&dw[i],a); }
}
__global__ void v_norm_fwd(const float* x, float* y, float* r_out, float* n_out, int n_heads, int head_dim){
    BLOCK_MUST_BE(NB_HEAD);
    int tok=blockIdx.x, head=blockIdx.y;
    const float* v=x+((size_t)tok*n_heads+head)*head_dim;
    float* yv=y+((size_t)tok*n_heads+head)*head_dim; float* nv=n_out+((size_t)tok*n_heads+head)*head_dim;
    __shared__ float rr;
    __shared__ float red_[NB_HEAD]; { float a_=0; for(int i=threadIdx.x;i<head_dim;i+=NB_HEAD) a_+=v[i]*v[i]; red_[threadIdx.x]=a_; __syncthreads();
      for(int k=NB_HEAD/2;k>0;k>>=1){ if(threadIdx.x<k) red_[threadIdx.x]+=red_[threadIdx.x+k]; __syncthreads(); } }
    if(threadIdx.x==0){ rr=rsqrtf(red_[0]/head_dim+g_eps_d); r_out[tok*n_heads+head]=rr; }
    __syncthreads();
    for(int i=threadIdx.x;i<head_dim;i+=NB_HEAD){ float n=v[i]*rr; nv[i]=n; yv[i]=n; }
}
__global__ void v_norm_bwd(const float* dy, const float* n, const float* r, float* dx, int n_heads, int head_dim){
    int tok=blockIdx.x, head=blockIdx.y;
    const float* dyv=dy+((size_t)tok*n_heads+head)*head_dim; const float* nv=n+((size_t)tok*n_heads+head)*head_dim;
    float* dxv=dx+((size_t)tok*n_heads+head)*head_dim;
    __shared__ float mean_dn_n;
    __shared__ float red_[64]; { float a_=0; for(int i=threadIdx.x;i<head_dim;i+=blockDim.x) a_+=dyv[i]*nv[i]; red_[threadIdx.x]=a_; __syncthreads();
      for(int k=blockDim.x/2;k>0;k>>=1){ if(threadIdx.x<k) red_[threadIdx.x]+=red_[threadIdx.x+k]; __syncthreads(); } }
    if(threadIdx.x==0) mean_dn_n=red_[0]/head_dim;
    __syncthreads();
    float rr=r[tok*n_heads+head];
    for(int i=threadIdx.x;i<head_dim;i+=blockDim.x) dxv[i]=rr*(dyv[i]-nv[i]*mean_dn_n);
}
// бш1
__device__ int* g_pos_d=nullptr; __device__ int* g_seg_d=nullptr;
// бэ1
__global__ void set_pos_seg(int* p, int* s){ g_pos_d=p; g_seg_d=s; }
// бю1
__global__ void rope_fwd(float* qk, int seq_len, int n_heads, int head_dim, int rotary_dim, float theta, const float* invf, int half, float msc){
    int tok=blockIdx.x, head=blockIdx.y; float* v=qk+((size_t)tok*n_heads+head)*head_dim;
    int active=rotary_dim/2;   // бю2
    for(int i=threadIdx.x;i<active;i+=blockDim.x){
        float freq=invf?invf[i]:powf(theta,-2.0f*i/head_dim); float ang=(g_pos_d?g_pos_d[tok]:tok)*freq; float c=cosf(ang)*msc,s=sinf(ang)*msc;
        float a=v[i],b=v[i+half]; v[i]=a*c-b*s; v[i+half]=a*s+b*c;
    }
}
__global__ void rope_bwd(float* dqk, int seq_len, int n_heads, int head_dim, int rotary_dim, float theta, const float* invf, int half, float msc){
    int tok=blockIdx.x, head=blockIdx.y; float* v=dqk+((size_t)tok*n_heads+head)*head_dim;
    int active=rotary_dim/2;   // бя1
    for(int i=threadIdx.x;i<active;i+=blockDim.x){
        float freq=invf?invf[i]:powf(theta,-2.0f*i/head_dim); float ang=(g_pos_d?g_pos_d[tok]:tok)*freq; float c=cosf(ang)*msc,s=-sinf(ang)*msc;
        float a=v[i],b=v[i+half]; v[i]=a*c-b*s; v[i+half]=a*s+b*c;
    }
}
__global__ void attention_fwd(const float* Q, const float* K, const float* V, float* out, float* P_out,
                               int seq_len, int n_heads, int n_kv_heads, int head_dim, int window, float cap){
    int tok=blockIdx.x, head=blockIdx.y; int kv_head=head/(n_heads/n_kv_heads);
    extern __shared__ float scores[];
    const float* q=Q+((size_t)tok*n_heads+head)*head_dim;
    int lo=(window>0)?max(0,tok-window+1):0; if(g_seg_d) lo=max(lo,g_seg_d[tok]); int n_keys=tok-lo+1;
    for(int j=threadIdx.x;j<n_keys;j+=blockDim.x){
        int kpos=lo+j; const float* k=K+((size_t)kpos*n_kv_heads+kv_head)*head_dim;
        float dot=0; for(int d=0;d<head_dim;++d) dot+=q[d]*k[d]; scores[j]=cap>0 ? cap*tanhf(dot/cap) : dot;   // ва1
    }
    __syncthreads();
    __shared__ float mx,sum;
    if(threadIdx.x==0){ mx=-1e30f; for(int j=0;j<n_keys;++j) mx=fmaxf(mx,scores[j]);
        sum=0; for(int j=0;j<n_keys;++j){ scores[j]=expf(scores[j]-mx); sum+=scores[j]; } }
    __syncthreads();
    float* Prow=P_out+((size_t)tok*n_heads+head)*seq_len;
    for(int j=threadIdx.x;j<n_keys;j+=blockDim.x) Prow[lo+j]=scores[j]/sum;
    __syncthreads();
    float* o=out+((size_t)tok*n_heads+head)*head_dim;
    for(int d=threadIdx.x;d<head_dim;d+=blockDim.x){
        float acc=0; for(int j=0;j<n_keys;++j) acc+=scores[j]*V[((size_t)(lo+j)*n_kv_heads+kv_head)*head_dim+d];
        o[d]=acc/sum;
    }
}
// ва2
__global__ void attention_bwd(const float* dO, const float* P, const float* O, const float* Q, const float* K, const float* V,
                               float* dQ, float* dS_out, int seq_len, int n_heads, int n_kv_heads, int head_dim, int window, float cap){
    int tok=blockIdx.x, head=blockIdx.y; int kv_head=head/(n_heads/n_kv_heads);
    const float* dOr=dO+((size_t)tok*n_heads+head)*head_dim; const float* Or=O+((size_t)tok*n_heads+head)*head_dim;
    const float* Prow=P+((size_t)tok*n_heads+head)*seq_len; float* dSrow=dS_out+((size_t)tok*n_heads+head)*seq_len;
    int lo=(window>0)?max(0,tok-window+1):0; if(g_seg_d) lo=max(lo,g_seg_d[tok]); int n_keys=tok-lo+1;
    __shared__ float red[64]; float a=0;
    for(int d=threadIdx.x;d<head_dim;d+=blockDim.x) a+=dOr[d]*Or[d];
    red[threadIdx.x]=a; __syncthreads();
    for(int k=blockDim.x/2;k>0;k>>=1){ if(threadIdx.x<k) red[threadIdx.x]+=red[threadIdx.x+k]; __syncthreads(); }
    float D=red[0];
    extern __shared__ float dS[];
    for(int j=threadIdx.x;j<n_keys;j+=blockDim.x){
        int kpos=lo+j; const float* v=V+((size_t)kpos*n_kv_heads+kv_head)*head_dim;
        float dP=0; for(int d=0;d<head_dim;++d) dP+=dOr[d]*v[d];
        float g=Prow[kpos]*(dP-D);
        if(cap>0){ const float* qr=Q+((size_t)tok*n_heads+head)*head_dim; const float* k=K+((size_t)kpos*n_kv_heads+kv_head)*head_dim;
            float s=0; for(int d=0;d<head_dim;++d) s+=qr[d]*k[d]; float t=tanhf(s/cap); g*=1.f-t*t; }   // вб1
        dS[j]=g; dSrow[kpos]=g;
    }
    __syncthreads();
    float* dqr=dQ+((size_t)tok*n_heads+head)*head_dim;
    for(int d=threadIdx.x;d<head_dim;d+=blockDim.x){
        float acc=0; for(int j=0;j<n_keys;++j) acc+=dS[j]*K[((size_t)(lo+j)*n_kv_heads+kv_head)*head_dim+d];
        dqr[d]=acc;
    }
}
// вб2
__global__ void attention_bwd_kv(const float* dO, const float* P, const float* dS, const float* Q, float* dK, float* dV,
                                  int seq_len, int n_heads, int n_kv_heads, int head_dim, int window){
    int kpos=blockIdx.x, kvh=blockIdx.y; int g=n_heads/n_kv_heads;
    for(int d=threadIdx.x;d<head_dim;d+=blockDim.x){
        float ak=0, av=0;
        for(int tok=kpos;tok<seq_len;++tok){
            int lo=(window>0)?max(0,tok-window+1):0; if(g_seg_d) lo=max(lo,g_seg_d[tok]);
            if(lo>kpos) break;                                              // вв1
            for(int h=kvh*g;h<kvh*g+g;++h){ size_t pr=((size_t)tok*n_heads+h)*seq_len+kpos, qr=((size_t)tok*n_heads+h)*head_dim+d;
                av+=P[pr]*dO[qr]; ak+=dS[pr]*Q[qr]; }
        }
        dK[((size_t)kpos*n_kv_heads+kvh)*head_dim+d]=ak; dV[((size_t)kpos*n_kv_heads+kvh)*head_dim+d]=av;
    }
}
__global__ void gelu_tanh_mul_fwd(const float* gate, const float* up, float* act, size_t n){
    size_t i=blockIdx.x*(size_t)blockDim.x+threadIdx.x; if(i>=n) return;
    float g=gate[i]; float c=0.7978845608f*(g+0.044715f*g*g*g);
    act[i]=0.5f*g*(1.0f+tanhf(c))*up[i];
}
__global__ void gelu_tanh_mul_bwd(const float* dact, const float* gate, const float* up, float* dgate, float* dup, size_t n){
    size_t i=blockIdx.x*(size_t)blockDim.x+threadIdx.x; if(i>=n) return;
    float g=gate[i], u=up[i];
    float inner=0.7978845608f*(g+0.044715f*g*g*g); float t=tanhf(inner); float gelu=0.5f*g*(1.0f+t);
    float dgelu_dg=0.5f*(1.0f+t) + 0.5f*g*(1.0f-t*t)*0.7978845608f*(1.0f+3.0f*0.044715f*g*g);
    dup[i]=dact[i]*gelu; dgate[i]=dact[i]*u*dgelu_dg;
}
__global__ void silu_mul_fwd(const float* gate, const float* up, float* act, size_t n){
    size_t i=blockIdx.x*(size_t)blockDim.x+threadIdx.x; if(i>=n) return; float g=gate[i]; act[i]=g/(1.0f+expf(-g))*up[i]; }
__global__ void silu_mul_bwd(const float* dact, const float* gate, const float* up, float* dgate, float* dup, size_t n){
    size_t i=blockIdx.x*(size_t)blockDim.x+threadIdx.x; if(i>=n) return; float g=gate[i], sg=1.0f/(1.0f+expf(-g)), si=g*sg;
    dup[i]=dact[i]*si; dgate[i]=dact[i]*up[i]*sg*(1.0f+g*(1.0f-sg)); }
__global__ void add_bias_k(float* y, const float* b, size_t rows, int n){ size_t i=blockIdx.x*(size_t)blockDim.x+threadIdx.x; if(i<rows*n) y[i]+=b[i%n]; }
// ви1
__global__ void add_bias_typed_k(float* y, const void* b, size_t rows, int n, int ty){ size_t i=blockIdx.x*(size_t)blockDim.x+threadIdx.x; if(i<rows*n) y[i]+=norm_weight_at(b,ty,(int)(i%n)); }
__global__ void scale_k(float* a, float s, size_t n){ size_t i=blockIdx.x*(size_t)blockDim.x+threadIdx.x; if(i<n) a[i]*=s; }
__global__ void add_kernel(float* a, const float* b, size_t n){ size_t i=blockIdx.x*(size_t)blockDim.x+threadIdx.x; if(i<n) a[i]+=b[i]; }
__global__ void mul_scalar_kernel(float* a, const float* s, size_t n){ size_t i=blockIdx.x*(size_t)blockDim.x+threadIdx.x; if(i<n) a[i]*=s[0]; }
// вн1
__global__ void mul_scalar_typed_kernel(float* a, const void* s, size_t n, int wt){ size_t i=blockIdx.x*(size_t)blockDim.x+threadIdx.x; if(i>=n) return;
    float v = wt==2 ? (float)((const tier_fp16*)s)[0] : (wt==1 ? (float)((const hip_bfloat16*)s)[0] : ((const float*)s)[0]); a[i]*=v; }
__global__ void softcap_kernel(float* logits, float cap, size_t n){ size_t i=blockIdx.x*(size_t)blockDim.x+threadIdx.x; if(i<n) logits[i]=cap*tanhf(logits[i]/cap); }
// вп1
__global__ void softcap_bwd_kernel(const float* logits_pre_cap, float* dlogits, float cap, size_t n){
    size_t i=blockIdx.x*(size_t)blockDim.x+threadIdx.x; if(i>=n) return;
    float t=tanhf(logits_pre_cap[i]/cap);
    dlogits[i] *= (1.0f - t*t);
}
// вп2
__global__ void dlogit_kernel(const float* logits_postcap, const double* running_max, const double* running_sum,
                               const int32_t* tokens, float* dlogit, int seq_len, int vc, int v0, int n_pred, const unsigned char* lmask, float cap){
    size_t i = blockIdx.x*(size_t)blockDim.x + threadIdx.x;
    if(i >= (size_t)seq_len*vc) return;
    int t = i / vc, v = i % vc;
    if(t >= seq_len-1 || !lmask[t]){ dlogit[i] = 0.0f; return; } // вр1
    double lse = running_max[t] + log(running_sum[t]);
    double softmax_v = exp((double)logits_postcap[(size_t)t*vc+v] - lse);
    int gv = v0 + v;
    double onehot = (gv == tokens[t+1]) ? 1.0 : 0.0;
    float y=cap>0 ? logits_postcap[(size_t)t*vc+v]/cap : 0.f; // вр2
    dlogit[i] = (float)((softmax_v - onehot) / n_pred) * (1.0f - y*y) * (float)lmask[t];   // вр3
}
// вс1
__global__ void lse_chunk_kernel(const float* lg, double* rmax, double* rsum, int* amax, float* tgt, const int32_t* tokens, const unsigned char* lmask, int vc, int v0){
    int t=blockIdx.x; if(!lmask[t]) return; const float* row=lg+(size_t)t*vc; __shared__ float sm[256]; __shared__ int si[256]; __shared__ double sd[256];
    float m=-1e30f; int mi=0x7fffffff; for(int v=threadIdx.x;v<vc;v+=blockDim.x){ float x=row[v]; if(x>m){ m=x; mi=v; } }
    sm[threadIdx.x]=m; si[threadIdx.x]=mi; __syncthreads();
    for(int o=blockDim.x/2;o>0;o>>=1){ if(threadIdx.x<o){ float a=sm[threadIdx.x+o]; int ai=si[threadIdx.x+o]; if(a>sm[threadIdx.x]||(a==sm[threadIdx.x]&&ai<si[threadIdx.x])){ sm[threadIdx.x]=a; si[threadIdx.x]=ai; } } __syncthreads(); }
    float bm=sm[0]; double s=0; for(int v=threadIdx.x;v<vc;v+=blockDim.x) s+=exp((double)row[v]-bm);
    sd[threadIdx.x]=s; __syncthreads(); for(int o=blockDim.x/2;o>0;o>>=1){ if(threadIdx.x<o) sd[threadIdx.x]+=sd[threadIdx.x+o]; __syncthreads(); }
    if(threadIdx.x==0){ double om=rmax[t], nm=fmax(om,(double)bm); if(bm>om) amax[t]=v0+si[0];
        rsum[t]=rsum[t]*exp(om-nm)+sd[0]*exp((double)bm-nm); rmax[t]=nm; int g=tokens[t+1]-v0; if(g>=0&&g<vc) tgt[t]=row[g]; }
}
// вт1
__device__ inline void atomic_add_f64(double* p, double v){
#if defined(TIER_CUDA) && defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 600
    unsigned long long* a=(unsigned long long*)p; unsigned long long old=*a, cur;
    do { cur=old; old=atomicCAS(a,cur,__double_as_longlong(v+__longlong_as_double(cur))); } while(cur!=old);
#else
    atomicAdd(p,v);
#endif
}
__global__ void sum_reduce_kernel(const float* a, const float* b, double* out, size_t n){
    size_t i=blockIdx.x*(size_t)blockDim.x+threadIdx.x; if(i<n) atomic_add_f64(out, (double)a[i]*(double)b[i]);
}

// вф1
template<bool BT, bool ACC, int BK=8, int UK=BK> __global__ __launch_bounds__(256) void sg(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C, int M, int N, int K, int kc=0){
    // вф2
    if(kc){ A+=(size_t)blockIdx.z*kc; B+= BT ? (size_t)blockIdx.z*kc : (size_t)blockIdx.z*kc*N; C+=(size_t)blockIdx.z*M*N; }
    const int KT=K, KL=kc?kc:K; // вф3
    __shared__ float As[2][BK][132], Bs[2][BK][132];
    const int tid=threadIdx.x, tx=tid&15, ty=tid>>4, bm=blockIdx.y*128, bn=blockIdx.x*128;
    constexpr int R=BK/8; // вф4
    float4 ra[R], rb[R];
    auto ld=[&](int k0){
        #pragma unroll
        for(int q=0;q<R;++q){ int e=tid+q*256, ar=e/(BK/4), ak=(e%(BK/4))*4, bk=e>>5, bc=(e&31)*4; int r=bm+ar;
            ra[q]= r<M ? *(const float4*)(A+(size_t)r*KT+k0+ak) : make_float4(0,0,0,0);
            rb[q]= BT ? *(const float4*)(B+(size_t)(bn+ar)*KT+k0+ak) : *(const float4*)(B+(size_t)(k0+bk)*N+bn+bc); } };
    auto st=[&](int s){
        #pragma unroll
        for(int q=0;q<R;++q){ int e=tid+q*256, ar=e/(BK/4), ak=(e%(BK/4))*4, bk=e>>5, bc=(e&31)*4;
            As[s][ak][ar]=ra[q].x; As[s][ak+1][ar]=ra[q].y; As[s][ak+2][ar]=ra[q].z; As[s][ak+3][ar]=ra[q].w;
            if(BT){ Bs[s][ak][ar]=rb[q].x; Bs[s][ak+1][ar]=rb[q].y; Bs[s][ak+2][ar]=rb[q].z; Bs[s][ak+3][ar]=rb[q].w; } else *(float4*)&Bs[s][bk][bc]=rb[q]; } };
    float acc[8][8];
    #pragma unroll
    for(int i=0;i<8;++i) for(int j=0;j<8;++j) acc[i][j]=0;
    ld(0); st(0); __syncthreads();
    const int ns=KL/BK;
    for(int t=0;t<ns;++t){ const int cur=t&1;
        if(t+1<ns) ld((t+1)*BK);
        #pragma unroll UK
        for(int k=0;k<BK;++k){
            float4 a0=*(const float4*)&As[cur][k][ty*4], a1=*(const float4*)&As[cur][k][64+ty*4];
            float4 b0=*(const float4*)&Bs[cur][k][tx*4], b1=*(const float4*)&Bs[cur][k][64+tx*4];
            float a[8]={a0.x,a0.y,a0.z,a0.w,a1.x,a1.y,a1.z,a1.w}, b[8]={b0.x,b0.y,b0.z,b0.w,b1.x,b1.y,b1.z,b1.w};
            #pragma unroll
            for(int i=0;i<8;++i)
                #pragma unroll
                for(int j=0;j<8;++j) acc[i][j]=fmaf(a[i],b[j],acc[i][j]); }
        if(t+1<ns) st(cur^1);
        __syncthreads(); }
    #pragma unroll
    for(int i=0;i<8;++i){ int m=bm+(i<4? ty*4+i : 64+ty*4+i-4); if(m>=M) continue;
        #pragma unroll
        for(int h=0;h<2;++h){ int n=bn+h*64+tx*4; float4* p=(float4*)(C+(size_t)m*N+n); float4 v=make_float4(acc[i][h*4],acc[i][h*4+1],acc[i][h*4+2],acc[i][h*4+3]);
            if(ACC){ float4 o=*p; v.x+=o.x; v.y+=o.y; v.z+=o.z; v.w+=o.w; } *p=v; } } }
__global__ void addparts(const float* P, float* C, size_t n, int S, int acc){ size_t i=blockIdx.x*(size_t)blockDim.x+threadIdx.x; if(i>=n) return; float v=acc?C[i]:0; for(int z=0;z<S;++z) v+=P[z*n+i]; C[i]=v; }
// вх1
static bool g_use_lt=true; static hipblasLtHandle_t g_lt=nullptr; static void* g_lt_ws=nullptr; static const size_t LT_WS=32u<<20; static size_t g_lt_forms=0;
struct LtPlan{ hipblasLtMatmulDesc_t d; hipblasLtMatrixLayout_t a,b,c; hipblasLtMatmulAlgo_t algo; bool ok; };
static std::map<std::tuple<int,int,int,int,int,int,int,int,int>,LtPlan> g_lt_plans; static std::map<std::tuple<int,int,int,int,int,int,int,int,int>,int> g_lt_rank;
static bool g_use_own=true; static float* g_own_part=nullptr; static const size_t OWN_PART=64u<<20; static float* g_own_tmp=nullptr; static const size_t OWN_TMP=32u<<20;
static inline int mbucket(int n){ return (n+127)/128*128; } // вч1
static std::map<std::tuple<int,int,int,int,int>,int> g_own_plan; // вч2
template<bool BT, int BK, int UK> static void own_k(dim3 g, bool acc1, const float* A, const float* B, float* out, int M, int N, int K, int kc){
    if(acc1) sg<BT,true,BK,UK><<<g,256>>>(A,B,out,M,N,K,kc); else sg<BT,false,BK,UK><<<g,256>>>(A,B,out,M,N,K,kc); }
static bool own_run(bool bt, int V, int M, int N, int K, const float* A, const float* B, float* C, bool acc){ // вэ1
    int S=V&7; bool b16=V&8; size_t ny=(size_t)M*N; if(S>1 && (S*ny*4>OWN_PART || (K/S)%(b16?16:8))) return false; if(!S||(K%(b16?16:8))) return false;
    dim3 g(N/128,(M+127)/128,S); int kc=S>1?K/S:0; float* out=S>1?g_own_part:C; bool a1=acc&&S==1;
    if(bt){ if(b16) own_k<true,16,2>(g,a1,A,B,out,M,N,K,kc); else own_k<true,8,8>(g,a1,A,B,out,M,N,K,kc); }
    else  { if(b16) own_k<false,16,2>(g,a1,A,B,out,M,N,K,kc); else own_k<false,8,8>(g,a1,A,B,out,M,N,K,kc); }
    if(S>1) addparts<<<(ny+255)/256,256>>>(g_own_part,C,ny,S,acc?1:0); return true; }
static void sgemm_lt(hipblasHandle_t h, hipblasOperation_t oa, hipblasOperation_t ob, int m, int n, int k, const float* A, int lda, const float* B, int ldb, float beta, float* C, int ldc);
static void sgemm(hipblasHandle_t h, hipblasOperation_t oa, hipblasOperation_t ob, int m, int n, int k, const float* A, int lda, const float* B, int ldb, float beta, float* C, int ldc){
    // вю1
    bool fwd=oa==HIPBLAS_OP_T&&ob==HIPBLAS_OP_N&&lda==k&&ldb==k&&ldc==m, bwd=oa==HIPBLAS_OP_N&&ob==HIPBLAS_OP_N&&lda==m&&ldb==k&&ldc==m;
    if(g_use_own && (fwd||bwd) && m%128==0 && k%8==0){
        if(!g_own_part){ HIP_CHECK(hipMalloc(&g_own_part,OWN_PART)); HIP_CHECK(hipMalloc(&g_own_tmp,OWN_TMP)); }
        auto key=std::make_tuple((int)fwd,m,mbucket(n),k,beta!=0); auto it=g_own_plan.find(key);
        if(it==g_own_plan.end()){ // вю2
            size_t ny=(size_t)m*n; int best=0; double bt=1e30; float* tmp= ny*4<=OWN_TMP ? g_own_tmp : nullptr;
            auto t_of=[&](auto fn){ hipStreamSynchronize(0); auto t0=std::chrono::steady_clock::now(); fn(); fn(); hipStreamSynchronize(0); return std::chrono::duration<double>(std::chrono::steady_clock::now()-t0).count(); };
            if(tmp){ for(int V : {1,2,4,9,10,12}){ bool ok=true; double t=t_of([&]{ ok=own_run(fwd,V,n,m,k,B,A,tmp,false); }); if(ok&&t<bt){ bt=t; best=V; } }
                double t=t_of([&]{ sgemm_lt(h,oa,ob,m,n,k,A,lda,B,ldb,0,tmp,ldc); }); if(t<bt){ bt=t; best=0; } }
            else best=1;
            it=g_own_plan.emplace(key,best).first; }
        if(it->second && own_run(fwd,it->second,n,m,k,B,A,C,beta!=0)) return; }
    sgemm_lt(h,oa,ob,m,n,k,A,lda,B,ldb,beta,C,ldc); }
static void sgemm_lt(hipblasHandle_t h, hipblasOperation_t oa, hipblasOperation_t ob, int m, int n, int k, const float* A, int lda, const float* B, int ldb, float beta, float* C, int ldc){
    float al=1;
    if(g_use_lt){ if(!g_lt){ hipblasLtCreate(&g_lt); HIP_CHECK(hipMalloc(&g_lt_ws,LT_WS)); }
        auto key=std::make_tuple((int)oa,(int)ob,m,n,k,lda,ldb,ldc,beta!=0); auto it=g_lt_plans.find(key);
        if(it==g_lt_plans.end()){ LtPlan P{}; hipblasLtMatmulDescCreate(&P.d,HIPBLAS_COMPUTE_32F,HIP_R_32F);
            hipblasLtMatmulDescSetAttribute(P.d,HIPBLASLT_MATMUL_DESC_TRANSA,&oa,sizeof oa); hipblasLtMatmulDescSetAttribute(P.d,HIPBLASLT_MATMUL_DESC_TRANSB,&ob,sizeof ob);
            hipblasLtMatrixLayoutCreate(&P.a,HIP_R_32F,oa==HIPBLAS_OP_N?m:k,oa==HIPBLAS_OP_N?k:m,lda); hipblasLtMatrixLayoutCreate(&P.b,HIP_R_32F,ob==HIPBLAS_OP_N?k:n,ob==HIPBLAS_OP_N?n:k,ldb);
            hipblasLtMatrixLayoutCreate(&P.c,HIP_R_32F,m,n,ldc);
            hipblasLtMatmulPreference_t pr; hipblasLtMatmulPreferenceCreate(&pr); size_t ws=LT_WS; hipblasLtMatmulPreferenceSetAttribute(pr,HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,&ws,sizeof ws);
            hipblasLtMatmulHeuristicResult_t hr[8]; int ret=0; hipblasLtMatmulAlgoGetHeuristic(g_lt,P.d,P.a,P.b,P.c,P.c,pr,8,hr,&ret); hipblasLtMatmulPreferenceDestroy(pr);
            int best=-1; double bt=1e30; auto bk=std::make_tuple((int)oa,(int)ob,m,mbucket(n),k,lda,ldb,ldc,beta!=0);
            if(auto r=g_lt_rank.find(bk); r!=g_lt_rank.end() && r->second<ret && hr[r->second].state==HIPBLAS_STATUS_SUCCESS) best=r->second, ret=0; // вя1
            for(int i=0;i<ret;++i){ if(hr[i].state!=HIPBLAS_STATUS_SUCCESS) continue; if(beta!=0){ best=i; break; } // вя2
                hipStreamSynchronize(0); auto t0=std::chrono::steady_clock::now();
                if(hipblasLtMatmul(g_lt,P.d,&al,A,P.a,B,P.b,&beta,C,P.c,C,P.c,&hr[i].algo,g_lt_ws,LT_WS,0)!=HIPBLAS_STATUS_SUCCESS) continue;
                hipStreamSynchronize(0); double t=std::chrono::duration<double>(std::chrono::steady_clock::now()-t0).count(); if(t<bt){ bt=t; best=i; } }
            P.ok=best>=0; if(P.ok){ P.algo=hr[best].algo; g_lt_rank[bk]=best; } it=g_lt_plans.emplace(key,P).first; ++g_lt_forms; }
        if(it->second.ok){ auto& P=it->second; if(hipblasLtMatmul(g_lt,P.d,&al,A,P.a,B,P.b,&beta,C,P.c,C,P.c,&P.algo,g_lt_ws,LT_WS,0)==HIPBLAS_STATUS_SUCCESS) return; P.ok=false; } }
    HIPBLAS_CHECK(hipblasSgemm(h,oa,ob,m,n,k,&al,A,lda,B,ldb,&beta,C,ldc)); } // вя3
__global__ void split3_k(const float* x, hip_bfloat16* h, hip_bfloat16* m, hip_bfloat16* l, size_t n){ size_t i=blockIdx.x*(size_t)blockDim.x+threadIdx.x; if(i>=n) return;
    float v=x[i]; hip_bfloat16 a=hip_bfloat16(v); float r=v-(float)a; hip_bfloat16 b=hip_bfloat16(r); h[i]=a; m[i]=b; l[i]=hip_bfloat16(r-(float)b); }
static hip_bfloat16* g_x3=nullptr; static size_t g_x3_n=0; static std::map<std::tuple<int,int,int,int,int,int,int,int>,LtPlan> g_bf_plans; static std::map<std::tuple<int,int,int,int,int,int>,int> g_bf_rank;
__global__ void split3_fp_k(const float* x, tier_fp16* h, tier_fp16* m, tier_fp16* l, size_t n){ size_t i=blockIdx.x*(size_t)blockDim.x+threadIdx.x; if(i>=n) return;
    float v=x[i]; tier_fp16 a= tier_fp16(v); float r=v-(float)a; tier_fp16 b=tier_fp16(r); h[i]=a; m[i]=b; l[i]=tier_fp16(r-(float)b); }
static tier_fp16* g_x3f=nullptr; static size_t g_x3f_n=0; static std::map<std::tuple<int,int,int,int,int,int,int,int>,LtPlan> g_fp_plans; static std::map<std::tuple<int,int,int,int,int,int>,int> g_fp_rank;
// гб1
static int g_tw=-1; static std::map<std::tuple<int,int,int,int,int>,int> g_tw_plan;
static bool tw_on(){ if(g_tw<0){ g_tw=0;
#if !defined(TIER_CUDA)
        hipDeviceProp_t p; if(hipGetDeviceProperties(&p,g_dev)==hipSuccess && !strncmp(p.gcnArchName,"gfx12",5)) g_tw=1;
        if(const char* e=getenv("TIER_WMMA")) g_tw = g_tw && atoi(e)!=0;
#endif
    } return g_tw==1; }
template<typename T, typename H> static bool tw_try(int ty, hipblasOperation_t oa, int m, int n, int k, const H* A, int lda, const H* X, float beta, float* C, int ldc){
#if !defined(TIER_CUDA)
    const bool tr = oa==HIPBLAS_OP_N && lda==m; if(!tw_on() || !((oa==HIPBLAS_OP_T && lda==k) || tr)) return false;
    auto it=g_tw_plan.find(std::make_tuple(ty+(tr?8:0),m,mbucket(n),k,beta!=0)); if(it==g_tw_plan.end() || it->second<0) return false;
    return tr ? tw::launch<T,true>(it->second,(const T*)X,(const T*)A,C,n,m,k,ldc,beta!=0) : tw::launch<T>(it->second,(const T*)X,(const T*)A,C,n,m,k,ldc,beta!=0);
#else
    return false;
#endif
}
template<typename T, typename H, typename F> static void tw_tune(int ty, hipblasOperation_t oa, int m, int n, int k, const H* A, int lda, const H* X, float beta, int ldc, F lt){
#if !defined(TIER_CUDA)
    const bool tr = oa==HIPBLAS_OP_N && lda==m; if(!tw_on() || !((oa==HIPBLAS_OP_T && lda==k) || tr)) return;
    auto key=std::make_tuple(ty+(tr?8:0),m,mbucket(n),k,beta!=0); if(g_tw_plan.count(key)) return;
    int best=-1; if((size_t)ldc*n*4<=OWN_TMP){
        auto tm=[&](auto fn){ hipStreamSynchronize(0); auto t0=std::chrono::steady_clock::now(); fn(); fn(); hipStreamSynchronize(0); return std::chrono::duration<double>(std::chrono::steady_clock::now()-t0).count(); };
        double bt=tm([&]{ lt(g_own_tmp); });
        for(int v=0; v<tw::NVAR; ++v){ bool ok=true; double t=tm([&]{ ok = tr ? tw::launch<T,true>(v,(const T*)X,(const T*)A,g_own_tmp,n,m,k,ldc,false) : tw::launch<T>(v,(const T*)X,(const T*)A,g_own_tmp,n,m,k,ldc,false); }); if(ok && t<bt){ bt=t; best=v; } } }
    g_tw_plan[key]=best;
#endif
}
static void bf3gemm(hipblasHandle_t h, hipblasOperation_t oa, int m, int n, int k, const hip_bfloat16* A, int lda, const float* B, float beta, float* C, int ldc){
    size_t nb=(size_t)k*n; if(nb>g_x3_n){ if(g_x3) HIP_CHECK(hipFree(g_x3)); HIP_CHECK(dmalloc(&g_x3,nb*3*2)); g_x3_n=nb; }
    if(!g_lt){ hipblasLtCreate(&g_lt); HIP_CHECK(hipMalloc(&g_lt_ws,LT_WS)); } if(!g_own_tmp){ HIP_CHECK(hipMalloc(&g_own_part,OWN_PART)); HIP_CHECK(hipMalloc(&g_own_tmp,OWN_TMP)); }
    hip_bfloat16 *xh=g_x3, *xm=g_x3+nb, *xl=g_x3+2*nb; split3_k<<<(nb+255)/256,256>>>(B,xh,xm,xl,nb);
    hipblasOperation_t ob=HIPBLAS_OP_N; auto key=std::make_tuple((int)oa,m,n,k,lda,ldc,0,0); auto it=g_bf_plans.find(key);
    if(it==g_bf_plans.end()){ LtPlan P{}; hipblasLtMatmulDescCreate(&P.d,HIPBLAS_COMPUTE_32F,HIP_R_32F);
        hipblasLtMatmulDescSetAttribute(P.d,HIPBLASLT_MATMUL_DESC_TRANSA,&oa,sizeof oa); hipblasLtMatmulDescSetAttribute(P.d,HIPBLASLT_MATMUL_DESC_TRANSB,&ob,sizeof ob);
        hipblasLtMatrixLayoutCreate(&P.a,HIP_R_16BF,oa==HIPBLAS_OP_N?m:k,oa==HIPBLAS_OP_N?k:m,lda); hipblasLtMatrixLayoutCreate(&P.b,HIP_R_16BF,k,n,k); hipblasLtMatrixLayoutCreate(&P.c,HIP_R_32F,m,n,ldc);
        hipblasLtMatmulPreference_t pr; hipblasLtMatmulPreferenceCreate(&pr); size_t ws=LT_WS; hipblasLtMatmulPreferenceSetAttribute(pr,HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,&ws,sizeof ws);
        hipblasLtMatmulHeuristicResult_t hr[16]; int ret=0; hipblasLtMatmulAlgoGetHeuristic(g_lt,P.d,P.a,P.b,P.c,P.c,pr,16,hr,&ret); hipblasLtMatmulPreferenceDestroy(pr);
        float al=1,b0=0; int best=-1; double bt=1e30; bool tmpok=(size_t)ldc*n*4<=OWN_TMP; auto bk=std::make_tuple((int)oa,m,mbucket(n),k,lda,ldc);
        if(auto r=g_bf_rank.find(bk); r!=g_bf_rank.end() && r->second<ret && hr[r->second].state==HIPBLAS_STATUS_SUCCESS) best=r->second, ret=0;
        for(int i=0;i<ret;++i){ if(hr[i].state!=HIPBLAS_STATUS_SUCCESS) continue; if(!tmpok){ best=i; break; } // ге1
            hipStreamSynchronize(0); auto t0=std::chrono::steady_clock::now();
            if(hipblasLtMatmul(g_lt,P.d,&al,A,P.a,xh,P.b,&b0,g_own_tmp,P.c,g_own_tmp,P.c,&hr[i].algo,g_lt_ws,LT_WS,0)!=HIPBLAS_STATUS_SUCCESS) continue;
            hipStreamSynchronize(0); double t=std::chrono::duration<double>(std::chrono::steady_clock::now()-t0).count(); if(t<bt){ bt=t; best=i; } }
        P.ok=best>=0; if(P.ok){ P.algo=hr[best].algo; g_bf_rank[bk]=best; } it=g_bf_plans.emplace(key,P).first; }
    auto& P=it->second; if(!P.ok){ fprintf(stderr,"BF16×3: отсутствует ядро для формы %d×%d×%d.\n",m,n,k); exit(1); }
    float al=1, b1=1; const hip_bfloat16* parts[3]={xl,xm,xh};
    if(tw_try<__bf16>(1,oa,m,n,k,A,lda,xh,beta,C,ldc)) return;
    auto lt=[&](float* out){ for(int q=0;q<3;++q){ const float* be= q==0 ? &beta : &b1; HIPBLAS_CHECK(hipblasLtMatmul(g_lt,P.d,&al,A,P.a,parts[q],P.b,be,out,P.c,out,P.c,&P.algo,g_lt_ws,LT_WS,0)); } };
    lt(C); tw_tune<__bf16>(1,oa,m,n,k,A,lda,xh,beta,ldc,lt); }
static void fp3gemm(hipblasHandle_t h, hipblasOperation_t oa, int m, int n, int k, const tier_fp16* A, int lda, const float* B, float beta, float* C, int ldc){
    size_t nb=(size_t)k*n; if(nb>g_x3f_n){ if(g_x3f) HIP_CHECK(hipFree(g_x3f)); HIP_CHECK(dmalloc(&g_x3f,nb*3*2)); g_x3f_n=nb; }
    if(!g_lt){ hipblasLtCreate(&g_lt); HIP_CHECK(hipMalloc(&g_lt_ws,LT_WS)); } if(!g_own_tmp){ HIP_CHECK(hipMalloc(&g_own_part,OWN_PART)); HIP_CHECK(hipMalloc(&g_own_tmp,OWN_TMP)); }
    tier_fp16 *xh=g_x3f, *xm=g_x3f+nb, *xl=g_x3f+2*nb; split3_fp_k<<<(nb+255)/256,256>>>(B,xh,xm,xl,nb);
    hipblasOperation_t ob=HIPBLAS_OP_N; auto key=std::make_tuple((int)oa,m,n,k,lda,ldc,0,0); auto it=g_fp_plans.find(key);
    if(it==g_fp_plans.end()){ LtPlan P{}; hipblasLtMatmulDescCreate(&P.d,HIPBLAS_COMPUTE_32F,HIP_R_32F);
        hipblasLtMatmulDescSetAttribute(P.d,HIPBLASLT_MATMUL_DESC_TRANSA,&oa,sizeof oa); hipblasLtMatmulDescSetAttribute(P.d,HIPBLASLT_MATMUL_DESC_TRANSB,&ob,sizeof ob);
        hipblasLtMatrixLayoutCreate(&P.a,HIP_R_16F,oa==HIPBLAS_OP_N?m:k,oa==HIPBLAS_OP_N?k:m,lda); hipblasLtMatrixLayoutCreate(&P.b,HIP_R_16F,k,n,k); hipblasLtMatrixLayoutCreate(&P.c,HIP_R_32F,m,n,ldc);
        hipblasLtMatmulPreference_t pr; hipblasLtMatmulPreferenceCreate(&pr); size_t ws=LT_WS; hipblasLtMatmulPreferenceSetAttribute(pr,HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,&ws,sizeof ws);
        hipblasLtMatmulHeuristicResult_t hr[16]; int ret=0; hipblasLtMatmulAlgoGetHeuristic(g_lt,P.d,P.a,P.b,P.c,P.c,pr,16,hr,&ret); hipblasLtMatmulPreferenceDestroy(pr);
        float al=1,b0=0; int best=-1; double bt=1e30; bool tmpok=(size_t)ldc*n*4<=OWN_TMP; auto bk=std::make_tuple((int)oa,m,mbucket(n),k,lda,ldc);
        if(auto r=g_fp_rank.find(bk); r!=g_fp_rank.end() && r->second<ret && hr[r->second].state==HIPBLAS_STATUS_SUCCESS) best=r->second, ret=0;
        for(int i=0;i<ret;++i){ if(hr[i].state!=HIPBLAS_STATUS_SUCCESS) continue; if(!tmpok){ best=i; break; } // гж1
            hipStreamSynchronize(0); auto t0=std::chrono::steady_clock::now();
            if(hipblasLtMatmul(g_lt,P.d,&al,A,P.a,xh,P.b,&b0,g_own_tmp,P.c,g_own_tmp,P.c,&hr[i].algo,g_lt_ws,LT_WS,0)!=HIPBLAS_STATUS_SUCCESS) continue;
            hipStreamSynchronize(0); double t=std::chrono::duration<double>(std::chrono::steady_clock::now()-t0).count(); if(t<bt){ bt=t; best=i; } }
        P.ok=best>=0; if(P.ok){ P.algo=hr[best].algo; g_fp_rank[bk]=best; } it=g_fp_plans.emplace(key,P).first; }
    auto& P=it->second; if(!P.ok){ fprintf(stderr,"FP16×3: отсутствует ядро для формы %d×%d×%d.\n",m,n,k); exit(1); }
    float al=1, b1=1; const tier_fp16* parts[3]={xl,xm,xh};
    if(tw_try<_Float16>(2,oa,m,n,k,A,lda,xh,beta,C,ldc)) return;
    auto lt=[&](float* out){ for(int q=0;q<3;++q){ const float* be= q==0 ? &beta : &b1; HIPBLAS_CHECK(hipblasLtMatmul(g_lt,P.d,&al,A,P.a,parts[q],P.b,be,out,P.c,out,P.c,&P.algo,g_lt_ws,LT_WS,0)); } };
    lt(C); tw_tune<_Float16>(2,oa,m,n,k,A,lda,xh,beta,ldc,lt); }
// гз1
static bool lt_probe(hipDataType in){
    if(!TIER_LT) return false;
    if(!g_lt){ if(hipblasLtCreate(&g_lt)!=HIPBLAS_STATUS_SUCCESS){ g_lt=nullptr; return false; }
        if(hipMalloc(&g_lt_ws,LT_WS)!=hipSuccess){ g_lt_ws=nullptr; g_lt=nullptr; return false; } }
    // гз2
    hipblasOperation_t oa=HIPBLAS_OP_T, ob=HIPBLAS_OP_N;
    const int m=128, n=32, k=128; const size_t na=(size_t)m*k, nb=(size_t)k*n, nc=(size_t)m*n;
    hipblasLtMatmulDesc_t d=nullptr; hipblasLtMatrixLayout_t a=nullptr,b=nullptr,c=nullptr; hipblasLtMatmulPreference_t pr=nullptr;
    void *da=nullptr,*db=nullptr,*dc=nullptr; bool ok=false;
    do {
        if(hipblasLtMatmulDescCreate(&d,HIPBLAS_COMPUTE_32F,HIP_R_32F)!=HIPBLAS_STATUS_SUCCESS) break;
        if(hipblasLtMatmulDescSetAttribute(d,HIPBLASLT_MATMUL_DESC_TRANSA,&oa,sizeof oa)!=HIPBLAS_STATUS_SUCCESS) break;
        if(hipblasLtMatmulDescSetAttribute(d,HIPBLASLT_MATMUL_DESC_TRANSB,&ob,sizeof ob)!=HIPBLAS_STATUS_SUCCESS) break;
        if(hipblasLtMatrixLayoutCreate(&a,in,k,m,k)!=HIPBLAS_STATUS_SUCCESS ||
           hipblasLtMatrixLayoutCreate(&b,in,k,n,k)!=HIPBLAS_STATUS_SUCCESS ||
           hipblasLtMatrixLayoutCreate(&c,HIP_R_32F,m,n,m)!=HIPBLAS_STATUS_SUCCESS) break;
        if(hipblasLtMatmulPreferenceCreate(&pr)!=HIPBLAS_STATUS_SUCCESS) break;
        size_t ws=LT_WS; hipblasLtMatmulPreferenceSetAttribute(pr,HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,&ws,sizeof ws);
        hipblasLtMatmulHeuristicResult_t hr[8]; int ret=0;
        if(hipblasLtMatmulAlgoGetHeuristic(g_lt,d,a,b,c,c,pr,8,hr,&ret)!=HIPBLAS_STATUS_SUCCESS || ret<=0) break;
        if(hipMalloc(&da,na*2)!=hipSuccess || hipMalloc(&db,nb*2)!=hipSuccess || hipMalloc(&dc,nc*4)!=hipSuccess) break;
        std::vector<uint16_t> ha(na), hb(nb); const uint16_t one=(in==HIP_R_16F)?0x3c00u:0x3f80u;
        std::fill(ha.begin(),ha.end(),one); std::fill(hb.begin(),hb.end(),one);
        if(hipMemcpy(da,ha.data(),na*2,hipMemcpyHostToDevice)!=hipSuccess ||
           hipMemcpy(db,hb.data(),nb*2,hipMemcpyHostToDevice)!=hipSuccess) break;
        float alpha=1.0f,beta=0.0f;
        for(int i=0;i<ret && !ok;++i){ if(hr[i].state!=HIPBLAS_STATUS_SUCCESS) continue;
            if(hipMemset(dc,0,nc*4)!=hipSuccess) continue;
            if(hipblasLtMatmul(g_lt,d,&alpha,da,a,db,b,&beta,dc,c,dc,c,&hr[i].algo,g_lt_ws,LT_WS,0)!=HIPBLAS_STATUS_SUCCESS) continue;
            if(hipDeviceSynchronize()!=hipSuccess){ hipGetLastError(); continue; }
            float result=0; if(hipMemcpy(&result,dc,sizeof(float),hipMemcpyDeviceToHost)!=hipSuccess) continue;
            ok=std::isfinite(result) && fabsf(result-(float)k)<0.05f*(float)k;
        }
    } while(false);
    if(da) hipFree(da); if(db) hipFree(db); if(dc) hipFree(dc);
    if(pr) hipblasLtMatmulPreferenceDestroy(pr); if(a) hipblasLtMatrixLayoutDestroy(a);
    if(b) hipblasLtMatrixLayoutDestroy(b); if(c) hipblasLtMatrixLayoutDestroy(c); if(d) hipblasLtMatmulDescDestroy(d);
    return ok;
}
static bool bf3_probe(){ return lt_probe(HIP_R_16BF); }
// гк1
static bool fp16_probe(){ return lt_probe(HIP_R_16F); }

struct TensorSlot { size_t offset_floats, n; int tier=-1; size_t rows=1, rlen=0, ntr=0; };
static std::map<std::string, TensorSlot> g_weight_layout; // гл1
// гл2
static char* g_rpool=nullptr; static size_t g_rpool_cap=0, g_rpool_off=0;
static hipError_t res_alloc(float** p, size_t bytes){
    if(!g_rpool){ size_t t=0; for(auto& [k,sl] : g_weight_layout) if(sl.tier==1){ size_t n=sl.ntr?sl.ntr:sl.n; t+=3*((n*4+255)/256*256); }
        if(t && hipMalloc((void**)&g_rpool,t)==hipSuccess) g_rpool_cap=t; else g_rpool=nullptr; }
    size_t a=(bytes+255)/256*256;
    if(g_rpool && g_rpool_off+a<=g_rpool_cap){ *p=(float*)(g_rpool+g_rpool_off); g_rpool_off+=a; return hipSuccess; }
    return cmalloc(p,bytes); }   // гм1
// гм2
static double g_cpu_secs = 0, g_cpu_wait = 0;      // гм3
static double g_gpu_frac = 0.95;                   // гм4

static bool ends_with(const std::string& s, const char* suf){
    size_t n=strlen(suf); return s.size()>=n && s.compare(s.size()-n,n,suf)==0;
}
// гн1
static bool g_split_ok=false;                     // гн2
static bool cpu_owned(const std::string& nm){
    if(g_cpu_thr<1 || !g_split_ok || tq_of(nm)) return false;   // го1
    return ends_with(nm,arch_tn("q").c_str()) || ends_with(nm,arch_tn("k").c_str()) || ends_with(nm,arch_tn("v").c_str());   // го2
}
// гп1
static int gpu_rows(int rows){ int R=(int)(rows*g_gpu_frac); if(R<0) R=0; if(R>rows) R=rows; return R; }

// гп2
static int g_pool_T=0;
// гп3
static std::mutex& g_pm = *new std::mutex();
static std::condition_variable& g_pcv = *new std::condition_variable();
static std::condition_variable& g_pdone = *new std::condition_variable();
static std::function<void(int,int)> g_pjob; static int g_pn=0, g_pleft=0; static uint64_t g_pgen=0; static bool g_pstop=false;
static void pool_worker(int idx){
    uint64_t seen=0;
    for(;;){
        std::function<void(int,int)> job; int T,n;
        { std::unique_lock<std::mutex> lk(g_pm);
          g_pcv.wait(lk,[&]{ return g_pstop || g_pgen!=seen; });
          if(g_pstop) return;
          seen=g_pgen; T=g_pool_T; n=g_pn; job=g_pjob; }
        int a=(int)((long long)n*idx/T), b=(int)((long long)n*(idx+1)/T);
        if(b>a) job(a,b);
        { std::lock_guard<std::mutex> lk(g_pm); if(--g_pleft==0) g_pdone.notify_all(); }
    }
}
static void pool_init(){
    if(g_pool_T==g_cpu_thr || g_cpu_thr<1) return;
    if(g_pool_T>0){ { std::lock_guard<std::mutex> lk(g_pm); g_pstop=true; ++g_pgen; } g_pcv.notify_all();
        std::this_thread::sleep_for(std::chrono::milliseconds(20)); g_pstop=false; }
    // гс1
    g_pool_T=g_cpu_thr; for(int i=0;i<g_pool_T;++i){ std::thread t(pool_worker,i); t.detach(); }
}
static void pool_run(int n_items, std::function<void(int,int)> f){
    if(g_pool_T<2){ f(0,n_items); return; }
    { std::lock_guard<std::mutex> lk(g_pm); g_pjob=std::move(f); g_pn=n_items; g_pleft=g_pool_T; ++g_pgen; }
    g_pcv.notify_all();
    std::unique_lock<std::mutex> lk(g_pm); g_pdone.wait(lk,[&]{ return g_pleft==0; });
}

// гу1
static void cpu_gemm_fwd(const float* X, const uint16_t* W, float* out, int M, int K, int N){
    pool_run(N, [&](int n0,int n1){
        const int B=4;                                      // гу2
        for(int n=n0;n<n1;++n){ const uint16_t* w=W+(size_t)n*K;
            for(int m=0;m<M;m+=B){ int mm=(M-m)<B?(M-m):B; float r[4]={0,0,0,0};
#if defined(__AVX2__)
                __m256 a0=_mm256_setzero_ps(),a1=a0,a2=a0,a3=a0; int k=0;
                for(;k+8<=K;k+=8){
                    __m256i e=_mm256_cvtepu16_epi32(_mm_loadu_si128((const __m128i*)(w+k)));
                    __m256 wf=_mm256_castsi256_ps(_mm256_slli_epi32(e,16));
                    a0=_mm256_fmadd_ps(wf,_mm256_loadu_ps(X+(size_t)(m+0)*K+k),a0);
                    if(mm>1) a1=_mm256_fmadd_ps(wf,_mm256_loadu_ps(X+(size_t)(m+1)*K+k),a1);
                    if(mm>2) a2=_mm256_fmadd_ps(wf,_mm256_loadu_ps(X+(size_t)(m+2)*K+k),a2);
                    if(mm>3) a3=_mm256_fmadd_ps(wf,_mm256_loadu_ps(X+(size_t)(m+3)*K+k),a3); }
                float sv[8];
                _mm256_storeu_ps(sv,a0); r[0]=sv[0]+sv[1]+sv[2]+sv[3]+sv[4]+sv[5]+sv[6]+sv[7];
                if(mm>1){_mm256_storeu_ps(sv,a1); r[1]=sv[0]+sv[1]+sv[2]+sv[3]+sv[4]+sv[5]+sv[6]+sv[7];}
                if(mm>2){_mm256_storeu_ps(sv,a2); r[2]=sv[0]+sv[1]+sv[2]+sv[3]+sv[4]+sv[5]+sv[6]+sv[7];}
                if(mm>3){_mm256_storeu_ps(sv,a3); r[3]=sv[0]+sv[1]+sv[2]+sv[3]+sv[4]+sv[5]+sv[6]+sv[7];}
                for(int mi=0;mi<mm;++mi){ const float* x=X+(size_t)(m+mi)*K; float acc=r[mi];
                    for(int kk=k;kk<K;++kk){ uint32_t u=(uint32_t)w[kk]<<16; float f; memcpy(&f,&u,4); acc+=f*x[kk]; }
                    out[(size_t)(m+mi)*N+n]=acc; }
#else
                for(int mi=0;mi<mm;++mi){ const float* x=X+(size_t)(m+mi)*K; float acc=0;
                    for(int k2=0;k2<K;++k2){ uint32_t u=(uint32_t)w[k2]<<16; float f; memcpy(&f,&u,4); acc+=f*x[k2]; }
                    out[(size_t)(m+mi)*N+n]=acc; }
#endif
            } } });
}
// гф1
static void cpu_gemm_dx(const float* dY, const uint16_t* W, float* dX, int M, int K, int N){
    int T=g_pool_T<1?1:g_pool_T;
    static std::vector<std::vector<float>> part; part.resize(T);
    pool_run(N, [&](int n0,int n1){
        int id=0; { /* гф2 */ for(int t=0;t<T;++t) if((int)((long long)N*t/T)==n0){ id=t; break; } }
        auto& acc=part[id]; acc.assign((size_t)M*K,0.0f);
        for(int n=n0;n<n1;++n){ const uint16_t* w=W+(size_t)n*K;
            for(int m=0;m<M;++m){ float c=dY[(size_t)m*N+n]; if(c==0.0f) continue; float* o=acc.data()+(size_t)m*K;
#if defined(__AVX2__)
                __m256 cb=_mm256_set1_ps(c); int k=0;
                for(;k+8<=K;k+=8){ __m256i e=_mm256_cvtepu16_epi32(_mm_loadu_si128((const __m128i*)(w+k)));
                    _mm256_storeu_ps(o+k,_mm256_fmadd_ps(cb,_mm256_castsi256_ps(_mm256_slli_epi32(e,16)),_mm256_loadu_ps(o+k))); }
                for(;k<K;++k){ uint32_t u=(uint32_t)w[k]<<16; float f; memcpy(&f,&u,4); o[k]+=c*f; }
#else
                for(int k=0;k<K;++k){ uint32_t u=(uint32_t)w[k]<<16; float f; memcpy(&f,&u,4); o[k]+=c*f; }
#endif
            } } });
    memset(dX,0,(size_t)M*K*sizeof(float));
    for(int t=0;t<T;++t){ if(part[t].size()!=(size_t)M*K) continue; const float* a=part[t].data();
        for(size_t i=0,n=(size_t)M*K;i<n;++i) dX[i]+=a[i]; }
}

// гф3
static std::vector<float> g_cpu_in, g_cpu_out;
static std::thread g_cpu_th; static bool g_cpu_busy=false;
static double cpu_clock(){ return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
static int g_co_M=0, g_co_cols=0, g_co_stride=0, g_co_off=0; static float* g_co_dev=nullptr; static bool g_co_add=false;
static float* g_cpu_dx_dev=nullptr; static size_t g_cpu_dx_n=0;

static void cpu_fwd_begin(const float* x_dev, const uint16_t* W_host, int M, int K, int N_cpu,
                          float* out_dev, int out_stride, int out_off){
    double t0=cpu_clock();
    g_cpu_in.resize((size_t)M*K); g_cpu_out.resize((size_t)M*N_cpu);
    HIP_CHECK(hipMemcpy(g_cpu_in.data(), x_dev, (size_t)M*K*4, hipMemcpyDeviceToHost));
    g_co_M=M; g_co_cols=N_cpu; g_co_stride=out_stride; g_co_off=out_off; g_co_dev=out_dev; g_co_add=false;
    g_cpu_th = std::thread([W_host,M,K,N_cpu]{ cpu_gemm_fwd(g_cpu_in.data(), W_host, g_cpu_out.data(), M, K, N_cpu); });
    g_cpu_busy=true; g_cpu_secs += cpu_clock()-t0;
}
static void cpu_dx_begin(const float* dY_dev, const uint16_t* W_host, int M, int K, int N_all, int N_cpu, int off,
                         float* dX_dev){
    double t0=cpu_clock();
    g_cpu_in.resize((size_t)M*N_cpu); g_cpu_out.resize((size_t)M*K);
    HIP_CHECK(hipMemcpy2D(g_cpu_in.data(), (size_t)N_cpu*4, dY_dev+off, (size_t)N_all*4,
                          (size_t)N_cpu*4, M, hipMemcpyDeviceToHost));
    g_co_M=M; g_co_cols=K; g_co_stride=K; g_co_off=0; g_co_dev=dX_dev; g_co_add=true;
    g_cpu_th = std::thread([W_host,M,K,N_cpu]{ cpu_gemm_dx(g_cpu_in.data(), W_host, g_cpu_out.data(), M, K, N_cpu); });
    g_cpu_busy=true; g_cpu_secs += cpu_clock()-t0;
}
__global__ void add_into(float* dst, const float* src, size_t n){
    size_t i=blockIdx.x*(size_t)blockDim.x+threadIdx.x; if(i<n) dst[i]+=src[i];
}
static void cpu_end(){
    if(!g_cpu_busy) return;
    double t0=cpu_clock(); g_cpu_th.join(); g_cpu_busy=false; double w=cpu_clock()-t0; g_cpu_wait+=w;
    if(g_co_add){                                   // гэ1
        size_t n=(size_t)g_co_M*g_co_cols;
        if(n>g_cpu_dx_n){ if(g_cpu_dx_dev) hipFree(g_cpu_dx_dev); HIP_CHECK(dmalloc(&g_cpu_dx_dev,n*4)); g_cpu_dx_n=n; }
        HIP_CHECK(hipMemcpy(g_cpu_dx_dev,g_cpu_out.data(),n*4,hipMemcpyHostToDevice));
        add_into<<<(n+255)/256,256>>>(g_co_dev,g_cpu_dx_dev,n); HIP_CHECK(hipStreamSynchronize(0));
    } else {                                        // гэ2
        HIP_CHECK(hipMemcpy2D(g_co_dev+g_co_off, (size_t)g_co_stride*4, g_cpu_out.data(), (size_t)g_co_cols*4,
                              (size_t)g_co_cols*4, g_co_M, hipMemcpyHostToDevice));
    }
    g_cpu_secs += cpu_clock()-t0;
}
// гэ3
static std::map<const uint16_t*, std::vector<float>> g_pend;   // гэ4
static std::vector<float> g_pend_x; static std::thread g_pend_th; static bool g_pend_busy=false;
static void cpu_group_begin(const float* x_dev, int M, int K, int layer){
    if(g_cpu_thr<1 || !g_split_ok) return;
    const std::string NM[3]={arch_tn("q"),arch_tn("k"),arch_tn("v")};
    static const char* KY[3]={"q_proj","k_proj","v_proj"};
    std::vector<std::tuple<const uint16_t*,int,int>> job;   // гю1
    for(int i=0;i<3;++i){
        std::string nm=LPX(layer)+NM[i];
        std::string ky="layer_"+std::to_string(layer)+"."+KY[i];
        auto sl=g_weight_layout.find(ky); auto rm=g_ram.find(nm);
        if(sl==g_weight_layout.end() || rm==g_ram.end() || !sl->second.rlen) continue;
        int rows=(int)sl->second.rows, KK=(int)sl->second.rlen, R=gpu_rows(rows);
        if(rows-R<=0 || KK!=K) continue;
        job.emplace_back((const uint16_t*)rm->second, R, rows); }
    if(job.empty()) return;
    g_pend.clear(); g_pend_x.resize((size_t)M*K);
    HIP_CHECK(hipMemcpy(g_pend_x.data(), x_dev, (size_t)M*K*4, hipMemcpyDeviceToHost));
    for(auto& [h,R,rows] : job) g_pend[h].resize((size_t)M*(rows-R));
    g_pend_th = std::thread([job,M,K]{
        for(auto& t : job){ const uint16_t* h=std::get<0>(t); int R=std::get<1>(t), rows=std::get<2>(t);
            cpu_gemm_fwd(g_pend_x.data(), h+(size_t)R*K, g_pend[h].data(), M, K, rows-R); } });
    g_pend_busy=true;
}
static void cpu_group_wait(){ if(!g_pend_busy) return;
    double t0=cpu_clock(); g_pend_th.join(); g_pend_busy=false; g_cpu_wait+=cpu_clock()-t0; }

// да1
__global__ void pack_cols(const float* src, float* dst, int M, int N, int R){
    size_t i=blockIdx.x*(size_t)blockDim.x+threadIdx.x; if(i>=(size_t)M*R) return;
    int m=(int)(i/R), r=(int)(i%R); dst[i]=src[(size_t)m*N+r];
}
static float* g_pk_dev=nullptr; static size_t g_pk_n=0;
static void pool_stop(){ if(g_pool_T<1) return; { std::lock_guard<std::mutex> lk(g_pm); g_pstop=true; ++g_pgen; }
    g_pcv.notify_all(); g_pool_T=0; }
// дб1
static float* g_low_expand=nullptr; static size_t g_low_expand_n=0;
static float* low_expand_buffer(const void* W, size_t n){
    if(n>g_low_expand_n){ if(g_low_expand) HIP_CHECK(hipFree(g_low_expand)); HIP_CHECK(dmalloc(&g_low_expand,n*sizeof(float))); g_low_expand_n=n; }
    low16_to_f32(W,g_low_expand,n); HIP_CHECK(hipStreamSynchronize(0)); return g_low_expand;
}
static void low3_linear(hipblasHandle_t h, hipblasOperation_t oa, int m, int n, int k, const void* W, const float* B, float beta, float* C, int ldc){
    // дг1
    const int lda=(oa==HIPBLAS_OP_N)?m:k;
    if(g_fp16w.count(W)){
        if(g_use_fp16_x3){ fp3gemm(h,oa,m,n,k,(const tier_fp16*)W,lda,B,beta,C,ldc); return; }
        if(g_use_bf16_x3){
            // дг2
            if(auto* wb=fp16_to_bf16_temp((const tier_fp16*)W,(size_t)m*k,0)){ bf3gemm(h,oa,m,n,k,wb,lda,B,beta,C,ldc); return; }
            fprintf(stderr,"BF16×3: не удалось выделить временную матрицу, используется запасной режим FP32.\n");
        }
        float* wf=low_expand_buffer(W,(size_t)m*k);
        sgemm(h,oa,HIPBLAS_OP_N,m,n,k,wf,lda,B,k,beta,C,ldc); return;
    }
    if(g_use_bf16_x3 || g_bf3) bf3gemm(h,oa,m,n,k,(const hip_bfloat16*)W,lda,B,beta,C,ldc);
    else { float* wf=low_expand_buffer(W,(size_t)m*k); sgemm(h,oa,HIPBLAS_OP_N,m,n,k,wf,lda,B,k,beta,C,ldc); }
}
static void linear(hipblasHandle_t h, const float* x, const float* W, float* out, int M, int K, int N){
    if(auto it=g_split.find(W); it!=g_split.end()){
        const RowSplit& sp=it->second; int nc=sp.rows-sp.R;
        if(sp.R>0) low3_linear(h,HIPBLAS_OP_T,sp.R,M,K,W,x,0,out,N);
        if(nc>0){ cpu_group_wait(); auto pit=g_pend.find(sp.host);
            if(pit!=g_pend.end() && pit->second.size()==(size_t)M*nc)
                HIP_CHECK(hipMemcpy2D(out+sp.R,(size_t)N*4,pit->second.data(),(size_t)nc*4,(size_t)nc*4,M,hipMemcpyHostToDevice));
            else { cpu_fwd_begin(x,sp.host+(size_t)sp.R*sp.K,M,K,nc,out,N,sp.R); cpu_end(); } }
        return; }
    if(g_bf16w.count(W)){
        if(g_bf3){ low3_linear(h,HIPBLAS_OP_T,N,M,K,W,x,0,out,N); return; }
        float* wf=low_expand_buffer(W,(size_t)N*K); sgemm(h,HIPBLAS_OP_T,HIPBLAS_OP_N,N,M,K,wf,K,x,K,0,out,N); return;
    }
    sgemm(h,HIPBLAS_OP_T,HIPBLAS_OP_N,N,M,K,W,K,x,K,0,out,N);
}
static void linear_dx(hipblasHandle_t h, const float* dY, const float* W, float* dX, int M, int K, int N){
    if(auto it=g_split.find(W); it!=g_split.end()){
        const RowSplit& sp=it->second; int nc=sp.rows-sp.R;
        if(nc>0) cpu_dx_begin(dY,sp.host+(size_t)sp.R*sp.K,M,K,N,nc,sp.R,dX);
        if(sp.R>0){ size_t need=(size_t)M*sp.R; if(need>g_pk_n){ if(g_pk_dev) hipFree(g_pk_dev); HIP_CHECK(dmalloc(&g_pk_dev,need*4)); g_pk_n=need; }
            pack_cols<<<(need+255)/256,256>>>(dY,g_pk_dev,M,N,sp.R); HIP_CHECK(hipStreamSynchronize(0));
            low3_linear(h,HIPBLAS_OP_N,K,M,sp.R,W,g_pk_dev,0,dX,K);
        } else HIP_CHECK(hipMemset(dX,0,(size_t)M*K*4));
        cpu_end(); return; }
    if(g_bf16w.count(W)){
        if(g_bf3){ low3_linear(h,HIPBLAS_OP_N,K,M,N,W,dY,0,dX,K); return; }
        float* wf=low_expand_buffer(W,(size_t)K*N); sgemm(h,HIPBLAS_OP_N,HIPBLAS_OP_N,K,M,N,wf,K,dY,N,0,dX,K); return;
    }
    sgemm(h,HIPBLAS_OP_N,HIPBLAS_OP_N,K,M,N,W,K,dY,N,0,dX,K);
}
static void linear_dx_acc(hipblasHandle_t h, const float* dY, const float* W, float* dX, int M, int K, int N){
    if(g_bf16w.count(W)){
        if(g_bf3){ low3_linear(h,HIPBLAS_OP_N,K,M,N,W,dY,1,dX,K); return; }
        float* wf=low_expand_buffer(W,(size_t)K*N); sgemm(h,HIPBLAS_OP_N,HIPBLAS_OP_N,K,M,N,wf,K,dY,N,1,dX,K); return;
    }
    sgemm(h,HIPBLAS_OP_N,HIPBLAS_OP_N,K,M,N,W,K,dY,N,1,dX,K);
}
static std::map<float*,size_t> g_gsz;   // дж1
// дж2
static bool g_nsgd=false, g_posdw=false;
struct Whiten{ int in=0, r=0; float* U=nullptr; float* f=nullptr; };   // дз1
static std::map<std::string,Whiten> g_wh; static std::string g_wh_path;
// дз2
static std::map<std::string,std::vector<int>> g_ridx; static std::map<std::string,int*> g_ridx_d;
static std::map<std::string,std::vector<float>> g_row_scores;
static const int* ridx_dev(const std::string& k){ auto it=g_ridx.find(k); if(it==g_ridx.end()||it->second.empty()) return nullptr;
    int*& d=g_ridx_d[k]; if(!d){ HIP_CHECK(hipMalloc(&d,it->second.size()*4)); HIP_CHECK(hipMemcpy(d,it->second.data(),it->second.size()*4,hipMemcpyHostToDevice)); } return d; }
__global__ void gather_rows_k(const void* src, int ty, float* dst, const int* idx, int r, size_t K){ size_t i=blockIdx.x*(size_t)blockDim.x+threadIdx.x; if(i>=(size_t)r*K) return;
    size_t row=i/K, c=i%K, o=(size_t)idx[row]*K+c; dst[i] = ty==2 ? (float)((const tier_fp16*)src)[o] : ty==1 ? (float)((const hip_bfloat16*)src)[o] : ((const float*)src)[o]; }
__global__ void scatter_rows_k(const float* src, void* dst, int ty, const int* idx, int r, size_t K){ size_t i=blockIdx.x*(size_t)blockDim.x+threadIdx.x; if(i>=(size_t)r*K) return;
    size_t row=i/K, c=i%K, o=(size_t)idx[row]*K+c; if(ty==2) ((tier_fp16*)dst)[o]=tier_fp16(src[i]); else if(ty==1) ((hip_bfloat16*)dst)[o]=hip_bfloat16(src[i]); else ((float*)dst)[o]=src[i]; }
__global__ void gather_cols_k(const float* y, float* out, const int* idx, int M, int r, int ld){ size_t i=blockIdx.x*(size_t)blockDim.x+threadIdx.x; if(i>=(size_t)M*r) return;
    size_t t=i/r, j=i%r; out[i]=y[t*ld+idx[j]]; }
__global__ void scatter_cols_k(const float* src, float* y, const int* idx, int M, int r, int ld){ size_t i=blockIdx.x*(size_t)blockDim.x+threadIdx.x; if(i>=(size_t)M*r) return;
    size_t t=i/r, j=i%r; y[t*ld+idx[j]]=src[i]; }
__global__ void rowsq_add_k(const float* g, float* acc, int rows, size_t K){ int row=blockIdx.x; float s=0; for(size_t c=threadIdx.x;c<K;c+=blockDim.x){ float v=g[(size_t)row*K+c]; s+=v*v; }
    __shared__ float red[256]; red[threadIdx.x]=s; __syncthreads(); for(int k=blockDim.x/2;k>0;k>>=1){ if(threadIdx.x<k) red[threadIdx.x]+=red[threadIdx.x+k]; __syncthreads(); } if(!threadIdx.x) acc[row]+=red[0]; }
static float *g_dwmask=nullptr, *g_dwtmp=nullptr, *g_wtmp=nullptr; static size_t g_dwmask_n=0, g_dwtmp_n=0, g_wtmp_n=0;
__global__ void rowmask_k(const float* s, const float* m, float* d, size_t rows, size_t cols, size_t ld){
    size_t i=blockIdx.x*(size_t)blockDim.x+threadIdx.x; if(i>=rows*cols) return; size_t r=i/cols, c=i%cols; d[i]=s[r*ld+c]*m[r]; }
static std::map<float*,std::string> g_gkey; static std::string g_rowprobe; static std::map<std::string,float*> g_rowacc;   // дп1
static void linear_dw(hipblasHandle_t h, const float* dY, const float* X, float* dW, int M, int K, int N){
    if(!dW) return; // др1
    const std::string* key=nullptr; if(auto gk=g_gkey.find(dW); gk!=g_gkey.end()) key=&gk->second;
    if(!g_rowprobe.empty() && key && g_weight_layout.at(*key).rlen>1){   // др2
        float*& acc=g_rowacc[*key]; if(!acc){ HIP_CHECK(hipMalloc(&acc,(size_t)N*4)); HIP_CHECK(hipMemset(acc,0,(size_t)N*4)); }
        float* full; HIP_CHECK(cmalloc(&full,(size_t)N*K*4)); sgemm(h,HIPBLAS_OP_N,HIPBLAS_OP_T,K,N,M,X,K,dY,N,0,full,K);
        rowsq_add_k<<<N,256>>>(full,acc,N,(size_t)K); cfree(full); }
    if(key){ if(const int* ri=ridx_dev(*key)){ int r=(int)g_ridx.at(*key).size(); size_t need=(size_t)M*r;   // др3
        if(need>g_dwtmp_n){ if(g_dwtmp) HIP_CHECK(hipFree(g_dwtmp)); HIP_CHECK(dmalloc(&g_dwtmp,need*sizeof(float))); g_dwtmp_n=need; }
        gather_cols_k<<<(unsigned)((need+255)/256),256>>>(dY,g_dwtmp,ri,M,r,N);
        if(g_posdw && g_dwmask && (size_t)M<=g_dwmask_n) rowmask_k<<<(unsigned)((need+255)/256),256>>>(g_dwtmp,g_dwmask,g_dwtmp,M,r,r);
        sgemm(h,HIPBLAS_OP_N,HIPBLAS_OP_T,K,r,M,X,K,g_dwtmp,r,0,dW,K); return; } }
    // др4
    int ld=N;
    if(auto it=g_gsz.find(dW); it!=g_gsz.end()){ int ns=(int)(it->second/(size_t)K); if(ns>0 && ns<N) N=ns; }
    if(g_posdw && g_dwmask && (size_t)M<=g_dwmask_n){   // др5
        size_t need=(size_t)M*N; if(need>g_dwtmp_n){ if(g_dwtmp) HIP_CHECK(hipFree(g_dwtmp)); HIP_CHECK(dmalloc(&g_dwtmp,need*sizeof(float))); g_dwtmp_n=need; }
        rowmask_k<<<(unsigned)((need+255)/256),256>>>(dY,g_dwmask,g_dwtmp,M,N,ld); dY=g_dwtmp; ld=N; }
    sgemm(h,HIPBLAS_OP_N,HIPBLAS_OP_T,K,N,M,X,K,dY,ld,0,dW,K);
}

// др6
__global__ void adam_step_kernel(const float* grad, float* weights, float* m, float* v,
                                  size_t n, float lr, float beta1, float beta2, float bc1, float bc2, float eps){
    size_t i = blockIdx.x*(size_t)blockDim.x + threadIdx.x;
    if(i >= n) return;
    float g = grad[i];
    float mi = beta1*m[i] + (1.0f-beta1)*g;
    m[i] = mi;
    v[i] = beta2*v[i] + (1.0f-beta2)*g*g;
    float m_hat = mi/bc1, v_hat = v[i]/bc2;
    weights[i] -= lr * m_hat / (sqrtf(v_hat) + eps);
}

// дт1
__global__ void nsgd_mom_k(const float* g, float* m, size_t n, float* acc){
    __shared__ float sh[256]; float a=0;
    for(size_t i=blockIdx.x*(size_t)blockDim.x+threadIdx.x;i<n;i+=(size_t)gridDim.x*blockDim.x){
        float b=0.9f*m[i]+g[i]; m[i]=b; a+=b*b; }
    sh[threadIdx.x]=a; __syncthreads();
    for(int o=128;o>0;o>>=1){ if(threadIdx.x<o) sh[threadIdx.x]+=sh[threadIdx.x+o]; __syncthreads(); }
    if(threadIdx.x==0) atomicAdd(acc,sh[0]);
}
__global__ void nsgd_step_k(float* w, const float* m, size_t n, float scale){
    size_t i=blockIdx.x*(size_t)blockDim.x+threadIdx.x; if(i<n) w[i]-=scale*m[i]; }
__global__ void colscale_k(float* T, const float* f, int r, size_t n){ size_t i=blockIdx.x*(size_t)blockDim.x+threadIdx.x; if(i<n) T[i]*=f[i%r]; }
static void whiten_load(){ static bool done=false; if(done || g_wh_path.empty()) return; done=true;
    FILE* f=fopen(g_wh_path.c_str(),"rb"); if(!f){ perror(g_wh_path.c_str()); exit(1); }
    int cnt=0; if(fread(&cnt,4,1,f)!=1){ fprintf(stderr,"Защита общих знаний: пустой файл %s.\n",g_wh_path.c_str()); exit(1); }
    for(int i=0;i<cnt;++i){ int kl=0; fread(&kl,4,1,f); std::string k(kl,'\0'); fread(&k[0],1,kl,f); Whiten w; fread(&w.in,4,1,f); fread(&w.r,4,1,f);
        std::vector<float> hf(w.r), hu((size_t)w.in*w.r);
        if(fread(hf.data(),4,w.r,f)!=(size_t)w.r || fread(hu.data(),4,hu.size(),f)!=hu.size()){ fprintf(stderr,"Защита общих знаний: файл оборван на тензоре %s.\n",k.c_str()); exit(1); }
        for(auto& x : hf) x=-x;                                  // дх1
        HIP_CHECK(hipMalloc(&w.f,w.r*sizeof(float))); HIP_CHECK(hipMalloc(&w.U,hu.size()*sizeof(float)));
        HIP_CHECK(hipMemcpy(w.f,hf.data(),w.r*sizeof(float),hipMemcpyHostToDevice)); HIP_CHECK(hipMemcpy(w.U,hu.data(),hu.size()*sizeof(float),hipMemcpyHostToDevice));
        g_wh[k]=w; }
    fclose(f); fprintf(stderr,"Защита общих знаний: матриц %zu, файл %s.\n",g_wh.size(),g_wh_path.c_str()); }
static const Whiten* whiten_for(const std::string& key){   // дц1
    whiten_load(); auto d=key.rfind('.'); if(d==std::string::npos) return nullptr; std::string t=key.substr(d+1), L=key.substr(0,d+1);
    std::string k = (t=="q_proj"||t=="k_proj"||t=="v_proj") ? L+"qkv" : (t=="gate_proj"||t=="up_proj") ? L+"gateup" : key;
    auto it=g_wh.find(k); return it==g_wh.end() ? nullptr : &it->second; }
static void sgemm(hipblasHandle_t h, hipblasOperation_t oa, hipblasOperation_t ob, int m, int n, int k, const float* A, int lda, const float* B, int ldb, float beta, float* C, int ldc);
static void whiten_apply(hipblasHandle_t h, const Whiten& w, float* G, size_t n){   // дч1
    int rows=(int)(n/(size_t)w.in); size_t need=(size_t)w.r*rows;
    if(need>g_wtmp_n){ if(g_wtmp) HIP_CHECK(hipFree(g_wtmp)); HIP_CHECK(dmalloc(&g_wtmp,need*sizeof(float))); g_wtmp_n=need; }
    sgemm(h,HIPBLAS_OP_T,HIPBLAS_OP_N,w.r,rows,w.in,w.U,w.in,G,w.in,0,g_wtmp,w.r);          // дч2
    colscale_k<<<(unsigned)((need+255)/256),256>>>(g_wtmp,w.f,w.r,need);                     // дч3
    sgemm(h,HIPBLAS_OP_N,HIPBLAS_OP_N,w.in,rows,w.r,w.U,w.in,g_wtmp,w.r,1,G,w.in); }          // дч4

struct Weights { float *iln=nullptr,*paln=nullptr,*pfln=nullptr,*pofln=nullptr,*q=nullptr,*k=nullptr,*v=nullptr,*o=nullptr,*qn=nullptr,*kn=nullptr,*gate=nullptr,*up=nullptr,*down=nullptr,*ls=nullptr,*qb=nullptr,*kb=nullptr,*vb=nullptr; };

// дш1
static size_t g_weight_layout_total_floats = 0;
static std::string g_checkpoint_path;
static bool g_checkpoint_exists_with_data = false; // дш2
// дш3
static bool g_compact=false; static std::map<std::string,size_t> g_c1off; static std::string g_c1hdr; static const size_t C1_HDR=64<<10;
static void pread_all(int fd, void* p, size_t n, size_t off){
    size_t t=0;
    while(t<n){ ssize_t r=pread(fd,(char*)p+t,n-t,off+t);
        if(r<0 && errno==EINTR) continue;
        if(r<=0){ if(r==0) fprintf(stderr,"Чтение (pread): неожиданный конец файла (смещение %zu, требуется байт: %zu).\n",off+t,n-t); else perror("Чтение (pread)"); exit(1); }
        t+=(size_t)r;
    }
}
static void pwrite_all(int fd, const void* p, size_t n, size_t off){
    size_t t=0;
    while(t<n){ ssize_t r=pwrite(fd,(const char*)p+t,n-t,off+t);
        if(r<0 && errno==EINTR) continue;
        if(r<=0){ if(r==0) fprintf(stderr,"Запись (pwrite): возвращён нулевой результат (смещение %zu, осталось байт: %zu).\n",off+t,n-t); else perror("Запись (pwrite)"); exit(1); }
        t+=(size_t)r;
    }
}

static void register_weight_slot(int layer_idx, const std::string& name, size_t n, size_t rows=0){
    std::string key = "layer_" + std::to_string(layer_idx) + "." + name;
    if(!rows) rows=n;                      // дя1
    g_weight_layout[key] = {g_weight_layout_total_floats, n, -1, rows, n/rows, 0};
    g_weight_layout_total_floats += n;
}

// дя2
static int g_sl_lo=0, g_sl_hi=1<<30;
static int layer_of(const std::string& k){ return (k.compare(0,6,"layer_")==0) ? atoi(k.c_str()+6) : -1; }
static bool in_slice(const std::string& k){ int l=layer_of(k); return l>=g_sl_lo && l<=g_sl_hi; }   // еб1
static bool g_row_global=false;   // еб2
static void assign_tiers_slice(size_t budget){
    size_t norms=0, mat=0;
    for(auto& [k,sl] : g_weight_layout){ if(!in_slice(k)) continue; (sl.rlen==1 ? norms : mat) += sl.n; }
    if(budget<=norms){ fprintf(stderr,"Срез строк: бюджет %zu меньше нормировок %zu, срез неприменим.\n",budget,norms); exit(1); }
    double frac = (double)(budget-norms)/(double)mat; if(frac>1) frac=1;
    size_t used=0, nm=0, nt=0, froz=0;
    for(auto& [k,sl] : g_weight_layout){
        if(!in_slice(k)){ sl.tier=2; sl.ntr=0; ++froz; continue; }
        if(sl.rlen==1){ sl.tier=1; sl.ntr=sl.n; used+=sl.n; ++nm; continue; }
        size_t r = (size_t)(sl.rows*frac); if(!r) r=1; if(r>sl.rows) r=sl.rows;
        sl.tier=1; sl.ntr=r*sl.rlen; used+=sl.ntr; ++nt;
        if(auto sc=g_row_scores.find(k); sc!=g_row_scores.end() && sc->second.size()==sl.rows && r<sl.rows){   // ев1
            std::vector<int> o(sl.rows); for(size_t i=0;i<sl.rows;++i) o[i]=(int)i;
            std::partial_sort(o.begin(),o.begin()+r,o.end(),[&](int x,int y){ return sc->second[x]>sc->second[y]; });
            o.resize(r); std::sort(o.begin(),o.end()); g_ridx[k]=o; }
    }
    if(g_row_global && !g_row_scores.empty()){   // ев2
        struct Rw{ float s; const std::string* k; int i; size_t len; }; std::vector<Rw> all; size_t mb=0;
        for(auto& [k,sl] : g_weight_layout){ if(!in_slice(k) || sl.rlen==1) continue; mb+=sl.ntr; auto sc=g_row_scores.find(k);
            if(sc==g_row_scores.end() || sc->second.size()!=sl.rows) continue; for(size_t i=0;i<sl.rows;++i) all.push_back({sc->second[i]/(float)sl.rlen,&k,(int)i,sl.rlen}); }
        std::sort(all.begin(),all.end(),[](const Rw& a,const Rw& b){ return a.s>b.s; });
        std::map<std::string,std::vector<int>> pick; size_t got=0; for(auto& r : all){ if(got+r.len>mb) continue; pick[*r.k].push_back(r.i); got+=r.len; }
        size_t nz=0; used=norms;
        for(auto& [k,sl] : g_weight_layout){ if(!in_slice(k) || sl.rlen==1 || !g_row_scores.count(k)) continue; auto& v=pick[k];
            if(v.empty()){ sl.tier=2; sl.ntr=0; g_ridx.erase(k); ++nz; continue; }
            std::sort(v.begin(),v.end()); if(v.size()>=sl.rows){ sl.ntr=sl.n; g_ridx.erase(k); } else { sl.ntr=v.size()*sl.rlen; g_ridx[k]=v; } used+=sl.ntr; }
        fprintf(stderr,"Срез строк по общему бюджету: обучается %zu чисел, матриц без строк %zu.\n",used,nz); }
    fprintf(stderr,"Срез строк: слои %d..%d, %.2f%% строк каждой матрицы, матриц %zu и нормировок %zu (заморожено %zu), всего параметров %zu из бюджета %zu.\n",
            g_sl_lo, (g_sl_hi>1<<20?N_LAYERS-1:g_sl_hi), 100*frac, nt, nm, froz, used, budget);
}
// ег1
static void save_tier_assignment(const std::string& path){
    FILE* f = fopen(path.c_str(), "w");
    fprintf(f, "# имя_тензора уровень всего_чисел обучается_чисел\n");
    for(auto& kv : g_weight_layout)
        fprintf(f, "%s %d %zu %zu\n", kv.first.c_str(), kv.second.tier, kv.second.n, kv.second.ntr);
    for(auto& [k,v] : g_ridx){ fprintf(f,"rows %s",k.c_str()); for(int x : v) fprintf(f," %d",x); fprintf(f,"\n"); }
    fclose(f);
    fprintf(stderr, "Раскладка среза записана: %s\n", path.c_str());
}

// ед1
static bool load_tier_assignment(const std::string& path){
    FILE* f = fopen(path.c_str(), "r");
    if(!f) return false;
    char* line=nullptr; size_t lcap=0; char name[256]; int tier; size_t n, ntr; double a,b,c;
    int loaded=0, sliced=0; g_ridx.clear();
    while(getline(&line, &lcap, f)>0){
        if(line[0]=='#') continue;
        if(!strncmp(line,"rows ",5)){ std::istringstream is(line+5); std::string k; is>>k; std::vector<int> v; int x; while(is>>x) v.push_back(x); if(!v.empty()) g_ridx[k]=v; continue; }   // ед2
        int cols=0; for(char* q=line; *q; ){ while(*q==' '||*q=='\t'||*q=='\n') ++q; if(*q){ ++cols; while(*q && *q!=' ' && *q!='\t' && *q!='\n') ++q; } }
        if(cols>=7){ if(sscanf(line, "%255s %d %le %le %le %zu %zu", name, &tier, &a, &b, &c, &n, &ntr)<7) continue; }   // ед3
        else if(sscanf(line, "%255s %d %zu %zu", name, &tier, &n, &ntr)<4) continue;
        if(!g_weight_layout.count(name)) continue;
        auto& sl = g_weight_layout[name]; sl.tier = tier;
        sl.ntr = (tier==1) ? (ntr ? ntr : sl.n) : 0;
        if(sl.ntr && sl.ntr < sl.n) ++sliced;
        loaded++;
    }
    fclose(f); free(line);
    for(auto it=g_ridx.begin(); it!=g_ridx.end();){ auto sl=g_weight_layout.find(it->first);   // ед4
        if(sl==g_weight_layout.end() || !sl->second.rlen || it->second.size()*sl->second.rlen!=sl->second.ntr) it=g_ridx.erase(it); else ++it; }
    fprintf(stderr, "Раскладка среза прочитана: %s (тензоров: %d%s%s).\n", path.c_str(), loaded, sliced?", часть — срезами строк":"", g_ridx.empty()?"":", строки выбраны замером");
    return loaded > 0;
}

// ее1
struct Resident{ float *w,*v,*m; };
static std::map<std::string,Resident> g_res;
static std::set<float*> g_res_ptrs;
static std::set<std::string> g_ckpt_written; // ее2
static bool g_train_tier2=false, g_train_tier3=false;
static void resident_register(const std::string& key, float* w);
// еж1
__global__ void f32_to_bf16_kernel(const float* src, hip_bfloat16* dst, size_t n){
    size_t i=blockIdx.x*(size_t)blockDim.x+threadIdx.x; if(i<n) dst[i]=hip_bfloat16(src[i]);
}
static float* load_tensor_or_checkpoint(const std::string& safetensors_name, const std::string& ckpt_key, void* stage, size_t cap){
    auto& sl=g_weight_layout.at(ckpt_key);
    bool sliced = sl.ntr && sl.ntr < sl.n;
    if(!sliced){ if(auto it=g_res.find(ckpt_key); it!=g_res.end()) return it->second.w; }
    float* d;
    if(sliced){   // ез1
        d = load_tensor_f32(safetensors_name, stage, cap, true);
        const int* ri = ridx_dev(ckpt_key); const int rr = ri ? (int)g_ridx.at(ckpt_key).size() : 0; const int dty = weight_type(d);
        auto it=g_res.find(ckpt_key);
        if(it==g_res.end()){                       // ез2
            float* w; HIP_CHECK(res_alloc(&w, sl.ntr*sizeof(float)));
            if(g_checkpoint_exists_with_data && g_compact && g_c1off.count(ckpt_key)){
                std::vector<float> h(sl.ntr); int fd=open(g_checkpoint_path.c_str(),O_RDONLY);
                pread_all(fd,h.data(),sl.ntr*sizeof(float),g_c1off.at(ckpt_key)); close(fd);
                HIP_CHECK(hipMemcpy(w,h.data(),sl.ntr*sizeof(float),hipMemcpyHostToDevice));
            } else if(ri){   // ез3
                if(tq_of(safetensors_name)){ std::vector<uint16_t> h(sl.ntr); int fd=open(g_path.c_str(),O_RDONLY); size_t rb=sl.rlen*2;
                    for(int i=0;i<rr;++i) pread_all(fd,h.data()+(size_t)i*sl.rlen,rb,g_tensors.at(safetensors_name).offset+(size_t)g_ridx.at(ckpt_key)[i]*rb); close(fd);
                    hip_bfloat16* tb; HIP_CHECK(cmalloc(&tb,sl.ntr*2)); HIP_CHECK(hipMemcpy(tb,h.data(),sl.ntr*2,hipMemcpyHostToDevice));
                    int th=256,bl=(int)((sl.ntr+th-1)/th); bf16_to_f32_kernel<<<bl,th>>>(tb,w,sl.ntr); HIP_CHECK(hipStreamSynchronize(0)); cfree(tb); }
                else gather_rows_k<<<(unsigned)((sl.ntr+255)/256),256>>>(d,dty,w,ri,rr,sl.rlen);
            } else if(tq_of(safetensors_name)){    // ез4
                std::vector<uint16_t> h(sl.ntr); int fd=open(g_path.c_str(),O_RDONLY); pread_all(fd,h.data(),sl.ntr*2,g_tensors.at(safetensors_name).offset); close(fd);
                hip_bfloat16* tb; HIP_CHECK(cmalloc(&tb,sl.ntr*2)); HIP_CHECK(hipMemcpy(tb,h.data(),sl.ntr*2,hipMemcpyHostToDevice));
                int th=256,bl=(int)((sl.ntr+th-1)/th); bf16_to_f32_kernel<<<bl,th>>>(tb,w,sl.ntr); HIP_CHECK(hipStreamSynchronize(0)); cfree(tb);
            } else if(g_fp16w.count(d)){ int th=256,bl=(int)((sl.ntr+th-1)/th); fp16_to_f32_kernel<<<bl,th>>>((const tier_fp16*)d,w,sl.ntr);
            } else if(g_bf16w.count(d)){ int th=256,bl=(int)((sl.ntr+th-1)/th); bf16_to_f32_kernel<<<bl,th>>>((const hip_bfloat16*)d,w,sl.ntr);
            } else HIP_CHECK(hipMemcpy(w,d,sl.ntr*sizeof(float),hipMemcpyDeviceToDevice));
            resident_register(ckpt_key,w); it=g_res.find(ckpt_key);
        }
        if(ri) scatter_rows_k<<<(unsigned)((sl.ntr+255)/256),256>>>(it->second.w,d,dty,ri,rr,sl.rlen);   // ез5
        else if(g_fp16w.count(d)){ int th=256,bl=(int)((sl.ntr+th-1)/th); f32_to_fp16_kernel<<<bl,th>>>(it->second.w,(tier_fp16*)d,sl.ntr); }
        else if(g_bf16w.count(d)){ int th=256,bl=(int)((sl.ntr+th-1)/th); f32_to_bf16_kernel<<<bl,th>>>(it->second.w,(hip_bfloat16*)d,sl.ntr); }
        else HIP_CHECK(hipMemcpy(d,it->second.w,sl.ntr*sizeof(float),hipMemcpyDeviceToDevice));
        return d;
    }
    bool frozen=(sl.tier==2&&!g_train_tier2)||(sl.tier==3&&!g_train_tier3); // ез6
    // ез7
    if(frozen && cpu_owned(safetensors_name) && g_bf3){
        auto rit=g_ram.find(safetensors_name);
        if(rit!=g_ram.end() && sl.rlen>0){
            int rows=(int)sl.rows, K=(int)sl.rlen;
            int R=(int)(rows*g_gpu_frac); if(R<0) R=0; if(R>rows) R=rows;
            float* d=nullptr;
            auto pf=g_pfm.find(safetensors_name); auto pr=g_pf_rows.find(safetensors_name);
            if(R>0 && pf!=g_pfm.end() && pr!=g_pf_rows.end() && pr->second==R){
                // ез8
                HIP_CHECK(hipStreamWaitEvent(0,g_pf_ev,0));
                HIP_CHECK(cmalloc(&d,(size_t)R*K*2));
                HIP_CHECK(hipMemcpyAsync(d,pf->second,(size_t)R*K*2,hipMemcpyDeviceToDevice,0));
                g_pfm.erase(pf); ++g_pf_hit;
            } else if(R>0){ HIP_CHECK(cmalloc(&d,(size_t)R*K*2));
                HIP_CHECK(hipMemcpy(d,rit->second,(size_t)R*K*2,hipMemcpyHostToDevice)); g_h2d+=(size_t)R*K*2;
                if(g_pf_on) ++g_pf_miss; }
            else { HIP_CHECK(cmalloc(&d,256)); }     // ез9
            if(R>0 && g_fp16_storage){ bf16_to_fp16_kernel<<<(unsigned)(((size_t)R*K+255)/256),256>>>((const hip_bfloat16*)d,(tier_fp16*)d,(size_t)R*K); HIP_CHECK(hipStreamSynchronize(0)); g_fp16w.insert(d); }
            g_bf16w.insert(d); g_split[d]={(const uint16_t*)rit->second,rows,R,K};
            return d; } }
    if(g_checkpoint_exists_with_data && !frozen){
        auto& slot = g_weight_layout.at(ckpt_key);
        std::vector<float> h(slot.n);
        int fd = open(g_checkpoint_path.c_str(), O_RDONLY);
        pread_all(fd, h.data(), slot.n*sizeof(float), g_compact ? g_c1off.at(ckpt_key) : slot.offset_floats*sizeof(float));
        close(fd);
        HIP_CHECK(cmalloc(&d, slot.n*sizeof(float)));
        HIP_CHECK(hipMemcpy(d, h.data(), slot.n*sizeof(float), hipMemcpyHostToDevice));
    } else d = load_tensor_f32(safetensors_name, stage, cap, frozen);
    if(g_weight_layout.at(ckpt_key).tier==1) resident_register(ckpt_key, d);
    return d;
}

static void checkpoint_store_weight(const std::string& ckpt_key, float* w){
    auto& slot = g_weight_layout.at(ckpt_key);
    std::vector<float> h(slot.n);
    HIP_CHECK(hipMemcpy(h.data(), w, slot.n*sizeof(float), hipMemcpyDeviceToHost));
    int fd = open(g_checkpoint_path.c_str(), O_RDWR);
    pwrite_all(fd, h.data(), slot.n*sizeof(float), slot.offset_floats*sizeof(float));
    close(fd);
}

static const char* PF_MN[7]={"q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"};
static void pf_list(int l, std::vector<std::pair<std::string,size_t>>& out){ out.clear(); // ел1
    for(int i=0;i<7;++i){ if(i==2 && !A.L[l].has_v) continue; std::string k="layer_"+std::to_string(l)+"."+PF_MN[i], nm=LPX(l)+arch_tn(SLOT7[i]);
        auto& sl=g_weight_layout.at(k);
        bool sliced = sl.ntr && sl.ntr < sl.n;                       // ел2
        if((g_res.count(k) && !sliced) || !g_ram.count(nm) || g_rbf.count(nm)) continue;
        if(!(sliced||(sl.tier==2&&!g_train_tier2)||(sl.tier==3&&!g_train_tier3))) continue;
        // ел3
        if(cpu_owned(nm) && sl.rlen){ int R=gpu_rows((int)sl.rows); if(R<=0) continue;
            g_pf_rows[nm]=R; out.push_back({nm,(size_t)R*sl.rlen*2}); continue; }
        const QRef* q=tq_of(nm); out.push_back({nm,q?q->size:g_tensors.at(nm).size}); } }
static const char* PF_MN0[7]={"q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"};
static void ram_prefill(int n_layers, bool tr2, bool tr3){
    arena_init(); if(!g_arena) return;
    std::vector<std::pair<uint64_t,std::string>> ord;   // ен1
    for(int l=0;l<n_layers;++l) for(int i=0;i<7;++i){
        if(i==2 && !A.L[l].has_v) continue;
        std::string k="layer_"+std::to_string(l)+"."+PF_MN0[i];
        std::string nm=LPX(l)+arch_tn(SLOT7[i]);
        auto sl=g_weight_layout.find(k); if(sl==g_weight_layout.end()||!g_tensors.count(nm)) continue;
        bool sliced=sl->second.ntr && sl->second.ntr<sl->second.n;   // ен2
        bool frozen=sliced||(sl->second.tier==2&&!tr2)||(sl->second.tier==3&&!tr3);
        if(frozen) ord.push_back({g_tensors.at(nm).offset,nm}); }
    // ен3
    { std::set<std::string> have; for(auto& e:ord) have.insert(e.second);
      for(auto& [nm,rf] : g_tensors){
          if(nm.compare(0,A.pre.size(),A.pre)!=0) continue;
          if(nm.find("embed_tokens")!=std::string::npos) continue;      // ен4
          if(nm.find("_proj.weight")!=std::string::npos) continue;      // ен5
          if(have.count(nm)) continue;
          ord.push_back({rf.offset,nm}); } }
    // ен6
    std::vector<std::pair<uint64_t,std::string>> ordq;
    for(auto it=ord.begin(); it!=ord.end();){ if(const QRef* q=tq_of(it->second)){ ordq.push_back({q->offset,it->second}); it=ord.erase(it); } else ++it; }
    if(ord.empty() && ordq.empty()) return;
    size_t total=0; for(auto& e:ord) total+=g_tensors.at(e.second).size; for(auto& e:ordq) total+=tq_of(e.second)->size;
    size_t done=0; auto clk=[]{ return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); };
    double t0=clk(); size_t reads=0, items=ord.size()+ordq.size(); bool direct_any=false;
    auto ref_of=[](const std::string& nm){ const QRef* q=tq_of(nm); return q ? TensorRef{q->offset,q->size} : g_tensors.at(nm); };
    auto pass=[&](const std::string& path, std::vector<std::pair<uint64_t,std::string>>& O){
    if(O.empty()) return;
    std::sort(O.begin(),O.end());
    // ен7
    const size_t AL=4096, CH=64ull<<20;
    const char* dv=getenv("TIER_DIRECT");
    int fd = (dv && atoi(dv)) ? open(path.c_str(),O_RDONLY|O_DIRECT) : -1;
    bool direct=fd>=0;
    if(!direct){ fd=open(path.c_str(),O_RDONLY); if(fd<0){ perror(path.c_str()); return; }
        posix_fadvise(fd,0,0,POSIX_FADV_SEQUENTIAL); }
    // ен8
    size_t big=CH; for(auto& e:O) big=std::max<size_t>(big,ref_of(e.second).size);
    void* bb=nullptr; if(posix_memalign(&bb,AL,big+2*AL)!=0){ close(fd); return; }
    size_t i=0;
    while(i<O.size()){
        size_t j=i; TensorRef r0=ref_of(O[i].second); uint64_t beg=r0.offset, end=beg+r0.size;
        while(j+1<O.size()){   // ен9
            TensorRef nx=ref_of(O[j+1].second);
            if(nx.offset<end || nx.offset+nx.size-beg>CH) break;
            end=nx.offset+nx.size; ++j; }
        uint64_t ab=beg/AL*AL, ae=(end+AL-1)/AL*AL;   // ен10
        uint64_t t=0, n=ae-ab;
        while(t<n){ ssize_t r=pread(fd,(char*)bb+t,n-t,ab+t);
            if(r<=0){ if(direct && t==0){ close(fd); fprintf(stderr,"Прямое чтение не поддерживается, используется обычное чтение.\n");
                    direct=false; fd=open(path.c_str(),O_RDONLY); if(fd<0){ free(bb); return; } posix_fadvise(fd,0,0,POSIX_FADV_SEQUENTIAL); continue; }
                if(r==0 && t>0 && ab+t>=end) break;    // ен11
                perror("Чтение кэша (pread)"); close(fd); free(bb); return; }
            t+=r; }
        ++reads;
        if(!direct) posix_fadvise(fd,ab,n,POSIX_FADV_DONTNEED);
        for(size_t k=i;k<=j;++k){ TensorRef rf=ref_of(O[k].second);
            if(g_arena_used+rf.size>g_arena_cap) continue;   // ен12
            char* dst=g_arena+g_arena_used; g_arena_used+=(rf.size+255)/256*256;
            memcpy(dst,(char*)bb+(rf.offset-ab),rf.size);
            g_ram[O[k].second]=dst; g_ram_bytes+=rf.size; done+=rf.size; }
        // ен13
        { double el=clk()-t0; static double last=-1e9;
          if(!g_to_file || el-last>=2.0 || done==total){ last=el;
              fprintf(stderr,"\rЗагрузка %.1f/%.1f ГиБ, %.0f МБ/с [%.1f с]%s",
                  done/1073741824.0,total/1073741824.0,done/1e6/(el>0?el:1e-9),el,g_to_file?"\n":""); } }
        i=j+1; }
    // ен14
    { int fv=open(path.c_str(),O_RDONLY); std::vector<char> a(4096); size_t bad=0; std::string first;
      for(auto& e:O){ auto it=g_ram.find(e.second); if(it==g_ram.end()) continue; TensorRef rf=ref_of(e.second);
          size_t w=std::min<size_t>(4096,rf.size), offs[2]={0,rf.size-w};
          for(size_t o : offs){ if(pread(fv,a.data(),w,rf.offset+o)!=(ssize_t)w || memcmp(a.data(),(char*)it->second+o,w)){ if(!bad++) first=e.second; break; } } }
      if(fv>=0) close(fv);
      if(bad){ fprintf(stderr,"Кэш весов не совпал с файлом: тензоров %zu, первый %s. Работа остановлена, чтобы не обучать модель на испорченных весах.\n",bad,first.c_str()); exit(1); } }
    posix_fadvise(fd,0,0,POSIX_FADV_DONTNEED);   // ен15
    close(fd); free(bb); direct_any|=direct; };
    pass(g_path,ord); pass(g_tq_path,ordq);
    double dt=clk()-t0;
    fprintf(stderr,"\rКэш весов: %.2f ГиБ за %.1f с (%.0f МБ/с, чтений: %zu вместо %zu, %s).\n",
            done/1073741824.0,dt,done/1048576.0/dt,reads,items,direct_any?"в обход кэша системы":"обычное чтение");
}
static void pf_enable(){ if(g_pf_on||!g_use_pf) return; std::vector<std::pair<std::string,size_t>> v; size_t mx=0;
    for(int l=0;l<N_LAYERS;++l){ pf_list(l,v); size_t t=0; for(auto& e:v) t+=(e.second+255)/256*256; mx=std::max(mx,t); }
    if(!mx) return; size_t dbf_freed=g_dbf_n; if(g_dbf){ HIP_CHECK(hipFree(g_dbf)); g_dbf=nullptr; g_dbf_n=0; } // ео1
    { size_t fr=0,tt=0; hipMemGetInfo(&fr,&tt); size_t tr=step_transient(tt-fr,dbf_freed), keep=(tr?tr:(512ull<<20))+vram_slack();   // ео2
      if(fr<mx+keep){ g_use_pf=false; fprintf(stderr,"Опережающая подкачка: требуется %.0f МиБ плюс запас %.0f МиБ, свободно %.0f МиБ. Опережающая подкачка отключена.\n",mx/1048576.0,keep/1048576.0,fr/1048576.0); return; } }
    if(hipMalloc((void**)&g_stg,mx)!=hipSuccess){ g_stg=nullptr; g_use_pf=false; fprintf(stderr,"Опережающая подкачка: не хватило видеопамяти на %.0f МиБ. Опережающая подкачка отключена.\n",mx/1048576.0); return; }
    g_stg_cap=mx; HIP_CHECK(hipStreamCreateWithFlags(&g_pf,hipStreamNonBlocking)); HIP_CHECK(hipEventCreateWithFlags(&g_pf_ev,hipEventDisableTiming)); HIP_CHECK(hipEventCreateWithFlags(&g_conv_ev,hipEventDisableTiming));
    g_pf_on=true; fprintf(stderr,"Опережающая подкачка: %.0f МиБ под слой.\n",mx/1048576.0); }
static void rbf_drop();
static bool vram_relief(){ if(!g_rbf.empty()){ rbf_drop(); return true; }   // еп1
    if(!g_pf_on) return false; HIP_CHECK(hipDeviceSynchronize()); g_pfm.clear();
    g_pf_lost=std::max(g_pf_lost,g_stg_cap); HIP_CHECK(hipFree(g_stg)); g_stg=nullptr; g_stg_cap=0; g_pf_on=false; g_use_pf=false;
    fprintf(stderr,"Опережающая подкачка отключена: не хватило видеопамяти, буфер освобождён (далее слои загружаются без опережающей подкачки).\n"); return true; }
static bool g_rbf_off=false;   // еп2
// ер1
static void rbf_boundary(){ g_rbf_inuse.clear(); if(g_rbf_zombie.empty()) return; HIP_CHECK(hipDeviceSynchronize());
    for(void* d : g_rbf_zombie){ g_rbf_ptr.erase(d); g_bf16w.erase(d); g_fp16w.erase(d); hipFree(d); } g_rbf_zombie.clear(); (void)hipGetLastError(); }
static void rbf_drop(){ g_rbf_off=true; if(g_rbf.empty()) return; HIP_CHECK(hipDeviceSynchronize()); size_t n=0, bytes=0;
    for(auto& [nm,d] : g_rbf){ ++n; (void)nm; if(g_rbf_inuse.count(d)){ g_rbf_zombie.push_back(d); continue; }   // ес1
        g_rbf_ptr.erase(d); g_bf16w.erase(d); g_fp16w.erase(d); hipFree(d); }
    g_rbf.clear(); g_rbf_bytes=0; (void)hipGetLastError(); (void)bytes;
    fprintf(stderr,"Постоянные веса в видеопамяти освобождены: матриц %zu (далее они передаются по шине, как без запаса).\n",n); }

static void rbf_fill(){ // ет1
    if((!g_bf3 && !g_fp16_storage) || !g_rbf.empty() || !g_pf_on || g_rbf_off) return; size_t fr=0,tt=0; if(hipMemGetInfo(&fr,&tt)!=hipSuccess) return; // ет2
    size_t tr=step_transient(tt-fr); const size_t KEEP=(tr?tr+tr/4:(1536ull<<20))+vram_slack()+(256ull<<20); /* ет3 */ /* ет4 */ if(fr<=KEEP) return; size_t spare=fr-KEEP, took=0;
    for(int l=0; l<N_LAYERS && spare>0; ++l){ std::vector<std::pair<std::string,size_t>> v; pf_list(l,v);
        for(auto& [nm,sz] : v){ if(cpu_owned(nm)) continue;   // ет5
            if(sz+(4ull<<20)>spare) continue; auto it=g_ram.find(nm); if(it==g_ram.end()) continue;
            void* d; if(hipMalloc(&d,sz)!=hipSuccess){ spare=0; break; }
            HIP_CHECK(hipMemcpy(d,it->second,sz,hipMemcpyHostToDevice)); g_h2d+=sz; g_rbf[nm]=d; g_rbf_ptr.insert(d);
            if(!tq_of(nm)){ g_bf16w.insert(d); if(g_fp16_storage){ bf16_to_fp16_kernel<<<(unsigned)((sz/2+255)/256),256>>>((const hip_bfloat16*)d,(tier_fp16*)d,sz/2); HIP_CHECK(hipStreamSynchronize(0)); g_fp16w.insert(d); } } // ет6
            spare-=sz; took+=sz; g_rbf_bytes+=sz; } }
    if(took) fprintf(stderr,"Постоянно в видеопамяти: замороженных матриц %zu, %.0f МиБ (на эту величину сокращается передача по шине на каждом шаге).\n",g_rbf.size(),took/1048576.0); }
static void pf_issue(int l){ // еу1
    if(!g_pf_on||l<0||l>=N_LAYERS) return; g_pfm.clear(); HIP_CHECK(hipEventRecord(g_conv_ev,0)); HIP_CHECK(hipStreamWaitEvent(g_pf,g_conv_ev,0));
    std::vector<std::pair<std::string,size_t>> v; pf_list(l,v); size_t off=0;
    for(auto& [nm,sz] : v){ if(off+sz>g_stg_cap) break; HIP_CHECK(hipMemcpyAsync(g_stg+off,g_ram.at(nm),sz,hipMemcpyHostToDevice,g_pf)); g_h2d+=sz; g_pfm[nm]=g_stg+off; off+=(sz+255)/256*256; }
    HIP_CHECK(hipEventRecord(g_pf_ev,g_pf)); }
// еф1
static const char* SLOT_KEY(const std::string& s){ static const std::map<std::string,std::string> m={{"iln","input_layernorm"},{"paln","post_attn_ln"},{"pfln","pre_ffn_ln"},
    {"pofln","post_ffn_ln"},{"qn","q_norm"},{"kn","k_norm"},{"q","q_proj"},{"k","k_proj"},{"v","v_proj"},{"o","o_proj"},{"gate","gate_proj"},{"up","up_proj"},{"down","down_proj"}};
    return m.at(s).c_str(); }
static float* load_slot(const std::string& slot, int l, void* stage, size_t cap){
    if(!arch_has(slot)) return nullptr; return load_tensor_or_checkpoint(LPX(l)+arch_tn(slot),"layer_"+std::to_string(l)+"."+SLOT_KEY(slot),stage,cap); }
static void load_weights_norms(Weights& w, void* stage, size_t cap, int layer_idx){
    w.iln=load_slot("iln",layer_idx,stage,cap);
    w.paln=A.post_attn ? load_slot("paln",layer_idx,stage,cap) : nullptr;
    w.pfln=load_slot("pfln",layer_idx,stage,cap);
    w.pofln=A.post_ffn ? load_slot("pofln",layer_idx,stage,cap) : nullptr;
    w.qn=A.qk_norm ? load_slot("qn",layer_idx,stage,cap) : nullptr;
    w.kn=A.qk_norm ? load_slot("kn",layer_idx,stage,cap) : nullptr;
    w.ls=A.layer_scalar ? load_tensor_f32(LPX(layer_idx)+arch_tn("ls"),stage,cap,true) : nullptr;
    if(A.qkv_bias){ w.qb=load_tensor_f32(LPX(layer_idx)+arch_tn("qb"),stage,cap,true); w.kb=load_tensor_f32(LPX(layer_idx)+arch_tn("kb"),stage,cap,true);
        w.vb=HAS_V_PROJ ? load_tensor_f32(LPX(layer_idx)+arch_tn("vb"),stage,cap,true) : nullptr; }
}
static void load_weights_mats(Weights& w, void* stage, size_t cap, int layer_idx){
    w.q=load_slot("q",layer_idx,stage,cap); w.k=load_slot("k",layer_idx,stage,cap);
    w.v=HAS_V_PROJ ? load_slot("v",layer_idx,stage,cap) : nullptr;
    w.o=load_slot("o",layer_idx,stage,cap); w.gate=load_slot("gate",layer_idx,stage,cap); w.up=load_slot("up",layer_idx,stage,cap); w.down=load_slot("down",layer_idx,stage,cap);
}
// еш1
static void bias_add(float* y, const float* b, int M, int n){ if(b){ size_t t=(size_t)M*n; add_bias_k<<<(unsigned)((t+255)/256),256>>>(y,b,M,n); } }
static void qscale(float* q, size_t n){ if(A.attn_scale!=1.0f) scale_k<<<(unsigned)((n+255)/256),256>>>(q,A.attn_scale,n); }
static void post_norm(const float* in, const float* w, float* out, float* r, float* n, int M, bool on){   // ею1
    if(on) rmsnorm_fwd<<<M, NB_ROW >>>(in,w,out,r,n,HIDDEN,weight_type(w)); else HIP_CHECK(hipMemcpyAsync(out,in,(size_t)M*HIDDEN*4,hipMemcpyDeviceToDevice,0)); }
static void act_fwd(const float* g, const float* u, float* a, size_t n){
    if(A.silu) silu_mul_fwd<<<(unsigned)((n+255)/256),256>>>(g,u,a,n); else gelu_tanh_mul_fwd<<<(unsigned)((n+255)/256),256>>>(g,u,a,n); }
static void act_bwd(const float* d, const float* g, const float* u, float* dg, float* du, size_t n){
    if(A.silu) silu_mul_bwd<<<(unsigned)((n+255)/256),256>>>(d,g,u,dg,du,n); else gelu_tanh_mul_bwd<<<(unsigned)((n+255)/256),256>>>(d,g,u,dg,du,n); }
static Weights load_weights(void* stage, size_t cap, int layer_idx){
    Weights w; load_weights_norms(w,stage,cap,layer_idx); load_weights_mats(w,stage,cap,layer_idx); return w;
}

// жб1
static std::string g_adam_path;

static void adam_load_mv(const std::string& key, float* m, float* v){
    auto& slot0=g_weight_layout.at(key);
    TensorSlot slot=slot0; if(slot0.ntr && slot0.ntr<slot0.n) slot.n=slot0.ntr;
    if(g_compact){
        std::vector<float> h(slot.n*2,0.f); int fd=open(g_checkpoint_path.c_str(),O_RDONLY);
        if(fd<0 || !g_checkpoint_exists_with_data){ if(fd>=0) close(fd); HIP_CHECK(hipMemset(m,0,slot.n*sizeof(float))); if(v) HIP_CHECK(hipMemset(v,0,slot.n*sizeof(float))); return; }
        pread_all(fd,h.data(),slot.n*2*sizeof(float),g_c1off.at(key)+slot.n*sizeof(float)); close(fd);
        HIP_CHECK(hipMemcpy(m,h.data(),slot.n*sizeof(float),hipMemcpyHostToDevice));
        if(v) HIP_CHECK(hipMemcpy(v,h.data()+slot.n,slot.n*sizeof(float),hipMemcpyHostToDevice));
        return;
    }
    int fd=open(g_adam_path.c_str(),O_RDONLY);
    if(fd<0){ HIP_CHECK(hipMemset(m,0,slot.n*sizeof(float))); if(v) HIP_CHECK(hipMemset(v,0,slot.n*sizeof(float))); return; }
    size_t offm=slot.offset_floats*sizeof(float), offv=(g_weight_layout_total_floats+slot.offset_floats)*sizeof(float);
    std::vector<float> hm(slot.n,0.f), hv(slot.n,0.f);
    pread_all(fd,hm.data(),slot.n*sizeof(float),offm); pread_all(fd,hv.data(),slot.n*sizeof(float),offv); close(fd);
    HIP_CHECK(hipMemcpy(m,hm.data(),slot.n*sizeof(float),hipMemcpyHostToDevice));
    if(v) HIP_CHECK(hipMemcpy(v,hv.data(),slot.n*sizeof(float),hipMemcpyHostToDevice));
}
static void adam_store_mv(const std::string& key, float* m, float* v){
    auto& slot0=g_weight_layout.at(key);
    TensorSlot slot=slot0; if(slot0.ntr && slot0.ntr<slot0.n) slot.n=slot0.ntr;
    std::vector<float> hm(slot.n), hv(slot.n,0.f);
    HIP_CHECK(hipMemcpy(hm.data(),m,slot.n*sizeof(float),hipMemcpyDeviceToHost));
    if(v) HIP_CHECK(hipMemcpy(hv.data(),v,slot.n*sizeof(float),hipMemcpyDeviceToHost));
    int fd=open(g_adam_path.c_str(),O_RDWR);
    if(fd<0){ perror(g_adam_path.c_str()); exit(1); }
    size_t offm=slot.offset_floats*sizeof(float), offv=(g_weight_layout_total_floats+slot.offset_floats)*sizeof(float);
    pwrite_all(fd,hm.data(),slot.n*sizeof(float),offm); pwrite_all(fd,hv.data(),slot.n*sizeof(float),offv);
    close(fd);
}

static bool g_is_ev=false; // жг1
static void resident_register(const std::string& key, float* w){
    auto& sl=g_weight_layout.at(key); size_t n = sl.ntr ? sl.ntr : sl.n; Resident r{w,nullptr,nullptr};
    if(g_eval && !g_is_ev){ g_res[key]=r; g_res_ptrs.insert(w); return; }   /* жд1 */
    HIP_CHECK(res_alloc(&r.m,n*sizeof(float)));
    if(!(g_nsgd && sl.rlen>1)) HIP_CHECK(res_alloc(&r.v,n*sizeof(float)));   // жд2
    adam_load_mv(key,r.m,r.v);
    g_res[key]=r; g_res_ptrs.insert(w);
}
static void checkpoint_store_weight(const std::string& ckpt_key, float* w);
static void resident_flush(){
    if(g_compact){ int fd=open(g_checkpoint_path.c_str(),O_RDWR); if(fd<0){ perror(g_checkpoint_path.c_str()); exit(1); }
        // же1
        static std::vector<float> buf[2]; static std::thread wr; int cur=0;
        for(auto& [k,r] : g_res){ auto& sk=g_weight_layout.at(k); size_t n = sk.ntr ? sk.ntr : sk.n; auto& h=buf[cur]; h.resize(n*3);
            HIP_CHECK(hipMemcpy(h.data(),r.w,n*4,hipMemcpyDeviceToHost));
            HIP_CHECK(hipMemcpy(h.data()+n,r.m,n*sizeof(float),hipMemcpyDeviceToHost)); // же2
            if(r.v) HIP_CHECK(hipMemcpy(h.data()+2*n,r.v,n*4,hipMemcpyDeviceToHost)); else std::fill(h.begin()+2*n,h.begin()+3*n,0.f);
            if(wr.joinable()) wr.join();
            size_t off=g_c1off.at(k), bytes=n*12; const void* ptr=h.data();
            wr=std::thread([fd,ptr,bytes,off]{ pwrite_all(fd,ptr,bytes,off); }); cur^=1; }
        if(wr.joinable()) wr.join();
        // же3
        fsync(fd); posix_fadvise(fd,0,0,POSIX_FADV_DONTNEED);
        close(fd); return; }
    for(auto& [k,r] : g_res){ adam_store_mv(k,r.m,r.v); checkpoint_store_weight(k,r.w); }
}

// же4
static FILE* g_log = nullptr; // же5
// жж1
static void lgv(const char* tag,int l,const float* d,size_t n,int rows){
    if(!g_log||!d) return; HIP_CHECK(hipStreamSynchronize(0)); std::vector<float> h(n); HIP_CHECK(hipMemcpy(h.data(),d,n*sizeof(float),hipMemcpyDeviceToHost));
    size_t w=n/rows, bad=0; std::vector<double> rn(rows); for(int t=0;t<rows;++t){ double a=0; for(size_t i=0;i<w;++i){ float v=h[t*w+i]; if(!std::isfinite(v)) ++bad; a+=(double)v*v; } rn[t]=sqrt(a); }
    int mt=0; for(int t=1;t<rows;++t) if(!(rn[t]<=rn[mt])) mt=t; std::vector<double> s=rn; std::sort(s.begin(),s.end());
    fprintf(g_log,"L%02d %-12s max %.3e @%-3d med %.3e min %.3e bad %zu\n",l,tag,rn[mt],mt,s[rows/2],s[0],bad);
}
// жз1
static void lgr(const char* tag,int l,const float* d,size_t n){
    if(!g_log) return; HIP_CHECK(hipStreamSynchronize(0)); std::vector<float> h(n); HIP_CHECK(hipMemcpy(h.data(),d,n*sizeof(float),hipMemcpyDeviceToHost));
    size_t im=0; for(size_t i=1;i<n;++i) if(!(h[i]<=h[im])) im=i; float mn=*std::min_element(h.begin(),h.end());
    fprintf(g_log,"L%02d %-12s r max %.3e @%zu  r min %.3e\n",l,tag,h[im],im,mn);
}
static int g_w1 = 1;        // жз2
static double g_damp = 0;   // жз3
// жз4
static int g_lrfind = 0;
// жз5
static int g_stop_after = 0;
static double g_gn_ema = 0, g_prev_gn = 0, g_lr_scale = 1.0;
static bool g_retier = false;   // жз6
static bool g_gdiag = false, g_prog = false; // жз7
static bool VERBOSE = false; // жз8
static double tensor_norm(const float* d, size_t n){   // жи1
    std::vector<float> h(n); HIP_CHECK(hipMemcpy(h.data(),d,n*sizeof(float),hipMemcpyDeviceToHost));
    double s=0; for(float x:h) s+=(double)x*x; return sqrt(s);
}
static std::string g_tier_file_path = "tier_assignment.txt"; // жи2

static int g_gen=0; static float g_temp=0; // жи3
// жк1
__global__ void sumsq_kernel(const float* g, size_t n, float* acc){
    __shared__ float sh[256]; float a=0;
    for(size_t i=blockIdx.x*(size_t)blockDim.x+threadIdx.x;i<n;i+=(size_t)gridDim.x*blockDim.x) a+=g[i]*g[i];
    sh[threadIdx.x]=a; __syncthreads();
    for(int o=128;o>0;o>>=1){ if(threadIdx.x<o) sh[threadIdx.x]+=sh[threadIdx.x+o]; __syncthreads(); }
    if(threadIdx.x==0) atomicAdd(acc,sh[0]);
}
static float* g_gn2=nullptr;
// жк2
static bool g_gprobe=false; static std::map<std::string,float*> g_gp;   // жк3
static std::map<std::string,FILE*> g_dump_act;   // жк4
static void dump_act(const std::string& key, const float* d, size_t rows, size_t dim){   // жл1
    if(g_dump_act.empty() || !g_eval || g_is_ev) return; auto it=g_dump_act.find(key); if(it==g_dump_act.end()) return;
    HIP_CHECK(hipStreamSynchronize(0)); std::vector<float> h(rows*dim); HIP_CHECK(hipMemcpy(h.data(),d,rows*dim*sizeof(float),hipMemcpyDeviceToHost));
    std::vector<uint16_t> b(rows*dim); for(size_t i=0;i<b.size();++i){ uint32_t u; memcpy(&u,&h[i],4); b[i]=(uint16_t)((u+0x7fffu+((u>>16)&1u))>>16); }
    if(ftell(it->second)==0){ int dd=(int)dim; fwrite(&dd,4,1,it->second); }
    fwrite(b.data(),2,b.size(),it->second); fflush(it->second); }
static double g_eval_sum=0, g_eval_tok=0, g_eval_hit=0;
// жм1
struct LoraP { int in=0, out=0; size_t off=0; float *A=nullptr,*B=nullptr,*mA=nullptr,*vA=nullptr,*mB=nullptr,*vB=nullptr; };
static int g_lora_r=0; static float g_lora_alpha=0, g_lora_drop=0, g_lora_s=0; static std::string g_lora_tg;
static std::map<std::string,LoraP> g_lora; static const size_t LORA_HDR=64u<<10;
static float *g_lu=nullptr,*g_lv=nullptr,*g_lxd=nullptr,*g_lga=nullptr,*g_lgb=nullptr; static size_t g_lu_n=0,g_lv_n=0,g_lxd_n=0,g_lga_n=0,g_lgb_n=0;
static const char* LORA_MN[7]={"q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"};
static const char LORA_CH[7]={'q','k','v','o','g','u','d'};
static void lbuf(float** p, size_t* cap, size_t n){ if(n<=*cap) return; if(*p) HIP_CHECK(hipFree(*p)); HIP_CHECK(hipMalloc(p,n*sizeof(float))); *cap=n; }
// жр1
__host__ __device__ inline unsigned lhash(unsigned x){ x^=x>>16; x*=0x7feb352dU; x^=x>>15; x*=0x846ca68bU; x^=x>>16; return x; }
__global__ void lora_drop_k(const float* x, float* y, size_t n, float p, unsigned seed){
    size_t i=blockIdx.x*(size_t)blockDim.x+threadIdx.x; if(i>=n) return;
    float u=(float)(lhash((unsigned)i*0x9E3779B9U ^ seed ^ lhash((unsigned)(i>>32)))>>8)*(1.0f/16777216.0f);
    y[i]= u<p ? 0.f : x[i]/(1.f-p); }
__global__ void lora_adam_k(float* w, const float* g, float* m, float* v, size_t n, float lr, float bc1, float bc2){
    size_t i=blockIdx.x*(size_t)blockDim.x+threadIdx.x; if(i>=n) return; float gi=g[i];
    float mi=0.9f*m[i]+0.1f*gi, vi=0.999f*v[i]+0.001f*gi*gi; m[i]=mi; v[i]=vi;
    w[i]-=lr*(mi/bc1)/(sqrtf(vi/bc2)+1e-8f); }
static const float* lora_in(const float* x, size_t n, int l, int k){   // жу1
    if(g_lora_drop<=0 || g_eval) return x;
    lbuf(&g_lxd,&g_lxd_n,n);
    unsigned seed=lhash((unsigned)g_cur_step*2654435761U ^ (unsigned)(l*16+k)*40503U ^ 0x5bd1e995U);
    lora_drop_k<<<(unsigned)((n+255)/256),256>>>(x,g_lxd,n,g_lora_drop,seed); return g_lxd; }
static LoraP* lora_of(int l, int k){ if(!g_lora_r) return nullptr; auto it=g_lora.find("layer_"+std::to_string(l)+"."+LORA_MN[k]); return it==g_lora.end()?nullptr:&it->second; }
// жх1
static void lora_fwd(hipblasHandle_t h, int l, int k, const float* x, float* y, int M){
    LoraP* P=lora_of(l,k); if(!P) return; int r=g_lora_r; const float one=1, zero=0, s=g_lora_s;
    const float* xi=lora_in(x,(size_t)M*P->in,l,k); lbuf(&g_lu,&g_lu_n,(size_t)M*r);
    HIPBLAS_CHECK(hipblasSgemm(h,HIPBLAS_OP_T,HIPBLAS_OP_N,r,M,P->in,&one,P->A,P->in,xi,P->in,&zero,g_lu,r));
    HIPBLAS_CHECK(hipblasSgemm(h,HIPBLAS_OP_T,HIPBLAS_OP_N,P->out,M,r,&s,P->B,r,g_lu,r,&one,y,P->out)); }
// жц1
static void lora_bwd(hipblasHandle_t h, int l, int k, const float* x, const float* dY, float* dX, int M, float lr, int step){
    if(g_eval) return; LoraP* P=lora_of(l,k); if(!P) return; int r=g_lora_r; const float one=1, zero=0, s=g_lora_s;
    const float* xi=lora_in(x,(size_t)M*P->in,l,k); lbuf(&g_lu,&g_lu_n,(size_t)M*r); lbuf(&g_lv,&g_lv_n,(size_t)M*r);
    size_t na=(size_t)r*P->in, nb=(size_t)P->out*r; lbuf(&g_lga,&g_lga_n,na); lbuf(&g_lgb,&g_lgb_n,nb);
    HIPBLAS_CHECK(hipblasSgemm(h,HIPBLAS_OP_T,HIPBLAS_OP_N,r,M,P->in,&one,P->A,P->in,xi,P->in,&zero,g_lu,r));        // жц2
    HIPBLAS_CHECK(hipblasSgemm(h,HIPBLAS_OP_N,HIPBLAS_OP_N,r,M,P->out,&one,P->B,r,dY,P->out,&zero,g_lv,r));         // жц3
    if(dX) HIPBLAS_CHECK(hipblasSgemm(h,HIPBLAS_OP_N,HIPBLAS_OP_N,P->in,M,r,&s,P->A,P->in,g_lv,r,&one,dX,P->in));    // жц4
    HIPBLAS_CHECK(hipblasSgemm(h,HIPBLAS_OP_N,HIPBLAS_OP_T,r,P->out,M,&s,g_lu,r,dY,P->out,&zero,g_lgb,r));          // жц5
    HIPBLAS_CHECK(hipblasSgemm(h,HIPBLAS_OP_N,HIPBLAS_OP_T,P->in,r,M,&s,xi,P->in,g_lv,r,&zero,g_lga,P->in));        // жц6
    if(g_gn2){ sumsq_kernel<<<(unsigned)std::min<size_t>((na+255)/256,1024),256>>>(g_lga,na,g_gn2);
               sumsq_kernel<<<(unsigned)std::min<size_t>((nb+255)/256,1024),256>>>(g_lgb,nb,g_gn2); }
    float bc1=1.0f-powf(0.9f,(float)step), bc2=1.0f-powf(0.999f,(float)step), lrs=lr*(float)g_lr_scale;
    lora_adam_k<<<(unsigned)((na+255)/256),256>>>(P->A,g_lga,P->mA,P->vA,na,lrs,bc1,bc2);
    lora_adam_k<<<(unsigned)((nb+255)/256),256>>>(P->B,g_lgb,P->mB,P->vB,nb,lrs,bc1,bc2); }
static bool lora_parse(const char* s){   // жч1
    char tg[32]={0}; float a=0, d=0; int r=0; if(sscanf(s,"%d,%f,%31[^,],%f",&r,&a,tg,&d)<3 || r<1 || r>1024 || a<=0) return false;
    g_lora_r=r; g_lora_alpha=a; g_lora_drop=std::max(0.f,std::min(0.9f,d)); g_lora_s=a/r; g_lora_tg=tg;
    for(char c : g_lora_tg) if(!strchr("qkvogud",c)) return false; return !g_lora_tg.empty(); }
static size_t lora_bytes(const LoraP& P){ return ((size_t)g_lora_r*(P.in+P.out)*3*4+4095)/4096*4096; }   // жш1
static void lora_layout(){   // жэ1
    size_t off=LORA_HDR;
    for(int l=0;l<N_LAYERS;++l){ bool full=!A.L[l].has_v; int hd=A.L[l].hd, nkv=A.L[l].nkv, qd=NUM_HEADS*hd, kd=nkv*hd;
        int io[7][2]={{HIDDEN,qd},{HIDDEN,kd},{HIDDEN,kd},{qd,HIDDEN},{HIDDEN,INTER},{HIDDEN,INTER},{INTER,HIDDEN}};
        for(int k=0;k<7;++k){ if(!strchr(g_lora_tg.c_str(),LORA_CH[k]) || (k==2&&full)) continue;
            LoraP P; P.in=io[k][0]; P.out=io[k][1]; P.off=off; off+=lora_bytes(P); g_lora["layer_"+std::to_string(l)+"."+LORA_MN[k]]=P; } } }
static std::string lora_header(){ char b[160]; snprintf(b,sizeof b,"LKL1 %d %g %g %s %zu\n",g_lora_r,g_lora_alpha,g_lora_drop,g_lora_tg.c_str(),g_lora.size());
    std::string t=b; for(auto& [k,P] : g_lora) t+=k+" "+std::to_string(P.in)+" "+std::to_string(P.out)+" "+std::to_string(P.off)+"\n";
    if(t.size()>=LORA_HDR){ fprintf(stderr,"Таблица LoRA не помещается в заголовок.\n"); exit(1); } t.resize(LORA_HDR,'\0'); return t; }
static size_t lora_params(){ size_t n=0; for(auto& [k,P] : g_lora) n+=(size_t)g_lora_r*(P.in+P.out); return n; }
static void lora_alloc(LoraP& P){ size_t na=(size_t)g_lora_r*P.in, nb=(size_t)P.out*g_lora_r;
    HIP_CHECK(hipMalloc(&P.A,na*4)); HIP_CHECK(hipMalloc(&P.mA,na*4)); HIP_CHECK(hipMalloc(&P.vA,na*4));
    HIP_CHECK(hipMalloc(&P.B,nb*4)); HIP_CHECK(hipMalloc(&P.mB,nb*4)); HIP_CHECK(hipMalloc(&P.vB,nb*4)); }
static void lora_init(){   // зб1
    for(auto& [k,P] : g_lora){ lora_alloc(P); size_t na=(size_t)g_lora_r*P.in, nb=(size_t)P.out*g_lora_r;
        std::vector<float> a(na); std::mt19937 rg(lhash((unsigned)std::hash<std::string>{}(k))); float bd=1.0f/sqrtf((float)P.in);
        std::uniform_real_distribution<float> U(-bd,bd); for(auto& v:a) v=U(rg);
        HIP_CHECK(hipMemcpy(P.A,a.data(),na*4,hipMemcpyHostToDevice));
        HIP_CHECK(hipMemset(P.mA,0,na*4)); HIP_CHECK(hipMemset(P.vA,0,na*4)); HIP_CHECK(hipMemset(P.B,0,nb*4)); HIP_CHECK(hipMemset(P.mB,0,nb*4)); HIP_CHECK(hipMemset(P.vB,0,nb*4)); } }
static void lora_save(const std::string& path){
    int fd=open(path.c_str(),O_RDWR|O_CREAT,0644); if(fd<0){ perror(path.c_str()); exit(1); }
    std::string h=lora_header(); pwrite_all(fd,h.data(),h.size(),0); std::vector<float> buf;
    for(auto& [k,P] : g_lora){ size_t na=(size_t)g_lora_r*P.in, nb=(size_t)P.out*g_lora_r, o=P.off;
        std::pair<float*,size_t> parts[6]={{P.A,na},{P.B,nb},{P.mA,na},{P.vA,na},{P.mB,nb},{P.vB,nb}};
        for(auto& q : parts){ buf.resize(q.second); HIP_CHECK(hipMemcpy(buf.data(),q.first,q.second*4,hipMemcpyDeviceToHost)); pwrite_all(fd,buf.data(),q.second*4,o); o+=q.second*4; } }
    fsync(fd); close(fd); }
static bool lora_load(const std::string& path){   // зг1
    int fd=open(path.c_str(),O_RDONLY); if(fd<0) return false; std::string h(LORA_HDR,'\0');
    if(pread(fd,h.data(),LORA_HDR,0)!=(ssize_t)LORA_HDR || h.compare(0,5,"LKL1 ")!=0){ close(fd); return false; }
    if(h!=lora_header()){ close(fd); fprintf(stderr,"%s: адаптеры другой раскладки (ранг, матрицы). Требуется удалить файл либо указать другой путь.\n",path.c_str()); exit(1); }
    std::vector<float> buf;
    for(auto& [k,P] : g_lora){ lora_alloc(P); size_t na=(size_t)g_lora_r*P.in, nb=(size_t)P.out*g_lora_r, o=P.off;
        std::pair<float*,size_t> parts[6]={{P.A,na},{P.B,nb},{P.mA,na},{P.vA,na},{P.mB,nb},{P.vB,nb}};
        for(auto& q : parts){ buf.resize(q.second); pread_all(fd,buf.data(),q.second*4,o); HIP_CHECK(hipMemcpy(q.first,buf.data(),q.second*4,hipMemcpyHostToDevice)); o+=q.second*4; } }
    close(fd); return true; }

static void adam_apply(hipblasHandle_t blas, const std::string& key, float* w, float* grad, size_t n, float lr, int step){
    if(g_eval) return;
    int tier=g_weight_layout.at(key).tier;
    if((tier==2&&!g_train_tier2)||(tier==3&&!g_train_tier3)){
        // зд1
        return;
    }
    { auto& sl=g_weight_layout.at(key);   // зд2
      if(sl.ntr && sl.ntr<sl.n){ auto r=g_res.find(key); if(r!=g_res.end()){ w=r->second.w; n=sl.ntr; } } }
    if(g_gn2) sumsq_kernel<<<std::min<size_t>((n+255)/256,1024),256>>>(grad,n,g_gn2);
    if(g_gprobe){ float*& p=g_gp[key]; if(!p){ HIP_CHECK(hipMalloc(&p,2*sizeof(float))); HIP_CHECK(hipMemset(p,0,2*sizeof(float))); }
        sumsq_kernel<<<std::min<size_t>((n+255)/256,1024),256>>>(grad,n,p); sumsq_kernel<<<std::min<size_t>((n+255)/256,1024),256>>>(w,n,p+1); }
    auto it=g_res.find(key); bool res=it!=g_res.end();
    const bool ns = g_nsgd && g_weight_layout.at(key).rlen>1;      // зд3
    float* m; float* v=nullptr;
    if(res && !it->second.m){ auto& r=it->second; HIP_CHECK(cmalloc(&r.m, n*sizeof(float))); if(!ns) HIP_CHECK(cmalloc(&r.v, n*sizeof(float))); adam_load_mv(key, r.m, r.v); }
    if(res){ m=it->second.m; v=it->second.v; }
    else { HIP_CHECK(cmalloc(&m, n*sizeof(float))); if(!ns) HIP_CHECK(cmalloc(&v, n*sizeof(float))); adam_load_mv(key, m, v); }
    if(ns){
        if(const Whiten* wh=whiten_for(key)){ if(n % (size_t)wh->in){ fprintf(stderr,"Защита общих знаний, %s: размер входа %d не делит %zu.\n",key.c_str(),wh->in,n); exit(1); } whiten_apply(blas,*wh,grad,n); }
        static float* acc=nullptr; if(!acc) HIP_CHECK(hipMalloc(&acc,sizeof(float)));
        HIP_CHECK(hipMemset(acc,0,sizeof(float)));
        nsgd_mom_k<<<(unsigned)std::min<size_t>((n+255)/256,4096),256>>>(grad,m,n,acc);
        float ss=0; HIP_CHECK(hipMemcpy(&ss,acc,sizeof(float),hipMemcpyDeviceToHost));
        float rms=sqrtf(ss/(float)n);
        if(rms>0 && std::isfinite(rms)) nsgd_step_k<<<(unsigned)((n+255)/256),256>>>(w,m,n,lr*(float)g_lr_scale/rms);
        HIP_CHECK(hipStreamSynchronize(0));
        if(VERBOSE) fprintf(g_log?g_log:stdout,"  [%s] nsgd n=%zu rms(m)=%.4e LR=%.3e%s\n",key.c_str(),n,rms,lr*(float)g_lr_scale,whiten_for(key)?", защита общих знаний":"");
        if(res) return;
        adam_store_mv(key,m,v); checkpoint_store_weight(key,w); cfree(m); return; }

    double w_norm_before=0, g_norm=0;
    if(VERBOSE){ w_norm_before=tensor_norm(w,n); g_norm=tensor_norm(grad,n); }

    int th=256, bl=(n+th-1)/th;
    float bc1 = 1.0f - powf(0.9f, step), bc2 = 1.0f - powf(0.999f, step);
    adam_step_kernel<<<bl,th>>>(grad, w, m, v, n, lr*(float)g_lr_scale, 0.9f, 0.999f, bc1, bc2, 1e-8f);
    HIP_CHECK(hipStreamSynchronize(0));

    if(VERBOSE){
        double w_norm_after = tensor_norm(w,n);
        double delta_ratio = fabs(w_norm_after-w_norm_before) / (w_norm_before+1e-12);
        fprintf(g_log?g_log:stdout,"  [%s] n=%zu |g|=%.4e |w|=%.4e->%.4e |dw|/|w|=%.4e\n", key.c_str(), n, g_norm, w_norm_before, w_norm_after, delta_ratio);
    }

    if(res) return;
    adam_store_mv(key, m, v);
    checkpoint_store_weight(key, w);
    cfree(m); cfree(v);
}

// зе1
static std::string vram_hogs(){ std::string out; DIR* d=opendir("/proc"); if(!d) return out;
    struct E{ size_t mb; std::string nm; }; std::vector<E> v; std::set<std::string> seen;
    while(dirent* e=readdir(d)){ if(e->d_name[0]<'0'||e->d_name[0]>'9') continue; std::string pid=e->d_name;
        std::string fdd="/proc/"+pid+"/fdinfo"; DIR* f=opendir(fdd.c_str()); if(!f) continue; size_t tot=0; std::string cid;
        while(dirent* q=readdir(f)){ if(q->d_name[0]=='.') continue; FILE* h=fopen((fdd+"/"+q->d_name).c_str(),"r"); if(!h) continue;
            char ln[256]; size_t mb=0; std::string id;
            while(fgets(ln,sizeof ln,h)){ if(!strncmp(ln,"drm-memory-vram:",16)) mb=strtoull(ln+16,nullptr,10)/1024;
                                          else if(!strncmp(ln,"drm-client-id:",14)) id=std::string(ln+14); }
            fclose(h); if(mb && !id.empty() && !seen.count(id)){ seen.insert(id); tot+=mb; } }
        closedir(f);
        if(tot>64){ std::string nm="?"; FILE* c=fopen(("/proc/"+pid+"/comm").c_str(),"r"); if(c){ char b[64]; if(fgets(b,64,c)){ b[strcspn(b,"\n")]=0; nm=b; } fclose(c); }
            if(nm!="one_full_traini") v.push_back({tot,nm+" ("+pid+")"}); } }
    closedir(d); std::sort(v.begin(),v.end(),[](const E& a,const E& b){ return a.mb>b.mb; });
    for(size_t i=0;i<v.size() && i<4;++i) out += (out.empty()?"":", ") + v[i].nm + " " + std::to_string(v[i].mb) + " MiB";
    return out; }
static void vram_wait(int step){ size_t fr=0,tt=0;
    if(hipMemGetInfo(&fr,&tt)!=hipSuccess) return;
    const size_t SLACK=128ull<<20; // зж1
    if(!g_free_ok || fr+SLACK>=g_free_ok) return;
    std::string h=vram_hogs();
    fprintf(stderr,"Пауза перед шагом %d: свободно %.0f МиБ, на прошлом шаге было %.0f МиБ%s%s.\n",step,fr/1048576.0,g_free_ok/1048576.0,
            h.empty()?"":"; занято: ",h.c_str());
    for(int w=1; w<=600; ++w){ sleep(1); if(hipMemGetInfo(&fr,&tt)!=hipSuccess) return;
        if(fr+SLACK>=g_free_ok){ fprintf(stderr,"Видеопамять освободилась. Продолжение шага %d.\n",step); return; }
        if(w%60==0) fprintf(stderr,"Ожидание видеопамяти: %.0f из %.0f МиБ (%d мин).\n",fr/1048576.0,g_free_ok/1048576.0,w/60); }
    fprintf(stderr,"Видеопамять не освободилась за 10 минут. Выполняется попытка продолжить расчёт.\n"); g_free_ok=fr; }
// зз1
static bool mem_guard(int step){
    size_t fr=0,tt=0;
    if(hipMemGetInfo(&fr,&tt)==hipSuccess && fr<cap_slack(tt) && !g_rbf.empty()){
        fprintf(stderr,"Потолок памяти %d%%: свободно %.0f МиБ видеопамяти из %.0f (шаг %d).\n",g_cap_pct,fr/1048576.0,tt/1048576.0,step); rbf_drop(); }
    HostMem m=host_mem(); if(!m.total) return false;
    const size_t need=cap_slack(m.total)+(256ull<<20);   // зз2
    if(m.avail>=need) return false;
    const char* wv=getenv("TIER_RAM_WAIT"); int wmax = wv ? atoi(wv) : 60;
    fprintf(stderr,"Потолок памяти %d%%: свободно %.0f МиБ ОЗУ, требуется не меньше %.0f. Ожидание до %d с (шаг %d).\n",g_cap_pct,m.avail/1048576.0,need/1048576.0,wmax,step);
    for(int w=1; w<=wmax; ++w){ sleep(1); m=host_mem(); if(m.avail>=need){ fprintf(stderr,"ОЗУ освободилась. Продолжение шага %d.\n",step); return false; } }
    return true; }
static float* galloc(const std::string& key, size_t n){
    auto& s=g_weight_layout.at(key);
    if((s.tier==2&&!g_train_tier2)||(s.tier==3&&!g_train_tier3)) return nullptr;
    if(s.ntr && s.ntr<n) n=s.ntr;          // зи1
    float* p; HIP_CHECK(cmalloc(&p,n*sizeof(float))); g_gsz[p]=n; g_gkey[p]=key; return p;
}
// зк1
static bool weight_trained(const std::string& key){ auto it=g_weight_layout.find(key); if(it==g_weight_layout.end()) return true;   // зк2
    const TensorSlot& s=it->second; return !((s.tier==2&&!g_train_tier2)||(s.tier==3&&!g_train_tier3)); }
// зл1
static int lowest_trained_layer(){
    for(int l=0;l<N_LAYERS;++l){ const std::string p="layer_"+std::to_string(l)+".";
        for(auto it=g_weight_layout.lower_bound(p); it!=g_weight_layout.end() && it->first.compare(0,p.size(),p)==0; ++it) if(weight_trained(it->first)) return l;
        if(g_lora_r) for(auto& [k,P] : g_lora) if(k.compare(0,p.size(),p)==0) return l; }
    return -1; }

// зм1
struct Conf { size_t vram_mib=0, headroom_mib=0, tier1_budget=0, ram_gib=0; int bf3=0, own=1, prefetch=1, cu=0, warp=TIER_WARP, pcie_gen=0, pcie_w=0, fp16=-1, mem_cap=98; std::string gpu="?", cal; std::map<std::string,std::pair<size_t,size_t>> per; };   // зм2
static Conf g_conf; static std::string g_conf_path; static bool g_conf_fresh=false;   // зм3
static std::string conf_path(){ if(const char* e=getenv("TIER_CONF")) return e;
    const char* h=getenv("HOME"); std::string d=std::string(h?h:".")+"/.config"; mkdir(d.c_str(),0755); return d+"/tier.conf"; }
// зн1
static int lm_chunk(int seq);
static size_t model_vram_need(int pack){
    std::map<int,size_t> per; for(auto& [k,r] : g_tensors){ size_t p=k.find(".layers."); if(p==std::string::npos) continue; per[atoi(k.c_str()+p+8)]+=(r.size+255)/256*256; }
    size_t stage=0; for(auto& [l,b] : per) stage=std::max(stage,b);
    size_t P = pack>0 ? pack : 512, H=HIDDEN, I=INTER, C=lm_chunk((int)P);
    size_t act = P*(8*H+3*I)*4 + 2*(size_t)NUM_HEADS*P*P*4 + (size_t)g_model_layers*P*H*4;
    size_t head = C*H*6 + 2*P*C*4, ws = 128ull<<20;
    return stage+act+head+ws; }
// зп1
static int lm_chunk(int seq){ static size_t tt=0; if(!tt){ size_t fr=0; if(hipMemGetInfo(&fr,&tt)!=hipSuccess) tt=0; } if(!tt) return 8192;
    size_t per = (size_t)HIDDEN*6 + (size_t)std::max(seq,1)*8; size_t c = per ? tt/36/per : 8192; c = c/128*128;
    size_t vmax = ((size_t)VOCAB+127)/128*128; return (int)std::max<size_t>(1024,std::min(c,vmax)); }
static long sysfs_int(const char* p){ FILE* f=fopen(p,"r"); if(!f) return 0; long v=0; if(fscanf(f,"%ld",&v)!=1) v=0; fclose(f); return v; }
static void conf_probe(Conf& c, bool quiet=false){
    hipDeviceProp_t pr; bool ok = hipGetDeviceProperties(&pr,g_dev)==hipSuccess;
    if(ok){
#if defined(TIER_CUDA)
        c.gpu=pr.name;
#else
        c.gpu=pr.gcnArchName[0]?pr.gcnArchName:pr.name;
#endif
        c.cu=pr.multiProcessorCount; c.warp=pr.warpSize; }
    size_t fr=0,tt=0; if(hipMemGetInfo(&fr,&tt)==hipSuccess) c.vram_mib=tt>>20;
    if(!c.vram_mib && ok) c.vram_mib=pr.totalGlobalMem>>20;   // зс1
    if(!c.vram_mib){ fprintf(stderr,"Объём видеопамяти не определён ни через hipMemGetInfo, ни через свойства карты. Обучаемая часть не назначается.\n"); c.tier1_budget=0; return; }
    // зс2
    if(!g_others0_set && fr && tt>fr){ g_others0=std::max(tt-fr,sys_vram_used()); g_others0_set=true; }
    size_t others = (fr && tt>fr) ? (tt-fr)>>20 : 0, step = model_vram_need(g_probe_pack)>>20, slack = std::max<size_t>(vram_slack(),256ull<<20)>>20;
    c.headroom_mib = others+step+slack;
    if(!quiet) fprintf(stderr,"Запас видеопамяти: %zu МиБ (занято другими %zu, рабочая память шага %zu, потолок памяти %zu).\n",c.headroom_mib,others,step,slack);
    c.tier1_budget = (c.vram_mib>c.headroom_mib ? c.vram_mib-c.headroom_mib : 0) * 1048576ULL / 12;  // зс3
    struct sysinfo si; if(sysinfo(&si)==0) c.ram_gib = (size_t)si.totalram*si.mem_unit/1073741824ULL;
#if defined(TIER_CUDA)
    c.bf3 = ok;   // зс4
#else
    // зс5
    c.bf3 = TIER_LT; // зс6
#endif
    c.prefetch=1; c.own=1;
    std::string dv = g_pci[0] ? std::string("/sys/bus/pci/devices/")+g_pci : "/sys/class/drm/card0/device";   // зс7
    { FILE* f=fopen((dv+"/current_link_speed").c_str(),"r");
      if(f){ float g=0; if(fscanf(f,"%f",&g)==1) c.pcie_gen = g>=63?6:(g>=31?5:(g>=15?4:(g>=7?3:(g>=4?2:1)))); fclose(f); } }
    c.pcie_w = (int)sysfs_int((dv+"/current_link_width").c_str());
}
static bool conf_load(Conf& c, const std::string& p){
    FILE* f=fopen(p.c_str(),"r"); if(!f) return false; char k[64], v[192];
    while(fscanf(f," %63s %191[^\n]",k,v)==2){ std::string K=k; long n=atol(v);
        if(K[0]=='#') continue;
        else if(K=="vram_mib") c.vram_mib=n; else if(K=="headroom_mib") c.headroom_mib=n;
        else if(K=="tier1_params") c.tier1_budget=n; else if(K=="bf16x3") c.bf3=n; else if(K=="fp16gemm") c.fp16=n;
        else if(K=="own_gemm") c.own=n; else if(K=="prefetch") c.prefetch=n;
        else if(K=="ram_gib") c.ram_gib=n; else if(K=="cu") c.cu=n; else if(K=="warp") c.warp=n;
        else if(K=="pcie_gen") c.pcie_gen=n; else if(K=="pcie_width") c.pcie_w=n; else if(K=="mem_cap_pct") c.mem_cap=n;
        else if(K=="model"){ std::istringstream is(v); std::string id; size_t h=0,b=0; if(is>>id>>h>>b) c.per[id]={h,b}; }
        else if(K=="calibrated"){ c.cal=v; while(!c.cal.empty()&&c.cal.back()==' ') c.cal.pop_back(); }
        else if(K=="gpu"){ c.gpu=v; while(!c.gpu.empty()&&c.gpu.back()==' ') c.gpu.pop_back(); } }
    fclose(f); return c.tier1_budget>0; }
static void conf_save(const Conf& c, const std::string& p){
    FILE* f=fopen(p.c_str(),"w"); if(!f){ fprintf(stderr,"Не удалось записать настройку %s.\n",p.c_str()); return; }
    fprintf(f,"# tier.conf: hardware probe, written on first run. Delete to re-probe. Hand edits are kept.\n");
    fprintf(f,"gpu %s\ncu %d\nwarp %d\nvram_mib %zu\nheadroom_mib %zu\ntier1_params %zu\nbf16x3 %d\nown_gemm %d\nprefetch %d\nram_gib %zu\npcie_gen %d\npcie_width %d\n",
            c.gpu.c_str(), c.cu, c.warp, c.vram_mib, c.headroom_mib, c.tier1_budget, c.bf3, c.own, c.prefetch, c.ram_gib, c.pcie_gen, c.pcie_w);
    if(c.fp16>=0) fprintf(f,"fp16gemm %d\n",c.fp16);
    fprintf(f,"mem_cap_pct %d\n",c.mem_cap);
    for(auto& [id,hb] : c.per) fprintf(f,"model %s %zu %zu\n",id.c_str(),hb.first,hb.second);
    fclose(f); }
// зу1
static std::string g_cal_path, g_cal_model, g_model_id;   // зу2
static void conf_calibrate(){
    if(!g_vram_peak||!g_conf.vram_mib) return;
    size_t t1=0; for(auto& [k,sl] : g_weight_layout) if(sl.tier==1) t1+= sl.ntr ? sl.ntr : sl.n;   // зф1
    size_t t1_mib = t1*12/1048576;                       // зф2
    // зф3
    size_t peak_mib = (g_vram_peak_need ? g_vram_peak_need : g_vram_peak)/1048576;
    if(peak_mib<=t1_mib) return;
    // зф4
    size_t jit = g_free_max>g_free_ok ? (g_free_max-g_free_ok)>>20 : 0, margin = std::max<size_t>({vram_slack()>>20, 128+jit});
    size_t need = peak_mib - t1_mib + margin + (g_pf_lost>>20);   // зф5
    size_t budget = (g_conf.vram_mib>need ? g_conf.vram_mib-need : 0)*1048576ULL/12;
    // зф6
    size_t full = (size_t)(g_probe_pack>0 ? g_probe_pack : 512);
    bool grow = need>g_conf.headroom_mib && ((need-g_conf.headroom_mib)*20 >= g_conf.headroom_mib || g_pf_lost);
    // зф7
    bool shrink = need<g_conf.headroom_mib && (g_conf.headroom_mib-need)*20 >= g_conf.headroom_mib && g_run_len*10 >= full*9 && g_pf_on && t1*10 >= g_conf.tier1_budget*9;
    if(!g_cal_path.empty()){   // зф8
        Conf c; if(!conf_load(c,g_cal_path)) c=g_conf; size_t pf = 0;   // зф9
        // зф10
        size_t h = (peak_mib-t1_mib)*6/5 + margin + pf, b = (c.vram_mib>h ? c.vram_mib-h : 0)*1048576ULL/12;
        c.per[g_cal_model]={h,b}; conf_save(c,g_cal_path); c.headroom_mib=h; c.tier1_budget=b;
        fprintf(stderr,"Замер видеопамяти: вне обучаемой части %zu МиБ, запас %zu МиБ, обучаемая часть %zu параметров (настройка %s).\n",peak_mib-t1_mib,c.headroom_mib,c.tier1_budget,g_cal_path.c_str());
        return; }
    if(!grow && !shrink) return;
    // зф11
    if(shrink && budget > t1*3/2){ budget=t1*3/2; need = g_conf.vram_mib > budget*12/1048576 ? g_conf.vram_mib-budget*12/1048576 : need; }
    fprintf(stderr,"Настройка уточнена по замеру: запас %zu → %zu МиБ, обучаемая часть %zu → %zu параметров.\n",
            g_conf.headroom_mib,need,g_conf.tier1_budget,budget);
    g_conf.headroom_mib=need; g_conf.tier1_budget=budget; if(g_conf.per.count(g_model_id)) g_conf.per[g_model_id]={need,budget}; conf_save(g_conf,g_conf_path);
}
// зх1
static void conf_grow_oom(size_t short_b){
    if(!g_conf.vram_mib || g_conf_path.empty()) return;   // зх2
    size_t add=((short_b+g_pf_lost)>>20)+256; g_conf.headroom_mib+=add;
    g_conf.tier1_budget=(g_conf.vram_mib>g_conf.headroom_mib ? g_conf.vram_mib-g_conf.headroom_mib : 0)*1048576ULL/12;
    if(g_conf.per.count(g_model_id)) g_conf.per[g_model_id]={g_conf.headroom_mib,g_conf.tier1_budget};   // зх3
    conf_save(g_conf,g_conf_path);
    fprintf(stderr,"Запас видеопамяти увеличен на %zu МиБ (до %zu), обучаемая часть %zu параметров. Требуется новая раскладка.\n",add,g_conf.headroom_mib,g_conf.tier1_budget);
    exit(76); }
static void conf_init(){
    g_conf_path=conf_path();
    bool have=conf_load(g_conf,g_conf_path); int cap_kept=g_conf.mem_cap;
    g_cap_pct=g_conf.mem_cap; if(const char* e=getenv("TIER_MEM_CAP_PCT")) g_cap_pct=atoi(e);   // зц1
    if(have){   // зц2
        Conf now; conf_probe(now,true);
        if(now.gpu!="?" && now.gpu!=g_conf.gpu){
            fprintf(stderr,"Настройка записана для %s, а установлена карта %s. Выполняется повторный осмотр.\n", g_conf.gpu.c_str(), now.gpu.c_str());
            have=false; }
        else if(now.vram_mib && now.vram_mib!=g_conf.vram_mib){
            fprintf(stderr,"Объём видеопамяти изменился: %zu МиБ вместо %zu. Выполняется повторный осмотр.\n", now.vram_mib, g_conf.vram_mib);
            have=false; }
        else if(now.ram_gib && g_conf.ram_gib && now.ram_gib!=g_conf.ram_gib){   // зц3
            fprintf(stderr,"Объём ОЗУ изменился: %zu ГиБ вместо %zu. Выполняется повторный осмотр.\n", now.ram_gib, g_conf.ram_gib);
            have=false; }
        else if(now.pcie_gen!=g_conf.pcie_gen || now.pcie_w!=g_conf.pcie_w){   // зц4
            g_conf.pcie_gen=now.pcie_gen; g_conf.pcie_w=now.pcie_w; conf_save(g_conf,g_conf_path); } }
    if(have) fprintf(stderr,"Настройка из %s: %s, %zu МиБ, обучаемая часть %zu параметров.\n",g_conf_path.c_str(),g_conf.gpu.c_str(),g_conf.vram_mib,g_conf.tier1_budget);
    else { g_conf=Conf(); g_conf.mem_cap=cap_kept; conf_probe(g_conf); conf_save(g_conf,g_conf_path); g_conf_fresh=true;   // зц5
        fprintf(stderr,"Осмотр машины выполнен: %s (%s), блоков %d, %zu МиБ видеопамяти, %zu ГиБ ОЗУ, PCIe %d.0 x%d → обучаемая часть %zu параметров, FP16×3: %s. Настройка записана в %s.\n",
                g_conf.gpu.c_str(), TIER_GPU, g_conf.cu, g_conf.vram_mib, g_conf.ram_gib, g_conf.pcie_gen, g_conf.pcie_w, g_conf.tier1_budget, g_conf.bf3?"да":"нет", g_conf_path.c_str()); }
    if(auto pm=g_conf.per.find(g_model_id); pm!=g_conf.per.end() && !g_model_id.empty()){   // зц6
        g_conf.headroom_mib=pm->second.first; g_conf.tier1_budget=pm->second.second;
        fprintf(stderr,"Запас видеопамяти для модели %s по замеру: %zu МиБ, обучаемая часть %zu параметров.\n",g_model_id.c_str(),g_conf.headroom_mib,g_conf.tier1_budget); }
    g_bf3 = g_conf.bf3!=0; g_use_own = g_conf.own!=0; g_snap_bytes=(size_t)g_conf.tier1_budget*12;
    g_cap_pct = g_conf.mem_cap; if(const char* e=getenv("TIER_MEM_CAP_PCT")) g_cap_pct=atoi(e);   // зц7
    if(g_conf.prefetch){ g_use_pf=true; g_use_arena=true; }
}


// зц8
static int g_chat_ws=32; static hipStream_t g_chat_cs, g_chat_ms; static hipblasHandle_t g_chat_blas;
// зч1
static bool chat_conf_bf3(){ std::string p; if(const char* e=getenv("TIER_CONF")) p=e; else { const char* h=getenv("HOME"); p=std::string(h?h:".")+"/.config/tier.conf"; }
    FILE* f=fopen(p.c_str(),"r"); if(!f) return true; char k[64],v[192]; bool r=true;
    while(fscanf(f," %63s %191[^\n]",k,v)==2) if(std::string(k)=="bf16x3") r=atoi(v)!=0;
    fclose(f); return r; }
static bool g_chat_no_bf3 = getenv("TIER_BF3") ? atoi(getenv("TIER_BF3"))==0 : !chat_conf_bf3();
static tier_fp16 chat_fp16_from_bf16(uint16_t b){ uint32_t u=(uint32_t)b<<16; float f; memcpy(&f,&u,4); return tier_fp16(f); }
static void chat_convert_bf16_host_to_fp16(void* p, size_t n){ auto* u=(uint16_t*)p; for(size_t i=0;i<n;++i) u[i]=chat_fp16_from_bf16(u[i]).bits; }
static double chat_now2(){ return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
// зю1
static hipblasLtHandle_t g_chat_lt=nullptr; static void* g_chat_ltws=nullptr; static const size_t LTWS=32<<20;
static hip_bfloat16* g_chat_xs=nullptr; static size_t g_chat_xs_n=0; static float* g_chat_ys=nullptr; static size_t g_chat_ys_n=0;
struct ChatBfKey{ int K,N,M; bool operator<(const ChatBfKey& o) const { return std::tie(K,N,M)<std::tie(o.K,o.N,o.M); } };
static std::map<ChatBfKey,hipblasLtMatmulAlgo_t> g_chat_bfalgo;
__global__ void split3s(const float* x, hip_bfloat16* d, size_t n, size_t stride){ size_t i=blockIdx.x*(size_t)blockDim.x+threadIdx.x; if(i>=n) return;
    float v=x[i]; hip_bfloat16 a=hip_bfloat16(v); float r=v-(float)a; hip_bfloat16 b=hip_bfloat16(r); float r2=r-(float)b;
    d[i]=hip_bfloat16(r2); d[i+stride]=b; d[i+2*stride]=a; }
__global__ void add3ld(const float* ys, float* y, int M, int N, int ldo){ size_t i=blockIdx.x*(size_t)blockDim.x+threadIdx.x; size_t n=(size_t)M*N; if(i>=n) return;
    int col=i/N, row=i%N; const float* p=ys+(size_t)col*N+row; size_t blk=(size_t)M*N;
    y[(size_t)col*ldo+row]=(p[0]+p[blk])+p[2*blk]; }
static bool chat_bf3(const hip_bfloat16* W, const float* x, float* y, int M, int K, int N, int ldo){
    if(!g_chat_lt){ if(hipblasLtCreate(&g_chat_lt)!=HIPBLAS_STATUS_SUCCESS) return false; if(hipMalloc(&g_chat_ltws,LTWS)!=hipSuccess) return false; }
    size_t nx=(size_t)M*K, ny=(size_t)M*N;
    if(nx*3>g_chat_xs_n){ if(g_chat_xs) HIP_CHECK(hipFree(g_chat_xs)); if(hipMalloc(&g_chat_xs,nx*3*2)!=hipSuccess){ g_chat_xs=nullptr; g_chat_xs_n=0; return false; } g_chat_xs_n=nx*3; }
    if(ny*3>g_chat_ys_n){ if(g_chat_ys) HIP_CHECK(hipFree(g_chat_ys)); if(hipMalloc(&g_chat_ys,ny*3*4)!=hipSuccess){ g_chat_ys=nullptr; g_chat_ys_n=0; return false; } g_chat_ys_n=ny*3; }
    hipblasOperation_t ta=HIPBLAS_OP_T, tn=HIPBLAS_OP_N; float al=1,be=0;
    hipblasLtMatmulDesc_t d; hipblasLtMatmulDescCreate(&d,HIPBLAS_COMPUTE_32F,HIP_R_32F);
    hipblasLtMatmulDescSetAttribute(d,HIPBLASLT_MATMUL_DESC_TRANSA,&ta,sizeof ta); hipblasLtMatmulDescSetAttribute(d,HIPBLASLT_MATMUL_DESC_TRANSB,&tn,sizeof tn);
    hipblasLtMatrixLayout_t la,lb,lc; hipblasLtMatrixLayoutCreate(&la,HIP_R_16BF,K,N,K); hipblasLtMatrixLayoutCreate(&lb,HIP_R_16BF,K,3*M,K); hipblasLtMatrixLayoutCreate(&lc,HIP_R_32F,N,3*M,N);
    ChatBfKey key{K,N,M}; auto it=g_chat_bfalgo.find(key); bool ok=true;
    split3s<<<(nx+255)/256,256,0,g_chat_cs>>>(x,g_chat_xs,nx,nx);
    if(it==g_chat_bfalgo.end()){ hipblasLtMatmulPreference_t pr; hipblasLtMatmulPreferenceCreate(&pr); size_t wsz=LTWS;
        hipblasLtMatmulPreferenceSetAttribute(pr,HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,&wsz,sizeof wsz);
        hipblasLtMatmulHeuristicResult_t hr[16]; int nr=0; hipblasLtMatmulAlgoGetHeuristic(g_chat_lt,d,la,lb,lc,lc,pr,16,hr,&nr); hipblasLtMatmulPreferenceDestroy(pr);
        double bt=1e30; int bi=-1;
        for(int i=0;i<nr;++i){ if(hr[i].state!=HIPBLAS_STATUS_SUCCESS) continue;
            if(hipblasLtMatmul(g_chat_lt,d,&al,W,la,g_chat_xs,lb,&be,g_chat_ys,lc,g_chat_ys,lc,&hr[i].algo,g_chat_ltws,LTWS,g_chat_cs)!=HIPBLAS_STATUS_SUCCESS) continue;
            HIP_CHECK(hipStreamSynchronize(g_chat_cs)); double t0=chat_now2();
            for(int q=0;q<3;++q) hipblasLtMatmul(g_chat_lt,d,&al,W,la,g_chat_xs,lb,&be,g_chat_ys,lc,g_chat_ys,lc,&hr[i].algo,g_chat_ltws,LTWS,g_chat_cs);
            HIP_CHECK(hipStreamSynchronize(g_chat_cs)); double t=chat_now2()-t0; if(t<bt){ bt=t; bi=i; } }
        if(bi<0) ok=false; else g_chat_bfalgo[key]=hr[bi].algo; it=g_chat_bfalgo.find(key); }
    if(ok) ok = hipblasLtMatmul(g_chat_lt,d,&al,W,la,g_chat_xs,lb,&be,g_chat_ys,lc,g_chat_ys,lc,&it->second,g_chat_ltws,LTWS,g_chat_cs)==HIPBLAS_STATUS_SUCCESS;
    hipblasLtMatrixLayoutDestroy(la); hipblasLtMatrixLayoutDestroy(lb); hipblasLtMatrixLayoutDestroy(lc); hipblasLtMatmulDescDestroy(d);
    if(!ok) return false;
    add3ld<<<(ny+255)/256,256,0,g_chat_cs>>>(g_chat_ys,y,M,N,ldo); return true; }
static tier_fp16* g_chat_xsf=nullptr; static size_t g_chat_xsf_n=0; static float* g_chat_ysf=nullptr; static size_t g_chat_ysf_n=0;
static std::map<ChatBfKey,hipblasLtMatmulAlgo_t> g_chat_fpalgo;
__global__ void split3s_fp16(const float* x, tier_fp16* d, size_t n, size_t stride){ size_t i=blockIdx.x*(size_t)blockDim.x+threadIdx.x; if(i>=n) return;
    float v=x[i]; tier_fp16 a=tier_fp16(v); float r=v-(float)a; tier_fp16 b=tier_fp16(r); float r2=r-(float)b;
    d[i]=tier_fp16(r2); d[i+stride]=b; d[i+2*stride]=a; }
static bool chat_fp3(const tier_fp16* W, const float* x, float* y, int M, int K, int N, int ldo){
    if(!g_chat_lt){ if(hipblasLtCreate(&g_chat_lt)!=HIPBLAS_STATUS_SUCCESS) return false; if(hipMalloc(&g_chat_ltws,LTWS)!=hipSuccess) return false; }
    size_t nx=(size_t)M*K, ny=(size_t)M*N;
    if(nx*3>g_chat_xsf_n){ if(g_chat_xsf) HIP_CHECK(hipFree(g_chat_xsf)); if(hipMalloc(&g_chat_xsf,nx*3*2)!=hipSuccess){ g_chat_xsf=nullptr; g_chat_xsf_n=0; return false; } g_chat_xsf_n=nx*3; }
    if(ny*3>g_chat_ysf_n){ if(g_chat_ysf) HIP_CHECK(hipFree(g_chat_ysf)); if(hipMalloc(&g_chat_ysf,ny*3*4)!=hipSuccess){ g_chat_ysf=nullptr; g_chat_ysf_n=0; return false; } g_chat_ysf_n=ny*3; }
    hipblasOperation_t ta=HIPBLAS_OP_T, tn=HIPBLAS_OP_N; float al=1,be=0;
    hipblasLtMatmulDesc_t d; hipblasLtMatmulDescCreate(&d,HIPBLAS_COMPUTE_32F,HIP_R_32F);
    hipblasLtMatmulDescSetAttribute(d,HIPBLASLT_MATMUL_DESC_TRANSA,&ta,sizeof ta); hipblasLtMatmulDescSetAttribute(d,HIPBLASLT_MATMUL_DESC_TRANSB,&tn,sizeof tn);
    hipblasLtMatrixLayout_t la,lb,lc; hipblasLtMatrixLayoutCreate(&la,HIP_R_16F,K,N,K); hipblasLtMatrixLayoutCreate(&lb,HIP_R_16F,K,3*M,K); hipblasLtMatrixLayoutCreate(&lc,HIP_R_32F,N,3*M,N);
    ChatBfKey key{K,N,M}; auto it=g_chat_fpalgo.find(key); bool ok=true;
    split3s_fp16<<<(nx+255)/256,256,0,g_chat_cs>>>(x,g_chat_xsf,nx,nx);
    if(it==g_chat_fpalgo.end()){ hipblasLtMatmulPreference_t pr; hipblasLtMatmulPreferenceCreate(&pr); size_t wsz=LTWS;
        hipblasLtMatmulPreferenceSetAttribute(pr,HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,&wsz,sizeof wsz);
        hipblasLtMatmulHeuristicResult_t hr[16]; int nr=0; hipblasLtMatmulAlgoGetHeuristic(g_chat_lt,d,la,lb,lc,lc,pr,16,hr,&nr); hipblasLtMatmulPreferenceDestroy(pr);
        double bt=1e30; int bi=-1;
        for(int i=0;i<nr;++i){ if(hr[i].state!=HIPBLAS_STATUS_SUCCESS) continue;
            if(hipblasLtMatmul(g_chat_lt,d,&al,W,la,g_chat_xsf,lb,&be,g_chat_ysf,lc,g_chat_ysf,lc,&hr[i].algo,g_chat_ltws,LTWS,g_chat_cs)!=HIPBLAS_STATUS_SUCCESS) continue;
            HIP_CHECK(hipStreamSynchronize(g_chat_cs)); double t0=chat_now2();
            for(int q=0;q<3;++q) hipblasLtMatmul(g_chat_lt,d,&al,W,la,g_chat_xsf,lb,&be,g_chat_ysf,lc,g_chat_ysf,lc,&hr[i].algo,g_chat_ltws,LTWS,g_chat_cs);
            HIP_CHECK(hipStreamSynchronize(g_chat_cs)); double t=chat_now2()-t0; if(t<bt){ bt=t; bi=i; } }
        if(bi<0) ok=false; else g_chat_fpalgo[key]=hr[bi].algo; it=g_chat_fpalgo.find(key); }
    if(ok) ok = hipblasLtMatmul(g_chat_lt,d,&al,W,la,g_chat_xsf,lb,&be,g_chat_ysf,lc,g_chat_ysf,lc,&it->second,g_chat_ltws,LTWS,g_chat_cs)==HIPBLAS_STATUS_SUCCESS;
    hipblasLtMatrixLayoutDestroy(la); hipblasLtMatrixLayoutDestroy(lb); hipblasLtMatrixLayoutDestroy(lc); hipblasLtMatmulDescDestroy(d);
    if(!ok) return false;
    add3ld<<<(ny+255)/256,256,0,g_chat_cs>>>(g_chat_ysf,y,M,N,ldo); return true; }
static bool chat_bf3_from_fp16(const tier_fp16* W, const float* x, float* y, int M, int K, int N, int ldo){
    // ие1
    hip_bfloat16* wb=fp16_to_bf16_temp(W,(size_t)K*N,g_chat_cs);
    return wb && chat_bf3(wb,x,y,M,K,N,ldo);
}
static void chat_linear_ld(hipblasHandle_t h, const float* x, const float* W, float* out, int M, int K, int N, int ldo){ // иж1
    float a=1,b=0; HIPBLAS_CHECK(hipblasSgemm(h,HIPBLAS_OP_T,HIPBLAS_OP_N,N,M,K,&a,W,K,x,K,&b,out,ldo)); }
static const int CHAT_EXP_ROWS=1024; // иж2
static void chat_linear(hipblasHandle_t h, const float* x, const float* W, float* out, int M, int K, int N){
    float a=1,b=0; HIPBLAS_CHECK(hipblasSgemm(h,HIPBLAS_OP_T,HIPBLAS_OP_N,N,M,K,&a,W,K,x,K,&b,out,N));
}
__global__ void rope_pos(float* qk, int M, int n_heads, int head_dim, int rotary_dim, float theta, int pos0, const float* invf, int half, float msc){
    int tok=blockIdx.x, head=blockIdx.y; float* v=qk+((size_t)tok*n_heads+head)*head_dim;
    int active=rotary_dim/2, pos=pos0+tok;
    for(int i=threadIdx.x;i<active;i+=blockDim.x){
        float freq=invf?invf[i]:powf(theta,-2.0f*i/head_dim); float ang=pos*freq; float c=cosf(ang)*msc,s=sinf(ang)*msc;
        float a=v[i],b=v[i+half]; v[i]=a*c-b*s; v[i+half]=a*s+b*c;
    }
}
// ик1
__global__ void attn_cache(const float* Q, const float* Kc, const float* Vc, float* out, int pos0, int n_heads, int n_kv_heads, int head_dim, int window, float cap){
    int tok=blockIdx.x, head=blockIdx.y; int kv_head=head/(n_heads/n_kv_heads); int p=pos0+tok;
    extern __shared__ float scores[];
    const float* q=Q+((size_t)tok*n_heads+head)*head_dim;
    int lo=(window>0)?max(0,p-window+1):0; int n_keys=p-lo+1;
    for(int j=threadIdx.x;j<n_keys;j+=blockDim.x){
        const float* k=Kc+((size_t)(lo+j)*n_kv_heads+kv_head)*head_dim;
        float dot=0; for(int d=0;d<head_dim;++d) dot+=q[d]*k[d]; scores[j]=cap>0 ? cap*tanhf(dot/cap) : dot;
    }
    __syncthreads();
    __shared__ float mx,sum;
    if(threadIdx.x==0){ mx=-1e30f; for(int j=0;j<n_keys;++j) mx=fmaxf(mx,scores[j]);
        sum=0; for(int j=0;j<n_keys;++j){ scores[j]=expf(scores[j]-mx); sum+=scores[j]; } }
    __syncthreads();
    float* o=out+((size_t)tok*n_heads+head)*head_dim;
    for(int d=threadIdx.x;d<head_dim;d+=blockDim.x){
        float acc=0; for(int j=0;j<n_keys;++j) acc+=scores[j]*Vc[((size_t)(lo+j)*n_kv_heads+kv_head)*head_dim+d];
        o[d]=acc/sum;
    }
}
// ил1
template<typename T> __global__ void gemv(const T* W, const float* x, float* y, int N, int K){
    int wpb=blockDim.x/warpSize, row=blockIdx.x*wpb+threadIdx.x/warpSize, lane=threadIdx.x%warpSize;
    if(row>=N) return;
    const T* w=W+(size_t)row*K; float acc=0;
    for(int k=lane;k<K;k+=warpSize) acc+=(float)w[k]*x[k];
    for(int o=warpSize/2;o>0;o>>=1) acc+=__shfl_down(acc,o);
    if(lane==0) y[row]=acc;
}
__global__ void embed_kernel(const void* E, const int32_t* tok, float* x, int M, int H, float sc, int etype){
    int t=blockIdx.x; size_t off=(size_t)tok[t]*H;
    for(int d=threadIdx.x;d<H;d+=blockDim.x) x[(size_t)t*H+d]=norm_weight_at(E,etype,(int)(off+d))*(sc<0?sqrtf((float)H):sc);
}

struct ChatMat { void* host; void* dev; size_t n, bytes; int rows, cols; bool f32; void* hostB; int rc; void* hostT=nullptr; float* la=nullptr; float* lb=nullptr;
                 int qt=0; size_t rb=0;                                   // ин1
                 int hr=0; void* hd=nullptr; float* hT=nullptr; float* hB=nullptr; const int* ri=nullptr; };   // ин2
struct ChatLayer { bool full; int hd, nkv, qdim, kvdim; ChatMat m[7]; float *qb=nullptr,*kb=nullptr,*vb=nullptr; };   // ио1
static const char* CHAT_MN[7]={"q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"};
static const char* CHAT_SLOT[7]={"q","k","v","o","gate","up","down"};
static std::string chat_tn(int i){ return arch_tn(CHAT_SLOT[i]); }   // ис1
static float* g_chat_expand;
// ис2
static std::vector<ChatMat*> g_chat_SQ; static void* chat_slot[2]; static hipEvent_t chat_ev_cp[2], chat_ev_done[2]; static long qi=0;
static size_t chat_gpu_bytes(const ChatMat* m); // ис3
static void chat_issue(long i){ int s=i%2; ChatMat* m=g_chat_SQ[i%g_chat_SQ.size()]; HIP_CHECK(hipStreamWaitEvent(g_chat_ms,chat_ev_done[s],0)); HIP_CHECK(hipMemcpyAsync(chat_slot[s],m->host,chat_gpu_bytes(m),hipMemcpyHostToDevice,g_chat_ms)); HIP_CHECK(hipEventRecord(chat_ev_cp[s],g_chat_ms)); } // ит1

// иу1
static bool cpu_has_avx2(){
#if TIER_X86
    return __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma");
#else
    return false;
#endif
}
static float g_chat_fcpu=0; static int g_chat_nt=6; static float *g_chat_xh, *g_chat_yh;
struct ChatJob{ const void* W; bool f32, fp16; int rows, K; const float* x; float* y; };
static ChatJob g_chat_job; static std::atomic<int> g_chat_gen{0}, g_chat_left{0};
#if TIER_X86
__attribute__((target("avx2,fma"))) static float chat_dot_bf16(const uint16_t* w, const float* x, int K){
    __m256 a0=_mm256_setzero_ps(), a1=_mm256_setzero_ps(); int k=0;
    for(; k+16<=K; k+=16){
        __m256 f0=_mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(_mm_loadu_si128((const __m128i*)(w+k))),16));
        __m256 f1=_mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(_mm_loadu_si128((const __m128i*)(w+k+8))),16));
        a0=_mm256_fmadd_ps(f0,_mm256_loadu_ps(x+k),a0); a1=_mm256_fmadd_ps(f1,_mm256_loadu_ps(x+k+8),a1); }
    a0=_mm256_add_ps(a0,a1); float t[8]; _mm256_storeu_ps(t,a0); float s=t[0]+t[1]+t[2]+t[3]+t[4]+t[5]+t[6]+t[7];
    for(; k<K; ++k){ uint32_t u=((uint32_t)w[k])<<16; float f; memcpy(&f,&u,4); s+=f*x[k]; } return s; }
#else
static float chat_dot_bf16(const uint16_t* w, const float* x, int K){ float s=0;
    for(int k=0;k<K;++k){ uint32_t u=((uint32_t)w[k])<<16; float f; memcpy(&f,&u,4); s+=f*x[k]; } return s; }
#endif
static float chat_dot_fp16(const uint16_t* w, const float* x, int K){ float s=0;
    for(int k=0;k<K;++k) s+=(float)tier_fp16::to_float(w[k])*x[k]; return s; }
#if TIER_X86
__attribute__((target("avx2,fma"))) static float chat_dot_f32(const float* w, const float* x, int K){
    __m256 a0=_mm256_setzero_ps(), a1=_mm256_setzero_ps(); int k=0;
    for(; k+16<=K; k+=16){ a0=_mm256_fmadd_ps(_mm256_loadu_ps(w+k),_mm256_loadu_ps(x+k),a0); a1=_mm256_fmadd_ps(_mm256_loadu_ps(w+k+8),_mm256_loadu_ps(x+k+8),a1); }
    a0=_mm256_add_ps(a0,a1); float t[8]; _mm256_storeu_ps(t,a0); float s=t[0]+t[1]+t[2]+t[3]+t[4]+t[5]+t[6]+t[7]; for(; k<K; ++k) s+=w[k]*x[k]; return s; }
#else
static float chat_dot_f32(const float* w, const float* x, int K){ float s=0; for(int k=0;k<K;++k) s+=w[k]*x[k]; return s; }
#endif
static void chat_worker(int w){ int seen=0;
    for(;;){ int sp=0; while(g_chat_gen.load(std::memory_order_acquire)==seen){ ++sp; if(sp>400000) std::this_thread::sleep_for(std::chrono::microseconds(200)); else if(sp>4000) std::this_thread::yield(); else _mm_pause(); } // ия1
        seen=g_chat_gen.load(std::memory_order_acquire); const ChatJob j=g_chat_job; int per=(j.rows+g_chat_nt-1)/g_chat_nt, a=w*per, b=std::min(j.rows,a+per);
        for(int r=a;r<b;++r) j.y[r]=j.f32 ? chat_dot_f32((const float*)j.W+(size_t)r*j.K,j.x,j.K) : (j.fp16 ? chat_dot_fp16((const uint16_t*)j.W+(size_t)r*j.K,j.x,j.K) : chat_dot_bf16((const uint16_t*)j.W+(size_t)r*j.K,j.x,j.K));
        g_chat_left.fetch_sub(1,std::memory_order_acq_rel); } }
static void chat_cpu_start(const ChatJob& j){ g_chat_job=j; g_chat_left.store(g_chat_nt,std::memory_order_release); g_chat_gen.fetch_add(1,std::memory_order_acq_rel); }
static void chat_cpu_wait(){ while(g_chat_left.load(std::memory_order_acquire)>0) _mm_pause(); }
static size_t chat_gpu_bytes(const ChatMat* m){ return m->qt ? m->bytes : (size_t)(m->rows-m->rc)*m->cols*(m->f32?4:2); }
static void chat_read_raw(void* dst, uint64_t off, size_t n){ int fd=open(g_path.c_str(),O_RDONLY); if(fd<0){ perror(g_path.c_str()); exit(1); } size_t t=0; while(t<n){ ssize_t r=pread(fd,(char*)dst+t,n-t,off+t); if(r<0 && errno==EINTR) continue; if(r<=0){ if(r==0) errno=EIO; perror("pread model tensor"); close(fd); exit(1); } t+=(size_t)r; } close(fd); }
static std::vector<float> chat_read_f32(const std::string& name){ auto& r=g_tensors.at(name); std::vector<uint16_t> b(r.size/2); chat_read_raw(b.data(),r.offset,r.size);
    std::vector<float> f(b.size()); for(size_t i=0;i<b.size();++i){ uint32_t u=((uint32_t)b[i])<<16; memcpy(&f[i],&u,4); } return f; }
static int g_chat_lr=0; static float g_chat_lalpha=0, g_chat_lk=1; static bool g_chat_lon=true; static float* g_chat_lu=nullptr; static size_t g_chat_lu_n=0;
static void chat_lora(const ChatMat& m, const float* x, float* y, int M){
    float s=g_chat_lon ? g_chat_lk*g_chat_lalpha/g_chat_lr : 0.f; if(s==0.f) return; const float one=1, zero=0; int r=g_chat_lr;
    if((size_t)M*r>g_chat_lu_n){ if(g_chat_lu) HIP_CHECK(hipFree(g_chat_lu)); g_chat_lu_n=(size_t)M*r; HIP_CHECK(hipMalloc(&g_chat_lu,g_chat_lu_n*4)); }
    HIPBLAS_CHECK(hipblasSgemm(g_chat_blas,HIPBLAS_OP_T,HIPBLAS_OP_N,r,M,m.cols,&one,m.la,m.cols,x,m.cols,&zero,g_chat_lu,r));
    HIPBLAS_CHECK(hipblasSgemm(g_chat_blas,HIPBLAS_OP_T,HIPBLAS_OP_N,m.rows,M,r,&s,m.lb,r,g_chat_lu,r,&one,y,m.rows)); }
static bool g_chat_heads=true;
static float* g_chat_htmp=nullptr; static size_t g_chat_htmp_n=0;
static void chat_heads(const ChatMat& m, const float* x, float* y, int M){
    if(m.ri){ size_t need=(size_t)M*m.hr; if(need>g_chat_htmp_n){ if(g_chat_htmp) HIP_CHECK(hipFree(g_chat_htmp)); HIP_CHECK(hipMalloc(&g_chat_htmp,need*4)); g_chat_htmp_n=need; }
        ChatMat t=m; t.ri=nullptr; t.rows=m.hr; chat_heads(t,x,g_chat_htmp,M);   // кж1
        scatter_cols_k<<<(unsigned)((need+255)/256),256,0,g_chat_cs>>>(g_chat_htmp,y,m.ri,M,m.hr,m.rows); return; }
    if(M==1){ int wpb=256/g_chat_ws;
        if(g_fp16_storage) gemv<tier_fp16><<<(m.hr+wpb-1)/wpb,256,0,g_chat_cs>>>((const tier_fp16*)m.hd,x,y,m.hr,m.cols);
        else gemv<hip_bfloat16><<<(m.hr+wpb-1)/wpb,256,0,g_chat_cs>>>((const hip_bfloat16*)m.hd,x,y,m.hr,m.cols); return; }
    for(int r0=0;r0<m.hr;r0+=CHAT_EXP_ROWS){ int rb=std::min(CHAT_EXP_ROWS,m.hr-r0); size_t ne=(size_t)rb*m.cols;
        if(g_fp16_storage) fp16_to_f32_kernel<<<(ne+255)/256,256,0,g_chat_cs>>>((const tier_fp16*)m.hd+(size_t)r0*m.cols,g_chat_expand,ne);
        else bf16_to_f32_kernel<<<(ne+255)/256,256,0,g_chat_cs>>>((const hip_bfloat16*)m.hd+(size_t)r0*m.cols,g_chat_expand,ne);
        chat_linear_ld(g_chat_blas,x,g_chat_expand,y+r0,M,m.cols,rb,m.rows); } }
static void chat_lin(ChatMat& m, const float* x, float* y, int M){
    const void* W=m.dev; long my=-1;
    int rc=(!m.dev)?m.rc:0, rg=m.rows-rc; size_t el=m.f32?4:2;
    if(!W){ my=qi++; HIP_CHECK(hipStreamWaitEvent(g_chat_cs,chat_ev_cp[my%2],0)); W=chat_slot[my%2]; }
    if(m.qt){   // кз1
        QRef Q{m.qt,m.rows,m.cols,0,m.bytes,m.rb};
        if(M==1){ int wpb=256/g_chat_ws; gemv_q<<<(m.rows+wpb-1)/wpb,256,0,g_chat_cs>>>((const uint8_t*)W,m.qt,m.rb,x,y,m.rows,m.cols); }
        else for(int r0=0;r0<m.rows;r0+=CHAT_EXP_ROWS){ int rb=std::min(CHAT_EXP_ROWS,m.rows-r0);
            dq_launch(W,Q,r0,rb,g_chat_expand,g_chat_cs); chat_linear_ld(g_chat_blas,x,g_chat_expand,y+r0,M,m.cols,rb,m.rows); }
    } else if(M==1){ int wpb=256/g_chat_ws, bl=(rg+wpb-1)/wpb;
        if(rc>0){ HIP_CHECK(hipMemcpyAsync(g_chat_xh,x,m.cols*4,hipMemcpyDeviceToHost,g_chat_cs)); HIP_CHECK(hipStreamSynchronize(g_chat_cs)); // кз2
            chat_cpu_start({(const char*)m.host+(size_t)rg*m.cols*el,m.f32,(!m.f32 && g_fp16_storage),rc,m.cols,g_chat_xh,g_chat_yh}); }
        if(m.f32) gemv<float><<<bl,256,0,g_chat_cs>>>((const float*)W,x,y,rg,m.cols); else if(g_fp16_storage) gemv<tier_fp16><<<bl,256,0,g_chat_cs>>>((const tier_fp16*)W,x,y,rg,m.cols); else gemv<hip_bfloat16><<<bl,256,0,g_chat_cs>>>((const hip_bfloat16*)W,x,y,rg,m.cols);
        if(rc>0){ chat_cpu_wait(); HIP_CHECK(hipMemcpyAsync(y+rg,g_chat_yh,rc*4,hipMemcpyHostToDevice,g_chat_cs)); } }
    else { if(rc>0) HIP_CHECK(hipMemcpyAsync((char*)W+(size_t)rg*m.cols*el,(const char*)m.host+(size_t)rg*m.cols*el,(size_t)rc*m.cols*el,hipMemcpyHostToDevice,g_chat_cs)); // кз3
        const float* Wf=(const float*)W;
        if(m.f32) chat_linear(g_chat_blas,x,Wf,y,M,m.cols,m.rows);
        else if(g_use_fp16_x3 && chat_fp3((const tier_fp16*)W,x,y,M,m.cols,m.rows,m.rows)) {}
        else if(g_use_bf16_x3 && chat_bf3_from_fp16((const tier_fp16*)W,x,y,M,m.cols,m.rows,m.rows)) {}
        else if(!g_fp16_storage && !g_chat_no_bf3 && chat_bf3((const hip_bfloat16*)W,x,y,M,m.cols,m.rows,m.rows)) {}
        else for(int r0=0;r0<m.rows;r0+=CHAT_EXP_ROWS){ int rb=std::min(CHAT_EXP_ROWS,m.rows-r0); size_t ne=(size_t)rb*m.cols;
            if(g_fp16_storage) fp16_to_f32_kernel<<<(ne+255)/256,256,0,g_chat_cs>>>((const tier_fp16*)W+(size_t)r0*m.cols,g_chat_expand,ne);
            else bf16_to_f32_kernel<<<(ne+255)/256,256,0,g_chat_cs>>>((const hip_bfloat16*)W+(size_t)r0*m.cols,g_chat_expand,ne);
            chat_linear_ld(g_chat_blas,x,g_chat_expand,y+r0,M,m.cols,rb,m.rows); } }
    if(m.hr && g_chat_heads) chat_heads(m,x,y,M);   // кз4
    if(m.la) chat_lora(m,x,y,M);
    if(my>=0){ HIP_CHECK(hipEventRecord(chat_ev_done[my%2],g_chat_cs)); chat_issue(my+2); }
}
static double chat_now(){ return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

int chat_main(int argc, char** argv){
    if(argc<4){ fprintf(stderr,"Использование: %s <model.safetensors> <weights_checkpoint.bin|-> <tier_file|-> [ctx]\n"
                               "Ввод (stdin): строки '<prompt.bin> <n> <max_new> <temp> [top_k top_p min_p штраф_повтора окно_повтора зерно присутствие частота]'; вывод: строки 'gen_token <id>' и 'gen_end'.\n",argv[0]); return 1; }
    setvbuf(stdout,nullptr,_IOLBF,0); double T0=chat_now();
    { size_t fr=0,tt=0; if(hipMemGetInfo(&fr,&tt)==hipSuccess && tt>fr){ g_others0=std::max(tt-fr,sys_vram_used()); g_others0_set=true; } }   // кк1
    parse_header(argv[1]); arch_load(argv[1]); int CTX=argc>4?atoi(argv[4]):768;
    for(int i=1;i+1<argc;++i) if(!strcmp(argv[i],"--qbase")){ if(!tq_load(argv[i+1])) return 1;   // кк2
        for(auto& [k,q] : g_tq){ auto it=g_tensors.find(k); if(it==g_tensors.end() || it->second.size!=(uint64_t)q.rows*q.cols*2 || (it->second.cols && it->second.cols!=q.cols)){ fprintf(stderr,"Квантованная база не от этой модели: %s.\n",k.c_str()); return 1; } } }
    gpu_pick(); hipDeviceProp_t pr; HIP_CHECK(hipGetDeviceProperties(&pr,g_dev)); g_chat_ws=pr.warpSize;
    { size_t mx=pr.sharedMemPerBlock/sizeof(float); if(mx>64 && (size_t)CTX>mx-64){   // кк3
        fprintf(stderr,"Контекст %d позиций не помещается в разделяемую память карты (%zu КиБ на блок), уменьшен до %zu.\n",CTX,(size_t)pr.sharedMemPerBlock>>10,mx-64); CTX=(int)(mx-64); } }
    arch_device();
    // кк4
    const bool fp16_x3_available=fp16_probe();
    const bool bf16_x3_available=bf3_probe();
    g_fp16_storage=true;
    if(const char* b=getenv("TIER_BF3")) g_chat_no_bf3=atoi(b)==0;
    g_use_fp16_x3=false; g_use_bf16_x3=false;
    if(!g_chat_no_bf3 && TIER_LT && fp16_x3_available) g_use_fp16_x3=true;
    else if(!g_chat_no_bf3 && TIER_LT && bf16_x3_available) g_use_bf16_x3=true;
    else if(!g_chat_no_bf3){ g_chat_no_bf3=true; fprintf(stderr,"FP16×3 и BF16×3 недоступны: в режиме разговора используется запасной режим FP32.\n"); }
    fprintf(stderr,"Хранение замороженных весов: FP16; режим ×3: %s.\n",
            g_use_fp16_x3?"FP16×3":(g_use_bf16_x3?"BF16×3 через временное преобразование FP16→BF16":"отключён (запасной режим FP32)"));
    // кк5
    std::map<std::string,std::pair<size_t,size_t>> lay; size_t tot=0;
    std::map<std::string,size_t> fulln; // кк6
    auto reg=[&](int l,const char* n,size_t c){ std::string k="layer_"+std::to_string(l)+"."+n; lay[k]={tot,c}; fulln[k]=c; tot+=c; };
    for(int l=0;l<N_LAYERS;++l){ bool full=!A.L[l].has_v; int hd=A.L[l].hd, nkv=A.L[l].nkv, qd=NUM_HEADS*hd, kd=nkv*hd;
        if(arch_has("iln")) reg(l,"input_layernorm",HIDDEN); if(A.post_attn) reg(l,"post_attn_ln",HIDDEN); if(arch_has("pfln")) reg(l,"pre_ffn_ln",HIDDEN); if(A.post_ffn) reg(l,"post_ffn_ln",HIDDEN);
        reg(l,"q_proj",(size_t)qd*HIDDEN); reg(l,"k_proj",(size_t)kd*HIDDEN); if(!full) reg(l,"v_proj",(size_t)kd*HIDDEN); reg(l,"o_proj",(size_t)HIDDEN*qd);
        if(A.qk_norm){ reg(l,"q_norm",A.qk_full?(size_t)qd:(size_t)hd); reg(l,"k_norm",A.qk_full?(size_t)kd:(size_t)hd); } reg(l,"gate_proj",(size_t)INTER*HIDDEN); reg(l,"up_proj",(size_t)INTER*HIDDEN); reg(l,"down_proj",(size_t)HIDDEN*INTER); }
    reg(-1,"final_norm",HIDDEN);
    std::set<std::string> t1; std::string ck=argv[2]; bool use_ck=(ck!="-") && access(ck.c_str(),R_OK)==0;
    if(use_ck && std::string(argv[3])!="-"){ FILE* f=fopen(argv[3],"r"); char line[512],nm[256]; int tr; double a,b,c; size_t n;
        while(f && fgets(line,sizeof line,f)) if(line[0]!='#' && strncmp(line,"rows ",5) && sscanf(line,"%255s %d %le %le %le %zu",nm,&tr,&a,&b,&c,&n)==6 && tr==1) t1.insert(nm); if(f) fclose(f);
        FILE* g=fopen(argv[3],"r"); char* ln=nullptr; size_t cap=0; g_ridx.clear();   // кк7
        while(g && getline(&ln,&cap,g)>0) if(!strncmp(ln,"rows ",5)){ std::istringstream is(ln+5); std::string k; is>>k; std::vector<int> v; int x; while(is>>x) v.push_back(x); if(!v.empty()) g_ridx[k]=v; }
        if(g) fclose(g); free(ln); }
    // кк8
    std::map<std::string,std::pair<float*,float*>> lora_ab;
    if(use_ck){ int fd=open(ck.c_str(),O_RDONLY); std::string h(64<<10,'\0'); ssize_t r=pread(fd,h.data(),h.size(),0);
        if(r>5 && h.compare(0,5,"LKL1 ")==0){ std::istringstream is(h.substr(0,h.find('\0'))); std::string tag, tg; float dr; size_t na; is>>tag>>g_chat_lr>>g_chat_lalpha>>dr>>tg>>na;
            std::string k; int in,out; size_t off; std::vector<float> buf;
            while(is>>k>>in>>out>>off){ float *A,*B; size_t a=(size_t)g_chat_lr*in, b=(size_t)out*g_chat_lr;
                HIP_CHECK(hipMalloc(&A,a*4)); HIP_CHECK(hipMalloc(&B,b*4));
                buf.resize(a); pread_all(fd,buf.data(),a*4,off); HIP_CHECK(hipMemcpy(A,buf.data(),a*4,hipMemcpyHostToDevice));
                buf.resize(b); pread_all(fd,buf.data(),b*4,off+a*4); HIP_CHECK(hipMemcpy(B,buf.data(),b*4,hipMemcpyHostToDevice));
                lora_ab[k]={A,B}; }
            fprintf(stderr,"LoRA r=%d alpha=%g: адаптеров %zu (%s).\n",g_chat_lr,g_chat_lalpha,lora_ab.size(),tg.c_str()); t1.clear(); }
        close(fd); }
    if(use_ck && lora_ab.empty()){ int fd=open(ck.c_str(),O_RDONLY); std::string h(64<<10,'\0'); ssize_t r=pread(fd,h.data(),h.size(),0); close(fd); // кк9
        if(r>5 && h.compare(0,5,"LKT1 ")==0){ t1.clear(); std::istringstream is(h.substr(0,h.find('\0'))); std::string k; size_t n,off; is>>k>>n; while(is>>k>>n>>off){ lay[k]={off/4,n}; t1.insert(k); } } }
    // кк10
    std::set<std::string> sl; for(const auto& k : t1){ auto f=fulln.find(k); if(f!=fulln.end() && lay.at(k).second < f->second) sl.insert(k); }
    if(!sl.empty()) fprintf(stderr,"Срез строк: матриц %zu, строки накладываются на BF16.\n", sl.size());
    fprintf(stderr,"%s, обучаемая часть FP32 из файла весов: тензоров %zu.\n", use_ck?"Обученная модель":"Исходная модель", t1.size());
    auto ck_read=[&](const std::string& key, void* dst){ auto p=lay.at(key); int fd=open(ck.c_str(),O_RDONLY); size_t n=p.second*4,t=0; while(t<n){ ssize_t r=pread(fd,(char*)dst+t,n-t,p.first*4+t); if(r<=0){perror("Снимок весов");exit(1);} t+=r; } close(fd); };
    auto normv=[&](const std::string& k,const std::string& st,size_t n,bool base)->float*{ std::vector<float> v(n);
        const bool from_ck=!base && t1.count(k); if(from_ck) ck_read(k,v.data()); else v=chat_read_f32(st);
        if(!from_ck && g_fp16_storage){ std::vector<tier_fp16> h(n); for(size_t i=0;i<n;++i) h[i]=tier_fp16(v[i]); tier_fp16* d; HIP_CHECK(hipMalloc(&d,n*sizeof(tier_fp16))); HIP_CHECK(hipMemcpy(d,h.data(),n*sizeof(tier_fp16),hipMemcpyHostToDevice)); g_fp16w.insert(d); return (float*)d; }
        float* d; HIP_CHECK(hipMalloc(&d,n*sizeof(float))); HIP_CHECK(hipMemcpy(d,v.data(),n*sizeof(float),hipMemcpyHostToDevice)); return d; };
    std::vector<std::array<float*,7>> nwT(N_LAYERS), nwB(N_LAYERS); std::vector<ChatLayer> L(N_LAYERS); size_t maxmb=0, maxn=0;
    for(int l=0;l<N_LAYERS;++l){ auto& y=L[l]; const LayerSpec& sp=A.L[l]; y.full=!sp.has_v; int qd=NUM_HEADS*sp.hd, kd=sp.nkv*sp.hd;
        int rc[7][2]={{qd,HIDDEN},{kd,HIDDEN},{kd,HIDDEN},{HIDDEN,qd},{INTER,HIDDEN},{INTER,HIDDEN},{HIDDEN,INTER}};
        for(int i=0;i<7;++i){ if(i==2&&y.full) continue; size_t n=(size_t)rc[i][0]*rc[i][1]; maxn=std::max(maxn,n); std::string kk="layer_"+std::to_string(l)+"."+CHAT_MN[i]; maxmb=std::max(maxmb,n*((t1.count(kk)&&!sl.count(kk))?4:2)); } }
    // кк11
    size_t arena=0;
    for(int l=0;l<N_LAYERS;++l){ auto& y=L[l]; const LayerSpec& sp=A.L[l]; y.full=!sp.has_v; y.hd=sp.hd; y.nkv=sp.nkv; y.qdim=NUM_HEADS*y.hd; y.kvdim=y.nkv*y.hd;
        std::string p=LPX(l), k="layer_"+std::to_string(l)+".";
        auto nv=[&](bool on,const char* key,const char* slot,size_t n,int b)->float*{ return on ? normv(k+key,p+arch_tn(slot),n,b) : nullptr; };
        for(int b=0;b<2;++b) (b?nwB:nwT)[l]={nv(arch_has("iln"),"input_layernorm","iln",HIDDEN,b),nv(A.post_attn,"post_attn_ln","paln",HIDDEN,b),
               nv(arch_has("pfln"),"pre_ffn_ln","pfln",HIDDEN,b),nv(A.post_ffn,"post_ffn_ln","pofln",HIDDEN,b),
               nv(A.qk_norm,"q_norm","qn",A.qk_full?y.qdim:y.hd,b),nv(A.qk_norm,"k_norm","kn",A.qk_full?y.kvdim:y.hd,b),A.layer_scalar?normv("-",p+arch_tn("ls"),1,b):nullptr};
        if(A.qkv_bias){ y.qb=normv("-",p+arch_tn("qb"),y.qdim,true); y.kb=normv("-",p+arch_tn("kb"),y.kvdim,true); if(sp.has_v) y.vb=normv("-",p+arch_tn("vb"),y.kvdim,true); }
        int rc[7][2]={{y.qdim,HIDDEN},{y.kvdim,HIDDEN},{y.kvdim,HIDDEN},{HIDDEN,y.qdim},{INTER,HIDDEN},{INTER,HIDDEN},{HIDDEN,INTER}};
        for(int i=0;i<7;++i){ ChatMat& m=y.m[i]; m={nullptr,nullptr,0,0,0,0,false,nullptr}; if(i==2&&y.full) continue;
            m.f32=t1.count(k+CHAT_MN[i])>0 && !sl.count(k+CHAT_MN[i]); m.rows=rc[i][0]; m.cols=rc[i][1]; m.n=(size_t)m.rows*m.cols; m.bytes=m.n*(m.f32?4:2);
            if(!m.f32) if(const QRef* q=tq_of(p+chat_tn(i))){ if(q->rows!=m.rows||q->cols!=m.cols){ fprintf(stderr,"Квантованная база не от этой модели: %s.\n",(p+chat_tn(i)).c_str()); exit(1); }
                m.qt=q->type; m.rb=q->rb; m.bytes=q->size; }   // кк12
            m.host=(void*)arena; arena+=(m.bytes+4095)/4096*4096; if(m.f32){ m.hostB=(void*)arena; arena+=(m.n*2+4095)/4096*4096; } } }
    // кк13
    { Conf cf; conf_load(cf,conf_path()); g_cap_pct=cf.mem_cap; if(const char* e=getenv("TIER_MEM_CAP_PCT")) g_cap_pct=atoi(e);   // кк14
      { FILE* o=fopen("/proc/self/oom_score_adj","w"); if(o){ fputs("300\n",o); fclose(o); } }   // кк15
      HostMem hm=host_mem(); size_t other=1ull<<30; if(const char* e=getenv("TIER_RAM_OTHER_GIB")) other=(size_t)(atof(e)*1073741824.0);
      size_t extra=0; for(const auto& k : sl) extra+=lay.at(k).second*(4+4+2);   // кк16
      { auto it=g_tensors.find(A.pre+arch_tn("embed")); size_t st=it!=g_tensors.end()?it->second.size:0;   // кк17
        if(!A.tied){ auto ih=g_tensors.find(A.head_name); if(ih!=g_tensors.end()) st=std::max(st,(size_t)ih->second.size); } extra+=st; }
      size_t need=arena+extra+other+cap_slack(hm.total);
      if(hm.total && hm.avail<need){
          fprintf(stderr,"Потолок памяти %d%%: для беседы требуется %.1f ГиБ ОЗУ под веса, плюс %.1f ГиБ процессу и %.1f ГиБ системе, всего %.1f ГиБ; свободно %.1f ГиБ. "
                         "Остановка до выделения памяти. Нужно закрыть часть приложений, использовать квантованную базу или поднять потолок памяти в настройках.\n",
                  g_cap_pct,arena/1073741824.0,(extra+other)/1073741824.0,cap_slack(hm.total)/1073741824.0,need/1073741824.0,hm.avail/1073741824.0);
          return 75; } }
    char* AR; if(hipHostMalloc((void**)&AR,arena)!=hipSuccess){ fprintf(stderr,"Не удалось закрепить %.1f ГиБ, используется обычная память.\n",arena/1073741824.0); AR=(char*)malloc(arena); }
    for(auto& y:L) for(auto& m:y.m) if(m.n){ m.host=AR+(size_t)m.host; if(m.hostB) m.hostB=AR+(size_t)m.hostB; }
    for(int l=0;l<N_LAYERS;++l) for(int i=0;i<7;++i){ auto it=lora_ab.find("layer_"+std::to_string(l)+"."+CHAT_MN[i]);
        if(it!=lora_ab.end() && L[l].m[i].n){ L[l].m[i].la=it->second.first; L[l].m[i].lb=it->second.second; } }
    fprintf(stderr,"ОЗУ под веса: %.2f ГиБ [%.1f с]\n",arena/1073741824.0,chat_now()-T0);
    // кк18
    { struct J{ int f; uint64_t off; size_t n; void* dst; }; std::vector<J> js; auto lo=[&](const std::string& key){ auto p=lay.at(key); return std::make_pair((uint64_t)p.first*4,(size_t)p.second*4); };
      for(int l=0;l<N_LAYERS;++l){ std::string p=LPX(l), k="layer_"+std::to_string(l)+".";
          for(int i=0;i<7;++i){ ChatMat& m=L[l].m[i]; if(!m.n) continue; auto& r=g_tensors.at(p+chat_tn(i));
              if(m.f32){ auto c=lo(k+CHAT_MN[i]); js.push_back({1,c.first,c.second,m.host}); js.push_back({0,r.offset,r.size,m.hostB}); }
              else if(m.qt){ const QRef* q=tq_of(p+chat_tn(i)); js.push_back({2,q->offset,q->size,m.host}); }
              else js.push_back({0,r.offset,r.size,m.host}); } }
      std::sort(js.begin(),js.end(),[](const J& x,const J& y){ return x.f!=y.f?x.f<y.f:x.off<y.off; });
      auto rot=[](const std::string& f){ struct stat sb; if(stat(f.c_str(),&sb)) return true; char q[128]; for(const char* sfx:{"/queue/rotational","/../queue/rotational"}){
          snprintf(q,sizeof q,"/sys/dev/block/%u:%u%s",major(sb.st_dev),minor(sb.st_dev),sfx); FILE* h=fopen(q,"r"); if(h){ int v=1; fscanf(h,"%d",&v); fclose(h); return v!=0; } } return true; };
      // кк19
      bool hdd=rot(g_path); int NT=getenv("TIER_THREADS")?atoi(getenv("TIER_THREADS")):1; bool dio=!getenv("TIER_DIRECT_IO")||atoi(getenv("TIER_DIRECT_IO")); size_t tot=0; for(auto& j:js) tot+=j.n;
      fprintf(stderr,"Диск: %s, потоков чтения: %d, %s.\n",hdd?"жёсткий":"SSD",NT,dio?"прямое чтение":"через кэш"); std::atomic<size_t> got{0}; std::vector<std::thread> th;
      for(int w=0;w<NT;++w) th.emplace_back([&,w]{ const size_t WIN=64<<20, AL=4096; int fl=O_RDONLY|(dio?O_DIRECT:0);
          bool uq=!g_tq_path.empty();   // кк20
          int fd[3]={open(g_path.c_str(),fl),use_ck?open(ck.c_str(),fl):-1,uq?open(g_tq_path.c_str(),fl):-1};
          if(dio && (fd[0]<0 || (use_ck && fd[1]<0) || (uq && fd[2]<0))){ for(int q=0;q<3;++q) if(fd[q]>=0) close(fd[q]); fl=O_RDONLY; dio=false;
              fd[0]=open(g_path.c_str(),fl); fd[1]=use_ck?open(ck.c_str(),fl):-1; fd[2]=uq?open(g_tq_path.c_str(),fl):-1; }
          char* bb=nullptr; if(dio) bb=(char*)aligned_alloc(AL,WIN);
          for(int q=0;q<3;++q) if(fd[q]>=0) posix_fadvise(fd[q],0,0,POSIX_FADV_SEQUENTIAL);
          auto prog=[&](size_t r){ size_t g=got+=r; if(w==0) fprintf(stderr,"Загрузка %.1f/%.1f ГиБ, %.0f МБ/с [%.1f с]\r",g/1073741824.0,tot/1073741824.0,g/1e6/(chat_now()-T0),chat_now()-T0); };
          for(size_t i=js.size()*w/NT;i<js.size()*(w+1)/NT;++i){ auto& j=js[i];
              if(!dio){ size_t t=0; while(t<j.n){ size_t c=std::min(j.n-t,WIN); ssize_t r=pread(fd[j.f],(char*)j.dst+t,c,j.off+t); if(r<=0){perror("Чтение (pread)");exit(1);} t+=r; prog(r); } continue; }
              uint64_t a=j.off/AL*AL, e=(j.off+j.n+AL-1)/AL*AL; // кк21
              for(uint64_t p=a;p<e;p+=WIN){ size_t c=std::min<uint64_t>(WIN,e-p); ssize_t r=pread(fd[j.f],bb,c,p); if(r<=0){perror("Чтение (pread, O_DIRECT)");exit(1);}
                  uint64_t s0=std::max<uint64_t>(p,j.off), s1=std::min<uint64_t>(p+r,j.off+j.n); if(s1>s0){ memcpy((char*)j.dst+(s0-j.off),bb+(s0-p),s1-s0); prog(s1-s0); }
                  if((size_t)r<c) break; } } // кк22
          free(bb); for(int q=0;q<3;++q) if(fd[q]>=0) close(fd[q]); }); for(auto& t:th) t.join(); fprintf(stderr,"\n"); }
    if(g_fp16_storage){
        for(auto& y:L) for(auto& m:y.m) if(m.n && !m.qt){
            if(m.f32){ if(m.hostB) chat_convert_bf16_host_to_fp16(m.hostB,m.n); }
            else chat_convert_bf16_host_to_fp16(m.host,m.n);
        }
    }
    // кк23
    size_t heads=0;
    if(!sl.empty()){ for(int l=0;l<N_LAYERS;++l){ std::string k="layer_"+std::to_string(l)+".", p=LPX(l);
        for(int i=0;i<7;++i){ ChatMat& m=L[l].m[i]; if(!m.n || !sl.count(k+CHAT_MN[i])) continue;
            size_t ntr=lay.at(k+CHAT_MN[i]).second; m.hr=(int)(ntr/m.cols);
            m.hT=(float*)malloc(ntr*4); m.hB=(float*)malloc(ntr*4); ck_read(k+CHAT_MN[i],m.hT);
            std::vector<uint16_t> b(ntr); auto gr=g_ridx.find(k+CHAT_MN[i]); bool byrow = gr!=g_ridx.end() && gr->second.size()==(size_t)m.hr;
            if(byrow){ m.ri=ridx_dev(k+CHAT_MN[i]);   // кк24
                for(int q=0;q<m.hr;++q){ size_t o=(size_t)gr->second[q]*m.cols;
                    if(m.qt) chat_read_raw(b.data()+(size_t)q*m.cols,g_tensors.at(p+chat_tn(i)).offset+o*2,(size_t)m.cols*2); else memcpy(b.data()+(size_t)q*m.cols,(const uint16_t*)m.host+o,(size_t)m.cols*2); } }
            else if(m.qt) chat_read_raw(b.data(),g_tensors.at(p+chat_tn(i)).offset,ntr*2); else memcpy(b.data(),m.host,ntr*2);
            for(size_t j=0;j<ntr;++j) m.hB[j]=(g_fp16_storage && !m.qt) ? (float)tier_fp16::to_float(b[j]) : q_b2f(b[j]);
            for(size_t j=0;j<ntr;++j){ if(g_fp16_storage) b[j]=tier_fp16(m.hT[j]).bits; else { uint32_t u; memcpy(&u,&m.hT[j],4); b[j]=(uint16_t)((u+0x7fffu+((u>>16)&1u))>>16); } }
            HIP_CHECK(hipMalloc(&m.hd,ntr*2)); HIP_CHECK(hipMemcpy(m.hd,b.data(),ntr*2,hipMemcpyHostToDevice)); heads+=ntr*2; } }
      fprintf(stderr,"Срез строк: обученные строки в VRAM, %.2f ГиБ.\n",heads/1073741824.0); }
    maxmb=0; for(auto& y:L) for(auto& m:y.m) if(m.n) maxmb=std::max(maxmb,m.qt?m.bytes:(size_t)m.n*(m.f32?4:2));   // кк25
    float* fnormT=normv("layer_-1.final_norm",A.pre+arch_tn("final"),HIDDEN,false), *fnormB=normv("-",A.pre+arch_tn("final"),HIDDEN,true);
    std::vector<std::array<float*,7>>* nwp=&nwT; float* fnorm=fnormT; bool trained=true;
    auto dev_low=[&](const std::string& nm)->void*{ void* d=nullptr; auto& r=g_tensors.at(nm); void* h; HIP_CHECK(hipHostMalloc(&h,r.size)); chat_read_raw(h,r.offset,r.size);
        if(g_fp16_storage) chat_convert_bf16_host_to_fp16(h,r.size/2);
        HIP_CHECK(hipMalloc(&d,r.size)); HIP_CHECK(hipMemcpy(d,h,r.size,hipMemcpyHostToDevice)); hipHostFree(h);
        if(g_fp16_storage) g_fp16w.insert(d); else g_bf16w.insert(d); return d; };
    void* E=dev_low(A.pre+arch_tn("embed"));
    void* HD=A.tied ? E : dev_low(A.head_name);   // кк26
    // кк27
    const int PC=std::min(CTX,256); size_t M=PC; // кк28
    float *x,*res,*nrm,*Q,*K,*V,*ao,*ap,*g,*u,*a,*mo,*rr,*nn,*h,*logits; int32_t* dtok;
    size_t QMAX=1, KVMAX=1, RRMAX=(size_t)NUM_HEADS;   // кк29
    for(int l=0;l<N_LAYERS;++l){ QMAX=std::max(QMAX,(size_t)NUM_HEADS*A.L[l].hd); KVMAX=std::max(KVMAX,(size_t)A.L[l].nkv*A.L[l].hd); RRMAX=std::max(RRMAX,(size_t)A.L[l].nkv); }
    for(auto pp : {std::make_pair(&x,M*HIDDEN),std::make_pair(&res,M*HIDDEN),std::make_pair(&nrm,M*HIDDEN),std::make_pair(&Q,M*QMAX),std::make_pair(&K,M*KVMAX),std::make_pair(&V,M*KVMAX),
                   std::make_pair(&ao,M*QMAX),std::make_pair(&ap,M*HIDDEN),std::make_pair(&g,M*INTER),std::make_pair(&u,M*INTER),std::make_pair(&a,M*INTER),std::make_pair(&mo,M*HIDDEN),
                   std::make_pair(&rr,M*RRMAX),std::make_pair(&nn,M*std::max((size_t)HIDDEN,QMAX)),std::make_pair(&h,(size_t)HIDDEN),std::make_pair(&logits,(size_t)VOCAB)}) HIP_CHECK(hipMalloc(pp.first,pp.second*4));
    HIP_CHECK(hipMalloc(&dtok,M*4)); { size_t mc=0; for(auto& y:L) for(auto& m:y.m) if(m.n) mc=std::max(mc,(size_t)m.cols); HIP_CHECK(hipMalloc(&g_chat_expand,std::min(maxn,(size_t)CHAT_EXP_ROWS*mc)*4)); }
    std::vector<float*> Kc(N_LAYERS),Vc(N_LAYERS); for(int l=0;l<N_LAYERS;++l){ HIP_CHECK(hipMalloc(&Kc[l],(size_t)CTX*L[l].kvdim*4)); HIP_CHECK(hipMalloc(&Vc[l],(size_t)CTX*L[l].kvdim*4)); }
    HIP_CHECK(hipMalloc(&chat_slot[0],maxmb)); HIP_CHECK(hipMalloc(&chat_slot[1],maxmb));
    // кк30
    size_t fr,tt; HIP_CHECK(hipMemGetInfo(&fr,&tt)); size_t reserve=std::max<size_t>(512ull<<20,vram_slack()), used=0, streamed=0; bool full_vram=false;   // кк31
    for(int l=0;l<N_LAYERS;++l) for(int i=0;i<7;++i){ ChatMat& m=L[l].m[i]; if(!m.n) continue;
        if(!full_vram && used+m.bytes+reserve<fr && hipMalloc(&m.dev,m.hostB?m.n*4:m.bytes)==hipSuccess){ /* кк32 */ HIP_CHECK(hipMemcpy(m.dev,m.host,m.bytes,hipMemcpyHostToDevice)); used+=m.bytes; }
        else { full_vram=true; m.dev=nullptr; g_chat_SQ.push_back(&m); streamed+=m.bytes; } }
    fprintf(stderr,"\nВ VRAM постоянно: матрицы и LM head, %.2f ГиБ; подкачка весов: %.2f ГиБ на проход (матриц: %zu); контекст: %d; загрузка: %.1f с.\n",used/1073741824.0,streamed/1073741824.0,g_chat_SQ.size(),CTX,chat_now()-T0);
    HIP_CHECK(hipStreamCreate(&g_chat_cs)); HIP_CHECK(hipStreamCreate(&g_chat_ms)); HIPBLAS_CHECK(hipblasCreate(&g_chat_blas)); HIPBLAS_CHECK(hipblasSetStream(g_chat_blas,g_chat_cs));
    for(int i=0;i<2;++i){ HIP_CHECK(hipEventCreateWithFlags(&chat_ev_cp[i],hipEventDisableTiming)); HIP_CHECK(hipEventCreateWithFlags(&chat_ev_done[i],hipEventDisableTiming)); HIP_CHECK(hipEventRecord(chat_ev_done[i],g_chat_cs)); }
    // кк33
    HIP_CHECK(hipHostMalloc((void**)&g_chat_xh,INTER*4)); HIP_CHECK(hipHostMalloc((void**)&g_chat_yh,INTER*4));
    g_chat_nt=getenv("TIER_CPU_THREADS")?atoi(getenv("TIER_CPU_THREADS")):std::max(1u,std::thread::hardware_concurrency()/2);
    for(int w=0;w<g_chat_nt;++w) std::thread(chat_worker,w).detach();
    ChatMat* big=nullptr; for(auto* m:g_chat_SQ) if(!m->qt && (!big || m->bytes>big->bytes)) big=m;   // кк34
    if(big){
        if(getenv("TIER_CPU")) g_chat_fcpu=atof(getenv("TIER_CPU"));
        if(!cpu_has_avx2() && g_chat_fcpu>0){ fprintf(stderr,"Процессор не поддерживает AVX2: расчёт выполняется только на карте.\n"); g_chat_fcpu=0; }
        else { big->rc=0; double t=chat_now(); for(int r=0;r<3;++r) HIP_CHECK(hipMemcpy(chat_slot[0],big->host,big->bytes,hipMemcpyHostToDevice)); double tb=(chat_now()-t)/3;
            for(int i=0;i<big->cols;++i) g_chat_xh[i]=1e-3f*(i%7); t=chat_now(); for(int r=0;r<3;++r){ chat_cpu_start({big->host,big->f32,(!big->f32 && g_fp16_storage),big->rows,big->cols,g_chat_xh,g_chat_yh}); chat_cpu_wait(); } double tc=(chat_now()-t)/3;
            double bw_pcie=big->bytes/tb, bw_cpu=big->bytes/tc; g_chat_fcpu=bw_cpu/(bw_cpu+bw_pcie);
            fprintf(stderr,"Шина: %.1f ГБ/с; процессор (потоков: %d): %.1f ГБ/с → доля процессора: %.0f%%.\n",bw_pcie/1e9,g_chat_nt,bw_cpu/1e9,100*g_chat_fcpu); }
        for(auto* m:g_chat_SQ){ int rc=m->qt?0:(int)(m->rows*g_chat_fcpu); rc=rc/64*64; m->rc=std::max(0,std::min(rc,m->rows-64)); } }
    if(!g_chat_SQ.empty()){ chat_issue(0); chat_issue(1); }
    std::vector<float> hl(VOCAB); std::mt19937 rng(std::random_device{}());   /* кк35 */
    double g_ent=0, g_top=0; int g_ntok=0;   /* кк36 */
    static const bool DBGC=getenv("TIER_CHAT_DEBUG")!=nullptr; static int dbg_calls=0;   // кк37
    auto dbgv=[&](const char* tag,int l,const float* d,size_t n){ if(!DBGC||dbg_calls>1||!d) return; HIP_CHECK(hipStreamSynchronize(g_chat_cs)); std::vector<float> h(n);
        HIP_CHECK(hipMemcpy(h.data(),d,n*4,hipMemcpyDeviceToHost)); size_t bad=0; double ss=0, mx=0; for(float v:h){ if(!std::isfinite(v)) ++bad; else { ss+=(double)v*v; mx=std::max(mx,(double)fabs(v)); } }
        fprintf(stderr,"[отладка] вызов %d слой %2d %-5s n=%zu нечисел %zu rms %.3e макс %.3e\n",dbg_calls,l,tag,n,bad,sqrt(ss/n),mx); };
    auto step=[&](int m, int pos0){ // кк38
        embed_kernel<<<m,256,0,g_chat_cs>>>(E,dtok,x,m,HIDDEN,A.emb_scale,weight_type(E)); size_t xn=(size_t)m*HIDDEN;
        for(int l=0;l<N_LAYERS;++l){ auto& y=L[l]; auto& N=(*nwp)[l];
            const LayerSpec& sp=A.L[l]; bool fl=y.full; float theta=sp.theta; int rot=sp.rot, win=sp.window;
            auto bias=[&](float* t,const float* b,int n){ if(b){ size_t c=(size_t)m*n; add_bias_typed_k<<<(unsigned)((c+255)/256),256,0,g_chat_cs>>>(t,b,m,n,weight_type(b)); } };
            HIP_CHECK(hipMemcpyAsync(res,x,xn*4,hipMemcpyDeviceToDevice,g_chat_cs));
            if(N[0]) rmsnorm_fwd<<<m, NB_ROW ,0,g_chat_cs>>>(x,N[0],nrm,rr,nn,HIDDEN,weight_type(N[0])); else HIP_CHECK(hipMemcpyAsync(nrm,x,xn*4,hipMemcpyDeviceToDevice,g_chat_cs));
            dbgv("nrm",l,nrm,xn);
            chat_lin(y.m[0],nrm,Q,m); chat_lin(y.m[1],nrm,K,m); bias(Q,y.qb,y.qdim); bias(K,y.kb,y.kvdim);
            dbgv("Q",l,Q,(size_t)m*y.qdim); dbgv("K",l,K,(size_t)m*y.kvdim);
            if(!fl){ chat_lin(y.m[2],nrm,V,m); bias(V,y.vb,y.kvdim); } else HIP_CHECK(hipMemcpyAsync(V,K,(size_t)m*y.kvdim*4,hipMemcpyDeviceToDevice,g_chat_cs));
            if(A.qk_norm && A.qk_full){ rmsnorm_fwd<<<m, NB_ROW ,0,g_chat_cs>>>(Q,N[4],Q,rr,nn,y.qdim,weight_type(N[4])); rmsnorm_fwd<<<m, NB_ROW ,0,g_chat_cs>>>(K,N[5],K,rr,nn,y.kvdim,weight_type(N[5])); }
            else if(A.qk_norm){ head_rmsnorm_fwd<<<dim3(m,NUM_HEADS), NB_HEAD ,0,g_chat_cs>>>(Q,N[4],Q,rr,nn,NUM_HEADS,y.hd,weight_type(N[4]));
                           head_rmsnorm_fwd<<<dim3(m,y.nkv), NB_HEAD ,0,g_chat_cs>>>(K,N[5],K,rr,nn,y.nkv,y.hd,weight_type(N[5])); }
            if(A.v_norm) v_norm_fwd<<<dim3(m,y.nkv), NB_HEAD ,0,g_chat_cs>>>(V,V,rr,nn,y.nkv,y.hd);
            rope_pos<<<dim3(m,NUM_HEADS),64,0,g_chat_cs>>>(Q,m,NUM_HEADS,y.hd,rot,theta,pos0,sp.d_invf,rope_half(y.hd,rot),A.rope_msc);
            rope_pos<<<dim3(m,y.nkv),64,0,g_chat_cs>>>(K,m,y.nkv,y.hd,rot,theta,pos0,sp.d_invf,rope_half(y.hd,rot),A.rope_msc);
            if(A.attn_scale!=1.0f){ size_t c=(size_t)m*y.qdim; scale_k<<<(unsigned)((c+255)/256),256,0,g_chat_cs>>>(Q,A.attn_scale,c); }
            HIP_CHECK(hipMemcpyAsync(Kc[l]+(size_t)pos0*y.kvdim,K,(size_t)m*y.kvdim*4,hipMemcpyDeviceToDevice,g_chat_cs));
            HIP_CHECK(hipMemcpyAsync(Vc[l]+(size_t)pos0*y.kvdim,V,(size_t)m*y.kvdim*4,hipMemcpyDeviceToDevice,g_chat_cs));
            attn_cache<<<dim3(m,NUM_HEADS),128,(size_t)(pos0+m)*4,g_chat_cs>>>(Q,Kc[l],Vc[l],ao,pos0,NUM_HEADS,y.nkv,y.hd,win,A.attn_cap);
            dbgv("ao",l,ao,(size_t)m*y.qdim);
            chat_lin(y.m[3],ao,ap,m);
            dbgv("ap",l,ap,xn);
            if(A.post_attn) rmsnorm_fwd<<<m, NB_ROW ,0,g_chat_cs>>>(ap,N[1],x,rr,nn,HIDDEN,weight_type(N[1])); else HIP_CHECK(hipMemcpyAsync(x,ap,xn*4,hipMemcpyDeviceToDevice,g_chat_cs));
            if(A.resid_mul!=1.f) scale_k<<<(unsigned)((xn+255)/256),256,0,g_chat_cs>>>(x,A.resid_mul,xn);
            add_kernel<<<(xn+255)/256,256,0,g_chat_cs>>>(x,res,xn);
            HIP_CHECK(hipMemcpyAsync(res,x,xn*4,hipMemcpyDeviceToDevice,g_chat_cs));
            if(N[2]) rmsnorm_fwd<<<m, NB_ROW ,0,g_chat_cs>>>(x,N[2],nrm,rr,nn,HIDDEN,weight_type(N[2])); else HIP_CHECK(hipMemcpyAsync(nrm,x,xn*4,hipMemcpyDeviceToDevice,g_chat_cs));
            chat_lin(y.m[4],nrm,g,m); chat_lin(y.m[5],nrm,u,m);
            size_t in=(size_t)m*INTER; if(A.silu) silu_mul_fwd<<<(in+255)/256,256,0,g_chat_cs>>>(g,u,a,in); else gelu_tanh_mul_fwd<<<(in+255)/256,256,0,g_chat_cs>>>(g,u,a,in);
            dbgv("g",l,g,in); dbgv("a",l,a,in);
            chat_lin(y.m[6],a,mo,m);
            dbgv("mo",l,mo,xn);
            if(A.post_ffn) rmsnorm_fwd<<<m, NB_ROW ,0,g_chat_cs>>>(mo,N[3],x,rr,nn,HIDDEN,weight_type(N[3])); else HIP_CHECK(hipMemcpyAsync(x,mo,xn*4,hipMemcpyDeviceToDevice,g_chat_cs));
            if(A.resid_mul!=1.f) scale_k<<<(unsigned)((xn+255)/256),256,0,g_chat_cs>>>(x,A.resid_mul,xn);
            add_kernel<<<(xn+255)/256,256,0,g_chat_cs>>>(x,res,xn); if(N[6]) mul_scalar_typed_kernel<<<(xn+255)/256,256,0,g_chat_cs>>>(x,N[6],xn,weight_type(N[6]));
            dbgv("x",l,x,xn);
        }
        rmsnorm_fwd<<<1, NB_ROW ,0,g_chat_cs>>>(x+(size_t)(m-1)*HIDDEN,fnorm,h,rr,nn,HIDDEN,weight_type(fnorm));
        if(g_fp16_storage) gemv<tier_fp16><<<(VOCAB+256/g_chat_ws-1)/(256/g_chat_ws),256,0,g_chat_cs>>>((const tier_fp16*)HD,h,logits,VOCAB,HIDDEN);
        else gemv<hip_bfloat16><<<(VOCAB+256/g_chat_ws-1)/(256/g_chat_ws),256,0,g_chat_cs>>>((const hip_bfloat16*)HD,h,logits,VOCAB,HIDDEN);
        if(A.logit_div!=1.f) scale_k<<<(VOCAB+255)/256,256,0,g_chat_cs>>>(logits,1.f/A.logit_div,VOCAB);
        if(SOFTCAP>0) softcap_kernel<<<(VOCAB+255)/256,256,0,g_chat_cs>>>(logits,SOFTCAP,VOCAB);
        dbgv("h",99,h,HIDDEN); dbgv("logit",99,logits,VOCAB); if(DBGC) ++dbg_calls;
        HIP_CHECK(hipMemcpyAsync(hl.data(),logits,VOCAB*4,hipMemcpyDeviceToHost,g_chat_cs)); HIP_CHECK(hipStreamSynchronize(g_chat_cs));
    };
    // кк39
    struct Samp { float temp=0, top_p=1, min_p=0, rep=1, pres=0, freq=0; int top_k=0, rep_n=256; long long seed=-1; };
    std::vector<int> cand; std::vector<double> spr;
    auto pick=[&](const Samp& S, const std::vector<int32_t>& hist){
        if(S.rep!=1.0f || S.pres!=0 || S.freq!=0){ std::unordered_map<int,int> cnt; int a=std::max(0,(int)hist.size()-S.rep_n);
            for(int i=a;i<(int)hist.size();++i){ int t=hist[i]; if(t>=0&&t<VOCAB) ++cnt[t]; }
            for(auto& [t,c] : cnt){ if(S.rep!=1.0f) hl[t]= hl[t]>0 ? hl[t]/S.rep : hl[t]*S.rep;     // кк40
                hl[t]-= S.pres + S.freq*c; } }                                                      // кк41
        int best=0; for(int v=1;v<VOCAB;++v) if(hl[v]>hl[best]) best=v;
        if(S.temp<=0) return best;
        float mx=hl[best]; spr.resize(VOCAB); double z=0; for(int v=0;v<VOCAB;++v){ spr[v]=exp((hl[v]-mx)/S.temp); z+=spr[v]; }
        double e=0, tp=0; for(int v=0;v<VOCAB;++v){ double q=spr[v]/z; if(q>tp) tp=q; if(q>1e-12) e-=q*log(q); }
        g_ent+=e; g_top+=tp; ++g_ntok;                                     // кк42
        cand.clear(); double floor_=S.min_p>0 ? S.min_p*spr[best] : 0;      // кк43
        for(int v=0;v<VOCAB;++v) if(spr[v]>=floor_ && spr[v]>0) cand.push_back(v);
        auto gt=[&](int a,int b){ return spr[a]>spr[b]; };
        if(S.top_k>0 && (int)cand.size()>S.top_k){ std::nth_element(cand.begin(),cand.begin()+S.top_k,cand.end(),gt); cand.resize(S.top_k); }
        if(S.top_p<1.0f){ std::sort(cand.begin(),cand.end(),gt); double tot=0; for(int v:cand) tot+=spr[v];
            double acc=0; size_t k=0; for(;k<cand.size();++k){ acc+=spr[cand[k]]; if(acc>=S.top_p*tot){ ++k; break; } } cand.resize(std::max<size_t>(1,std::min(k,cand.size()))); }
        double zz=0; for(int v:cand) zz+=spr[v];
        double r=std::uniform_real_distribution<double>(0,zz)(rng); for(int v:cand){ r-=spr[v]; if(r<=0) return v; }
        return cand.back(); };
    // кк44
    std::vector<int32_t> ctx; char line[1024];
    fprintf(stderr,"Готово к вопросам.\n"); printf("ready\n");
    // кк45
    float g_scale=1.0f; std::vector<float> blend; 
    auto set_scale=[&](float k){
        if(!lora_ab.empty()){ g_chat_lk=k; g_scale=k; ctx.clear(); return; }   // кк46
        if(t1.empty()) return; HIP_CHECK(hipDeviceSynchronize()); g_scale=k;
        for(auto& y:L) for(auto& m:y.m) if(m.hr){   // кк47
            size_t nh=(size_t)m.hr*m.cols; std::vector<uint16_t> b(nh);
            for(size_t j=0;j<nh;++j){ float v=m.hB[j]+k*(m.hT[j]-m.hB[j]); if(g_fp16_storage) b[j]=tier_fp16(v).bits; else { uint32_t u; memcpy(&u,&v,4); b[j]=(uint16_t)((u+0x7fffu+((u>>16)&1u))>>16); } }
            HIP_CHECK(hipMemcpy(m.hd,b.data(),nh*2,hipMemcpyHostToDevice)); }
        for(auto& y:L) for(auto& m:y.m){ if(!m.hostB || !m.f32) continue;
            if(!m.hostT){ m.hostT=malloc(m.n*4); if(!m.hostT){ fprintf(stderr,"Недостаточно ОЗУ для хранения исходной версии обученной части.\n"); return; } memcpy(m.hostT,m.host,m.n*4); }
            const float* tr=(const float*)m.hostT; const uint16_t* ba=(const uint16_t*)m.hostB;
            blend.resize(m.n);
            std::vector<std::thread> th; int nt=std::max(1u,std::thread::hardware_concurrency());
            for(int w=0;w<nt;++w) th.emplace_back([&,w]{ for(size_t i=m.n*w/nt;i<m.n*(w+1)/nt;++i){ float b=g_fp16_storage ? (float)tier_fp16::to_float(ba[i]) : q_b2f(ba[i]); blend[i]=b+k*(tr[i]-b); } });
            for(auto& t:th) t.join();
            if(m.dev) HIP_CHECK(hipMemcpy(m.dev,blend.data(),m.n*4,hipMemcpyHostToDevice));
            memcpy(m.host,blend.data(),m.n*4); } // кк48
        ctx.clear(); if(!g_chat_SQ.empty()){ chat_issue(qi); chat_issue(qi+1); } HIP_CHECK(hipDeviceSynchronize()); };
    auto set_mode=[&](bool tr){ // кк49
        if(!lora_ab.empty()){ if(tr!=trained){ trained=tr; g_chat_lon=tr; ctx.clear(); } return; }   // кк50
        if(tr==trained || t1.empty()) return; HIP_CHECK(hipDeviceSynchronize()); trained=tr; g_chat_heads=tr;   // кк51
        for(auto& y:L) for(auto& m:y.m) if(m.hostB){ std::swap(m.host,m.hostB); m.f32=tr; m.bytes=m.n*(tr?4:2); if(m.dev) HIP_CHECK(hipMemcpy(m.dev,m.host,m.bytes,hipMemcpyHostToDevice)); }
        nwp=tr?&nwT:&nwB; fnorm=tr?fnormT:fnormB; ctx.clear();
        if(!g_chat_SQ.empty()){ chat_issue(qi); chat_issue(qi+1); } HIP_CHECK(hipDeviceSynchronize()); }; // кк52
    while(fgets(line,sizeof line,stdin)){
        if(!strncmp(line,"cpu ",4)){ HIP_CHECK(hipDeviceSynchronize()); g_chat_fcpu=atof(line+4); // кк53
            for(auto* m:g_chat_SQ){ int rc=m->qt?0:(int)(m->rows*g_chat_fcpu); rc=rc/64*64; m->rc=std::max(0,std::min(rc,m->rows-64)); }
            if(!g_chat_SQ.empty()){ chat_issue(qi); chat_issue(qi+1); } HIP_CHECK(hipDeviceSynchronize()); fprintf(stderr,"Доля процессора: %.0f%%.\n",100*g_chat_fcpu); printf("cpu_ok\n"); continue; }
        if(!strncmp(line,"scale ",6)){ float k=atof(line+6); set_scale(k); fprintf(stderr,"Сила обучения: %.2f.\n",k); printf("scale_ok %.2f\n",k); continue; }
        if(!strncmp(line,"mode ",5)){ set_mode(line[5]=='t'); printf("mode_ok %s\n",trained?"trained":"base"); continue; }
        // кк54
        char path[900]; int n,maxn_; Samp SP; if(sscanf(line,"%899s %d %d %f %d %f %f %f %d %lld %f %f",path,&n,&maxn_,&SP.temp,&SP.top_k,&SP.top_p,&SP.min_p,&SP.rep,&SP.rep_n,&SP.seed,&SP.pres,&SP.freq)<4) continue;
        if(SP.seed>=0) rng.seed((unsigned)SP.seed);
        std::vector<int32_t> tk(n); { FILE* f=fopen(path,"rb"); if(!f){ perror(path); printf("gen_end\n"); continue; } fread(tk.data(),4,n,f); fclose(f); }
        if(n+maxn_>CTX){ fprintf(stderr,"Запрос и ответ (%d) превышают контекст (%d).\n",n+maxn_,CTX); printf("gen_end\n"); continue; }
        int cp=0; while(cp<(int)ctx.size() && cp<n-1 && ctx[cp]==tk[cp]) ++cp;
        ctx=tk; double t1=chat_now();
        g_ent=g_top=0; g_ntok=0;
        int nt=0; for(int o=cp;o<n;o+=PC){ int c=std::min(PC,n-o); HIP_CHECK(hipMemcpy(dtok,tk.data()+o,c*4,hipMemcpyHostToDevice)); step(c,o); } nt=pick(SP,tk); /* кк55 */ double tp=chat_now()-t1; printf("gen_token %d\n",nt); int made=1; double t2=chat_now();
        while(made<maxn_ && !is_eos(nt)){ ctx.push_back(nt); HIP_CHECK(hipMemcpy(dtok,&nt,4,hipMemcpyHostToDevice)); step(1,(int)ctx.size()-1); nt=pick(SP,ctx); printf("gen_token %d\n",nt); ++made; }
        double td=chat_now()-t2; if(g_ntok) fprintf(stderr,"Уверенность: средняя энтропия %.2f бита, вероятность лучшего токена %.0f%%.\n",g_ent/g_ntok/log(2.0),100*g_top/g_ntok);
        printf("gen_end\n");
        fprintf(stderr,"Запрос %d токенов (из них %d уже в кэше): %.2f с | Ответ %d токенов: %.2f с (%.2f с/токен)\n",n,cp,tp,made,td,made>1?td/(made-1):0);
    }
    return 0;
}


// кл1
static int quant_main(int argc, char** argv){
    if(argc<5){ fprintf(stderr,"%s --quantize <model.safetensors> <out.tq> <Q8_0|Q6_K|Q5_K_M|Q5_K_S|Q4_K_M|Q4_K_S>\n",argv[0]); return 1; }
    parse_header(argv[2]); arch_load(argv[2]); std::string out=argv[3], rec=argv[4], tmp=out+".part";
    int base= rec=="Q8_0"?QT_Q8_0 : rec=="Q6_K"?QT_Q6_K : rec.rfind("Q5_K",0)==0?QT_Q5_K : rec.rfind("Q4_K",0)==0?QT_Q4_K : -1;
    if(base<0){ fprintf(stderr,"Неизвестный рецепт квантования %s. Доступные: Q8_0, Q6_K, Q5_K_M, Q5_K_S, Q4_K_M, Q4_K_S.\n",rec.c_str()); return 1; }
    const bool mm = rec.size()>2 && rec.compare(rec.size()-2,2,"_M")==0, ms = rec.size()>2 && rec.compare(rec.size()-2,2,"_S")==0;
    struct It{ std::string nm; int type, rows, cols; const TensorRef* r; uint64_t off; double err; };
    std::vector<It> L; int nl=0, nv=0, iv=0;
    nl=N_LAYERS; for(int l=0;l<nl;++l) if(A.L[l].has_v) ++nv;   // кл2
    auto more=[](int i, int n){ return i<n/8 || i>=7*n/8 || (i-n/8)%3==2; };   // кл3
    for(int l=0;l<nl;++l) for(int i=0;i<7;++i){ if(i==2 && !A.L[l].has_v) continue; std::string nm=LPX(l)+arch_tn(SLOT7[i]);
        auto it=g_tensors.find(nm); if(it==g_tensors.end()) continue;
        int type=base;
        if(i==2){ if(mm && more(iv,nv)) type=QT_Q6_K; else if(ms && base==QT_Q4_K && iv<4) type=QT_Q5_K; ++iv; }
        if(i==6){ if(mm && more(l,nl)) type=QT_Q6_K; else if(ms && base==QT_Q4_K && l<nl/8) type=QT_Q5_K; }
        L.push_back({nm,type,0,0,&it->second,0,0}); }
    // кл4
    for(auto& e : L){ size_t n=e.r->size/2; int l=atoi(e.nm.c_str()+A.lp.size()); int hd=(l>=0&&l<(int)A.L.size())?A.L[l].hd:HIDDEN/NUM_HEADS;
        int cols= e.nm.find("down_proj")!=std::string::npos ? INTER : e.nm.find("o_proj")!=std::string::npos ? NUM_HEADS*hd : HIDDEN;
        if(e.r->cols>0) cols=(int)e.r->cols;   // кл5
        e.cols=cols; e.rows=(int)(n/cols);
        if((size_t)e.rows*cols!=n || !q_cols_ok(e.type,cols)){ fprintf(stderr,"%s: форма %zu/%d не делится на блоки %s.\n",e.nm.c_str(),n,cols,q_info(e.type)->name); return 1; } }
    char b[512]; size_t data=0;
    for(auto& e : L){ e.off=data; data+=(q_row_bytes(e.type,e.cols)*e.rows+4095)/4096*4096; }
    size_t hb=(L.size()*200+4096+4095)/4096*4096;   // кл6
    for(auto& e : L) e.off+=hb;
    struct stat st; stat(argv[2],&st);
    int fi=open(argv[2],O_RDONLY); int fo=open(tmp.c_str(),O_RDWR|O_CREAT|O_TRUNC,0644); if(fi<0||fo<0){ perror("Квантование"); return 1; }
    if(ftruncate(fo,hb+data)!=0){ perror("ftruncate"); return 1; }
    int NT=std::max(1u,std::thread::hardware_concurrency()); double t0=chat_now(), sq=0, se=0; size_t done=0, tot=0; for(auto& e:L) tot+=e.r->size;
    fprintf(stderr,"Квантование %s: матриц %zu, потоков %d.\n",rec.c_str(),L.size(),NT);
    for(size_t k=0;k<L.size();++k){ auto& e=L[k]; size_t n=(size_t)e.rows*e.cols, rb=q_row_bytes(e.type,e.cols);
        std::vector<uint16_t> src(n); pread_all(fi,src.data(),n*2,e.r->offset);
        std::vector<uint8_t> q(rb*e.rows); std::vector<double> es(NT,0), ss(NT,0); std::vector<std::thread> th;
        for(int w=0;w<NT;++w) th.emplace_back([&,w]{ std::vector<float> x(e.cols);
            for(int r=(int)((size_t)e.rows*w/NT); r<(int)((size_t)e.rows*(w+1)/NT); ++r){ const uint16_t* s=src.data()+(size_t)r*e.cols;
                for(int c=0;c<e.cols;++c) x[c]=q_b2f(s[c]);
                q_row(e.type,x.data(),q.data()+(size_t)r*rb,e.cols);
                for(int c=0;c<e.cols;++c){ double d=q_val(q.data()+(size_t)r*rb,e.type,c)-x[c]; es[w]+=d*d; ss[w]+=(double)x[c]*x[c]; } } });
        for(auto& t:th) t.join(); double E=0,S=0; for(int w=0;w<NT;++w){ E+=es[w]; S+=ss[w]; } e.err=S>0?std::sqrt(E/S):0; se+=E; sq+=S;
        pwrite_all(fo,q.data(),q.size(),e.off); done+=e.r->size;
        fprintf(stderr,"Квантование %zu/%zu %s %s, ошибка %.2f%% [%.0f с]\n",k+1,L.size(),e.nm.c_str()+28,q_info(e.type)->name,100*e.err,chat_now()-t0); }
    std::string h; snprintf(b,sizeof b,"TQB1 %zu %zu\n#from tierlm %s\n#src %lld %lld\n#err %.6f\n",hb,L.size(),rec.c_str(),(long long)st.st_size,(long long)st.st_mtime,sq>0?std::sqrt(se/sq):0.0); h=b;
    for(auto& e : L){ snprintf(b,sizeof b,"%s %s %d %d %llu %llu %.6f\n",e.nm.c_str(),q_info(e.type)->name,e.rows,e.cols,(unsigned long long)e.off,(unsigned long long)(q_row_bytes(e.type,e.cols)*e.rows),e.err); h+=b; }
    if(h.size()>=hb){ fprintf(stderr,"Заголовок квантованной базы не помещается.\n"); return 1; } h.resize(hb,'\0'); pwrite_all(fo,h.data(),hb,0);
    fsync(fo); close(fo); close(fi);
    if(rename(tmp.c_str(),out.c_str())!=0){ perror("rename"); return 1; }
    fprintf(stderr,"Готово: %s, %.2f ГиБ, общая ошибка весов %.2f%%, %.0f с.\n",out.c_str(),(hb+data)/1073741824.0,sq>0?100*std::sqrt(se/sq):0.0,chat_now()-t0);
    return 0; }

// км1
static int tqcheck_main(int argc, char** argv){
    if(argc<4){ fprintf(stderr,"%s --tq-check <model.safetensors> <база.tq>\n",argv[0]); return 1; }
    parse_header(argv[2]); arch_load(argv[2]); if(!tq_load(argv[3])) return 1;
    int fi=open(argv[2],O_RDONLY), fq=open(argv[3],O_RDONLY); if(fi<0||fq<0){ perror("Сверка"); return 1; }
    double SE=0,SS=0, worst=1; std::string wn; size_t k=0, bad=0;
    for(auto& [nm,q] : g_tq){ ++k; auto it=g_tensors.find(nm);
        if(it==g_tensors.end() || it->second.size!=(uint64_t)q.rows*q.cols*2 || (it->second.cols && it->second.cols!=q.cols)){ printf("Сверка %zu/%zu %s: в модели нет такой матрицы или у неё другая форма.\n",k,g_tq.size(),nm.c_str()); ++bad; continue; }
        int S=std::min(q.rows,48); double se=0,ss=0,sxy=0,sx=0,sy=0,sxx=0,syy=0; size_t n=0;
        std::vector<uint16_t> o(q.cols); std::vector<uint8_t> b(q.rb);
        for(int j=0;j<S;++j){ int r=(int)((int64_t)j*(q.rows-1)/std::max(1,S-1));
            pread_all(fi,o.data(),(size_t)q.cols*2,it->second.offset+(size_t)r*q.cols*2); pread_all(fq,b.data(),q.rb,q.offset+(size_t)r*q.rb);
            for(int c=0;c<q.cols;++c){ double x=q_b2f(o[c]), y=q_val(b.data(),q.type,c), d=y-x; se+=d*d; ss+=x*x; sx+=x; sy+=y; sxy+=x*y; sxx+=x*x; syy+=y*y; ++n; } }
        double cov=sxy-sx*sy/n, vx=sxx-sx*sx/n, vy=syy-sy*sy/n, cr=(vx>0&&vy>0)?cov/std::sqrt(vx*vy):0, er=ss>0?std::sqrt(se/ss):0;
        SE+=se; SS+=ss; if(cr<worst){ worst=cr; wn=nm; } if(cr<0.5) ++bad;
        printf("Сверка %zu/%zu %s %s, ошибка %.2f%%, корреляция %.4f\n",k,g_tq.size(),nm.c_str()+(nm.compare(0,A.lp.size(),A.lp)?0:A.lp.size()),q_info(q.type)->name,100*er,cr); }
    printf("Итог: ошибка весов %.2f%%, худшая корреляция %.4f (%s)%s\n",SS>0?100*std::sqrt(SE/SS):0.0,worst,wn.c_str(),bad?", обнаружены матрицы от другой модели":"");
    close(fi); close(fq); return bad?2:0; }

int main(int argc, char** argv){
    if(argc>1 && std::string(argv[1])=="--quantize") return quant_main(argc,argv);   // кн1
    if(argc>1 && std::string(argv[1])=="--tq-check") return tqcheck_main(argc,argv);
    // кн2
    for(int i=1;i<argc;++i) if(std::string(argv[i])=="--chat"){
        std::vector<char*> a; a.push_back(argv[0]);
        for(int j=1;j<argc;++j) if(j!=i) a.push_back(argv[j]);
        return chat_main((int)a.size(), a.data()); }
    if(argc<6){ fprintf(stderr,
        "Обучение: %s <model.safetensors> <token_ids.bin> <n_tokens> <adam_state.bin> <tier1.bin> [lr] [слоёв] [шаг] [ключи]\n"
        "Разговор: %s --chat <model.safetensors> <tier1.bin|-> <раскладка|-> [контекст]\n", argv[0], argv[0]); return 1; }
    setvbuf(stdout,nullptr,_IOLBF,0);
    { FILE* o=fopen("/proc/self/oom_score_adj","w"); if(o){ fputs("300\n",o); fclose(o); } } // кн3
    gpu_pick();    // кн4
    parse_header(argv[1]); arch_load(argv[1]);
    { std::string mp=argv[1]; size_t e=mp.find_last_of('/'); std::string dir= e==std::string::npos ? std::string(".") : mp.substr(0,e); size_t b=dir.find_last_of('/'); g_model_id = b==std::string::npos ? dir : dir.substr(b+1); }
    for(int i=1;i+1<argc;++i) if(std::string(argv[i])=="--pack") g_probe_pack=atoi(argv[i+1]);
    for(int i=1;i+2<argc;++i) if(std::string(argv[i])=="--calibrate-conf"){ g_cal_path=argv[i+1]; g_cal_model=argv[i+2]; g_rbf_off=true; }   // кн5
    conf_init();   // кн6
    arch_device();
    int seq_len = atoi(argv[3]), t0 = 0; // кн7
    g_adam_path = argv[4];
    g_checkpoint_path = argv[5];
    float lr = argc>6 ? atof(argv[6]) : 1e-5f;
    N_LAYERS = std::min(g_model_layers, argc>7 ? atoi(argv[7]) : g_model_layers);   // кн8
    int step0 = argc>8 ? atoi(argv[8]) : 1, n_steps=1;
    for(int i=1;i<argc;++i) if(std::string(argv[i])=="-v") VERBOSE=true;
    for(int i=1;i<argc;++i) if(std::string(argv[i])=="--gdiag") g_gdiag=true;
    for(int i=1;i<argc;++i) if(std::string(argv[i])=="--eval") g_eval=true;
    for(int i=1;i<argc;++i) if(std::string(argv[i])=="--progress") g_prog=true;
    for(int i=1;i<argc;++i) if(std::string(argv[i])=="--arena") g_use_arena=true;
    for(int i=1;i<argc;++i) if(std::string(argv[i])=="--hipblas") g_use_lt=g_use_own=false;
    for(int i=1;i<argc;++i) if(std::string(argv[i])=="--no-own") g_use_own=false; // кн9
    bool force_no_bf3=false;
    for(int i=1;i<argc;++i) if(std::string(argv[i])=="--bf3") g_bf3=true;
    for(int i=1;i<argc;++i) if(std::string(argv[i])=="--no-bf3"){ g_bf3=false; force_no_bf3=true; } // кн10
    const bool fp16_x3_available=fp16_probe();
    const bool bf16_x3_available=bf3_probe();
    g_fp16_storage=true; // кн11
    g_use_fp16_x3=false; g_use_bf16_x3=false;
    g_conf.fp16=fp16_x3_available?1:0; // кн12
    if(force_no_bf3) g_bf3=false;
    else if(g_bf3 && fp16_x3_available) g_use_fp16_x3=true;
    else if(g_bf3 && bf16_x3_available) g_use_bf16_x3=true;
    else if(g_bf3){ fprintf(stderr,"FP16×3 и BF16×3 недоступны: используется хранение FP16 с запасным режимом FP32.\n"); g_bf3=false; }
    if(!TIER_LT){ g_bf3=false; g_use_fp16_x3=g_use_bf16_x3=false; g_use_lt=false; }
    if(g_conf_fresh){   // кн13
        size_t fr=0,tt=0; if(hipMemGetInfo(&fr,&tt)==hipSuccess && tt>fr){ size_t others=(tt-fr)>>20, step=model_vram_need(g_probe_pack)>>20, slack=std::max<size_t>(vram_slack(),256ull<<20)>>20;
            size_t lazy = std::max<size_t>(others*3, 256);   // кн14
            g_conf.headroom_mib = others+lazy+step+slack; g_conf.tier1_budget = (g_conf.vram_mib>g_conf.headroom_mib ? g_conf.vram_mib-g_conf.headroom_mib : 0)*1048576ULL/12;
            g_snap_bytes=(size_t)g_conf.tier1_budget*12;
            fprintf(stderr,"Запас видеопамяти после проб: %zu МиБ (занято средой выполнения и другими %zu, ещё ядра по ходу шага %zu, рабочая память шага %zu, потолок памяти %zu) → обучаемая часть %zu параметров.\n",
                    g_conf.headroom_mib,others,lazy,step,slack,g_conf.tier1_budget); } }
    conf_save(g_conf,g_conf_path);
    const char* x3_mode=g_use_fp16_x3?"FP16×3":(g_use_bf16_x3?"BF16×3 через временное преобразование FP16→BF16":"отключён (запасной режим FP32)");
    fprintf(stderr,"Хранение нормировок, эмбеддингов и замороженных весов: FP16; режим ×3: %s.\n",x3_mode);
    { const char* dt=getenv("TIER_DT"); if(dt && !strcmp(dt,"fp16") && !fp16_x3_available)
        fprintf(stderr,"TIER_DT=fp16: хранение FP16 включено, проверка FP16×3 не пройдена; выбран %s.\n",g_use_bf16_x3?"запасной режим BF16×3":"запасной режим FP32"); }
    for(int i=1;i<argc;++i) if(std::string(argv[i])=="--no-prefetch") g_use_pf=false; // кн15
    for(int i=1;i<argc;++i) if(std::string(argv[i])=="--prefetch") g_use_pf=g_use_arena=true; // кн16
    for(int i=1;i<argc;++i) if(std::string(argv[i])=="--gen" && i+1<argc){ g_gen=atoi(argv[i+1]); g_eval=true; }
    for(int i=1;i<argc;++i) if(std::string(argv[i])=="--temp" && i+1<argc) g_temp=atof(argv[i+1]);
    if(g_gen) n_steps=g_gen;
    for(int i=1;i<argc;++i) if(std::string(argv[i])=="--log" && i+1<argc){ g_log=fopen(argv[i+1],"w"); if(!g_log){ perror(argv[i+1]); return 1; } g_gdiag=VERBOSE=true; }
    for(int i=1;i+1<argc;++i) if(std::string(argv[i])=="--lr-find") g_lrfind=atoi(argv[i+1]);
    for(int i=1;i+1<argc;++i) if(std::string(argv[i])=="--stop-after") g_stop_after=atoi(argv[i+1]);
    for(int i=1;i+1<argc;++i) if(std::string(argv[i])=="--cpu-share"){ g_cpu_thr=atoi(argv[i+1]);
        if(!cpu_has_avx2()){ fprintf(stderr,"Процессор не поддерживает AVX2: доля процессора отключена.\n"); g_cpu_thr=0; } else pool_init(); }
    for(int i=1;i+1<argc;++i) if(std::string(argv[i])=="--first-weight") g_w1=atoi(argv[i+1]);
    for(int i=1;i+1<argc;++i) if(std::string(argv[i])=="--damp") g_damp=atof(argv[i+1]);
    for(int i=1;i<argc;++i) if(std::string(argv[i])=="--retier") g_retier=true;
    for(int i=1;i<argc;++i) if(std::string(argv[i])=="--row-global") g_row_global=true;
    for(int i=1;i+1<argc;++i) if(std::string(argv[i])=="--mem-cap") g_cap_pct=atoi(argv[i+1]);
    { HostMem hm=host_mem(); size_t vf=0,vt=0; if(hipMemGetInfo(&vf,&vt)==hipSuccess && hm.total) fprintf(stderr,"Потолок памяти %d%%: ОЗУ свободно %.1f из %.1f ГиБ (системе остаётся не меньше %.1f), видеопамять свободна %.0f из %.0f МиБ (остаётся не меньше %.0f).\n",g_cap_pct,hm.avail/1073741824.0,hm.total/1073741824.0,cap_slack(hm.total)/1073741824.0,vf/1048576.0,vt/1048576.0,cap_slack(vt)/1048576.0); }
    for(int i=1;i<argc;++i) if(std::string(argv[i])=="--nsgd") g_nsgd=true;
    for(int i=1;i<argc;++i) if(std::string(argv[i])=="--pos-dw") g_posdw=true;
    for(int i=1;i<argc;++i) if(std::string(argv[i])=="--grad-probe") g_gprobe=true;
    for(int i=1;i+1<argc;++i) if(std::string(argv[i])=="--row-probe") g_rowprobe=argv[i+1];
    for(int i=1;i+1<argc;++i) if(std::string(argv[i])=="--whiten") g_wh_path=argv[i+1];
    for(int i=1;i+2<argc;++i) if(std::string(argv[i])=="--dump-act"){   // кн17
        std::string spec=argv[i+1], pre=argv[i+2]; size_t a=0;
        while(a<spec.size()){ size_t b=spec.find(';',a); if(b==std::string::npos) b=spec.size(); std::string part=spec.substr(a,b-a); a=b+1;
            size_t c=part.find(':'); if(c==std::string::npos) continue; std::string kind=part.substr(0,c), ls=part.substr(c+1);
            std::string nm = kind=="down" ? "down_proj" : kind=="o" ? "o_proj" : kind=="qkv" ? "qkv" : kind;
            size_t p=0; while(p<ls.size()){ size_t q=ls.find(',',p); if(q==std::string::npos) q=ls.size(); int L=atoi(ls.substr(p,q-p).c_str()); p=q+1;
                std::string key="layer_"+std::to_string(L)+"."+nm, path=pre+"."+key+".bf16";
                FILE* f=fopen(path.c_str(),"wb"); if(!f){ perror(path.c_str()); return 1; } g_dump_act[key]=f; } } }
    for(int i=1;i+1<argc;++i) if(std::string(argv[i])=="--slice-range"){        // кн18
        const char* s=argv[i+1]; g_sl_lo=atoi(s); const char* c=strchr(s,':'); if(c) g_sl_hi=atoi(c+1); }
    for(int i=1;i<argc;++i) if(std::string(argv[i])=="--tier-file" && i+1<argc) g_tier_file_path = argv[i+1];
    for(int i=1;i<argc;++i) if(std::string(argv[i])=="--steps" && i+1<argc) n_steps = atoi(argv[i+1]);
    for(int i=1;i<argc;++i) if(std::string(argv[i])=="--train-tier2") g_train_tier2=true;
    for(int i=1;i<argc;++i) if(std::string(argv[i])=="--train-tier3") g_train_tier3=true;
    for(int i=1;i+1<argc;++i) if(std::string(argv[i])=="--qbase"){ if(!tq_load(argv[i+1])) return 1;   // кн19
        for(auto& [k,q] : g_tq){ auto it=g_tensors.find(k); if(it==g_tensors.end() || it->second.size!=(uint64_t)q.rows*q.cols*2 || (it->second.cols && it->second.cols!=q.cols)){ fprintf(stderr,"Квантованная база не от этой модели: %s.\n",k.c_str()); return 1; } } }
    for(int i=1;i+1<argc;++i) if(std::string(argv[i])=="--lora" && !lora_parse(argv[i+1])){
        fprintf(stderr,"Формат --lora: r,alpha,матрицы[,dropout]; например 16,32,qkvogud,0.05 (q k v o — внимание, g u d — gate up down).\n"); return 1; }

    hwmon_init(); fprintf(stderr, g_hwmon.empty() ? "Датчики hwmon amdgpu не найдены, температуры недоступны.\n" : "Датчики hwmon: %s.\n", g_hwmon.c_str());
    auto t_start = std::chrono::steady_clock::now();
    auto elapsed = [&](){ return std::chrono::duration<double>(std::chrono::steady_clock::now()-t_start).count(); };

    // кн20
    std::vector<std::vector<int32_t>> ds, ds_lab; std::vector<int> ds_p;
    auto read_ex=[](FILE* f, std::vector<std::vector<int32_t>>& D, std::vector<std::vector<int32_t>>& LB, std::vector<int>& P){
        int len,pl; fread(&len,4,1,f); fread(&pl,4,1,f); D.emplace_back(len); fread(D.back().data(),4,len,f); LB.emplace_back();
        if(pl<0){ pl=-pl-1; LB.back().resize(len); fread(LB.back().data(),4,len,f); } P.push_back(pl); };
    for(int i=1;i<argc;++i) if(std::string(argv[i])=="--data" && i+1<argc){
        FILE* f=fopen(argv[i+1],"rb"); if(!f){ perror(argv[i+1]); return 1; }
        int n; fread(&n,4,1,f);
        for(int k=0;k<n;++k) read_ex(f,ds,ds_lab,ds_p);
        fclose(f); printf("Набор данных %s: примеров %zu.\n", argv[i+1], ds.size());
    }
    int g_pack=0; for(int i=1;i<argc;++i) if(std::string(argv[i])=="--pack" && i+1<argc) g_pack=atoi(argv[i+1]); // кн21
    std::vector<std::vector<int>> packs;
    if(g_pack>0 && !ds.empty()){ std::vector<int> cur; int ct=0;
        for(int k=0;k<(int)ds.size();++k){ int L=ds[k].size(); if(!cur.empty() && ct+L>g_pack){ packs.push_back(cur); cur.clear(); ct=0; } cur.push_back(k); ct+=L; }
        if(!cur.empty()) packs.push_back(cur); printf("Упаковок: %zu (до %d токенов в шаге).\n", packs.size(), g_pack); }
    // кн22
    std::vector<std::vector<int32_t>> ds_e, ds_elab; std::vector<int> ds_ep; FILE* dump_f=nullptr;
    std::vector<std::vector<int>> packs_e; std::vector<int> ev_end; int eval_every=0, flush_every=0; bool eval_first=false, shuffle=false; std::string step_file, best_path;   // кн23
    for(int i=1;i+1<argc;++i) if(std::string(argv[i])=="--dump-pred"){ dump_f=fopen(argv[i+1],"wb"); if(!dump_f){ perror(argv[i+1]); return 1; } } // кн24
    for(int i=1;i<argc;++i){ std::string a=argv[i]; bool nx=i+1<argc;
        if(a=="--eval-every"&&nx) eval_every=atoi(argv[i+1]); if(a=="--flush-every"&&nx) flush_every=atoi(argv[i+1]); if(a=="--eval-first") eval_first=true; if(a=="--shuffle") shuffle=true;
        if(a=="--step-file"&&nx) step_file=argv[i+1]; if(a=="--best"&&nx) best_path=argv[i+1];
        if(a=="--eval-data"&&nx){ FILE* f=fopen(argv[i+1],"rb"); if(!f){ perror(argv[i+1]); return 1; } int n; fread(&n,4,1,f);
            const int b0=(int)ds_e.size(), p0=(int)packs_e.size(); for(int k=0;k<n;++k) read_ex(f,ds_e,ds_elab,ds_ep); fclose(f);
            std::vector<int> cur; int ct=0; for(int k=b0;k<b0+n;++k){ int L=ds_e[k].size(); if(!cur.empty() && (g_pack<=0 || ct+L>g_pack)){ packs_e.push_back(cur); cur.clear(); ct=0; } cur.push_back(k); ct+=L; }
            if(!cur.empty()) packs_e.push_back(cur); ev_end.push_back((int)packs_e.size());
            printf("Набор для валидации %s: примеров %d, упаковок %zu.\n", argv[i+1], n, packs_e.size()-p0); } }
    const bool eval_mode=g_eval; double best_loss=1e30; std::vector<int> perm; int perm_epoch=-1;
    { hipDeviceProp_t dp; size_t lim=0; if(hipGetDeviceProperties(&dp,g_dev)==hipSuccess) lim=dp.sharedMemPerBlock/sizeof(float);
      size_t mx=0; auto len_of=[&](const std::vector<std::vector<int32_t>>& D, const std::vector<std::vector<int>>& P){ if(P.empty()){ for(auto& e:D) mx=std::max(mx,e.size()); } else for(auto& pk:P){ size_t t=0; for(int k:pk) t+=D[k].size(); mx=std::max(mx,t); } };
      len_of(ds,packs); len_of(ds_e,packs_e);
      if(lim && mx>lim){ fprintf(stderr,"Упаковка %zu токенов длиннее, чем помещается во внимание на этой карте (%zu позиций: %zu КиБ разделяемой памяти на блок). Требуется меньшая упаковка (PACK) или более короткие примеры.\n",mx,lim,(size_t)dp.sharedMemPerBlock>>10); return 1; } }
    std::vector<std::pair<int,int>> sched; // кн25
    auto add_eval=[&](int st){ for(int e=0;e<(int)packs_e.size();++e) sched.push_back({st,e}); };
    if(!packs_e.empty() && eval_first && step0==1) add_eval(0);
    for(int si=0; si<n_steps; ++si){ int st=step0+si; sched.push_back({st,-1}); if(!packs_e.empty() && eval_every>0 && (st%eval_every==0 || si==n_steps-1)) add_eval(st); }
    std::vector<int> pos_h, seg_h, cur_ex; std::vector<int32_t> tg_h; std::vector<unsigned char> lm_h; int *d_pos=nullptr,*d_seg=nullptr; size_t d_pcap=0;
    std::vector<int32_t> tokens(seq_len);
    if(ds.empty()){ FILE* tf=fopen(argv[2],"rb"); if(!tf){ perror(argv[2]); return 1; } fread(tokens.data(),sizeof(int32_t),seq_len,tf); fclose(tf); }

    // кн26
    { struct stat st; g_checkpoint_exists_with_data = (stat(g_checkpoint_path.c_str(), &st) == 0 && st.st_size > 0); }
    printf(g_checkpoint_exists_with_data ? "Снимок весов найден, обучение продолжается с него.\n" : "Снимка весов ещё нет, обучение начинается с исходного model.safetensors.\n");

    // кн27
    for(int l=0; l<N_LAYERS; ++l){
        const LayerSpec& sp=A.L[l];
        int head_dim = sp.hd, n_kv=sp.nkv;
        int q_dim=NUM_HEADS*head_dim, kv_dim=n_kv*head_dim;
        if(arch_has("iln")) register_weight_slot(l,"input_layernorm",HIDDEN);
        if(A.post_attn) register_weight_slot(l,"post_attn_ln",HIDDEN);
        if(arch_has("pfln")) register_weight_slot(l,"pre_ffn_ln",HIDDEN);
        if(A.post_ffn) register_weight_slot(l,"post_ffn_ln",HIDDEN);
        register_weight_slot(l,"q_proj",(size_t)q_dim*HIDDEN,q_dim);
        register_weight_slot(l,"k_proj",(size_t)kv_dim*HIDDEN,kv_dim);
        if(sp.has_v) register_weight_slot(l,"v_proj",(size_t)kv_dim*HIDDEN,kv_dim);
        register_weight_slot(l,"o_proj",(size_t)HIDDEN*q_dim,HIDDEN);
        if(A.qk_norm){ register_weight_slot(l,"q_norm",(size_t)(A.qk_full?q_dim:head_dim)); register_weight_slot(l,"k_norm",(size_t)(A.qk_full?kv_dim:head_dim)); }
        register_weight_slot(l,"gate_proj",(size_t)INTER*HIDDEN,INTER);
        register_weight_slot(l,"up_proj",(size_t)INTER*HIDDEN,INTER);
        register_weight_slot(l,"down_proj",(size_t)HIDDEN*INTER,HIDDEN);
    }
    // кн28
    register_weight_slot(-1, "final_norm", HIDDEN);

    // кн29
    for(int i=1;i+1<argc;++i) if(std::string(argv[i])=="--row-scores"){ FILE* f=fopen(argv[i+1],"r"); if(f){ char* ln=nullptr; size_t cap=0;   // кн30
            while(getline(&ln,&cap,f)>0){ std::istringstream is(ln); std::string k; is>>k; std::vector<float> v; float x; while(is>>x) v.push_back(x); if(!v.empty()) g_row_scores[k]=v; }
            free(ln); fclose(f); fprintf(stderr,"Замер строк: матриц %zu, строки среза выбираются по нему.\n",g_row_scores.size()); } }
    if(g_retier){ g_ridx.clear(); double sf=1;   // кн31
        for(int i=1;i+1<argc;++i) if(std::string(argv[i])=="--slice-frac"){ sf=atof(argv[i+1]); if(!(sf>0) || sf>1) sf=1; }
        assign_tiers_slice((size_t)(g_conf.tier1_budget*sf)); save_tier_assignment(g_tier_file_path); return 0; }

    load_tier_assignment(g_tier_file_path); // кн32
    if(g_lora_r){   // кн33
        for(auto& [k,sl] : g_weight_layout){ sl.tier=3; sl.ntr=0; } g_train_tier2=g_train_tier3=false; g_nsgd=g_posdw=false;
        lora_layout();
        if(lora_load(g_checkpoint_path)) printf("LoRA: адаптеры загружены из %s, обучение продолжается.\n", g_checkpoint_path.c_str());
        else { lora_init(); if(!g_eval) lora_save(g_checkpoint_path); }
        printf("LoRA r=%d alpha=%g (s=%.4g), dropout %g, матрицы %s: адаптеров %zu, чисел %zu, %.2f ГиБ видеопамяти с AdamW.\n",
               g_lora_r,g_lora_alpha,g_lora_s,g_lora_drop,g_lora_tg.c_str(),g_lora.size(),lora_params(),lora_params()*12/1073741824.0); }

    size_t weight_file_bytes = g_weight_layout_total_floats * sizeof(float); // кн34
    size_t adam_file_bytes = g_weight_layout_total_floats * 2 * sizeof(float); // кн35
    printf("Параметров весов: %zu.\n", g_weight_layout_total_floats);

    // кн36
    { bool ok=true; size_t n1=0; for(auto& [k,sl] : g_weight_layout){ bool fr=(sl.tier==2&&!g_train_tier2)||(sl.tier==3&&!g_train_tier3); if(sl.tier==1) ++n1; else if(!fr) ok=false; }
        if(ok && n1){ g_compact=true; std::string t="LKT1 "+std::to_string(n1)+"\n"; size_t off=C1_HDR;
            for(auto& [k,sl] : g_weight_layout) if(sl.tier==1){ size_t nn = sl.ntr ? sl.ntr : sl.n;   // кн37
                g_c1off[k]=off; t+=k+" "+std::to_string(nn)+" "+std::to_string(off)+"\n"; off+=(nn*12+4095)/4096*4096; }
            if(t.size()>C1_HDR){ fprintf(stderr,"Таблица обучаемой части не помещается в заголовок.\n"); exit(1); } g_c1hdr=t; g_c1hdr.resize(C1_HDR,'\0');
            struct stat st; bool ex=stat(g_checkpoint_path.c_str(),&st)==0 && st.st_size>0;
            if(ex){ std::string h(C1_HDR,'\0'); int fd=open(g_checkpoint_path.c_str(),O_RDONLY); pread_all(fd,h.data(),std::min<size_t>(C1_HDR,st.st_size),0); close(fd);
                if(h!=g_c1hdr){ fprintf(stderr,"%s: файл обучаемой части не соответствует этой раскладке (старый формат или другое распределение). Требуется удалить файл либо указать другой путь.\n",g_checkpoint_path.c_str()); exit(1); } }
            else if(!g_eval){ int fd=open(g_checkpoint_path.c_str(),O_RDWR|O_CREAT,0644); if(fd<0||ftruncate(fd,off)!=0){ perror("ftruncate"); exit(1); } pwrite_all(fd,g_c1hdr.data(),C1_HDR,0); close(fd); }
            g_checkpoint_exists_with_data=ex;
            printf("Файл обучаемой части: %s, тензоров %zu, %.2f ГиБ (вес и два момента)%s.\n",g_checkpoint_path.c_str(),n1,off/1073741824.0,ex?", продолжение":""); } }
    // кн38
    if(!g_compact && !g_lora_r){
        int fd = open(g_adam_path.c_str(), O_RDWR|O_CREAT, 0644);
        if(ftruncate(fd, adam_file_bytes) != 0){ perror("ftruncate"); exit(1); }
        close(fd);
        printf("Файл состояния Adam выделен заранее: %s (%.2f ГиБ).\n", g_adam_path.c_str(), adam_file_bytes/1024.0/1024/1024);
    }
    // кн39
    if(!g_compact && !g_lora_r && !g_checkpoint_exists_with_data && !g_eval){ // кн40
        int fd = open(g_checkpoint_path.c_str(), O_RDWR|O_CREAT, 0644);
        if(ftruncate(fd, weight_file_bytes) != 0){ perror("ftruncate"); exit(1); }
        close(fd);
        printf("Файл снимка весов выделен заранее: %s (%.2f ГиБ).\n", g_checkpoint_path.c_str(), weight_file_bytes/1024.0/1024/1024);
    }

    // кн41
    size_t cap=0; for(int l=0;l<N_LAYERS;++l) for(const char* sl : {"q","k","v","o","gate","up","down"}){ if(!arch_has(sl)) continue; auto it=g_tensors.find(LPX(l)+arch_tn(sl)); if(it!=g_tensors.end()) cap=std::max<size_t>(cap,it->second.size); }
    cap=std::max<size_t>(cap,(size_t)HIDDEN*4); cap=(cap+(1u<<20)-1)/(1u<<20)*(1u<<20)+(5u<<20);
    void* stage; HIP_CHECK(hipHostMalloc(&stage,cap));
    hipblasHandle_t blas; HIPBLAS_CHECK(hipblasCreate(&blas));

    // кн42
    if(!g_gn2){ HIP_CHECK(cmalloc(&g_gn2,sizeof(float))); }
    bool stop_early=false;
    // кн43
    const bool full_bwd = g_gdiag || getenv("TIER_FULL_BWD");   // кн44
    const int lo_tr = full_bwd ? 0 : std::max(0, lowest_trained_layer());
    if(lo_tr>0) printf("Обратный проход: слои %d..%d (ниже ничего не обучается, пропущено слоёв: %d из %d; пересчёт, передача весов и градиент).\n", N_LAYERS-1, lo_tr, lo_tr, N_LAYERS);
    for(size_t qi=0; qi<sched.size(); ++qi){ int step=sched[qi].first, ev=sched[qi].second; bool is_ev=ev>=0; g_is_ev=is_ev; g_eval=eval_mode||is_ev;
    if(is_ev && (ev==0 || std::count(ev_end.begin(),ev_end.end(),ev))){ g_eval_sum=0; g_eval_tok=0; g_eval_hit=0; }   // кн45
    if(!is_ev) vram_wait(step);
    tguard_init();
    if(!is_ev && (mem_guard(step) || g_thermal_stop)){   // кн46
        printf("Остановлено по потолку памяти: свободной ОЗУ не осталось, шаг %d не начат, снимок весов записывается.\n",step); fflush(stdout);
        resident_flush(); if(g_lora_r) lora_save(g_checkpoint_path);
        if(!step_file.empty()){ FILE* sf=fopen(step_file.c_str(),"w"); if(sf){ fprintf(sf,"%d\n",step); fclose(sf); } }
        printf("Снимок весов записан, повторный запуск продолжится с шага %d.\n",step); fflush(stdout); exit(75); }
    g_cur_step=is_ev?-1:step; double t_step0=elapsed(); HIP_CHECK(hipMemset(g_gn2,0,sizeof(float)));
    // кн47
    g_h2d=0; g_cpu_secs=0; g_cpu_wait=0;
    g_lr_scale=1.0;
    if(g_lrfind>0 && !is_ev){   // кн48
        double t=(g_lrfind>1)?(double)(step-step0)/(g_lrfind-1):0.0;
        g_lr_scale = exp(log(1.0/300.0) + t*(log(30.0)-log(1.0/300.0)));
    }
    else if(g_damp>0 && g_gn_ema>0 && g_prev_gn>g_damp*g_gn_ema){ g_lr_scale=g_damp*g_gn_ema/g_prev_gn;
        if(!is_ev) printf("Шаг ограничен: норма %.3e при среднем %.3e, масштаб шага ×%.3f.\n", g_prev_gn, g_gn_ema, g_lr_scale); }
    int ex=-1; pos_h.clear(); seg_h.clear();
    if(is_ev || !packs.empty()){ auto& DS=is_ev?ds_e:ds; auto& DP=is_ev?ds_ep:ds_p; auto& PK=is_ev?packs_e:packs; int pi=is_ev?ev:(step-1)%(int)packs.size();
        if(!is_ev && shuffle){ int epoch=(step-1)/(int)packs.size(); if(epoch!=perm_epoch){ perm.resize(packs.size()); for(size_t i=0;i<perm.size();++i) perm[i]=i;
            std::mt19937 rg(1000+epoch); std::shuffle(perm.begin(),perm.end(),rg); perm_epoch=epoch; } pi=perm[pi]; }
        auto& pk=PK[pi]; ex=pk[0]; tokens.clear(); lm_h.clear(); tg_h.clear(); cur_ex=pk; auto& LBS=is_ev?ds_elab:ds_lab;
        for(int k : pk){ int o=tokens.size(), L=DS[k].size(), p=DP[k]; tokens.insert(tokens.end(),DS[k].begin(),DS[k].end()); auto& lb=LBS[k];
            for(int t=0;t<L;++t){ pos_h.push_back(t); seg_h.push_back(o); tg_h.push_back(lb.empty()||lb[t]<0 ? DS[k][t] : lb[t]);
                lm_h.push_back(t>=p-1 && t<=L-2 && (lb.empty() || lb[t+1]>=0)); }
            // кн49
            if(g_w1>1 && p-1>=0 && p-1<=L-2 && lm_h[lm_h.size()-L+(p-1)]) lm_h[lm_h.size()-L+(p-1)] = (unsigned char)g_w1; }
        seq_len=(int)tokens.size(); t0=0; }
    else { tg_h.clear(); cur_ex.clear(); if(!ds.empty()){ ex=(step-1)%(int)ds.size(); if(!(g_gen && step>step0)) tokens=ds[ex];   /* кн50 */
            seq_len=(int)tokens.size(); t0=ds_p[ex]-1; cur_ex.push_back(ex); }
        if(g_gen) t0=seq_len-1; // кн51
        lm_h.assign(seq_len,0); for(int t=t0;t<seq_len-1;++t) lm_h[t]=1;
        if(ex>=0 && !g_gen && !ds_lab[ex].empty()){ auto& lb=ds_lab[ex]; tg_h=tokens; for(int t=0;t<seq_len;++t) if(lb[t]>=0) tg_h[t]=lb[t];
            for(int t=0;t<seq_len-1;++t) if(lb[t+1]<0) lm_h[t]=0; }
        if(g_w1>1 && t0>=0 && t0<seq_len-1 && lm_h[t0]) lm_h[t0]=(unsigned char)g_w1; }   // кн52
    if(g_posdw){   // кн53
        std::vector<float> mk(seq_len,0.f); for(int t=0;t<seq_len && t<(int)lm_h.size();++t) mk[t]= lm_h[t]?1.f:0.f;
        if((size_t)seq_len>g_dwmask_n){ if(g_dwmask) HIP_CHECK(hipFree(g_dwmask)); g_dwmask_n=(size_t)seq_len*2; HIP_CHECK(dmalloc(&g_dwmask,g_dwmask_n*sizeof(float))); }
        HIP_CHECK(hipMemcpy(g_dwmask,mk.data(),seq_len*sizeof(float),hipMemcpyHostToDevice)); }
    { int* pz=nullptr; int* sz=nullptr;
      if(!pos_h.empty()){ if(d_pcap<pos_h.size()){ if(d_pos){cfree(d_pos);cfree(d_seg);} d_pcap=pos_h.size()*2; HIP_CHECK(cmalloc(&d_pos,d_pcap*4)); HIP_CHECK(cmalloc(&d_seg,d_pcap*4)); }
        HIP_CHECK(hipMemcpy(d_pos,pos_h.data(),pos_h.size()*4,hipMemcpyHostToDevice)); HIP_CHECK(hipMemcpy(d_seg,seg_h.data(),seg_h.size()*4,hipMemcpyHostToDevice)); pz=d_pos; sz=d_seg; }
      set_pos_seg<<<1,1>>>(pz,sz); HIP_CHECK(hipGetLastError()); HIP_CHECK(hipDeviceSynchronize()); }
    std::vector<float> lastrow(g_gen?VOCAB:0); bool gen_stop=false;
    auto& embed_ref = g_tensors.at(A.pre+arch_tn("embed"));
    static void* embed_h=nullptr;
    if(!embed_h){ if(hipHostMalloc(&embed_h,embed_ref.size)!=hipSuccess) embed_h=malloc(embed_ref.size); // кн54
        int _fd=open(g_path.c_str(),O_RDONLY); size_t _t=0;
        auto _c0=std::chrono::steady_clock::now();
        posix_fadvise(_fd,embed_ref.offset,embed_ref.size,POSIX_FADV_SEQUENTIAL);   // кн55
        while(_t<embed_ref.size){ ssize_t _r=pread(_fd,(char*)embed_h+_t,embed_ref.size-_t,embed_ref.offset+_t); if(_r<=0){perror("Чтение эмбеддингов (pread)");exit(1);} _t+=_r; }
        posix_fadvise(_fd,embed_ref.offset,embed_ref.size,POSIX_FADV_DONTNEED);   // кн56
        close(_fd);
        if(g_fp16_storage){ auto* raw=(uint16_t*)embed_h; auto* half=(tier_fp16*)embed_h; size_t count=embed_ref.size/2;
            for(size_t i=0;i<count;++i){ uint32_t u=(uint32_t)raw[i]<<16; float f; memcpy(&f,&u,4); half[i]=tier_fp16(f); }
        }
        fprintf(stderr,"Эмбеддинги в ОЗУ: %.2f ГиБ за %.1f с%s.\n",embed_ref.size/1073741824.0,
            std::chrono::duration<double>(std::chrono::steady_clock::now()-_c0).count(),g_fp16_storage?" (FP16)":" (BF16)");
    }
    size_t row_bytes = (size_t)HIDDEN*2;
    std::vector<float> h_x((size_t)seq_len*HIDDEN);
    { std::vector<uint16_t> row(HIDDEN);
      for(int t=0;t<seq_len;++t){ memcpy(row.data(),(const char*)embed_h+(size_t)tokens[t]*row_bytes,row_bytes);
        for(int d=0;d<HIDDEN;++d){ float f;
            if(g_fp16_storage) f=(float)((tier_fp16*)row.data())[d];
            else { uint32_t u=((uint32_t)row[d])<<16; memcpy(&f,&u,4); }
            h_x[(size_t)t*HIDDEN+d]=f*(A.emb_scale<0?sqrtf((float)HIDDEN):A.emb_scale); } } }
    static void* head_h=nullptr;   // кн57
    if(!head_h){ if(A.tied) head_h=embed_h; else { auto& hr=g_tensors.at(A.head_name);
        if(hipHostMalloc(&head_h,hr.size)!=hipSuccess) head_h=malloc(hr.size); int fd=open(g_path.c_str(),O_RDONLY); pread_all(fd,head_h,hr.size,hr.offset); close(fd);
        if(g_fp16_storage){ auto* raw=(uint16_t*)head_h; auto* half=(tier_fp16*)head_h; for(size_t i=0;i<hr.size/2;++i){ uint32_t u=(uint32_t)raw[i]<<16; float f; memcpy(&f,&u,4); half[i]=tier_fp16(f); } }
    } }
    float* x; HIP_CHECK(cmalloc(&x,(size_t)seq_len*HIDDEN*sizeof(float)));
    HIP_CHECK(hipMemcpy(x,h_x.data(),(size_t)seq_len*HIDDEN*sizeof(float),hipMemcpyHostToDevice));

    size_t x_n=(size_t)seq_len*HIDDEN;
    g_split_ok=true;   // кн58
    std::vector<std::vector<float>> layer_in(N_LAYERS); // кн59
    for(int l=0; l<N_LAYERS; ++l){
        bool is_full=layer_set(l);

        // кн60
        if(l==0){ static bool prefilled=false; if(!prefilled){ prefilled=true; ram_prefill(N_LAYERS,g_train_tier2,g_train_tier3); } }
        if(g_pf_ready) pf_enable(); // кн61
        rbf_boundary(); if(g_use_arena) layer_begin();
        Weights W; load_weights_norms(W,stage,cap,l);            // кн62
        if(!g_eval && l>=lo_tr){ layer_in[l].resize(x_n); HIP_CHECK(hipMemcpy(layer_in[l].data(),x,x_n*sizeof(float),hipMemcpyDeviceToHost)); } // кн63
        float *normed,*residual; HIP_CHECK(cmalloc(&normed,x_n*sizeof(float))); HIP_CHECK(cmalloc(&residual,x_n*sizeof(float)));
        float *r1,*n1; HIP_CHECK(cmalloc(&r1,seq_len*sizeof(float))); HIP_CHECK(cmalloc(&n1,x_n*sizeof(float)));
        HIP_CHECK(hipMemcpy(residual,x,x_n*sizeof(float),hipMemcpyDeviceToDevice));
        post_norm(x,W.iln,normed,r1,n1,seq_len,W.iln!=nullptr); HIP_CHECK(hipStreamSynchronize(0));   // кн64
        dump_act("layer_"+std::to_string(l)+".qkv",normed,seq_len,HIDDEN);   // кн65
        cpu_group_begin(normed,seq_len,HIDDEN,l);                // кн66
        load_weights_mats(W,stage,cap,l);                        // кн67
        pf_issue(l<N_LAYERS-1 ? l+1 : (g_eval ? 0 : N_LAYERS-1)); // кн68
        std::string ap="layer_"+std::to_string(l)+".";

        float *Q,*K,*V; HIP_CHECK(cmalloc(&Q,(size_t)seq_len*Q_DIM*sizeof(float)));
        HIP_CHECK(cmalloc(&K,(size_t)seq_len*KV_DIM*sizeof(float))); HIP_CHECK(cmalloc(&V,(size_t)seq_len*KV_DIM*sizeof(float)));
        linear(blas,normed,W.q,Q,seq_len,HIDDEN,Q_DIM); linear(blas,normed,W.k,K,seq_len,HIDDEN,KV_DIM);
        lora_fwd(blas,l,0,normed,Q,seq_len); lora_fwd(blas,l,1,normed,K,seq_len);
        if(HAS_V_PROJ){ linear(blas,normed,W.v,V,seq_len,HIDDEN,KV_DIM); lora_fwd(blas,l,2,normed,V,seq_len); }
        bias_add(Q,W.qb,seq_len,Q_DIM); bias_add(K,W.kb,seq_len,KV_DIM); bias_add(V,W.vb,seq_len,KV_DIM);
        if(!HAS_V_PROJ) HIP_CHECK(hipMemcpy(V,K,(size_t)seq_len*KV_DIM*sizeof(float),hipMemcpyDeviceToDevice));
        size_t qn_=(size_t)seq_len*Q_DIM, kn_=(size_t)seq_len*KV_DIM;
        float *rQ,*rK,*rV,*Qn,*Kn,*Vn;
        HIP_CHECK(cmalloc(&rQ,(size_t)seq_len*NUM_HEADS*sizeof(float))); HIP_CHECK(cmalloc(&rK,(size_t)seq_len*NUM_KV_HEADS*sizeof(float)));
        HIP_CHECK(cmalloc(&rV,(size_t)seq_len*NUM_KV_HEADS*sizeof(float)));
        HIP_CHECK(cmalloc(&Qn,(size_t)seq_len*Q_DIM*sizeof(float))); HIP_CHECK(cmalloc(&Kn,(size_t)seq_len*KV_DIM*sizeof(float))); HIP_CHECK(cmalloc(&Vn,(size_t)seq_len*KV_DIM*sizeof(float)));
        if(A.qk_norm){ if(A.qk_full){ rmsnorm_fwd<<<seq_len, NB_ROW >>>(Q,W.qn,Q,rQ,Qn,Q_DIM,weight_type(W.qn)); rmsnorm_fwd<<<seq_len, NB_ROW >>>(K,W.kn,K,rK,Kn,KV_DIM,weight_type(W.kn)); }
        else { head_rmsnorm_fwd<<<dim3(seq_len,NUM_HEADS), NB_HEAD >>>(Q,W.qn,Q,rQ,Qn,NUM_HEADS,HEAD_DIM,weight_type(W.qn));
                       head_rmsnorm_fwd<<<dim3(seq_len,NUM_KV_HEADS), NB_HEAD >>>(K,W.kn,K,rK,Kn,NUM_KV_HEADS,HEAD_DIM,weight_type(W.kn)); } }
        if(A.v_norm) v_norm_fwd<<<dim3(seq_len,NUM_KV_HEADS), NB_HEAD >>>(V,V,rV,Vn,NUM_KV_HEADS,HEAD_DIM);
        HIP_CHECK(hipStreamSynchronize(0));
        rope_fwd<<<dim3(seq_len,NUM_HEADS),64>>>(Q,seq_len,NUM_HEADS,HEAD_DIM,ROTARY_DIM,ROPE_THETA,ROPE_INVF,rope_half(HEAD_DIM,ROTARY_DIM),A.rope_msc);
        rope_fwd<<<dim3(seq_len,NUM_KV_HEADS),64>>>(K,seq_len,NUM_KV_HEADS,HEAD_DIM,ROTARY_DIM,ROPE_THETA,ROPE_INVF,rope_half(HEAD_DIM,ROTARY_DIM),A.rope_msc);
        qscale(Q,(size_t)seq_len*Q_DIM);
        HIP_CHECK(hipStreamSynchronize(0));

        float *attn_out,*P; HIP_CHECK(cmalloc(&attn_out,(size_t)seq_len*Q_DIM*sizeof(float)));
        HIP_CHECK(cmalloc(&P,(size_t)seq_len*NUM_HEADS*seq_len*sizeof(float)));
        size_t shmem=seq_len*sizeof(float);
        attention_fwd<<<dim3(seq_len,NUM_HEADS),128,shmem>>>(Q,K,V,attn_out,P,seq_len,NUM_HEADS,NUM_KV_HEADS,HEAD_DIM,WINDOW,A.attn_cap);
        HIP_CHECK(hipStreamSynchronize(0));
        dump_act("layer_"+std::to_string(l)+".o_proj",attn_out,seq_len,Q_DIM);

        float *attn_proj; HIP_CHECK(cmalloc(&attn_proj,x_n*sizeof(float)));
        linear(blas,attn_out,W.o,attn_proj,seq_len,Q_DIM,HIDDEN); lora_fwd(blas,l,3,attn_out,attn_proj,seq_len);
        post_norm(attn_proj,W.paln,x,r1,n1,seq_len,A.post_attn); // кн69
        if(A.resid_mul!=1.f) scale_k<<<(unsigned)((x_n+255)/256),256>>>(x,A.resid_mul,x_n);
        HIP_CHECK(hipStreamSynchronize(0));

        add_kernel<<<(x_n+255)/256,256>>>(x,residual,x_n); HIP_CHECK(hipStreamSynchronize(0));

        HIP_CHECK(hipMemcpy(residual,x,x_n*sizeof(float),hipMemcpyDeviceToDevice));
        post_norm(x,W.pfln,normed,r1,n1,seq_len,W.pfln!=nullptr); HIP_CHECK(hipStreamSynchronize(0));

        float *gate,*up,*act; size_t inter_n=(size_t)seq_len*INTER;
        HIP_CHECK(cmalloc(&gate,inter_n*sizeof(float))); HIP_CHECK(cmalloc(&up,inter_n*sizeof(float))); HIP_CHECK(cmalloc(&act,inter_n*sizeof(float)));
        linear(blas,normed,W.gate,gate,seq_len,HIDDEN,INTER); linear(blas,normed,W.up,up,seq_len,HIDDEN,INTER);
        lora_fwd(blas,l,4,normed,gate,seq_len); lora_fwd(blas,l,5,normed,up,seq_len);

        act_fwd(gate,up,act,inter_n); HIP_CHECK(hipStreamSynchronize(0));
        dump_act("layer_"+std::to_string(l)+".down_proj",act,seq_len,INTER);
        float* mlp_out; HIP_CHECK(cmalloc(&mlp_out,x_n*sizeof(float)));
        linear(blas,act,W.down,mlp_out,seq_len,INTER,HIDDEN); lora_fwd(blas,l,6,act,mlp_out,seq_len);
        post_norm(mlp_out,W.pofln,x,r1,n1,seq_len,A.post_ffn);
        if(A.resid_mul!=1.f) scale_k<<<(unsigned)((x_n+255)/256),256>>>(x,A.resid_mul,x_n);
        HIP_CHECK(hipStreamSynchronize(0));

        add_kernel<<<(x_n+255)/256,256>>>(x,residual,x_n); HIP_CHECK(hipStreamSynchronize(0));
        if(W.ls){ mul_scalar_kernel<<<(x_n+255)/256,256>>>(x,W.ls,x_n); HIP_CHECK(hipStreamSynchronize(0)); }

        vram_sample();
        for(float* p : {normed,residual,r1,n1,Q,K,V,rQ,rK,rV,Qn,Kn,Vn,attn_out,P,attn_proj,gate,up,act,mlp_out,
                         W.iln,W.paln,W.pfln,W.pofln,W.q,W.k,W.v,W.o,W.qn,W.kn,W.gate,W.up,W.down,W.ls,W.qb,W.kb,W.vb})
            if(p && !g_res_ptrs.count(p)) cfree(p);
        if(g_use_arena) layer_end();
        if(!VERBOSE&&!g_gdiag&&g_tty) fprintf(stderr, "forward layer %d/47 done\r", l);
        if(g_prog && !g_is_ev) printf("prog f %d\n", l+1);
    }
    fprintf(stderr, "\n");

    double t_f=elapsed(); // кн70
    float* w_norm = nullptr;
    float *normed_final=nullptr, *r_final=nullptr, *n_final=nullptr, *d_normed_final=nullptr, *d_x=nullptr;

    if(N_LAYERS < g_model_layers){
        // кн71
        std::mt19937 rng(123); std::normal_distribution<float> dist(0.0f,0.02f);
        std::vector<float> h_R(x_n); for(auto& v:h_R) v=dist(rng);
        float* R; HIP_CHECK(cmalloc(&R,x_n*sizeof(float))); HIP_CHECK(hipMemcpy(R,h_R.data(),x_n*sizeof(float),hipMemcpyHostToDevice));
        double* d_loss; HIP_CHECK(cmalloc(&d_loss,sizeof(double))); HIP_CHECK(hipMemset(d_loss,0,sizeof(double)));
        sum_reduce_kernel<<<(x_n+255)/256,256>>>(x,R,d_loss,x_n); HIP_CHECK(hipStreamSynchronize(0));
        double loss; HIP_CHECK(hipMemcpy(&loss,d_loss,sizeof(double),hipMemcpyDeviceToHost));
        printf("=== DIAGNOSTIC MODE (N_LAYERS=%d): loss = sum(x*R) = %.6f ===\n", N_LAYERS, loss);
        d_x = R; // кн72
        cfree(d_loss);
    } else {
    w_norm = load_tensor_or_checkpoint(A.pre+arch_tn("final"), "layer_-1.final_norm", stage, cap);
    HIP_CHECK(cmalloc(&normed_final,x_n*sizeof(float)));
    HIP_CHECK(cmalloc(&r_final,seq_len*sizeof(float))); HIP_CHECK(cmalloc(&n_final,x_n*sizeof(float)));
    rmsnorm_fwd<<<seq_len, NB_ROW >>>(x,w_norm,normed_final,r_final,n_final,HIDDEN,weight_type(w_norm));
    if(A.logit_div!=1.f) scale_k<<<(unsigned)((x_n+255)/256),256>>>(normed_final,1.f/A.logit_div,x_n);   // кн73
    HIP_CHECK(hipStreamSynchronize(0));

    // кн74
    const int CHUNK=lm_chunk(seq_len);
    rbf_boundary(); if(g_use_arena) layer_begin();
    int n_pred = seq_len-1, n_loss = 0; for(int t=0;t<n_pred;++t) n_loss+=(lm_h[t]!=0);   // кн75
    std::vector<double> running_max(n_pred,-1e30), running_sum(n_pred,0.0); std::vector<int> amax(n_pred,-1); // кн76
    std::vector<float> target_logit(n_pred,0.0f);
    // кн77
    HIP_CHECK(cmalloc(&d_normed_final,x_n*sizeof(float))); HIP_CHECK(hipMemset(d_normed_final,0,x_n*sizeof(float)));   // кн78

    int fd=open(g_path.c_str(),O_RDONLY);
    float *d_chunk_bf,*d_chunk_f32,*d_logits;
    HIP_CHECK(cmalloc(&d_chunk_bf,(size_t)CHUNK*HIDDEN*2)); HIP_CHECK(cmalloc(&d_chunk_f32,(size_t)CHUNK*HIDDEN*sizeof(float)));
    HIP_CHECK(cmalloc(&d_logits,(size_t)seq_len*CHUNK*sizeof(float)));
    int32_t* d_tokens; HIP_CHECK(cmalloc(&d_tokens, seq_len*sizeof(int32_t)));
    if((int)tg_h.size()!=seq_len) tg_h=tokens;   // кн79
    HIP_CHECK(hipMemcpy(d_tokens, tg_h.data(), seq_len*sizeof(int32_t), hipMemcpyHostToDevice));
    unsigned char* d_lmask; HIP_CHECK(cmalloc(&d_lmask, seq_len)); HIP_CHECK(hipMemcpy(d_lmask, lm_h.data(), seq_len, hipMemcpyHostToDevice));
    double *d_running_max, *d_running_sum; int* d_amax; float* d_tgt;
    HIP_CHECK(cmalloc(&d_running_max, n_pred*sizeof(double))); HIP_CHECK(cmalloc(&d_running_sum, n_pred*sizeof(double)));
    HIP_CHECK(cmalloc(&d_amax, n_pred*sizeof(int))); HIP_CHECK(cmalloc(&d_tgt, n_pred*sizeof(float)));
    HIP_CHECK(hipMemcpy(d_running_max, running_max.data(), n_pred*sizeof(double), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_running_sum, running_sum.data(), n_pred*sizeof(double), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_amax, amax.data(), n_pred*sizeof(int), hipMemcpyHostToDevice)); HIP_CHECK(hipMemset(d_tgt,0,n_pred*sizeof(float)));

    // кн80
    for(int v0=0; v0<VOCAB; v0+=CHUNK){
        int vc=std::min(CHUNK,VOCAB-v0);
        HIP_CHECK(hipMemcpy(d_chunk_bf,(const char*)head_h+(size_t)v0*row_bytes,(size_t)vc*row_bytes,hipMemcpyHostToDevice));
        int th=256; size_t ne=(size_t)vc*HIDDEN;
        if(g_fp16_storage) fp16_to_f32_kernel<<<(unsigned)((ne+th-1)/th),th>>>((tier_fp16*)d_chunk_bf,d_chunk_f32,ne);
        else bf16_to_f32_kernel<<<(unsigned)((ne/4+th)/th),th>>>((hip_bfloat16*)d_chunk_bf,d_chunk_f32,ne);
        HIP_CHECK(hipStreamSynchronize(0));
        linear(blas,normed_final,d_chunk_f32,d_logits,seq_len,HIDDEN,vc);
        size_t ln=(size_t)seq_len*vc;
        if(SOFTCAP>0) softcap_kernel<<<(ln+255)/256,256>>>(d_logits,SOFTCAP,ln);
        if(n_pred>0) lse_chunk_kernel<<<n_pred,256>>>(d_logits,d_running_max,d_running_sum,d_amax,d_tgt,d_tokens,d_lmask,vc,v0);
        if(g_gen) HIP_CHECK(hipMemcpy(lastrow.data()+v0,d_logits+(size_t)(seq_len-1)*vc,vc*sizeof(float),hipMemcpyDeviceToHost));
        if(g_prog && !g_is_ev) printf("prog h %d\n", v0/CHUNK+1);
    }
    HIP_CHECK(hipStreamSynchronize(0));
    HIP_CHECK(hipMemcpy(running_max.data(), d_running_max, n_pred*sizeof(double), hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(running_sum.data(), d_running_sum, n_pred*sizeof(double), hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(amax.data(), d_amax, n_pred*sizeof(int), hipMemcpyDeviceToHost)); HIP_CHECK(hipMemcpy(target_logit.data(), d_tgt, n_pred*sizeof(float), hipMemcpyDeviceToHost));
    double total_loss=0;
    int hit=0; for(int t=0;t<n_pred;++t) if(lm_h[t]){ double lse=running_max[t]+log(running_sum[t]); total_loss+=lse-target_logit[t]; hit+=(amax[t]==tg_h[t+1]); }
    if(dump_f && g_eval && !g_is_ev){ int ne=(int)cur_ex.size(); fwrite(&ne,4,1,dump_f); fwrite(cur_ex.data(),4,ne,dump_f); fwrite(&seq_len,4,1,dump_f);
        for(int t=0;t<n_pred;++t){ int a=lm_h[t]?amax[t]:-1; float pt=lm_h[t]?(float)(1.0/running_sum[t]):0.f; fwrite(&a,4,1,dump_f); fwrite(&pt,4,1,dump_f); } fflush(dump_f); }
    if(g_eval && !g_is_ev && getenv("TIER_EVAL_EACH") && cur_ex.size()==1) printf("evex %d %.6f %d\n",cur_ex[0],total_loss,n_loss);   // кн81
    double avg_loss=total_loss/n_loss; g_eval_sum+=total_loss; g_eval_tok+=n_loss; g_eval_hit+=hit;
    // кн82
    if(g_lrfind>0 && n_loss>0 && !g_is_ev) printf("lrfind %d %.6e %.6f\n", step-step0+1, lr*g_lr_scale, avg_loss);
    if(n_loss>0 && !g_is_ev) printf("=== forward loss: %.4f nats (perplexity %.2f) === [ex %d, n_ex %d, len %d, loss on %d, acc %.4f] [%.1fs elapsed]\n", avg_loss, exp(avg_loss), ex, (int)(packs.empty()?1:packs[(step-1)%packs.size()].size()), seq_len, n_loss, (double)hit/n_loss, elapsed());
    if(g_log){ fprintf(g_log,"\n===== step %d ex %d len %d t0 %d loss %.4f\ntokens:",step,ex,seq_len,t0,avg_loss); for(int t=0;t<seq_len;++t) fprintf(g_log," %d",tokens[t]); fprintf(g_log,"\n"); }

    // кн83

    // кн84
    if(g_gen){ int best=0; // кн85
        if(g_temp<=0){ for(int v=1;v<VOCAB;++v) if(lastrow[v]>lastrow[best]) best=v; }
        else { static std::mt19937 grng(42); float mx=*std::max_element(lastrow.begin(),lastrow.end()); std::vector<double> p(VOCAB); double z=0;
            for(int v=0;v<VOCAB;++v){ p[v]=exp((lastrow[v]-mx)/g_temp); z+=p[v]; } std::uniform_real_distribution<double> u(0,z); double r=u(grng);
            for(best=0;best<VOCAB-1;++best){ r-=p[best]; if(r<=0) break; } }
        printf("gen_token %d\n",best); tokens.push_back(best); gen_stop=is_eos(best); }
    float* d_dlogit; HIP_CHECK(cmalloc(&d_dlogit,(size_t)seq_len*CHUNK*sizeof(float)));
    for(int v0=0; v0<VOCAB; v0+=CHUNK){
        if(g_eval) break; // кн86
        int vc=std::min(CHUNK,VOCAB-v0);
        HIP_CHECK(hipMemcpy(d_chunk_bf,(const char*)head_h+(size_t)v0*row_bytes,(size_t)vc*row_bytes,hipMemcpyHostToDevice));
        int th=256; size_t ne=(size_t)vc*HIDDEN;
        if(g_fp16_storage) fp16_to_f32_kernel<<<(unsigned)((ne+th-1)/th),th>>>((tier_fp16*)d_chunk_bf,d_chunk_f32,ne);
        else bf16_to_f32_kernel<<<(unsigned)((ne/4+th)/th),th>>>((hip_bfloat16*)d_chunk_bf,d_chunk_f32,ne);
        linear(blas,normed_final,d_chunk_f32,d_logits,seq_len,HIDDEN,vc);
        size_t ln=(size_t)seq_len*vc;
        if(SOFTCAP>0) softcap_kernel<<<(ln+255)/256,256>>>(d_logits,SOFTCAP,ln);

        size_t dl_n = (size_t)seq_len*vc; // кн87
        dlogit_kernel<<<(dl_n+255)/256,256>>>(d_logits, d_running_max, d_running_sum, d_tokens, d_dlogit, seq_len, vc, v0, n_loss, d_lmask, SOFTCAP);

        // кн88
        linear_dx_acc(blas,d_dlogit,d_chunk_f32,d_normed_final,seq_len,HIDDEN,vc);
        if(!VERBOSE&&!g_gdiag&&g_tty) fprintf(stderr, "LM head backward chunk %d/%d\r", v0/CHUNK+1, (VOCAB+CHUNK-1)/CHUNK);
        if(g_prog && !g_is_ev) printf("prog g %d\n", v0/CHUNK+1);
    }
    fprintf(stderr, "\n");
    close(fd);
    vram_sample(); HIP_CHECK(hipStreamSynchronize(0)); cfree(d_running_max); cfree(d_running_sum); cfree(d_amax); cfree(d_tgt); cfree(d_tokens); cfree(d_lmask); cfree(d_chunk_bf); cfree(d_chunk_f32); cfree(d_logits); cfree(d_dlogit);

    if(g_use_arena) layer_end(); // кн89
    // кн90
    HIP_CHECK(cmalloc(&d_x,x_n*sizeof(float)));
    float* g_norm_dummy; HIP_CHECK(cmalloc(&g_norm_dummy,HIDDEN*sizeof(float))); HIP_CHECK(hipMemset(g_norm_dummy,0,HIDDEN*sizeof(float)));
    if(A.logit_div!=1.f) scale_k<<<(unsigned)((x_n+255)/256),256>>>(d_normed_final,1.f/A.logit_div,x_n);
    rmsnorm_bwd<<<seq_len,256>>>(d_normed_final,n_final,w_norm,r_final,d_x,g_norm_dummy,HIDDEN,weight_type(w_norm)); HIP_CHECK(hipStreamSynchronize(0));
    adam_apply(blas, "layer_-1.final_norm", w_norm, g_norm_dummy, HIDDEN, lr, step);
    cfree(g_norm_dummy);
    } // кн91

    if(g_gdiag){ std::vector<float> h(x_n); HIP_CHECK(hipMemcpy(h.data(),d_x,x_n*sizeof(float),hipMemcpyDeviceToHost)); double mx=0; int mt=0;
        for(int t=0;t<seq_len;++t){ double a=0; for(int d=0;d<HIDDEN;++d){ double v=h[(size_t)t*HIDDEN+d]; a+=v*v; } if(sqrt(a)>mx){ mx=sqrt(a); mt=t; } }
        fprintf(g_log?g_log:stdout,"gdiag TOP (вход в L%d после final_norm) |d_x| max %.3e @pos %d\n", N_LAYERS-1, mx, mt); }
    printf("d(final_norm_input) computed, starting %d-layer backward (%d -> %d)... [%.1fs elapsed]\n", N_LAYERS-lo_tr, N_LAYERS-1, lo_tr, elapsed());

    // кн92
    double t_h=elapsed(); // кн93
    bool bwd_ran=false;
    for(int l=N_LAYERS-1; l>=lo_tr; --l){
        if(g_eval) break; bwd_ran=true; // кн94
        bool is_full=layer_set(l);

        rbf_boundary(); if(g_use_arena) layer_begin();
        // кн95
        float* layer_x; HIP_CHECK(cmalloc(&layer_x,x_n*sizeof(float)));
        HIP_CHECK(hipMemcpy(layer_x,layer_in[l].data(),x_n*sizeof(float),hipMemcpyHostToDevice));

        g_split_ok=false;  // кн96
        Weights W=load_weights(stage,cap,l);
        pf_issue(l>lo_tr ? l-1 : 0); // кн97
        // кн98
        const bool skip_dx0 = (l==lo_tr) && !full_bwd && !g_log && !weight_trained("layer_"+std::to_string(l)+".input_layernorm");
        // кн99
        float *residual1,*normed1,*n1,*r1; HIP_CHECK(cmalloc(&residual1,x_n*sizeof(float)));
        HIP_CHECK(cmalloc(&normed1,x_n*sizeof(float))); HIP_CHECK(cmalloc(&n1,x_n*sizeof(float))); HIP_CHECK(cmalloc(&r1,seq_len*sizeof(float)));
        HIP_CHECK(hipMemcpy(residual1,layer_x,x_n*sizeof(float),hipMemcpyDeviceToDevice));
        post_norm(layer_x,W.iln,normed1,r1,n1,seq_len,W.iln!=nullptr); HIP_CHECK(hipStreamSynchronize(0));

        float *Q,*K,*V; HIP_CHECK(cmalloc(&Q,(size_t)seq_len*Q_DIM*sizeof(float)));
        HIP_CHECK(cmalloc(&K,(size_t)seq_len*KV_DIM*sizeof(float))); HIP_CHECK(cmalloc(&V,(size_t)seq_len*KV_DIM*sizeof(float)));
        linear(blas,normed1,W.q,Q,seq_len,HIDDEN,Q_DIM); linear(blas,normed1,W.k,K,seq_len,HIDDEN,KV_DIM);
        lora_fwd(blas,l,0,normed1,Q,seq_len); lora_fwd(blas,l,1,normed1,K,seq_len);
        if(HAS_V_PROJ){ linear(blas,normed1,W.v,V,seq_len,HIDDEN,KV_DIM); lora_fwd(blas,l,2,normed1,V,seq_len); }
        bias_add(Q,W.qb,seq_len,Q_DIM); bias_add(K,W.kb,seq_len,KV_DIM); bias_add(V,W.vb,seq_len,KV_DIM);
        if(!HAS_V_PROJ) HIP_CHECK(hipMemcpy(V,K,(size_t)seq_len*KV_DIM*sizeof(float),hipMemcpyDeviceToDevice));

        float *Q_rope,*K_rope,*Qn,*Kn,*Vn,*rQ,*rK,*rV;
        HIP_CHECK(cmalloc(&Q_rope,(size_t)seq_len*Q_DIM*sizeof(float))); HIP_CHECK(cmalloc(&K_rope,(size_t)seq_len*KV_DIM*sizeof(float)));
        HIP_CHECK(cmalloc(&Qn,(size_t)seq_len*Q_DIM*sizeof(float))); HIP_CHECK(cmalloc(&Kn,(size_t)seq_len*KV_DIM*sizeof(float))); HIP_CHECK(cmalloc(&Vn,(size_t)seq_len*KV_DIM*sizeof(float)));
        HIP_CHECK(cmalloc(&rQ,(size_t)seq_len*NUM_HEADS*sizeof(float))); HIP_CHECK(cmalloc(&rK,(size_t)seq_len*NUM_KV_HEADS*sizeof(float))); HIP_CHECK(cmalloc(&rV,(size_t)seq_len*NUM_KV_HEADS*sizeof(float)));
        if(A.qk_norm){ if(A.qk_full){ rmsnorm_fwd<<<seq_len, NB_ROW >>>(Q,W.qn,Q_rope,rQ,Qn,Q_DIM,weight_type(W.qn)); rmsnorm_fwd<<<seq_len, NB_ROW >>>(K,W.kn,K_rope,rK,Kn,KV_DIM,weight_type(W.kn)); }
        else { head_rmsnorm_fwd<<<dim3(seq_len,NUM_HEADS), NB_HEAD >>>(Q,W.qn,Q_rope,rQ,Qn,NUM_HEADS,HEAD_DIM,weight_type(W.qn));
                       head_rmsnorm_fwd<<<dim3(seq_len,NUM_KV_HEADS), NB_HEAD >>>(K,W.kn,K_rope,rK,Kn,NUM_KV_HEADS,HEAD_DIM,weight_type(W.kn)); } }
        else { HIP_CHECK(hipMemcpy(Q_rope,Q,(size_t)seq_len*Q_DIM*4,hipMemcpyDeviceToDevice)); HIP_CHECK(hipMemcpy(K_rope,K,(size_t)seq_len*KV_DIM*4,hipMemcpyDeviceToDevice)); }
        if(A.v_norm) v_norm_fwd<<<dim3(seq_len,NUM_KV_HEADS), NB_HEAD >>>(V,V,rV,Vn,NUM_KV_HEADS,HEAD_DIM);
        HIP_CHECK(hipStreamSynchronize(0));
        rope_fwd<<<dim3(seq_len,NUM_HEADS),64>>>(Q_rope,seq_len,NUM_HEADS,HEAD_DIM,ROTARY_DIM,ROPE_THETA,ROPE_INVF,rope_half(HEAD_DIM,ROTARY_DIM),A.rope_msc);
        rope_fwd<<<dim3(seq_len,NUM_KV_HEADS),64>>>(K_rope,seq_len,NUM_KV_HEADS,HEAD_DIM,ROTARY_DIM,ROPE_THETA,ROPE_INVF,rope_half(HEAD_DIM,ROTARY_DIM),A.rope_msc);
        qscale(Q_rope,(size_t)seq_len*Q_DIM);
        HIP_CHECK(hipStreamSynchronize(0));

        float *attn_out,*P; HIP_CHECK(cmalloc(&attn_out,(size_t)seq_len*Q_DIM*sizeof(float)));
        HIP_CHECK(cmalloc(&P,(size_t)seq_len*NUM_HEADS*seq_len*sizeof(float)));
        size_t shmem=seq_len*sizeof(float);
        attention_fwd<<<dim3(seq_len,NUM_HEADS),128,shmem>>>(Q_rope,K_rope,V,attn_out,P,seq_len,NUM_HEADS,NUM_KV_HEADS,HEAD_DIM,WINDOW,A.attn_cap);
        HIP_CHECK(hipStreamSynchronize(0));

        float *attn_proj,*normed2,*n2,*r2; HIP_CHECK(cmalloc(&attn_proj,x_n*sizeof(float)));
        linear(blas,attn_out,W.o,attn_proj,seq_len,Q_DIM,HIDDEN); lora_fwd(blas,l,3,attn_out,attn_proj,seq_len);
        HIP_CHECK(cmalloc(&normed2,x_n*sizeof(float))); HIP_CHECK(cmalloc(&n2,x_n*sizeof(float))); HIP_CHECK(cmalloc(&r2,seq_len*sizeof(float)));
        post_norm(attn_proj,W.paln,normed2,r2,n2,seq_len,A.post_attn); HIP_CHECK(hipStreamSynchronize(0));
        float* residual2; HIP_CHECK(cmalloc(&residual2,x_n*sizeof(float)));
        HIP_CHECK(hipMemcpy(residual2,normed2,x_n*sizeof(float),hipMemcpyDeviceToDevice));
        if(A.resid_mul!=1.f) scale_k<<<(unsigned)((x_n+255)/256),256>>>(residual2,A.resid_mul,x_n);
        add_kernel<<<(x_n+255)/256,256>>>(residual2,residual1,x_n); HIP_CHECK(hipStreamSynchronize(0));

        float *normed3,*n3,*r3; HIP_CHECK(cmalloc(&normed3,x_n*sizeof(float)));
        HIP_CHECK(cmalloc(&n3,x_n*sizeof(float))); HIP_CHECK(cmalloc(&r3,seq_len*sizeof(float)));
        post_norm(residual2,W.pfln,normed3,r3,n3,seq_len,W.pfln!=nullptr); HIP_CHECK(hipStreamSynchronize(0));
        float *gate,*up,*act; size_t inter_n=(size_t)seq_len*INTER;
        HIP_CHECK(cmalloc(&gate,inter_n*sizeof(float))); HIP_CHECK(cmalloc(&up,inter_n*sizeof(float))); HIP_CHECK(cmalloc(&act,inter_n*sizeof(float)));
        linear(blas,normed3,W.gate,gate,seq_len,HIDDEN,INTER); linear(blas,normed3,W.up,up,seq_len,HIDDEN,INTER);
        lora_fwd(blas,l,4,normed3,gate,seq_len); lora_fwd(blas,l,5,normed3,up,seq_len);
        act_fwd(gate,up,act,inter_n); HIP_CHECK(hipStreamSynchronize(0));
        float *mlp_out,*normed4,*n4,*r4; HIP_CHECK(cmalloc(&mlp_out,x_n*sizeof(float)));
        linear(blas,act,W.down,mlp_out,seq_len,INTER,HIDDEN); lora_fwd(blas,l,6,act,mlp_out,seq_len);
        HIP_CHECK(cmalloc(&normed4,x_n*sizeof(float))); HIP_CHECK(cmalloc(&n4,x_n*sizeof(float))); HIP_CHECK(cmalloc(&r4,seq_len*sizeof(float)));
        post_norm(mlp_out,W.pofln,normed4,r4,n4,seq_len,A.post_ffn); HIP_CHECK(hipStreamSynchronize(0));
        float* pre_scalar; HIP_CHECK(cmalloc(&pre_scalar,x_n*sizeof(float)));
        HIP_CHECK(hipMemcpy(pre_scalar,normed4,x_n*sizeof(float),hipMemcpyDeviceToDevice));
        if(A.resid_mul!=1.f) scale_k<<<(unsigned)((x_n+255)/256),256>>>(pre_scalar,A.resid_mul,x_n);
        add_kernel<<<(x_n+255)/256,256>>>(pre_scalar,residual2,x_n); HIP_CHECK(hipStreamSynchronize(0));

        // кн100
        float* dpre; HIP_CHECK(cmalloc(&dpre,x_n*sizeof(float))); HIP_CHECK(hipMemcpy(dpre,d_x,x_n*sizeof(float),hipMemcpyDeviceToDevice));
        if(W.ls){ mul_scalar_kernel<<<(x_n+255)/256,256>>>(dpre,W.ls,x_n); HIP_CHECK(hipStreamSynchronize(0)); }

        float* d_mlp_out; HIP_CHECK(cmalloc(&d_mlp_out,x_n*sizeof(float)));
        float* g_pofln; HIP_CHECK(cmalloc(&g_pofln,HIDDEN*sizeof(float))); HIP_CHECK(hipMemset(g_pofln,0,HIDDEN*sizeof(float)));
        float* dbr=dpre; if(A.resid_mul!=1.f){ HIP_CHECK(cmalloc(&dbr,x_n*sizeof(float))); HIP_CHECK(hipMemcpy(dbr,dpre,x_n*4,hipMemcpyDeviceToDevice)); scale_k<<<(unsigned)((x_n+255)/256),256>>>(dbr,A.resid_mul,x_n); }
        if(A.post_ffn) rmsnorm_bwd<<<seq_len,256>>>(dbr,n4,W.pofln,r4,d_mlp_out,g_pofln,HIDDEN,weight_type(W.pofln)); else HIP_CHECK(hipMemcpy(d_mlp_out,dbr,x_n*4,hipMemcpyDeviceToDevice));
        if(dbr!=dpre){ HIP_CHECK(hipStreamSynchronize(0)); cfree(dbr); }
        HIP_CHECK(hipStreamSynchronize(0));

        float* d_act; HIP_CHECK(cmalloc(&d_act,inter_n*sizeof(float)));
        std::string kp="layer_"+std::to_string(l)+".";
        float* g_down=galloc(kp+"down_proj",(size_t)INTER*HIDDEN);
        linear_dw(blas,d_mlp_out,act,g_down,seq_len,INTER,HIDDEN);
        linear_dx(blas,d_mlp_out,W.down,d_act,seq_len,INTER,HIDDEN); lora_bwd(blas,l,6,act,d_mlp_out,d_act,seq_len,lr,step);

        float *d_gate,*d_up; HIP_CHECK(cmalloc(&d_gate,inter_n*sizeof(float))); HIP_CHECK(cmalloc(&d_up,inter_n*sizeof(float)));
        act_bwd(d_act,gate,up,d_gate,d_up,inter_n); HIP_CHECK(hipStreamSynchronize(0));

        float *d_normed3_a,*d_normed3_b,*d_normed3;
        HIP_CHECK(cmalloc(&d_normed3_a,x_n*sizeof(float))); HIP_CHECK(cmalloc(&d_normed3_b,x_n*sizeof(float))); HIP_CHECK(cmalloc(&d_normed3,x_n*sizeof(float)));
        float *g_gate=galloc(kp+"gate_proj",(size_t)INTER*HIDDEN),*g_up=galloc(kp+"up_proj",(size_t)INTER*HIDDEN);
        linear_dw(blas,d_gate,normed3,g_gate,seq_len,HIDDEN,INTER); linear_dx(blas,d_gate,W.gate,d_normed3_a,seq_len,HIDDEN,INTER);
        linear_dw(blas,d_up,normed3,g_up,seq_len,HIDDEN,INTER); linear_dx(blas,d_up,W.up,d_normed3_b,seq_len,HIDDEN,INTER);
        lora_bwd(blas,l,4,normed3,d_gate,d_normed3_a,seq_len,lr,step); lora_bwd(blas,l,5,normed3,d_up,d_normed3_b,seq_len,lr,step);
        HIP_CHECK(hipMemcpy(d_normed3,d_normed3_a,x_n*sizeof(float),hipMemcpyDeviceToDevice));
        add_kernel<<<(x_n+255)/256,256>>>(d_normed3,d_normed3_b,x_n); HIP_CHECK(hipStreamSynchronize(0));

        float* d_residual2_mlp; HIP_CHECK(cmalloc(&d_residual2_mlp,x_n*sizeof(float)));
        float* g_pfln; HIP_CHECK(cmalloc(&g_pfln,HIDDEN*sizeof(float))); HIP_CHECK(hipMemset(g_pfln,0,HIDDEN*sizeof(float)));
        if(W.pfln) rmsnorm_bwd<<<seq_len,256>>>(d_normed3,n3,W.pfln,r3,d_residual2_mlp,g_pfln,HIDDEN,weight_type(W.pfln)); else HIP_CHECK(hipMemcpy(d_residual2_mlp,d_normed3,x_n*4,hipMemcpyDeviceToDevice));
        HIP_CHECK(hipStreamSynchronize(0));

        float* d_residual2; HIP_CHECK(cmalloc(&d_residual2,x_n*sizeof(float)));
        HIP_CHECK(hipMemcpy(d_residual2,dpre,x_n*sizeof(float),hipMemcpyDeviceToDevice));
        add_kernel<<<(x_n+255)/256,256>>>(d_residual2,d_residual2_mlp,x_n); HIP_CHECK(hipStreamSynchronize(0));

        float* d_attn_proj; HIP_CHECK(cmalloc(&d_attn_proj,x_n*sizeof(float)));
        float* g_paln; HIP_CHECK(cmalloc(&g_paln,HIDDEN*sizeof(float))); HIP_CHECK(hipMemset(g_paln,0,HIDDEN*sizeof(float)));
        float* dbr2=d_residual2; if(A.resid_mul!=1.f){ HIP_CHECK(cmalloc(&dbr2,x_n*sizeof(float))); HIP_CHECK(hipMemcpy(dbr2,d_residual2,x_n*4,hipMemcpyDeviceToDevice)); scale_k<<<(unsigned)((x_n+255)/256),256>>>(dbr2,A.resid_mul,x_n); }
        if(A.post_attn) rmsnorm_bwd<<<seq_len,256>>>(dbr2,n2,W.paln,r2,d_attn_proj,g_paln,HIDDEN,weight_type(W.paln)); else HIP_CHECK(hipMemcpy(d_attn_proj,dbr2,x_n*4,hipMemcpyDeviceToDevice));
        if(dbr2!=d_residual2){ HIP_CHECK(hipStreamSynchronize(0)); cfree(dbr2); }
        HIP_CHECK(hipStreamSynchronize(0));

        float* d_attn_out; HIP_CHECK(cmalloc(&d_attn_out,(size_t)seq_len*Q_DIM*sizeof(float)));
        float* g_o=galloc(kp+"o_proj",(size_t)Q_DIM*HIDDEN);
        linear_dw(blas,d_attn_proj,attn_out,g_o,seq_len,Q_DIM,HIDDEN); linear_dx(blas,d_attn_proj,W.o,d_attn_out,seq_len,Q_DIM,HIDDEN);
        lora_bwd(blas,l,3,attn_out,d_attn_proj,d_attn_out,seq_len,lr,step);

        float *dQ,*dK,*dV; HIP_CHECK(cmalloc(&dQ,(size_t)seq_len*Q_DIM*sizeof(float)));
        HIP_CHECK(cmalloc(&dK,(size_t)seq_len*KV_DIM*sizeof(float))); HIP_CHECK(cmalloc(&dV,(size_t)seq_len*KV_DIM*sizeof(float)));
        HIP_CHECK(hipMemset(dK,0,(size_t)seq_len*KV_DIM*sizeof(float))); HIP_CHECK(hipMemset(dV,0,(size_t)seq_len*KV_DIM*sizeof(float)));
        size_t bshmem=seq_len*sizeof(float);
        float* dSb; HIP_CHECK(cmalloc(&dSb,(size_t)seq_len*NUM_HEADS*seq_len*sizeof(float)));
        attention_bwd<<<dim3(seq_len,NUM_HEADS),64,bshmem>>>(d_attn_out,P,attn_out,Q_rope,K_rope,V,dQ,dSb,seq_len,NUM_HEADS,NUM_KV_HEADS,HEAD_DIM,WINDOW,A.attn_cap);
        attention_bwd_kv<<<dim3(seq_len,NUM_KV_HEADS),64>>>(d_attn_out,P,dSb,Q_rope,dK,dV,seq_len,NUM_HEADS,NUM_KV_HEADS,HEAD_DIM,WINDOW);
        HIP_CHECK(hipStreamSynchronize(0)); cfree(dSb);
        qscale(dQ,(size_t)seq_len*Q_DIM);   // кн101
        rope_bwd<<<dim3(seq_len,NUM_HEADS),64>>>(dQ,seq_len,NUM_HEADS,HEAD_DIM,ROTARY_DIM,ROPE_THETA,ROPE_INVF,rope_half(HEAD_DIM,ROTARY_DIM),A.rope_msc);
        rope_bwd<<<dim3(seq_len,NUM_KV_HEADS),64>>>(dK,seq_len,NUM_KV_HEADS,HEAD_DIM,ROTARY_DIM,ROPE_THETA,ROPE_INVF,rope_half(HEAD_DIM,ROTARY_DIM),A.rope_msc);
        HIP_CHECK(hipStreamSynchronize(0));

        float *dQ_pre,*dK_pre,*dV_pre;
        HIP_CHECK(cmalloc(&dQ_pre,(size_t)seq_len*Q_DIM*sizeof(float))); HIP_CHECK(cmalloc(&dK_pre,(size_t)seq_len*KV_DIM*sizeof(float)));
        HIP_CHECK(cmalloc(&dV_pre,(size_t)seq_len*KV_DIM*sizeof(float)));
        const int QNW=A.qk_full?Q_DIM:HEAD_DIM, KNW=A.qk_full?KV_DIM:HEAD_DIM;   // кн102
        float *g_qn,*g_kn; HIP_CHECK(cmalloc(&g_qn,QNW*sizeof(float))); HIP_CHECK(hipMemset(g_qn,0,QNW*sizeof(float)));
        HIP_CHECK(cmalloc(&g_kn,KNW*sizeof(float))); HIP_CHECK(hipMemset(g_kn,0,KNW*sizeof(float)));
        if(A.qk_norm && A.qk_full){ rmsnorm_bwd<<<seq_len,256>>>(dQ,Qn,W.qn,rQ,dQ_pre,g_qn,Q_DIM,weight_type(W.qn)); rmsnorm_bwd<<<seq_len,256>>>(dK,Kn,W.kn,rK,dK_pre,g_kn,KV_DIM,weight_type(W.kn)); }
        else if(A.qk_norm){ head_rmsnorm_bwd<<<dim3(seq_len,NUM_HEADS),64>>>(dQ,Qn,W.qn,rQ,dQ_pre,g_qn,NUM_HEADS,HEAD_DIM,weight_type(W.qn));
            head_rmsnorm_bwd<<<dim3(seq_len,NUM_KV_HEADS),64>>>(dK,Kn,W.kn,rK,dK_pre,g_kn,NUM_KV_HEADS,HEAD_DIM,weight_type(W.kn));
            head_norm_dw<<<64,256>>>(dQ,Qn,g_qn,seq_len*NUM_HEADS,HEAD_DIM); head_norm_dw<<<64,256>>>(dK,Kn,g_kn,seq_len*NUM_KV_HEADS,HEAD_DIM); }
        else { HIP_CHECK(hipMemcpy(dQ_pre,dQ,(size_t)seq_len*Q_DIM*4,hipMemcpyDeviceToDevice)); HIP_CHECK(hipMemcpy(dK_pre,dK,(size_t)seq_len*KV_DIM*4,hipMemcpyDeviceToDevice)); }
        if(A.v_norm) v_norm_bwd<<<dim3(seq_len,NUM_KV_HEADS),64>>>(dV,Vn,rV,dV_pre,NUM_KV_HEADS,HEAD_DIM);
        else HIP_CHECK(hipMemcpy(dV_pre,dV,(size_t)seq_len*KV_DIM*4,hipMemcpyDeviceToDevice));
        HIP_CHECK(hipStreamSynchronize(0));

        float *d_normed1_q,*d_normed1_k,*d_normed1;
        HIP_CHECK(cmalloc(&d_normed1_q,x_n*sizeof(float))); HIP_CHECK(cmalloc(&d_normed1_k,x_n*sizeof(float))); HIP_CHECK(cmalloc(&d_normed1,x_n*sizeof(float)));
        float* g_q=galloc(kp+"q_proj",(size_t)Q_DIM*HIDDEN);
        float* g_k=galloc(kp+"k_proj",(size_t)KV_DIM*HIDDEN);
        float* g_v=nullptr;

        if(HAS_V_PROJ){
            float* d_normed1_v; HIP_CHECK(cmalloc(&d_normed1_v,x_n*sizeof(float)));
            g_v=galloc(kp+"v_proj",(size_t)KV_DIM*HIDDEN);
            linear_dw(blas,dQ_pre,normed1,g_q,seq_len,HIDDEN,Q_DIM); if(!skip_dx0) linear_dx(blas,dQ_pre,W.q,d_normed1_q,seq_len,HIDDEN,Q_DIM);
            linear_dw(blas,dK_pre,normed1,g_k,seq_len,HIDDEN,KV_DIM); if(!skip_dx0) linear_dx(blas,dK_pre,W.k,d_normed1_k,seq_len,HIDDEN,KV_DIM);
            linear_dw(blas,dV_pre,normed1,g_v,seq_len,HIDDEN,KV_DIM); if(!skip_dx0) linear_dx(blas,dV_pre,W.v,d_normed1_v,seq_len,HIDDEN,KV_DIM);
            lora_bwd(blas,l,0,normed1,dQ_pre,skip_dx0?nullptr:d_normed1_q,seq_len,lr,step); lora_bwd(blas,l,1,normed1,dK_pre,skip_dx0?nullptr:d_normed1_k,seq_len,lr,step);
            lora_bwd(blas,l,2,normed1,dV_pre,skip_dx0?nullptr:d_normed1_v,seq_len,lr,step);
            if(!skip_dx0){ HIP_CHECK(hipMemcpy(d_normed1,d_normed1_q,x_n*sizeof(float),hipMemcpyDeviceToDevice));
            add_kernel<<<(x_n+255)/256,256>>>(d_normed1,d_normed1_k,x_n);
            add_kernel<<<(x_n+255)/256,256>>>(d_normed1,d_normed1_v,x_n); HIP_CHECK(hipStreamSynchronize(0)); }
            cfree(d_normed1_v);
        } else {
            add_kernel<<<((size_t)seq_len*KV_DIM+255)/256,256>>>(dK_pre,dV_pre,(size_t)seq_len*KV_DIM); HIP_CHECK(hipStreamSynchronize(0));
            linear_dw(blas,dQ_pre,normed1,g_q,seq_len,HIDDEN,Q_DIM); if(!skip_dx0) linear_dx(blas,dQ_pre,W.q,d_normed1_q,seq_len,HIDDEN,Q_DIM);
            linear_dw(blas,dK_pre,normed1,g_k,seq_len,HIDDEN,KV_DIM); if(!skip_dx0) linear_dx(blas,dK_pre,W.k,d_normed1_k,seq_len,HIDDEN,KV_DIM);
            lora_bwd(blas,l,0,normed1,dQ_pre,skip_dx0?nullptr:d_normed1_q,seq_len,lr,step); lora_bwd(blas,l,1,normed1,dK_pre,skip_dx0?nullptr:d_normed1_k,seq_len,lr,step);   // кн103
            if(!skip_dx0){ HIP_CHECK(hipMemcpy(d_normed1,d_normed1_q,x_n*sizeof(float),hipMemcpyDeviceToDevice));
            add_kernel<<<(x_n+255)/256,256>>>(d_normed1,d_normed1_k,x_n); HIP_CHECK(hipStreamSynchronize(0)); }
        }

        float* d_x_new; HIP_CHECK(cmalloc(&d_x_new,x_n*sizeof(float)));
        float* g_iln; HIP_CHECK(cmalloc(&g_iln,HIDDEN*sizeof(float))); HIP_CHECK(hipMemset(g_iln,0,HIDDEN*sizeof(float)));
        if(!skip_dx0){ if(W.iln) rmsnorm_bwd<<<seq_len,256>>>(d_normed1,n1,W.iln,r1,d_x_new,g_iln,HIDDEN,weight_type(W.iln)); else HIP_CHECK(hipMemcpy(d_x_new,d_normed1,x_n*4,hipMemcpyDeviceToDevice));
            HIP_CHECK(hipStreamSynchronize(0));
        add_kernel<<<(x_n+255)/256,256>>>(d_x_new,d_residual2,x_n); HIP_CHECK(hipStreamSynchronize(0)); }

        // кн104
        std::string pfx = "layer_"+std::to_string(l)+".";
        if(W.iln) adam_apply(blas,pfx+"input_layernorm",W.iln,g_iln,HIDDEN,lr,step);
        if(A.post_attn) adam_apply(blas,pfx+"post_attn_ln",W.paln,g_paln,HIDDEN,lr,step);
        if(W.pfln) adam_apply(blas,pfx+"pre_ffn_ln",W.pfln,g_pfln,HIDDEN,lr,step);
        if(A.post_ffn) adam_apply(blas,pfx+"post_ffn_ln",W.pofln,g_pofln,HIDDEN,lr,step);
        adam_apply(blas,pfx+"q_proj",W.q,g_q,(size_t)Q_DIM*HIDDEN,lr,step);
        adam_apply(blas,pfx+"k_proj",W.k,g_k,(size_t)KV_DIM*HIDDEN,lr,step);
        if(HAS_V_PROJ) adam_apply(blas,pfx+"v_proj",W.v,g_v,(size_t)KV_DIM*HIDDEN,lr,step);
        adam_apply(blas,pfx+"o_proj",W.o,g_o,(size_t)HIDDEN*Q_DIM,lr,step);
        if(A.qk_norm){ adam_apply(blas,pfx+"q_norm",W.qn,g_qn,QNW,lr,step); adam_apply(blas,pfx+"k_norm",W.kn,g_kn,KNW,lr,step); }
        adam_apply(blas,pfx+"gate_proj",W.gate,g_gate,(size_t)INTER*HIDDEN,lr,step);
        adam_apply(blas,pfx+"up_proj",W.up,g_up,(size_t)INTER*HIDDEN,lr,step);
        adam_apply(blas,pfx+"down_proj",W.down,g_down,(size_t)HIDDEN*INTER,lr,step);
        // кн105

        if(!skip_dx0) HIP_CHECK(hipMemcpy(d_x,d_x_new,x_n*sizeof(float),hipMemcpyDeviceToDevice));
        if(g_gdiag){ std::vector<float> h(x_n); HIP_CHECK(hipMemcpy(h.data(),d_x,x_n*sizeof(float),hipMemcpyDeviceToHost));
            std::vector<double> rn(seq_len,0); for(int t=0;t<seq_len;++t){ double a=0; for(int d=0;d<HIDDEN;++d){ double v=h[(size_t)t*HIDDEN+d]; a+=v*v; } rn[t]=sqrt(a); }
            int mt=0; for(int t=1;t<seq_len;++t) if(!(rn[t]<=rn[mt])) mt=t;
            std::vector<double> srt=rn; std::sort(srt.begin(),srt.end());
            fprintf(g_log?g_log:stdout,"gdiag L%02d %s |d_x| max %.3e @pos %d (tok %d) | pos0 %.3e | median %.3e | pos t0=%d %.3e\n", l, is_full?"full":"slid", rn[mt], mt, tokens[mt], rn[0], srt[seq_len/2], t0, rn[t0]); }

        if(g_log){
            lgv("x_in",l,layer_x,x_n,seq_len); lgv("dpre",l,dpre,x_n,seq_len); lgr("r4 post_ffn",l,r4,seq_len); lgv("d_mlp_out",l,d_mlp_out,x_n,seq_len);
            lgv("d_act",l,d_act,inter_n,seq_len); lgv("d_gate",l,d_gate,inter_n,seq_len); lgv("d_up",l,d_up,inter_n,seq_len);
            lgr("r3 pre_ffn",l,r3,seq_len); lgv("d_normed3",l,d_normed3,x_n,seq_len); lgv("d_res2_mlp",l,d_residual2_mlp,x_n,seq_len); lgv("d_res2",l,d_residual2,x_n,seq_len);
            lgr("r2 post_attn",l,r2,seq_len); lgv("d_attn_proj",l,d_attn_proj,x_n,seq_len); lgv("d_attn_out",l,d_attn_out,(size_t)seq_len*Q_DIM,seq_len);
            lgv("dQ",l,dQ,(size_t)seq_len*Q_DIM,seq_len); lgv("dK",l,dK,(size_t)seq_len*KV_DIM,seq_len); lgv("dV",l,dV,(size_t)seq_len*KV_DIM,seq_len);
            lgr("rQ",l,rQ,(size_t)seq_len*NUM_HEADS); lgr("rK",l,rK,(size_t)seq_len*NUM_KV_HEADS); lgr("rV",l,rV,(size_t)seq_len*NUM_KV_HEADS);
            lgv("dQ_pre",l,dQ_pre,(size_t)seq_len*Q_DIM,seq_len); lgv("dK_pre",l,dK_pre,(size_t)seq_len*KV_DIM,seq_len); lgv("dV_pre",l,dV_pre,(size_t)seq_len*KV_DIM,seq_len);
            lgr("r1 input",l,r1,seq_len); lgv("d_normed1",l,d_normed1,x_n,seq_len); lgv("d_x_new",l,d_x_new,x_n,seq_len); fflush(g_log);
        }
        vram_sample();
        for(float* p : {layer_x,residual1,normed1,n1,r1,Q,K,V,Q_rope,K_rope,Qn,Kn,Vn,rQ,rK,rV,attn_out,P,
                         attn_proj,normed2,n2,r2,residual2,normed3,n3,r3,gate,up,act,mlp_out,normed4,n4,r4,
                         pre_scalar,dpre,d_mlp_out,g_pofln,d_act,g_down,d_gate,d_up,d_normed3_a,d_normed3_b,d_normed3,
                         g_gate,g_up,d_residual2_mlp,g_pfln,d_residual2,d_attn_proj,g_paln,d_attn_out,g_o,dQ,dK,dV,
                         dQ_pre,dK_pre,dV_pre,g_qn,g_kn,d_normed1_q,d_normed1_k,d_normed1,g_q,g_k,g_v,d_x_new,g_iln,
                         W.iln,W.paln,W.pfln,W.pofln,W.q,W.k,W.v,W.o,W.qn,W.kn,W.gate,W.up,W.down,W.ls,W.qb,W.kb,W.vb})
            if(p && !g_res_ptrs.count(p)) cfree(p);
        if(g_use_arena) layer_end();
        if(!VERBOSE&&!g_gdiag&&g_tty) fprintf(stderr, "backward+Adam layer %d/0 done\r", l);
        if(g_prog && !g_is_ev) printf("prog b %d\n", N_LAYERS-l);
    }
    if(bwd_ran && lo_tr>0 && g_prog && !g_is_ev) printf("prog b %d\n", N_LAYERS);   // кн106
    fprintf(stderr, "\n");
    if(bwd_ran && g_use_arena) g_arena_ok=true;
    if(bwd_ran) g_run_len=std::max(g_run_len,(size_t)seq_len);
    if(bwd_ran){ size_t fr=0,tt=0; if(hipMemGetInfo(&fr,&tt)==hipSuccess && step>2){ if(!g_free_ok||fr<g_free_ok) g_free_ok=fr; if(fr>g_free_max) g_free_max=fr; } rbf_fill(); } // кн107
    if(bwd_ran) g_pf_ready=true; // кн108
    for(float* p : {x,normed_final,r_final,n_final,d_normed_final,d_x,w_norm}) if(p && !g_res_ptrs.count(p)) cfree(p);
    float gn2=0; HIP_CHECK(hipMemcpy(&gn2,g_gn2,sizeof(float),hipMemcpyDeviceToHost));
    // кн109
    if(!is_ev && g_cpu_thr>0){ double st=elapsed()-t_step0; static double best=1e30, best_f=1.0; static bool lock=false;
        double per=st/std::max(1,seq_len);
        if(st>0 && !lock){ if(per<best){ best=per; best_f=g_gpu_frac; }
            else if(per>1.10*best && step>2){ g_gpu_frac=std::min(1.0,best_f+0.04); lock=true;
                printf("Процессор: доля карты зафиксирована на %.0f%%, дальнейшее участие процессора замедляло расчёт.\n",100*g_gpu_frac); }
            if(!lock){ if(g_cpu_wait > 0.03*st) g_gpu_frac += 0.02; else if(g_cpu_wait < 0.005*st) g_gpu_frac -= 0.02; }
            if(g_gpu_frac>1.0) g_gpu_frac=1.0; if(g_gpu_frac<0.70) g_gpu_frac=0.70; } }
    if(!is_ev){ g_prev_gn=sqrt((double)gn2);                       // кн110
        g_gn_ema = (g_gn_ema<=0) ? g_prev_gn : 0.9*g_gn_ema + 0.1*g_prev_gn; }
    if(!is_ev){ std::map<int,std::pair<int,size_t>> lk; for(auto& kv:g_live) if(kv.second.step==step && qi>0){ auto& e=lk[kv.second.line]; e.first++; e.second+=kv.second.n; }
      for(auto& [ln,e] : lk) fprintf(stderr,"Утечка на шаге %d: строка %d, блоков %d, %.2f МиБ.\n", step, ln, e.first, e.second/1048576.0); }
    if(!is_ev){ std::vector<std::pair<size_t,int>> pk; for(auto& kv:g_peak_by_line) pk.push_back({kv.second,kv.first});
        std::sort(pk.rbegin(),pk.rend()); std::string pb;
        for(size_t i=0;i<pk.size() && i<6;++i) pb += (i?", ":"") + std::to_string(pk[i].second) + ":" + std::to_string(pk[i].first>>20) + "M";
        { std::vector<std::pair<size_t,int>> dk; size_t dt=0; for(auto& kv:g_dpeak){ dk.push_back({kv.second,kv.first}); dt+=kv.second; } std::sort(dk.rbegin(),dk.rend()); std::string db;
          for(size_t i=0;i<dk.size() && i<8;++i) db += (i?", ":"") + std::to_string(dk[i].second) + ":" + std::to_string(dk[i].first>>20) + "M";
          printf("Выделения видеопамяти в пик: всего %zu МиБ, по строкам %s\n", dt>>20, db.c_str()); }
        printf("Пик %zu МиБ: собственные выделения %zu МиБ по строкам %s; обучаемая часть %zu МиБ, подкачка весов %zu МиБ, остальное — драйвер и рабочий стол.\n",
               g_vram_peak>>20, g_peak_live>>20, pb.c_str(), (g_res.size()? [&]{ size_t b=0; for(auto&[k,rr]:g_res) { auto& sl=g_weight_layout.at(k); b+=(sl.ntr?sl.ntr:sl.n)*12; } return b>>20; }() : 0ul), (size_t)(g_stg_cap>>20));
    }
    if(!is_ev){ size_t lv=0,pl=0; for(auto& kv:g_live) lv+=kv.second.n; pl=g_ar_cap; // кн111
      printf("mem live %.1f MiB (%zu блоков), pool %.1f MiB (%zu) prefetch %zu/%zu lt %zu own %zu\n", lv/1048576.0, g_live.size(), pl/1048576.0, g_ar_miss, g_pf_hit, g_pf_miss, g_use_lt?g_lt_plans.size():0, g_own_plan.size()); g_pf_hit=g_pf_miss=0; }
    if(!is_ev && g_gprobe){ for(auto& [k,p] : g_gp){ float h[2]; HIP_CHECK(hipMemcpy(h,p,sizeof h,hipMemcpyDeviceToHost)); HIP_CHECK(hipMemset(p,0,sizeof h));
        printf("gprobe %d %s %.6e %.6e\n", step, k.c_str(), h[0], h[1]); } }
    if(!is_ev) printf("--- step %d done [%.1fs, total %.1fs, gnorm %.4e, resident %zu тензоров, RAM cache %.2f ГиБ, VRAM peak %.0f/%.0f МиБ, max edge/junc/mem %d/%d/%d C, fan max %d об/мин, по шине %.2f ГиБ, процессор %.2f с (ожидание %.2f с, доля карты %.0f%% строк), phases %.2f/%.2f/%.2f s] ---\n", step, elapsed()-t_step0, elapsed(), sqrt(gn2), g_res.size(), g_ram_bytes/1073741824.0, g_vram_peak/1048576.0, g_vram_total/1048576.0, g_t_edge/1000, g_t_junc/1000, g_t_mem/1000, g_fan, g_h2d/1073741824.0, g_cpu_secs, g_cpu_wait, 100*g_gpu_frac, t_f-t_step0, t_h-t_f, elapsed()-t_h);
    if(g_gen){ seq_len=(int)tokens.size(); if(gen_stop) break; }
    const int evs = is_ev ? (int)(std::upper_bound(ev_end.begin(),ev_end.end(),ev)-ev_end.begin()) : -1;   // кн112
    if(evs>0 && ev==ev_end[evs]-1){ double L=g_eval_sum/g_eval_tok;   // кн113
        printf("=== EVAL step %d set %d: %d packs, %.0f answer tokens, mean loss %.4f nats (perplexity %.2f), acc %.4f ===\n", step, evs, ev_end[evs]-ev_end[evs-1], g_eval_tok, L, exp(L), g_eval_hit/g_eval_tok);
        fflush(stdout); }
    if(evs==0 && ev==ev_end[0]-1){ double L=g_eval_sum/g_eval_tok;
        printf("=== EVAL step %d: %d packs, %.0f answer tokens, mean loss %.4f nats (perplexity %.2f), acc %.4f ===\n", step, ev_end[0], g_eval_tok, L, exp(L), g_eval_hit/g_eval_tok);
        // кн114
        if(step==0 && !getenv("TIER_NO_SELFCHECK") && L > 0.6*log((double)VOCAB) && g_eval_hit/g_eval_tok < 0.05){
            printf("Самопроверка не пройдена: потеря (loss) исходной модели %.2f, при случайном угадывании %.2f (словарь %d). "
                   "Движок считает эту модель неверно, обучение остановлено. К обращению рекомендуется приложить журнал запуска.\n", L, log((double)VOCAB), VOCAB);
            fflush(stdout); exit(3); }
        if(!best_path.empty() && step>0 && L<best_loss){ best_loss=L; resident_flush(); if(g_lora_r) lora_save(g_checkpoint_path); std::string c="cp --sparse=always '"+std::string(argv[5])+"' '"+best_path+"'";
            int rc=system(c.c_str()); printf("Лучший снимок весов: потеря (loss) %.4f на шаге %d, файл %s%s\n", L, step, best_path.c_str(), rc?" (копирование не выполнено)":""); }
        else if(step>0 && L<best_loss) best_loss=L;
        if(g_stop_after>0 && step>0){
            static int idle=0; static double prev_best=1e30;
            if(L < prev_best-1e-4){ prev_best=L; idle=0; } else ++idle;
            if(idle>=g_stop_after){
                printf("Остановка: потеря (loss) на валидации не улучшилась (валидаций подряд: %d), лучшее значение %.4f.\n", idle, best_loss);
                fflush(stdout); stop_early=true; } }
        fflush(stdout); }
    if(stop_early && (!is_ev || ev==(int)packs_e.size()-1)) break;   // кн115
    if(!is_ev && !eval_mode && flush_every>0 && (step%flush_every==0 || step==step0+n_steps-1)){ double tf=elapsed(); resident_flush(); if(g_lora_r) lora_save(g_checkpoint_path);
        if(!step_file.empty()){ FILE* sf=fopen(step_file.c_str(),"w"); if(sf){ fprintf(sf,"%d\n",step+1); fclose(sf); } }
        printf("Снимок весов сохранён: шаг %d [%.1f с]\n", step, elapsed()-tf); fflush(stdout); }
    g_checkpoint_exists_with_data = true; // кн116
    }

    g_eval=eval_mode; g_is_ev=false;
    if(g_eval&&!g_gen) printf("=== EVAL: %d examples, %.0f answer tokens, mean loss %.4f nats (perplexity %.2f), acc %.4f ===\n", n_steps, g_eval_tok, g_eval_sum/g_eval_tok, exp(g_eval_sum/g_eval_tok), g_eval_hit/g_eval_tok);
    if(!g_eval && !g_res.empty()){ double t0=elapsed(); resident_flush(); printf("Обучаемая часть записана на диск: тензоров %zu [%.1f с].\n", g_res.size(), elapsed()-t0);
        conf_calibrate(); }
    if(!g_rowprobe.empty()){ FILE* f=fopen(g_rowprobe.c_str(),"w"); size_t nm=0;   // кн117
        if(f){ for(auto& [k,d] : g_rowacc){ size_t rows=g_weight_layout.at(k).rows; std::vector<float> h(rows); HIP_CHECK(hipMemcpy(h.data(),d,rows*4,hipMemcpyDeviceToHost));
                fprintf(f,"%s",k.c_str()); for(float x : h) fprintf(f," %.4g",x); fprintf(f,"\n"); ++nm; } fclose(f); }
        fprintf(stderr,"Замер строк записан: %s (матриц %zu).\n",g_rowprobe.c_str(),nm); }
    if(!g_eval && g_lora_r){ lora_save(g_checkpoint_path); printf("LoRA: адаптеры записаны в %s.\n", g_checkpoint_path.c_str()); }
    {
        printf("=== one full training step complete === [%.1fs total, step=%d, lr=%g]\n", elapsed(), step0+n_steps-1, lr);
    }

    hipblasDestroy(blas);
    return 0;
}
