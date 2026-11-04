#!/usr/bin/env python3
# а1
from pathlib import Path
import subprocess, tempfile, sys
ROOT=Path(__file__).resolve().parent
SRC=(ROOT/'tier.cpp').read_text()
def fn(token):
    start=SRC.index(token); brace=SRC.index('{',start); depth=0
    for i in range(brace,len(SRC)):
        if SRC[i]=='{': depth+=1
        elif SRC[i]=='}':
            depth-=1
            if depth==0: return SRC[start:i+1]
    raise RuntimeError('unterminated '+token)
funcs='\n\n'.join(fn(x) for x in [
    'static void pread_all(', 'static void pwrite_all(',
    'static void adam_load_mv(', 'static void adam_store_mv(', 'static void resident_flush('
])
h=r'''#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <iostream>
#include <limits>
#include <map>
#include <string>
#include <thread>
#include <unistd.h>
#include <sys/stat.h>
#include <vector>
using std::size_t;
struct TensorSlot { size_t offset_floats=0,n=0; int tier=-1; size_t rows=1,rlen=0,ntr=0; };
static std::map<std::string,TensorSlot> g_weight_layout;
static size_t g_weight_layout_total_floats=0;
static std::string g_adam_path,g_checkpoint_path;
static bool g_compact=false,g_checkpoint_exists_with_data=false;
static std::map<std::string,size_t> g_c1off;
static const size_t C1_HDR=64u<<10;
struct Resident { float *w=nullptr,*v=nullptr,*m=nullptr; };
static std::map<std::string,Resident> g_res;
using hipError_t=int;
static constexpr int hipSuccess=0,hipMemcpyHostToDevice=1,hipMemcpyDeviceToHost=2;
static int hipMemcpy(void* d,const void* s,size_t n,int){std::memcpy(d,s,n);return hipSuccess;}
static int hipMemset(void* d,int x,size_t n){std::memset(d,x,n);return hipSuccess;}
#define HIP_CHECK(x) do { if((x)!=hipSuccess){std::cerr<<"CPU shim memory op failed\\n";std::exit(2);} } while(0)
static void checkpoint_store_weight(const std::string&,float*);
''' + funcs + r'''
static void checkpoint_store_weight(const std::string&,float*){}
static void ck(bool x,const char* s){if(!x){std::cerr<<"FAIL: "<<s<<"\\n";std::exit(1);}}
static std::string tempfile(size_t n){char p[]="/tmp/tier_adam_io_XXXXXX";int fd=mkstemp(p);ck(fd>=0,"mkstemp");ck(ftruncate(fd,(off_t)n)==0,"ftruncate");close(fd);return p;}
int main(){
  const std::string key="layer_0.test";
  // Non-compact format: m and v are separate contiguous FP32 arrays at their defined offsets.
  g_compact=false;g_weight_layout_total_floats=16;g_checkpoint_exists_with_data=false;
  TensorSlot s;s.offset_floats=2;s.n=4;g_weight_layout[key]=s;
  g_adam_path=tempfile(256);
  float m[4]={1e-31f,-2.5e-20f,0.33333334f,std::numeric_limits<float>::denorm_min()};
  float v[4]={2e-30f,1e-18f,0.25f,3*std::numeric_limits<float>::denorm_min()};
  adam_store_mv(key,m,v);float mr[4]={},vr[4]={};adam_load_mv(key,mr,vr);
  ck(std::memcmp(m,mr,sizeof m)==0,"noncompact FP32 m round trip");
  ck(std::memcmp(v,vr,sizeof v)==0,"noncompact FP32 v round trip");
  unlink(g_adam_path.c_str());

  // Sliced tensor: I/O must honour ntr and not require storage for the frozen tail.
  s.n=5;s.ntr=2;s.offset_floats=1;g_weight_layout[key]=s;g_adam_path=tempfile(256);
  float ms[2]={1e-35f,0.125f},vs[2]={1e-30f,0.75f};
  adam_store_mv(key,ms,vs);float mrs[2]={},vrs[2]={};adam_load_mv(key,mrs,vrs);
  ck(std::memcmp(ms,mrs,sizeof ms)==0,"sliced FP32 m round trip");
  ck(std::memcmp(vs,vrs,sizeof vs)==0,"sliced FP32 v round trip");
  unlink(g_adam_path.c_str());

  // Compact Tier1 checkpoint: actual resident_flush writes [w,m,v]; actual adam_load_mv reloads m and v.
  s.n=4;s.ntr=0;s.offset_floats=0;g_weight_layout[key]=s;
  g_compact=true;g_checkpoint_exists_with_data=true;g_c1off[key]=C1_HDR;
  g_checkpoint_path=tempfile(C1_HDR+4*3*sizeof(float));
  float w[4]={0.25f,-0.5f,1.125f,-2.0f};
  float cm[4]={1e-32f,-1e-19f,0.875f,std::numeric_limits<float>::denorm_min()};
  float cv[4]={1e-28f,2e-17f,0.0625f,5*std::numeric_limits<float>::denorm_min()};
  g_res.clear();g_res[key]=Resident{w,cv,cm};
  resident_flush();
  float cmr[4]={},cvr[4]={},wr[4]={};adam_load_mv(key,cmr,cvr);
  ck(std::memcmp(cm,cmr,sizeof cm)==0,"compact FP32 m round trip");
  ck(std::memcmp(cv,cvr,sizeof cv)==0,"compact FP32 v round trip");
  int fd=open(g_checkpoint_path.c_str(),O_RDONLY);ck(fd>=0,"open compact checkpoint");pread_all(fd,wr,sizeof wr,C1_HDR);close(fd);
  ck(std::memcmp(w,wr,sizeof w)==0,"compact checkpoint weight layout");
  unlink(g_checkpoint_path.c_str());
  std::cout<<"PASS actual adam_store_mv/adam_load_mv non-compact FP32 state\n";
  std::cout<<"PASS actual Adam state I/O with sliced ntr\n";
  std::cout<<"PASS actual resident_flush + adam_load_mv compact [w|m|v] layout\n";
  std::cout<<"PASS subnormal and small FP32 moments survive checkpoint/file round trip\n";
  return 0;
}
'''
with tempfile.TemporaryDirectory(prefix='tier_adam_io_') as td:
    cpp=Path(td)/'test.cpp'; exe=Path(td)/'test'; cpp.write_text(h)
    c=subprocess.run(['g++','-D_GNU_SOURCE','-O2','-std=c++17','-pthread',str(cpp),'-o',str(exe)],capture_output=True,text=True)
    if c.returncode: print(c.stdout+c.stderr,file=sys.stderr);sys.exit(c.returncode)
    r=subprocess.run([str(exe)],capture_output=True,text=True)
    print(r.stdout,end='')
    if r.returncode: print(r.stderr,file=sys.stderr);sys.exit(r.returncode)
