#!/bin/bash
# а1
cd "$(dirname "$0")"
F=tier.cpp; W=both
for a in "$@"; do case "$a" in amd|nvidia) W=$a;; *.cpp) F=$a;; esac; done
RC=0
run(){ local name=$1; shift
  local out; out=$("$@" 2>&1)
  local e; e=$(printf '%s\n' "$out" | grep -c 'error:')
  if [ "$e" = 0 ]; then echo "$F под $name — разбирается начисто"
  else echo "$F под $name — ошибок: $e"
       printf '%s\n' "$out" | grep 'error:' | head -20; RC=1; fi }
[ "$W" = both ] || [ "$W" = amd ] && run AMD \
  clang++ -x hip -nogpuinc -nogpulib --offload-arch=gfx1200 -fsyntax-only -std=c++17 -Istub -I. "$F"
[ "$W" = both ] || [ "$W" = amd ] && run "AMD без hipBLASLt" \
  clang++ -x hip -nogpuinc -nogpulib --offload-arch=gfx1030 -fsyntax-only -std=c++17 -DTIER_NO_LT -Istub -I. "$F"
[ "$W" = both ] || [ "$W" = nvidia ] && run nvidia \
  clang++ -x cuda -nogpuinc -nogpulib --cuda-gpu-arch=sm_86 -fsyntax-only -std=c++17 -DTIER_CUDA -Istub_cuda -I. "$F"
exit $RC
