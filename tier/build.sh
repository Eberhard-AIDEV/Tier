#!/bin/bash
# а1
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
OUT="${TIER_OUT:-${TIER_DATA:-$HERE}/tier}"
# а2
[ -z "${NVCC_CCBIN:-}" ] && [ -f /etc/profile.d/cuda.sh ] && . /etc/profile.d/cuda.sh
for d in /opt/rocm/bin /opt/cuda/bin /usr/local/cuda/bin; do [ -d "$d" ] && case ":$PATH:" in *":$d:"*) ;; *) PATH="$PATH:$d" ;; esac; done
export PATH
# а3
CPUF=; grep -qw avx2 /proc/cpuinfo 2>/dev/null && grep -qw fma /proc/cpuinfo 2>/dev/null && CPUF="-mavx2 -mfma"

# б1
vendor(){ local amd=0 nv=0 m c
    [ -n "${TIER_VENDOR:-}" ] && { echo "$TIER_VENDOR"; return; }
    for c in /sys/class/drm/card[0-9]*/device; do
        case "$(cat "$c/vendor" 2>/dev/null)" in
            0x1002) m=$(cat "$c/mem_info_vram_total" 2>/dev/null || echo 1); [ "$m" -gt "$amd" ] && amd=$m ;;
            0x10de) nv=1 ;;
        esac; done
    if [ "$nv" = 1 ]; then m=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | sort -n | tail -1 | tr -d ' ')
        case "$m" in ''|*[!0-9]*) m=1 ;; esac; nv=$(( m * 1048576 )); fi
    if [ "$amd" -eq 0 ] && [ "$nv" -eq 0 ]; then       # б2
        { [ -d /sys/module/amdgpu ] || command -v rocminfo >/dev/null; } && { echo amd; return; }
        { [ -d /proc/driver/nvidia ] || command -v nvidia-smi >/dev/null; } && echo nvidia; return; fi
    [ "$nv" -gt "$amd" ] && echo nvidia || echo amd; }

# в1
amd_arch(){ local best=-1 arch= n v mem
    for n in /sys/class/kfd/kfd/topology/nodes/*; do
        v=$(awk '$1=="gfx_target_version"{print $2}' "$n/properties" 2>/dev/null); [ -n "$v" ] && [ "$v" != 0 ] || continue
        # в2
        mem=$(for b in "$n"/mem_banks/*/properties; do awk '$1=="heap_type"{h=$2} $1=="size_in_bytes"{s=$2} END{if(h==1||h==2) print s+0; else print 0}' "$b" 2>/dev/null; done | awk '{s+=$1} END{print s+0}')
        [ "$mem" -gt "$best" ] || continue
        best=$mem; arch=$(printf 'gfx%d%d%x' $((v/10000)) $((v/100%100)) $((v%100)))
    done
    [ -n "$arch" ] || arch=$( (amdgpu-arch || rocm_agent_enumerator || rocminfo) 2>/dev/null | grep -o 'gfx[0-9a-f]\+' | grep -v '^gfx000$' | head -1)
    echo "$arch"; }

[ "${1:-}" = --vendor ] && { vendor; exit 0; }       # в3
# в4
U_VENDOR=${TIER_VENDOR:-}; U_ARCH=${TIER_ARCH:-}; U_OV=${HSA_OVERRIDE_GFX_VERSION:-}   # в5
eval "$(python3 "$HERE/hwprobe.py" --env 2>/dev/null)"
if [ -n "$U_ARCH" ]; then TIER_ARCH=$U_ARCH; TIER_GPU_ARCH=$U_ARCH; HSA_OVERRIDE_GFX_VERSION=$U_OV; TIER_ARCH_ERR=; fi
[ -n "$U_VENDOR" ] && TIER_VENDOR=$U_VENDOR
mkdir -p "$(dirname "$OUT")"; ENVTXT=
exec 9> "$OUT.lock"; flock -n 9 || { echo "Движок уже собирается в другом окне или другой командой. Требуется дождаться завершения сборки."; exit 1; }   # в6
V=$(vendor)
[ -n "$V" ] || { command -v hipcc >/dev/null && V=amd || V=nvidia; }
# в7
if [ "$V" = amd ] && ! command -v hipcc >/dev/null; then
    echo "Карта AMD обнаружена, компилятор hipcc отсутствует. Выполняется установка ROCm."; python3 "$HERE/hwprobe.py" --deps
    command -v hipcc >/dev/null || { echo "Установка ROCm не удалась, причина указана выше."; exit 1; }; fi
if [ "$V" = nvidia ] && ! command -v nvcc >/dev/null; then
    echo "Карта nvidia обнаружена, компилятор nvcc отсутствует. Выполняется установка CUDA."; python3 "$HERE/hwprobe.py" --deps
    command -v nvcc >/dev/null || { echo "Установка CUDA не удалась, причина указана выше."; exit 1; }; fi

if [ "$V" = amd ]; then
    # в8
    [ -n "${TIER_ARCH_ERR:-}" ] && { echo "$TIER_ARCH_ERR"; exit 1; }
    BUILD=${TIER_ARCH:-$(amd_arch)}; ARCH=${TIER_GPU_ARCH:-$BUILD}; OV=${HSA_OVERRIDE_GFX_VERSION:-}
    [ -n "$OV" ] && ENVTXT="HSA_OVERRIDE_GFX_VERSION=$OV"
    ROC=$(hipconfig --rocmpath 2>/dev/null || echo /opt/rocm)
    LT=-lhipblaslt; DEF=
    if ! { [ -f "$ROC/include/hipblaslt/hipblaslt.h" ] || [ -f /usr/include/hipblaslt/hipblaslt.h ]; }; then
        LT=; DEF=-DTIER_NO_LT; echo "hipBLASLt не найден: умножение FP32 выполняется собственными ядрами и hipBLAS, режим FP16×3 недоступен."; fi
    echo "Карта AMD ${ARCH:-?}${OV:+ (сборка как $BUILD, HSA_OVERRIDE_GFX_VERSION=$OV)}, сборка через hipcc."
    if ! hipcc -std=c++17 -O3 -march=native -Wno-unused-result $DEF ${BUILD:+--offload-arch=$BUILD} -I"$HERE" "$HERE/tier.cpp" -o "$OUT" -lhipblas $LT -pthread 2> "$OUT.log"; then
        if grep -q "invalid target ID\|unsupported.*arch\|unknown target CPU" "$OUT.log"; then
            echo "Установленный ROCm не поддерживает $BUILD: требуется более новая версия ROCm (gfx12 — с 6.4, gfx1150/1151 — с 6.3)."; fi
        grep -v "warning\|^ *[0-9]* |\|^ *|\|^ *\^\|warnings generated" "$OUT.log" | head -40; exit 1; fi
else
    # в9
    CC=${TIER_ARCH#sm_}; [ -n "$CC" ] || CC=$(nvidia-smi --query-gpu=memory.total,compute_cap --format=csv,noheader,nounits 2>/dev/null | sort -t, -k1 -n | tail -1 | cut -d, -f2 | tr -d ' .')
    NV=$(nvcc --version | sed -n 's/.*release \([0-9]*\)\..*/\1/p')
    if [ -n "$CC" ]; then
        GEN="-gencode arch=compute_$CC,code=[sm_$CC,compute_$CC]"     # в10
        # в11
        KNOWN=$(nvcc --list-gpu-arch 2>/dev/null | sed -n 's/^compute_\([0-9]*\)$/\1/p' | sort -n)
        if [ -n "$KNOWN" ] && ! printf '%s\n' $KNOWN | grep -qx "$CC"; then
            LOW=$(printf '%s\n' $KNOWN | awk -v c="$CC" '$1<=c' | tail -1)      # в12
            if [ -z "$LOW" ]; then
                echo "CUDA $NV не поддерживает карты с вычислительной способностью $((CC/10)).$((CC%10)): самая старая, которую она знает, — $(printf '%s\n' $KNOWN | head -1 | sed 's/\(.*\)\(.\)$/\1.\2/'). Требуется более старая CUDA."; exit 1; fi
            echo "CUDA $NV не знает sm_$CC: выполняется сборка PTX compute_$LOW, драйвер преобразует его под карту при первом запуске."
            GEN="-gencode arch=compute_$LOW,code=compute_$LOW"; fi
    else
        GEN="-arch=native"; echo "nvidia-smi не ответил: архитектура определяется средствами nvcc."; fi
    echo "Карта nvidia, вычислительная способность ${CC:-?}, CUDA ${NV:-?}, сборка через nvcc."
    try(){ nvcc -x cu -DTIER_CUDA -std=c++17 -O3 -Xcompiler "$CPUF -pthread -Wno-unused-result" -diag-suppress 177,1650 "$@" -I"$HERE" $GEN "$HERE/tier.cpp" -o "$OUT" -lcublas -lcublasLt 2> "$OUT.log"; }
    if ! try; then
        # г1
        if grep -q "unsupported GNU version\|unsupported clang version" "$OUT.log"; then ok=
            for c in $(compgen -c g++- | grep -E '^g\+\+-[0-9]+$' | sort -t- -k2 -nr | uniq); do command -v $c >/dev/null || continue   # г2
                echo "nvcc не принял системный g++, выполняется попытка с $c."; try -ccbin "$(command -v $c)" && { ok=1; break; }; done
            [ -n "$ok" ] || { echo "Подходящий g++ не найден, сборка выполняется с ключом -allow-unsupported-compiler."; try -allow-unsupported-compiler || { cat "$OUT.log"; exit 1; }; }
        else cat "$OUT.log"; exit 1; fi
    fi
    grep -i "error" "$OUT.log" | grep -v perror || true
fi
printf '%s' "${ENVTXT:+$ENVTXT
}" > "$OUT.env"
cat "$HERE"/{tier.cpp,gpu_compat.h,tier_wmma.h,quant.h,quant_tables.h} | sha1sum | cut -d" " -f1 > "$OUT.src"   # г3
python3 "$HERE/hwprobe.py" --fingerprint > "$OUT.hw" 2>/dev/null || : > "$OUT.hw"   # г4
echo "Движок собран: $OUT"
