#!/usr/bin/env python3
# а1
import subprocess, sys, tempfile, os
import numpy as np
from gguf import quants, constants

T = constants.GGMLQuantizationType
QTEST = r"""// проверка quant.h на процессоре: dq <тип> <in.bin> <cols> <out.f32> — разобрать блоки; q <тип> <in.f32> <cols> <out.bin> — квантовать
#include "quant.h"
#include <cstdio>
#include <vector>
#include <cstdlib>
int main(int c, char** v){
    int t=atoi(v[2]); int cols=atoi(v[4]); FILE* f=fopen(v[3],"rb"); std::vector<uint8_t> in; { uint8_t b[65536]; size_t n; while((n=fread(b,1,sizeof b,f))>0) in.insert(in.end(),b,b+n); } fclose(f);
    FILE* o=fopen(v[5],"wb");
    if(v[1][0]=='d'){ size_t rb=q_row_bytes(t,cols); size_t rows=in.size()/rb; std::vector<float> y(cols);
        for(size_t r=0;r<rows;++r){ for(int k=0;k<cols;++k) y[k]=q_val(in.data()+r*rb,t,k); fwrite(y.data(),4,cols,o); } }
    else { const float* x=(const float*)in.data(); size_t rows=in.size()/4/cols; std::vector<uint8_t> y(q_row_bytes(t,cols));
        for(size_t r=0;r<rows;++r){ if(!q_row(t,x+r*cols,y.data(),cols)) return 2; fwrite(y.data(),1,y.size(),o); } }
    fclose(o); return 0; }
"""
_qd = tempfile.mkdtemp()
QT = os.path.join(_qd, 'qtest')
open(QT + '.cpp', 'w').write(QTEST)
if subprocess.run(['g++', '-O2', '-std=c++17', '-I', os.path.dirname(os.path.abspath(__file__)), QT + '.cpp', '-o', QT]).returncode:
    sys.exit('Не удалось собрать проверочную программу.')
rng = np.random.default_rng(7)
COLS = 512


def blocks(t, nb):
    bs, by = constants.GGML_QUANT_SIZES[t]
    b = rng.integers(0, 256, size=(nb, by), dtype=np.uint8)
    halfs = {T.Q4_0: [0], T.Q4_1: [0, 2], T.Q5_0: [0], T.Q5_1: [0, 2], T.Q8_0: [0], T.Q2_K: [80, 82], T.Q3_K: [108], T.Q4_K: [0, 2],
             T.Q5_K: [0, 2], T.Q6_K: [208], T.IQ2_XXS: [0], T.IQ2_XS: [0], T.IQ2_S: [0], T.IQ3_XXS: [0], T.IQ3_S: [0], T.IQ1_S: [0],
             T.IQ4_NL: [0], T.IQ4_XS: [0], T.TQ1_0: [52], T.TQ2_0: [64]}.get(t, [])
    for o in halfs:
        h = (rng.standard_normal(nb) * 0.02).astype(np.float16).view(np.uint8).reshape(nb, 2)
        b[:, o:o + 2] = h
    if t == T.IQ1_M:   # б1
        h = (rng.standard_normal(nb) * 0.02).astype(np.float16).view(np.uint16)
        sc = b[:, 48:56].copy().view(np.uint16).reshape(nb, 4)
        for k, sh in enumerate((12, 8, 4, 0)):
            sc[:, k] = (sc[:, k] & 0x0FFF) | (((h >> (12 - 4 * k)) & 0xF) << 12)
        b[:, 48:56] = sc.view(np.uint8).reshape(nb, 8)
    if t == T.MXFP4:
        b[:, 0] = rng.integers(100, 140, size=nb)
    return b


def ref(t, b):
    if t == T.IQ1_M:
        # в1
        cls = quants.IQ1_M
        cls.init_grid()
        n_blocks = b.shape[0]
        qs, rest = np.hsplit(b, [256 // 8])
        qh, scales = np.hsplit(rest, [256 // 16])
        scales = scales.copy().view(np.uint16)
        d = (scales.reshape((n_blocks, 4)) & np.uint16(0xF000)) >> np.array([12, 8, 4, 0], dtype=np.uint16).reshape((1, 4))
        d = d[..., 0] | d[..., 1] | d[..., 2] | d[..., 3]
        d = d.view(np.float16).astype(np.float32).reshape((n_blocks, 1))
        scales = scales.reshape(n_blocks, -1, 1) >> np.array([0, 3, 6, 9], dtype=np.uint16).reshape((1, 1, 4))
        scales = (scales & 0x07).reshape((n_blocks, -1))
        dl = (d * (2 * scales + 1)).reshape((n_blocks, -1, 2, 1, 1))
        qh = qh.reshape((n_blocks, -1, 1)) >> np.array([0, 4], dtype=np.uint8).reshape((1, 1, 2))
        q2 = qs.astype(np.uint16) | ((qh & 0x07).astype(np.uint16) << 8).reshape((n_blocks, -1))
        delta = np.where((qh & 0x08) == 0, cls.delta, -cls.delta).reshape((n_blocks, -1, 2, 2, 1))
        grid = np.take_along_axis(cls.grid, q2.reshape((n_blocks, -1, 1, 1)), axis=-2).reshape((n_blocks, -1, 2, 2, 8))
        return (dl * (grid + delta)).reshape(-1).astype(np.float32)
    return quants.dequantize(b.reshape(-1), t).reshape(-1).astype(np.float32)


def run(mode, t, data, cols):
    with tempfile.TemporaryDirectory() as d:
        a, o = os.path.join(d, 'in'), os.path.join(d, 'out')
        data.tofile(a)
        r = subprocess.run([QT, mode, str(int(t)), a, str(cols), o])
        if r.returncode:
            return None
        return np.fromfile(o, dtype=np.uint8 if mode == 'q' else np.float32)


bad = 0
for t in [T.F32, T.F16, T.BF16, T.Q4_0, T.Q4_1, T.Q5_0, T.Q5_1, T.Q8_0, T.Q2_K, T.Q3_K, T.Q4_K, T.Q5_K, T.Q6_K, T.IQ2_XXS, T.IQ2_XS, T.IQ2_S,
          T.IQ3_XXS, T.IQ3_S, T.IQ1_S, T.IQ1_M, T.IQ4_NL, T.IQ4_XS, T.TQ1_0, T.TQ2_0, T.MXFP4]:
    bs, by = constants.GGML_QUANT_SIZES[t]
    if bs == 1:
        x = rng.standard_normal(COLS * 8).astype({T.F32: np.float32, T.F16: np.float16}.get(t, np.float32))
        b = x.view(np.uint8).reshape(-1, by) if t != T.BF16 else (x.view(np.uint32) >> 16).astype(np.uint16).view(np.uint8).reshape(-1, 2)
    else:
        b = blocks(t, COLS // bs * 8)
    want = ref(t, b)
    got = run('dq', t, b, COLS)
    same = got is not None and got.shape == want.shape and np.array_equal(got.view(np.uint32), want.view(np.uint32))
    close = got is not None and got.shape == want.shape and np.allclose(got, want, rtol=1e-6, atol=0, equal_nan=True)
    print(f'{t.name:8} Разбор: ' + ('бит в бит' if same else 'совпадает до 1e-6' if close else 'расхождение'))
    if not close:
        bad += 1
        if got is not None and got.shape == want.shape:
            i = int(np.argmax(np.abs(got - want)))
            print('   Первое расхождение:', i, got[i], want[i])

# г1
x = (rng.standard_normal((32, 4096)) * 0.02).astype(np.float32)   # г2
for t in [T.Q8_0, T.Q6_K, T.Q5_K, T.Q4_K]:
    q = run('q', t, x, 4096)
    if q is None:
        print(t.name, 'Квантование: не выполнено'); bad += 1; continue
    back = quants.dequantize(q, t).reshape(x.shape)
    rel = float(np.sqrt(((back - x) ** 2).sum() / (x ** 2).sum()))
    extra = ''
    if t == T.Q8_0:
        extra = ', байты как у gguf-py' if np.array_equal(q, quants.quantize(x, t).reshape(-1)) else ', байты не совпали'
        if 'НЕ' in extra.upper():
            bad += 1
    lim = {T.Q8_0: 0.007, T.Q6_K: 0.022, T.Q5_K: 0.042, T.Q4_K: 0.08}[t]
    if rel > lim:
        bad += 1
    print(f'{t.name:8} Квантование: ошибка {100 * rel:.2f} % (предел {100 * lim:.1f} %){extra}')
print('Итог:', 'всё сходится' if not bad else f'расхождений {bad}')
sys.exit(1 if bad else 0)
