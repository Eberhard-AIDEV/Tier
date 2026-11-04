#!/usr/bin/env python3
# а1
import json, os, re, struct, subprocess, sys, time

GGML = {0: 'F32', 1: 'F16', 2: 'Q4_0', 3: 'Q4_1', 6: 'Q5_0', 7: 'Q5_1', 8: 'Q8_0', 10: 'Q2_K', 11: 'Q3_K', 12: 'Q4_K', 13: 'Q5_K', 14: 'Q6_K',
        16: 'IQ2_XXS', 17: 'IQ2_XS', 18: 'IQ3_XXS', 19: 'IQ1_S', 20: 'IQ4_NL', 21: 'IQ3_S', 22: 'IQ2_S', 23: 'IQ4_XS', 29: 'IQ1_M', 30: 'BF16',
        34: 'TQ1_0', 35: 'TQ2_0', 39: 'MXFP4'}
BLK = {'F32': (1, 4), 'F16': (1, 2), 'BF16': (1, 2), 'Q4_0': (32, 18), 'Q4_1': (32, 20), 'Q5_0': (32, 22), 'Q5_1': (32, 24), 'Q8_0': (32, 34),
       'Q2_K': (256, 84), 'Q3_K': (256, 110), 'Q4_K': (256, 144), 'Q5_K': (256, 176), 'Q6_K': (256, 210), 'IQ2_XXS': (256, 66), 'IQ2_XS': (256, 74),
       'IQ2_S': (256, 82), 'IQ3_XXS': (256, 98), 'IQ3_S': (256, 110), 'IQ1_S': (256, 50), 'IQ1_M': (256, 56), 'IQ4_NL': (32, 18), 'IQ4_XS': (256, 136),
       'TQ1_0': (256, 54), 'TQ2_0': (256, 66), 'MXFP4': (32, 17), 'NF4': (64, 36), 'FP4': (64, 36)}
# а2
GG = {'attn_q': 'self_attn.q_proj', 'attn_k': 'self_attn.k_proj', 'attn_v': 'self_attn.v_proj', 'attn_output': 'self_attn.o_proj',
      'ffn_gate': 'mlp.gate_proj', 'ffn_up': 'mlp.up_proj', 'ffn_down': 'mlp.down_proj'}
HF = re.compile(r'^(?:model\.language_model\.|model\.|language_model\.model\.)?layers\.(\d+)\.(self_attn\.[qkvo]_proj|mlp\.(?:gate|up|down)_proj)\.weight$')


def say(*a):
    print(*a, flush=True)


SIDE = re.compile(r'vision|visual|audio|encoder|mm_proj|multi_modal|speech|image')
ANY = re.compile(r'^(.*?)layers\.(\d+)\.(self_attn\.[qkvo]_proj|mlp\.(?:gate|up|down)_proj)\.weight$')


def fit(items, model):
    # в1
    if not model:
        return items
    M = {k: v[2] for k, v in st_open(model).items()}
    idx = {}
    for k in M:
        m = ANY.match(k)
        if m and not SIDE.search(k):
            idx.setdefault((int(m.group(2)), m.group(3)), []).append(k)
    idx = {kk: min(v, key=lambda k: ('language_model' not in k, len(k))) for kk, v in idx.items()}
    keep, extra = [], []
    for it in items:
        m = ANY.match(it[0])
        nm = idx.get((int(m.group(2)), m.group(3))) if m else None
        if nm is None:
            extra.append(it[0])
            continue
        sh = M[nm]
        if list(sh) != [it[2], it[3]]:
            sys.exit(f'{nm}: В кванте {it[2]}×{it[3]}, в модели {sh[0]}×{sh[1]}. Квант относится к другой модели.')
        keep.append((nm,) + tuple(it[1:]))
    if extra:
        say(f'Отсутствуют в модели, пропущено {len(extra)}: {", ".join(short(x) for x in extra[:4])}{" и другие" if len(extra) > 4 else ""}')
    if not keep:
        sys.exit('Ни одна матрица не подходит к модели.')
    return keep


def short(nm):
    m = ANY.match(nm)
    return f'{m.group(2)}.{m.group(3)}' if m else nm


def write_tq(out, items, meta):
    # д1
    hb = (len(items) * 200 + 4096 + 4095) // 4096 * 4096
    off, lay = hb, []
    for nm, ty, rows, cols, src in items:
        b, by = BLK[ty]
        if cols % b:
            sys.exit(f'{nm}: Число столбцов {cols} не делится на размер блока {ty} ({b}).')
        size = cols // b * by * rows
        lay.append((nm, ty, rows, cols, off, size, src))
        off += (size + 4095) // 4096 * 4096
    tmp = out + '.part'
    t0, done, tot = time.time(), 0, sum(x[5] for x in lay)
    with open(tmp, 'wb') as f:
        f.truncate(off)
        for i, (nm, ty, rows, cols, o, size, src) in enumerate(lay):
            f.seek(o)
            if callable(src):
                data = src()
                if len(data) != size:
                    sys.exit(f'{nm}: Получено {len(data)} байт вместо {size}.')
                f.write(data)
            else:
                fn, so, sn = src
                if sn != size:
                    sys.exit(f'{nm}: В источнике {sn} байт, а для {ty} {rows}×{cols} требуется {size}.')
                with open(fn, 'rb') as g:
                    g.seek(so)
                    left = size
                    while left:
                        c = g.read(min(left, 64 << 20))
                        if not c:
                            sys.exit(f'{nm}: Данные источника оборваны.')
                        f.write(c)
                        left -= len(c)
            done += size
            say(f'Перенос {i + 1}/{len(lay)} {short(nm)} {ty} '
                f'[{done / 2**30:.2f}/{tot / 2**30:.2f} ГиБ, {time.time() - t0:.0f} с]')
        h = f'TQB1 {hb} {len(lay)}\n' + ''.join(f'#{k} {v}\n' for k, v in meta.items())
        h += ''.join(f'{nm} {ty} {rows} {cols} {o} {size} -1\n' for nm, ty, rows, cols, o, size, _ in lay)
        if len(h.encode()) >= hb:
            sys.exit('Заголовок не помещается в отведённое место.')
        f.seek(0)
        f.write(h.encode().ljust(hb, b'\0'))
    os.replace(tmp, out)
    say(f'Готово: {out}, {off / 2**30:.2f} ГиБ, матриц: {len(lay)}')


# е1
def gguf_read(path):
    f = open(path, 'rb')
    if f.read(4) != b'GGUF':
        sys.exit(f'{path}: Файл не является GGUF.')
    ver, = struct.unpack('<I', f.read(4))
    if ver == 1:
        nt, nkv = struct.unpack('<II', f.read(8))
    else:
        nt, nkv = struct.unpack('<QQ', f.read(16))
    rd = lambda fmt: struct.unpack('<' + fmt, f.read(struct.calcsize(fmt)))[0]
    def s():
        n = rd('Q' if ver > 1 else 'I')
        return f.read(n).decode('utf-8', 'replace')
    SC = {0: 'B', 1: 'b', 2: 'H', 3: 'h', 4: 'I', 5: 'i', 6: 'f', 7: '?', 10: 'Q', 11: 'q', 12: 'd'}
    def val(t):
        if t in SC:
            return rd(SC[t])
        if t == 8:
            return s()
        if t == 9:
            it, n = rd('I'), rd('Q' if ver > 1 else 'I')
            if it in SC and n > 64:          # е2
                f.seek(n * struct.calcsize(SC[it]), 1)
                return None
            return [val(it) for _ in range(n)]
        sys.exit(f'{path}: Неизвестный тип значения {t}.')
    kv = {}
    for _ in range(nkv):
        k = s()
        kv[k] = val(rd('I'))
    tens = []
    for _ in range(nt):
        nm = s()
        nd = rd('I')
        dims = [rd('Q' if ver > 1 else 'I') for _ in range(nd)]
        ty, off = rd('I'), rd('Q')
        tens.append((nm, dims, ty, off))
    al = kv.get('general.alignment') or 32
    base = (f.tell() + al - 1) // al * al
    f.close()
    return kv, tens, base


def gguf_shards(path):
    m = re.match(r'^(.*)-(\d{5})-of-(\d{5})\.gguf$', path)
    if not m:
        return [path]
    n = int(m.group(3))
    out = [f'{m.group(1)}-{i:05d}-of-{n:05d}.gguf' for i in range(1, n + 1)]
    for p in out:
        if not os.path.exists(p):
            sys.exit(f'Нет части {os.path.basename(p)}: GGUF состоит из {n} частей, все они должны находиться в одной папке.')
    return out


ROPE_PERM = {'llama', 'mistral', 'granite'}


def unperm(src, rows, nh):
    # з1
    fn, so, sn = src
    if rows % nh or (rows // nh) % 2 or sn % rows:
        sys.exit(f'{fn}: Число строк {rows} не делится на число голов {nh}.')
    hd, rb = rows // nh, sn // rows
    half = hd // 2
    order = [h * hd + 2 * j + t for h in range(nh) for t in range(2) for j in range(half)]   # з2

    def get():
        with open(fn, 'rb') as g:
            g.seek(so)
            raw = g.read(sn)
        if len(raw) != sn:
            sys.exit(f'{fn}: Источник оборвался.')
        mv = memoryview(raw)
        return b''.join(mv[o * rb:(o + 1) * rb] for o in order)
    return get


def from_gguf(path, out, model=None):
    items, types, arch = [], {}, ''
    for p in gguf_shards(path):
        kv, tens, base = gguf_read(p)
        arch = arch or kv.get('general.architecture', '')
        sizes = {}
        for nm, dims, ty, off in tens:
            m = re.match(r'^blk\.(\d+)\.(attn_q|attn_k|attn_v|attn_output|ffn_gate|ffn_up|ffn_down)\.weight$', nm)
            if not m:
                continue
            if ty not in GGML:
                sys.exit(f'{nm}: Тип ggml {ty} не поддерживается движком.')
            t = GGML[ty]
            cols, rows = dims[0], (dims[1] if len(dims) > 1 else 1)
            b, by = BLK[t]
            size = cols // b * by * rows
            src = (p, base + off, size)
            if arch in ROPE_PERM and m.group(2) in ('attn_q', 'attn_k'):
                # и1
                nh = kv.get(f'{arch}.attention.head_count') if m.group(2) == 'attn_q' else kv.get(f'{arch}.attention.head_count_kv') or kv.get(f'{arch}.attention.head_count')
                src = unperm(src, rows, int(nh))
            items.append((f'model.layers.{m.group(1)}.{GG[m.group(2)]}.weight', t, rows, cols, src))
    if not items:
        sys.exit('В файле нет матриц слоёв (blk.N.attn_q.weight …).')
    items = fit(sorted(items, key=lambda x: (int(x[0].split('.')[2]), x[0])), model)
    types = {}
    for it in items:
        types[it[1]] = types.get(it[1], 0) + 1
    st = os.stat(path)
    mix = ','.join(f'{k}x{v}' for k, v in sorted(types.items(), key=lambda kv: -kv[1]))
    say(f'GGUF {os.path.basename(path)}: {arch}, матриц: {len(items)} ({mix})')
    write_tq(out, items, {'from': f'gguf {os.path.basename(path)}', 'file': f'{st.st_size} {int(st.st_mtime)}', 'arch': arch, 'mix': mix})


# к1
def st_open(path):
    # к2
    files = []
    if os.path.isdir(path):
        idx = os.path.join(path, 'model.safetensors.index.json')
        if os.path.exists(idx):
            files = sorted({os.path.join(path, v) for v in json.load(open(idx))['weight_map'].values()})
        elif os.path.exists(os.path.join(path, 'model.safetensors')):
            files = [os.path.join(path, 'model.safetensors')]
    else:
        files = [path]
    if not files:
        sys.exit(f'{path}: Файлы safetensors не найдены.')
    T = {}
    for fn in files:
        with open(fn, 'rb') as f:
            n, = struct.unpack('<Q', f.read(8))
            h = json.loads(f.read(n))
        for k, v in h.items():
            if k != '__metadata__':
                T[k] = (fn, v['dtype'], v['shape'], 8 + n + v['data_offsets'][0], v['data_offsets'][1] - v['data_offsets'][0])
    return T


def st_get(T, k, np, raw=False):
    fn, dt, shape, off, n = T[k]
    if raw:                                                      # л1
        with open(fn, 'rb') as f:
            f.seek(off)
            return np.frombuffer(f.read(n), dtype=np.uint8)
    D = {'U8': np.uint8, 'I8': np.int8, 'F32': np.float32, 'F16': np.float16, 'BF16': np.uint16, 'I32': np.int32, 'I64': np.int64}
    with open(fn, 'rb') as f:
        f.seek(off)
        a = np.frombuffer(f.read(n), dtype=D[dt])
    if dt == 'BF16':
        a = (a.astype(np.uint32) << 16).view(np.float32)
    return a


def from_bnb(path, out, model=None):
    try:
        import numpy as np
    except ImportError:
        sys.exit('Для переноса bitsandbytes требуется numpy.')
    T = st_open(path)
    NF4 = np.array([-1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453, -0.28444138169288635, -0.18477343022823334,
                    -0.09105003625154495, 0.0, 0.07958029955625534, 0.16093020141124725, 0.24611230194568634, 0.33791524171829224,
                    0.44070982933044434, 0.5626170039176941, 0.7229568362236023, 1.0], dtype=np.float32)
    items, kinds = [], {}
    for k in sorted(T):
        m = HF.match(k)
        if not m:
            continue
        qs = [x for x in T if x.startswith(k + '.quant_state.bitsandbytes__')]
        if not qs:
            continue                                             # м1
        st = json.loads(bytes(st_get(T, qs[0], np)).decode())
        kind = st.get('quant_type', qs[0].rsplit('__', 1)[1]).upper()
        if kind not in ('NF4', 'FP4'):
            sys.exit(f'{k}: Вид {kind} не поддерживается.')
        if st.get('blocksize', 64) != 64:
            sys.exit(f'{k}: Размер блока {st.get("blocksize")}, движок поддерживает только 64.')
        rows, cols = st['shape']
        if cols % 64:
            sys.exit(f'{k}: Число столбцов {cols} не делится на 64.')
        if kind == 'NF4' and k + '.quant_map' in T and not np.allclose(st_get(T, k + '.quant_map', np), NF4, atol=1e-6):
            sys.exit(f'{k}: Таблица NF4 нестандартная.')
        name = f'model.layers.{m.group(1)}.{m.group(2)}.weight'

        def blocks(k=k, st=st, rows=rows, cols=cols):
            a = st_get(T, k + '.absmax', np)
            if a.dtype == np.uint8:                              # м2
                code, a2 = st_get(T, k + '.nested_quant_map', np), st_get(T, k + '.nested_absmax', np)
                nb2 = st.get('nested_blocksize', 256)
                a = code[a.astype(np.int64)] * np.repeat(a2, nb2)[:a.size] + np.float32(st.get('nested_offset', 0))
            a = a.astype(np.float32)
            w = st_get(T, k, np, raw=True)
            nb = rows * cols // 64
            if a.size != nb or w.size != nb * 32:
                sys.exit(f'{k}: Размеры не согласуются ({a.size} масштабов, {w.size} байт).')
            return np.concatenate([a.view(np.uint8).reshape(nb, 4), w.reshape(nb, 32)], axis=1).tobytes()
        items.append((name, kind, rows, cols, blocks))
    if not items:
        sys.exit('В файлах нет матриц bitsandbytes 4 бита (…weight.quant_state.bitsandbytes__nf4).')
    items = fit(sorted(items, key=lambda x: (int(x[0].split('.')[2]), x[0])), model)
    for it in items:
        kinds[it[1]] = kinds.get(it[1], 0) + 1
    mix = ','.join(f'{k}x{v}' for k, v in kinds.items())
    say(f'bitsandbytes {os.path.basename(path.rstrip("/"))}: матриц: {len(items)} ({mix})')
    write_tq(out, items, {'from': f'bnb {os.path.basename(path.rstrip("/"))}', 'mix': mix})


def info(path):
    with open(path, 'rb') as f:
        first = f.readline().decode()
        m = re.match(r'^TQB1 (\d+) (\d+)', first)
        if not m:
            sys.exit('Файл не в формате .tq.')
        f.seek(0)
        h = f.read(int(m.group(1))).split(b'\0', 1)[0].decode()
    meta, mats, types, errs = {}, 0, {}, {}
    for line in h.splitlines()[1:]:
        if line.startswith('#'):
            k, _, v = line[1:].partition(' ')
            meta[k] = v
            continue
        p = line.split()
        if len(p) >= 6:
            mats += 1
            types[p[1]] = types.get(p[1], 0) + 1
            if len(p) >= 7 and float(p[6]) >= 0:
                errs[p[0]] = float(p[6])
    return {'мета': meta, 'матриц': mats, 'типы': types, 'байт': os.path.getsize(path), 'ошибки': errs}


def check(engine, model, tq):
    # о1
    say('Сверка с исходной моделью')
    p = subprocess.Popen([engine, '--tq-check', model, tq], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    last = ''
    for line in p.stdout:
        say(line.rstrip('\n'))
        if line.lower().startswith('итог:'):
            last = line
    rc = p.wait()
    m = re.search(r'ошибка весов ([\d.]+)%, худшая корреляция (-?[\d.]+) \((.*?)\)', last, re.I)
    if rc == 2 or (m and float(m.group(2)) < 0.5):
        for f in (tq, tq + '.check'):
            if os.path.exists(f):
                os.remove(f)
        sys.exit(f'Матрицы не относятся к этой модели: худшая корреляция {m.group(2) if m else "?"} у {m.group(3) if m else "?"}. Файл удалён.')
    if rc or not m:
        sys.exit('Сверка не выполнена.')
    st = os.stat(model)
    json.dump({'ошибка': float(m.group(1)) / 100, 'худшая': float(m.group(2)), 'модель': f'{st.st_size} {int(st.st_mtime)}'},
              open(tq + '.check', 'w', encoding='utf-8'), ensure_ascii=False)
    say(f'Годна: ошибка весов {m.group(1)} %, худшая корреляция {m.group(2)}')


def import_(src, out, model, engine):
    if os.path.isfile(src) and src.endswith('.gguf'):
        from_gguf(src, out, model)
    else:
        from_bnb(src, out, model)
    check(engine, model, out)


if __name__ == '__main__':
    a = sys.argv[1:]
    if len(a) >= 3 and a[0] == 'gguf':
        from_gguf(a[1], a[2], a[3] if len(a) > 3 else None)
    elif len(a) >= 3 and a[0] == 'bnb':
        from_bnb(a[1], a[2], a[3] if len(a) > 3 else None)
    elif len(a) >= 4 and a[0] == 'check':
        check(a[1], a[2], a[3])
    elif len(a) >= 5 and a[0] == 'import':
        import_(a[1], a[2], a[3], a[4])
    elif len(a) >= 2 and a[0] == 'info':
        print(json.dumps(info(a[1]), ensure_ascii=False))
    else:
        print(__doc__ if __doc__ else open(__file__).read().split('\nimport', 1)[0])
        sys.exit(1)
