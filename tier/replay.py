# а1
import hashlib, json, math, os, random, re, struct, subprocess, sys, tempfile
# а2
try: _CPU = str(max(1, int(os.environ['TIER_CPU_THREADS'])))
except (KeyError, ValueError): _CPU = ''
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'BLIS_NUM_THREADS'):
    if _CPU: os.environ[_v] = _CPU                                   # а3
    else: os.environ.setdefault(_v, str(max(1, (os.cpu_count() or 2) // 2)))

HERE = os.path.dirname(os.path.abspath(__file__))
E = os.environ.get
CORPUS = E('REPLAY_DATA', os.path.join(HERE, 'replay.jsonl'))
THR = float(E('REPLAY_THR', '0.3'))   # а4


def read_bin(path):
    out = []
    with open(path, 'rb') as f:
        n, = struct.unpack('<i', f.read(4))
        for _ in range(n):
            L, p = struct.unpack('<ii', f.read(8))
            t = list(struct.unpack('<%di' % L, f.read(4 * L)))
            lab = None
            if p < 0:
                p = -p - 1
                lab = list(struct.unpack('<%di' % L, f.read(4 * L)))
            out.append((t, p, lab))
    return out


def write_bin(path, recs):
    tmp = path + '.tmp'
    with open(tmp, 'wb') as f:
        f.write(struct.pack('<i', len(recs)))
        for t, p, lab in recs:
            f.write(struct.pack('<ii', len(t), p if lab is None else -p - 1))
            f.write(struct.pack('<%di' % len(t), *t))
            if lab is not None:
                f.write(struct.pack('<%di' % len(t), *lab))
    os.replace(tmp, path)


def packs_of(recs, pack):
    # г1
    if pack <= 0:
        return [[i] for i in range(len(recs))]
    out, cur, ct = [], [], 0
    for i, (t, _p, _l) in enumerate(recs):
        if cur and ct + len(t) > pack:
            out.append(cur); cur, ct = [], 0
        cur.append(i); ct += len(t)
    if cur:
        out.append(cur)
    return out


def loss_pos(rec):
    t, p, lab = rec
    if lab is None:
        return max(0, len(t) - p)
    return sum(1 for j in range(p, len(t)) if lab[j] >= 0)


def engine_id():
    # е1
    b = E('B', '')
    try: return open(b + '.src').read().strip()
    except OSError:
        try: st = os.stat(b); return f'{st.st_size}|{int(st.st_mtime)}'
        except OSError: return ''


def cache_path(scenes=None):
    m = E('M', '')
    st = os.stat(m)
    h = hashlib.sha1()
    h.update(f'{os.path.realpath(m)}|{st.st_size}|{int(st.st_mtime)}|{THR}|{engine_id()}'.encode())
    h.update(open(CORPUS, 'rb').read())
    if scenes: h.update(b'\0scenes\0' + json.dumps(sorted(scenes), ensure_ascii=False).encode())   # ж1
    d = E('REPLAY_DIR') or os.path.join(os.path.dirname(os.path.dirname(m)) or '.', 'replay')
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, h.hexdigest()[:16] + '.bin')


PK = 2048   # ж2
def eval_pack(model):
    # з1
    try:
        sys.path.insert(0, HERE); import hwprobe
        d, free = hwprobe.arch_dims(model), hwprobe.vram_free_mib()
        if d and free: return hwprobe.pack_fit((free << 20) // hwprobe.EVAL_ACT_SHARE, d)
    except Exception: pass
    return PK


def prog(f, cap):
    # и1
    pf = E('STAGE_PROG')
    if pf:
        with open(pf + '.tmp', 'w') as o: o.write(f'{f:.4f}\t{cap}\n')
        os.replace(pf + '.tmp', pf)


def corpus_pass(td, extra, what, share=1.0, scenes=None):
    # к1
    B, M, MODEL = E('B'), E('M'), E('MODEL') or os.path.dirname(E('M'))
    raw = os.path.join(td, 'in.bin'); corpus = CORPUS
    if scenes:
        rs = random.Random(1); names = [t for t, _n in scenes]; ws = [n for _t, n in scenes]
        corpus = os.path.join(td, 'corpus.jsonl')
        with open(corpus, 'w', encoding='utf-8') as fo:
            for l in open(CORPUS, encoding='utf-8'):
                if l.strip(): r = json.loads(l); r['instruction'] = rs.choices(names, ws)[0]; fo.write(json.dumps(r, ensure_ascii=False) + '\n')
    env = dict(os.environ, NAME='_replay', WORK=td, MODEL=MODEL, OUT=raw, EVAL=os.path.join(td, 'e.bin'), HOLD='1000000', INSTR='1' if scenes else '0')
    subprocess.run([sys.executable, os.path.join(HERE, 'prepare.py'), corpus, raw], env=env, check=True, stdout=subprocess.DEVNULL)
    pk = eval_pack(MODEL); recs = read_bin(raw); np_ = len(packs_of(recs, pk)); tier = os.path.join(td, 'tier.txt')
    X = [x for x in (E('X') or '').split() if x not in ('--nsgd', '--pos-dw')]
    # к2
    base = [B, M, '/dev/null', '8', os.path.join(td, 'adam.bin'), os.path.join(td, 'нет.bin'), '1e-4', '9999', '1']
    for attempt in range(4):   # к3
      subprocess.run(base + ['--retier', '--tier-file', tier] + X, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
      print(f'{what}: Исходная модель читает {len(recs)} обычных ответов ({np_} проходов).', file=sys.stderr, flush=True)
      # к4
      pr = subprocess.Popen(base + X + ['--tier-file', tier, '--data', raw, '--steps', str(np_), '--pack', str(pk), '--eval'] + extra,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
      tail, buf, done = [], b'', 0
      prog(0, 'Загрузка весов')
      while True:
          ch = pr.stdout.read1(65536) if hasattr(pr.stdout, 'read1') else pr.stdout.read(4096)
          if not ch: break
          buf += ch; parts = re.split(rb'[\r\n]', buf); buf = parts.pop()
          for ln in parts:
              s = ln.decode('utf-8', 'replace').strip()
              if not s: continue
              tail = (tail + [s])[-40:]
              m = re.match(r'загрузка ([\d.]+)/([\d.]+) ГиБ', s, re.I)
              if m and float(m.group(2)) > 0:
                  prog(share * .5 * float(m.group(1)) / float(m.group(2)), f'Загрузка весов {m.group(1)}/{m.group(2)} ГиБ')
              elif s.startswith('=== forward loss'):
                  done += 1; prog(share * (.5 + .5 * done / max(1, np_)), f'Проход {done} из {np_}')
      rc = pr.wait()
      if rc == 76 and pk > 32 and attempt < 3:
          pk //= 2; np_ = len(packs_of(recs, pk)); print(f'{what}: видеопамяти не хватило, упаковка {pk}, повтор.', file=sys.stderr, flush=True); continue
      if rc != 0:
          sys.stderr.write('\n'.join(tail) + '\n'); sys.exit(f'{what}: Движок завершился с кодом {rc}.')
      break
    return recs


def label(scenes=None):
    out = cache_path(scenes)
    if os.path.exists(out):
        return out
    with tempfile.TemporaryDirectory(dir=os.path.dirname(out)) as td:
        pred = os.path.join(td, 'pred.bin')
        recs = corpus_pass(td, ['--dump-pred', pred], 'Подмешивание' + (' под системными инструкциями данных' if scenes else ''), scenes=scenes)
        labs = [[-1] * len(t) for t, _p, _l in recs]
        agree = kept = tot = 0
        with open(pred, 'rb') as f:
            while True:
                h = f.read(4)
                if not h:
                    break
                ne, = struct.unpack('<i', h)
                ids = struct.unpack('<%di' % ne, f.read(4 * ne))
                sl, = struct.unpack('<i', f.read(4))
                pr = struct.unpack('<' + 'if' * (sl - 1), f.read(8 * (sl - 1)))
                o = 0
                for k in ids:
                    t = recs[k][0]
                    for j in range(len(t) - 1):
                        a, pt = pr[2 * (o + j)], pr[2 * (o + j) + 1]
                        if a < 0:
                            continue
                        tot += 1; agree += (a == t[j + 1])
                        if pt >= THR:
                            labs[k][j + 1] = a; kept += 1
                    o += len(t)
        if not tot:
            sys.exit('Подмешивание: движок не выдал ни одного предсказания.')
        res = [(t, p, lb) for (t, p, _l), lb in zip(recs, labs) if sum(1 for x in lb[p:] if x >= 0) >= 4]
        write_bin(out, res)
        json.dump({'примеров': len(res), 'позиций': kept, 'из': tot, 'совпало_с_текстом': round(agree / tot, 3), 'порог': THR},
                  open(out + '.json', 'w'), ensure_ascii=False)
        print(f'Подмешивание: {len(res)} примеров, сохранено {kept} из {tot} позиций; модель согласна с текстом набора на {100 * agree / tot:.0f} %.', file=sys.stderr)
    return out


def qtext(rec, tk):
    # м1
    t = tk.decode(list(rec[0][:rec[1]]), skip_special_tokens=True).strip()
    roles = {'user', 'model', 'assistant', 'system', 'human', 'gpt', 'bot', 'ai'}   # м2
    ls = [x.strip() for x in t.split('\n') if x.strip() and x.strip().lower().rstrip(':') not in roles]
    return ls[-1] if ls else ''

NUMW = {'ноль': 0, 'один': 1, 'одна': 1, 'два': 2, 'две': 2, 'три': 3, 'четыре': 4, 'пять': 5, 'шесть': 6, 'семь': 7, 'восемь': 8, 'девять': 9,
        'десять': 10, 'zero': 0, 'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5, 'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10}

def qwords(t):
    # н1
    out = set()
    for w in re.findall(r'\w+', t.lower().replace('ё', 'е')):
        if w.isdigit() or w in NUMW: out.add('#'); continue
        if len(w) > 1: out.add(w[:5])
    return out

def topics(user_bin, out):
    # о1
    from tokenizers import Tokenizer
    tk = Tokenizer.from_file(os.path.join(E('MODEL') or os.path.dirname(E('M')), 'tokenizer.json'))
    pool = read_bin(label()); user = read_bin(user_bin)
    pq = [qtext(x, tk) for x in pool]; pw = [qwords(t) - {'#'} for t in pq]
    uw = set().union(*[qwords(qtext(x, tk)) - {'#'} for x in user]) if user else set()
    df = {}
    for q in pw:
        for w in q: df[w] = df.get(w, 0) + 1
    n = len(pw) or 1; bad = []
    for t, q in zip(pq, pw):
        if not q: continue
        key = max(q, key=lambda w: (math.log(n / df.get(w, 1)), w))
        if key in uw: bad.append(t)
    with open(out, 'w', encoding='utf-8') as f:
        for t in sorted(set(bad)): f.write(t + '\n')
    print(f'Противоречия подмешивания по теме вопроса: {len(set(bad))} из {len(set(pq))}.', file=sys.stderr)
    return len(set(bad))

def conflicts(base_each, probe_each, out):
    # п1
    from tokenizers import Tokenizer
    tk = Tokenizer.from_file(os.path.join(E('MODEL') or os.path.dirname(E('M')), 'tokenizer.json'))
    rd = lambda f: {int(w[1]): float(w[2]) / max(1, int(w[3])) for w in (l.split() for l in open(f)) if len(w) == 4 and w[0] == 'evex'}
    b, p = rd(base_each), rd(probe_each); pool = read_bin(label())
    d = sorted((p[i] - b[i], i) for i in b if i in p)
    if not d: return 0
    ds = [x for x, _ in d]; med = ds[len(ds) // 2]; mad = sorted(abs(x - med) for x in ds)[len(ds) // 2]
    thr = max(0.5, med + 4 * mad)
    bad = [i for x, i in d if x >= thr and i < len(pool)]
    with open(out, 'w', encoding='utf-8') as f:
        for i in bad: f.write(qtext(pool[i], tk) + '\n')
    print(f'Противоречия подмешивания: {len(bad)} из {len(d)} (рост потери от {thr:.2f} нат на токен).', file=sys.stderr)
    return len(bad)

def mix(user_bin, out_bin, R, pack, seed=42):
    user = read_bin(user_bin)
    pools = [(read_bin(label()), 1.0)]
    # р1
    try:
        sc = [x for x in json.load(open(E('REPLAY_SCENES', ''), encoding='utf-8')) if isinstance(x, dict) and isinstance(x.get('сцена'), str)]
    except (OSError, ValueError): sc = []
    tot = sum(int(x.get('примеров') or 0) for x in sc) or 1
    ok = sorted(((x['сцена'], int(x.get('примеров') or 0)) for x in sc if x['сцена'] and len(x['сцена']) <= 2000 and int(x.get('примеров') or 0) > 0),
                key=lambda t: (-t[1], t[0]))[:1000]
    share = sum(n for _t, n in ok) / tot
    if ok and share >= 0.05:
        try:
            sp = read_bin(label(ok))
            base = max(float(E('REPLAY_PLAIN_MIN', '0.5')), 1 - share)         # р2
            pools = [(pools[0][0], base), (sp, 1 - base)]
        except (SystemExit, subprocess.CalledProcessError, OSError, ValueError) as e:
            print(f'Подмешивание под системными инструкциями не выполнено ({e}), остаётся обычное.', file=sys.stderr)
    # р3
    exf = E('REPLAY_EXCLUDE', ''); hof = E('REPLAY_HOLDOUT', '')
    if hof and os.path.exists(hof):   # р4
        if exf and os.path.exists(exf):
            both = exf + '.all'; open(both, 'w', encoding='utf-8').write(open(exf, encoding='utf-8').read() + open(hof, encoding='utf-8').read()); exf = both
        else: exf = hof
    if exf and os.path.exists(exf):   # р5
        try:
            from tokenizers import Tokenizer
            tk = Tokenizer.from_file(os.path.join(E('MODEL') or os.path.dirname(E('M')), 'tokenizer.json'))
            ex = {l.strip() for l in open(exf, encoding='utf-8') if l.strip()}; dropped = 0; npools = []
            for p, w in pools:
                keep = [r for r in p if qtext(r, tk) not in ex]; dropped += len(p) - len(keep)
                if keep: npools.append((keep, w))
            if npools: pools = npools
            print(f'Подмешивание: убрано вопросов, спорящих с данными: {dropped}.', file=sys.stderr)
        except Exception as e: print(f'Подмешивание: список противоречий не применён ({e}).', file=sys.stderr)
    thr = 0
    if thr > 0:
        try:
            from tokenizers import Tokenizer
            tk = Tokenizer.from_file(os.path.join(E('MODEL') or os.path.dirname(E('M')), 'tokenizer.json'))
            NUM = {'ноль': 0, 'один': 1, 'одна': 1, 'два': 2, 'две': 2, 'три': 3, 'четыре': 4, 'пять': 5, 'шесть': 6, 'семь': 7, 'восемь': 8, 'девять': 9, 'десять': 10,
                   'zero': 0, 'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5, 'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10}
            def words(rec):
                t = tk.decode(list(rec[0][:rec[1]]), skip_special_tokens=True).lower().replace('ё', 'е')
                t = t.rsplit('\n', 1)[-1] if '\n' in t else t     # р6
                return {str(NUM[w]) if w in NUM else w[:5] for w in re.findall(r'\w+', t) if len(w) > 1 or w.isdigit()}
            uq = [words(u) for u in user]; pq = [[words(r) for r in p] for p, _w in pools]
            df = {}
            for q in uq + [q for qs_ in pq for q in qs_]:
                for t in q: df[t] = df.get(t, 0) + 1
            Nq = len(uq) + sum(len(x) for x in pq); wt = lambda t: math.log(Nq / df.get(t, 1))
            inv = {}
            for i, q in enumerate(uq):
                for t in q: inv.setdefault(t, []).append(i)
            def conflict(q):
                if not q: return False
                key = max(q, key=wt)
                for i in inv.get(key, []):
                    u = uq[i]; uni = sum(wt(t) for t in q | u)
                    if uni > 0 and sum(wt(t) for t in q & u) / uni >= thr: return True
                return False
            dropped = 0; npools = []
            for (p, w), qs_ in zip(pools, pq):
                keep = [r for r, q in zip(p, qs_) if not conflict(q)]; dropped += len(p) - len(keep)
                if keep: npools.append((keep, w))
            if npools: pools = npools
            print(f'Подмешивание: убрано вопросов, спорящих с данными: {dropped}.', file=sys.stderr)
        except Exception as e:
            print(f'Подмешивание: проверка противоречий не выполнена ({e}).', file=sys.stderr)
    rng = random.Random(seed)
    orders = [list(range(len(p))) for p, _w in pools]; qs = [0] * len(pools)
    for o in orders: rng.shuffle(o)
    def draw():
        x = rng.random() * sum(w for _p, w in pools); i = 0
        while i < len(pools) - 1 and x >= pools[i][1]: x -= pools[i][1]; i += 1
        if qs[i] == len(orders[i]): rng.shuffle(orders[i]); qs[i] = 0
        r = pools[i][0][orders[i][qs[i]]]; qs[i] += 1; return r
    # р7
    near = float(E('REPLAY_NEAR', '0') or 0); nbr = {}
    if near > 0:
        cand = [r for p, _w in pools for r in p]
        qs_ = lambda rec: rec[0][:rec[1]]
        df = {}
        for rec in cand + user:
            for t in set(qs_(rec)): df[t] = df.get(t, 0) + 1
        lim = (len(cand) + len(user)) / 3
        sig = lambda rec: {t for t in qs_(rec) if df.get(t, 0) <= lim}
        cs = [sig(c) for c in cand]
        for i, u in enumerate(user):
            su = sig(u)
            if not su: continue
            sc_ = sorted(((len(su & c) / len(su | c), j) for j, c in enumerate(cs) if su & c), reverse=True)[:8]
            if sc_: nbr[i] = [cand[j] for _s, j in sc_]
    res, groups = [], []
    for i, u in enumerate(user):
        g = [u]; need = R * loss_pos(u); got = 0; k = 0; nb = nbr.get(i, [])
        while got < need or len(g) == 1:
            if nb and got < near * need: r = nb[k % len(nb)]; k += 1   # р8
            else: r = draw()
            g.append(r); got += loss_pos(r)
        groups.append(g)
    for g in groups:
        res += g
    if pack <= 0:   # р9
        pack = max(sum(len(t) for t, _p, _l in g) for g in groups)
    write_bin(out_bin, res)
    return pack, len(packs_of(res, pack)), sum(len(g) - 1 for g in groups)


def mats_spec(mats):
    # с1
    kinds, ls = mats.split(':'); out = {}
    for k in kinds.split('+'):
        for e in {'down_proj': ['down'], 'attn': ['qkv', 'o'], 'q_proj': ['qkv'], 'k_proj': ['qkv'], 'v_proj': ['qkv'], 'o_proj': ['o']}.get(k, []):
            out[e] = ls
    return ';'.join(f'{k}:{v}' for k, v in out.items())


def pick_layers(kind, n_layers, lines, k=3):
    # т1
    import math, re as _re
    acc = {}
    for ln in lines:
        w = ln.split()
        if len(w) != 5 or w[0] != 'gprobe': continue
        m = _re.match(r'layer_(\d+)\.(\w+)$', w[2])
        if not m: continue
        l, mat = int(m.group(1)), m.group(2)
        if (kind == 'down_proj' and mat != 'down_proj') or (kind == 'attn' and mat not in ('q_proj', 'o_proj')): continue
        g2, w2 = float(w[3]), float(w[4])
        if w2 > 0: acc.setdefault(l, []).append(math.sqrt(g2 / w2))
    lo, hi = ((0.15, 0.85) if kind == 'down_proj' else (0.10, 0.60))
    cand = sorted(((sum(v) / len(v), l) for l, v in acc.items() if lo * n_layers <= l <= hi * n_layers), reverse=True)
    gap = max(1, n_layers // 12); out = []
    for sc, l in cand:
        if all(abs(l - o) >= gap for o in out): out.append(l)
        if len(out) == k: break
    return sorted(out), {l: sum(v) / len(v) for l, v in acc.items()}


def top_dirs(Xb, r, it=5, seed=0):
    # у1
    import numpy as np
    N, dim = Xb.shape
    def mm(Q):                                   # у2
        acc = np.zeros((dim, Q.shape[1]), np.float32)
        for i in range(0, N, 4096):
            x = (Xb[i:i + 4096].astype(np.uint32) << 16).view(np.float32); acc += x.T @ (x @ Q)
        return acc / N
    k = min(dim, r + max(64, r // 2)); Q = np.random.default_rng(seed).standard_normal((dim, k)).astype(np.float32)
    for _ in range(it): Q, _ = np.linalg.qr(mm(Q))
    ev, V = np.linalg.eigh(Q.T @ mm(Q)); ev, V = ev[::-1][:r], V[:, ::-1][:, :r]
    tr = 0.0
    for i in range(0, N, 4096):
        x = (Xb[i:i + 4096].astype(np.uint32) << 16).view(np.float32); tr += float((x.astype(np.float64) ** 2).sum())
    return (Q @ V).astype(np.float32), ev.astype(np.float64) / (tr / N / dim)        # у3


ENERGY = float(E('WHITEN_ENERGY', '0.9'))


def whiten(mats, a=0.3, frac=0.1):
    import numpy as np
    spec = mats_spec(mats)
    if not spec: sys.exit(f'Защита общих знаний: значение MATS={mats} не распознано.')
    m = E('M', ''); st = os.stat(m); h = hashlib.sha1()
    h.update(f'{os.path.realpath(m)}|{st.st_size}|{int(st.st_mtime)}|{spec}|{a}|{frac}|{ENERGY}|{engine_id()}'.encode()); h.update(open(CORPUS, 'rb').read())
    d = E('REPLAY_DIR') or os.path.join(os.path.dirname(os.path.dirname(m)) or '.', 'replay'); os.makedirs(d, exist_ok=True)
    out = os.path.join(d, h.hexdigest()[:16] + '.wh')
    if os.path.exists(out): return out
    with tempfile.TemporaryDirectory(dir=d) as td:
        pre = os.path.join(td, 'act')
        corpus_pass(td, ['--dump-act', spec, pre], 'Защита общих знаний', .8)
        ents = []; fl = [fn for fn in sorted(os.listdir(td)) if fn.startswith('act.') and fn.endswith('.bf16')]
        for i, fn in enumerate(fl):
            prog(.8 + .2 * i / len(fl), f'Главные направления: матрица {i + 1} из {len(fl)}')
            key = fn[4:-5]; raw = np.fromfile(os.path.join(td, fn), dtype=np.uint16)
            dim = int(raw[:2].view(np.int32)[0]); Xb = raw[2:].reshape(-1, dim)
            # ф1
            rmax = max(1, int(frac * dim)); r = min(rmax, int(E('WHITEN_START', '64')))
            while True:
                U, ev = top_dirs(Xb, r)
                cum = np.cumsum(np.clip(ev, 0, None)) / dim             # ф2
                if cum[-1] >= ENERGY or r >= rmax: break
                r = min(rmax, r * 4)
            k = int(np.searchsorted(cum, ENERGY)) + 1                   # ф3
            if k < r: r = min(r, max(8, k)); U, ev = U[:, :r], ev[:r]   # ф4
            f = (1 - 1 / (1 + np.clip(ev, 0, None) / a)).astype(np.float32)
            print(f'Защита общих знаний: {key}: {Xb.shape[0]} позиций × {dim}, {r} направлений, '
                  f'ослабление до {100 * f[0]:.0f} % по главному и {100 * f[-1]:.0f} % по последнему направлению.', file=sys.stderr, flush=True)
            ents.append((key, dim, r, f, U))
        if not ents: sys.exit('Защита общих знаний: движок не выгрузил ни одной матрицы.')
        tmp = out + '.tmp'
        with open(tmp, 'wb') as fo:
            fo.write(struct.pack('<i', len(ents)))
            for key, dim, r, f, U in ents:
                kb = key.encode(); fo.write(struct.pack('<i', len(kb))); fo.write(kb); fo.write(struct.pack('<ii', dim, r))
                fo.write(f.tobytes()); fo.write(np.ascontiguousarray(U.T).tobytes())          # ф5
        os.replace(tmp, out)
    return out


if __name__ == '__main__':
    if len(sys.argv) >= 2 and sys.argv[1] == 'label':
        print(label())
    elif len(sys.argv) >= 3 and sys.argv[1] == 'whiten':
        print(whiten(sys.argv[2], *(float(x) for x in sys.argv[3:5])))
    elif len(sys.argv) >= 4 and sys.argv[1] == 'layers':      # ф6
        ls, sc = pick_layers(sys.argv[2], int(sys.argv[3]), sys.stdin.read().splitlines())
        if not ls: sys.exit('Замер не дал слоёв.')
        print(','.join(map(str, ls)))
        print('Сила по слоям: ' + ' '.join(f'{l}:{v:.2e}' for l, v in sorted(sc.items())), file=sys.stderr)
    elif len(sys.argv) >= 4 and sys.argv[1] == 'topics':   # ф7
        print(topics(sys.argv[2], sys.argv[3]))
    elif len(sys.argv) >= 5 and sys.argv[1] == 'conflicts':   # ф8
        print(conflicts(sys.argv[2], sys.argv[3], sys.argv[4]))
    elif len(sys.argv) >= 2 and sys.argv[1] == 'pool':   # ф9
        print(label())
    elif len(sys.argv) >= 4 and sys.argv[1] == 'retain':   # ф10
        recs = read_bin(label()); exf = E('REPLAY_EXCLUDE', '')
        if exf and os.path.exists(exf):   # ф11
            from tokenizers import Tokenizer
            tk = Tokenizer.from_file(os.path.join(E('MODEL') or os.path.dirname(E('M')), 'tokenizer.json'))
            ex = {l.strip() for l in open(exf, encoding='utf-8') if l.strip()}; recs = [x for x in recs if qtext(x, tk) not in ex]
        rng = random.Random(7); rng.shuffle(recs); keep = recs[:int(sys.argv[3])]; write_bin(sys.argv[2], keep)
        try:   # ф12
            from tokenizers import Tokenizer
            tk = Tokenizer.from_file(os.path.join(E('MODEL') or os.path.dirname(E('M')), 'tokenizer.json'))
            with open(sys.argv[2] + '.txt', 'w', encoding='utf-8') as f:
                for x in keep: f.write(qtext(x, tk) + '\n')
        except Exception as e: print(f'Замер порчи: список вопросов не записан ({e}).', file=sys.stderr)
        print(len(keep))
    elif len(sys.argv) >= 6 and sys.argv[1] == 'mix':
        pk, st, nr = mix(sys.argv[2], sys.argv[3], float(sys.argv[4]), int(sys.argv[5]))
        print(f'pack {pk}\nsteps {st}\nreplay {nr}')
    else:
        sys.exit('Формат вызова: replay.py mix <токены.bin> <выход.bin> <R> <упаковка> | label | whiten <MATS> [a] [доля]')
