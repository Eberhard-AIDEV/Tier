#!/usr/bin/env python3
# а1
import json, math, os, re, struct, sys

def load_json(p, d=None):
    try: return json.load(open(p, encoding='utf-8'))
    except (OSError, ValueError): return d

def st_header(path):
    with open(path, 'rb') as f:
        n = struct.unpack('<Q', f.read(8))[0]; h = json.loads(f.read(n))
    h.pop('__metadata__', None); return h, 8 + n

def tensors(mdir):
    # г1
    one = os.path.join(mdir, 'model.safetensors')
    if os.path.exists(one): return {k: (v['dtype'], v['shape']) for k, v in st_header(one)[0].items()}, [one]
    idx = load_json(os.path.join(mdir, 'model.safetensors.index.json'))
    if not idx: raise SystemExit(f'{mdir}: Файл model.safetensors не найден.')
    files = sorted(set(idx['weight_map'].values())); out = {}
    for fn in files: out.update({k: (v['dtype'], v['shape']) for k, v in st_header(os.path.join(mdir, fn))[0].items()})
    return out, [os.path.join(mdir, f) for f in files]

# г2
CFG_DEFAULTS = {
    'gemma3_text': dict(rope_theta=1e6, rope_local_base_freq=1e4, query_pre_attn_scalar=256, sliding_window=4096, sliding_window_pattern=6,
                        rms_norm_eps=1e-6, hidden_activation='gelu_pytorch_tanh', head_dim=256, num_attention_heads=8, num_key_value_heads=4,
                        max_position_embeddings=131072),
    'gemma4_text': dict(sliding_window=512, rms_norm_eps=1e-6, hidden_activation='gelu_pytorch_tanh', head_dim=256, num_attention_heads=8,
                        num_key_value_heads=4, max_position_embeddings=131072),
    'gemma2': dict(rope_theta=1e4, query_pre_attn_scalar=256, sliding_window=4096, rms_norm_eps=1e-6, final_logit_softcapping=30.0,
                   attn_logit_softcapping=50.0, hidden_activation='gelu_pytorch_tanh', head_dim=256, max_position_embeddings=8192),
    'gemma': dict(rope_theta=1e4, rms_norm_eps=1e-6, hidden_act='gelu_pytorch_tanh', head_dim=256, max_position_embeddings=8192),
    'llama': dict(rope_theta=1e4, rms_norm_eps=1e-6, hidden_act='silu', max_position_embeddings=2048),
    'qwen2': dict(rope_theta=1e4, rms_norm_eps=1e-6, hidden_act='silu', use_sliding_window=False, max_position_embeddings=32768),
    'qwen3': dict(rope_theta=1e4, rms_norm_eps=1e-6, hidden_act='silu', use_sliding_window=False, head_dim=128, max_position_embeddings=32768),
    'mistral': dict(rope_theta=1e4, rms_norm_eps=1e-6, hidden_act='silu', sliding_window=4096, max_position_embeddings=131072),
}
CFG_DEFAULTS['gemma3'] = CFG_DEFAULTS['gemma3_text']; CFG_DEFAULTS['gemma4'] = CFG_DEFAULTS['gemma4_text']

def text_cfg(c):
    tc = c.get('text_config') or {}
    mt = str(tc.get('model_type') or c.get('model_type') or '')
    t = dict(CFG_DEFAULTS.get(mt, {})); t.update(c); t.update(tc); return t

def rope_for(cfg, ltype):
    # е1
    rp = cfg.get('rope_parameters')
    if isinstance(rp, dict) and ltype in rp and isinstance(rp[ltype], dict): rp = rp[ltype]
    elif isinstance(rp, dict) and ('rope_theta' in rp or 'rope_type' in rp): pass
    else: rp = None
    theta = (rp or {}).get('rope_theta')
    if theta is None:
        theta = cfg.get('rope_local_base_freq') if ltype == 'sliding_attention' and cfg.get('rope_local_base_freq') else cfg.get('rope_theta', 10000.0)
    sc = dict(rp) if rp else (cfg.get('rope_scaling') or {})
    if ltype == 'sliding_attention' and not rp and str(cfg.get('model_type', '')).startswith('gemma3'): sc = {}   # е2
    frac = (rp or {}).get('partial_rotary_factor', cfg.get('partial_rotary_factor', 1.0))
    return float(theta), sc, float(frac)

def inv_freq(theta, sc, hd, rot, den=None, cfg=None):
    # ж1
    den = den or hd; cfg = cfg or {}
    kind = (sc.get('rope_type') or sc.get('type') or 'default')
    base = [theta ** (-2.0 * i / den) for i in range(rot // 2)]
    if kind in ('default', 'proportional', None, 'dynamic'):   # ж2
        return (None if den == hd else base), 1.0, None
    if kind == 'linear': return [f / float(sc['factor']) for f in base], 1.0, None
    if kind == 'llama3':
        fct, lo, hi, om = float(sc['factor']), float(sc.get('low_freq_factor', 1)), float(sc.get('high_freq_factor', 4)), float(sc['original_max_position_embeddings'])
        lo_w, hi_w, out = om / lo, om / hi, []
        for f in base:
            w = 2 * math.pi / f
            if w < hi_w: out.append(f)
            elif w > lo_w: out.append(f / fct)
            else: s_ = (om / w - lo) / (hi - lo); out.append((1 - s_) * f / fct + s_ * f)
        return out, 1.0, None
    if kind in ('longrope', 'su'):
        # ж3
        om = float(sc.get('original_max_position_embeddings') or cfg.get('original_max_position_embeddings') or cfg.get('max_position_embeddings') or 4096)
        mx = float(cfg.get('max_position_embeddings') or om); fct = mx / om
        sf = sc.get('short_factor') or [1.0] * (rot // 2)
        if len(sf) != rot // 2: return None, 1.0, f'LongRoPE: short_factor длины {len(sf)}, ожидалось {rot // 2}.'
        att = sc.get('attention_factor') or (math.sqrt(1 + math.log(fct) / math.log(om)) if fct > 1 else 1.0)
        return [f / float(x) for f, x in zip(base, sf)], float(att), None
    if kind == 'yarn':
        fct = float(sc['factor']); om = float(sc.get('original_max_position_embeddings') or 4096)
        bf, bs = float(sc.get('beta_fast') or 32), float(sc.get('beta_slow') or 1)
        dim_of = lambda nr: (den * math.log(om / (nr * 2 * math.pi))) / (2 * math.log(theta))
        lo, hi = max(math.floor(dim_of(bf)), 0), min(math.ceil(dim_of(bs)), rot // 2 - 1)
        out = []
        for i, f in enumerate(base):
            ramp = min(max((i - lo) / max(hi - lo, 1e-3), 0), 1); ext = 1 - ramp      # ж4
            out.append(f / fct * (1 - ext) + f * ext)
        att = sc.get('attention_factor') or (0.1 * math.log(fct) + 1.0 if fct > 1 else 1.0)
        return out, float(att), None
    return None, 1.0, f'RoPE «{kind}» пока не поддерживается.'

# ж5
_ATT = r'(?:self_attn|self_attention|attention|attn|attn\.attention|mixer)'
_MLP = r'(?:mlp|feed_forward|ffn|block_sparse_moe)'
ROLES = {
    'q': _ATT + r'\.(?:q_proj|wq|query|q)', 'k': _ATT + r'\.(?:k_proj|wk|key|k)', 'v': _ATT + r'\.(?:v_proj|wv|value|v)',
    'o': _ATT + r'\.(?:o_proj|wo|out_proj|dense|o)', 'qkv': _ATT + r'\.(?:qkv_proj|Wqkv|query_key_value|wqkv|c_attn)',
    'gate': _MLP + r'\.(?:gate_proj|w1|c_fc_0|fc_gate)', 'up': _MLP + r'\.(?:up_proj|w3|c_fc_1|fc_up)',
    'down': _MLP + r'\.(?:down_proj|w2|c_proj|fc2|dense_4h_to_h)', 'gate_up': _MLP + r'\.(?:gate_up_proj)',
    'fc1': _MLP + r'\.(?:fc1|c_fc|dense_h_to_4h)',
    'iln': r'(?:input_layernorm|attention_norm|ln_1|input_norm|pre_attention_layernorm|norm1)',
    'pal': r'(?:post_attention_layernorm|ffn_norm|ln_2|norm2)',                 # ж6
    'psal': r'(?:post_self_attn_layernorm|post_attn_norm)',                     # ж7
    'pfln': r'(?:pre_feedforward_layernorm)', 'pofln': r'(?:post_feedforward_layernorm|post_mlp_layernorm|post_ffn_norm)',
    'qn': _ATT + r'\.(?:q_norm|q_layernorm|query_layernorm)', 'kn': _ATT + r'\.(?:k_norm|k_layernorm|key_layernorm)',
    'ls': r'layer_scalar',
}
EMBED = r'(?:embed_tokens|tok_embeddings|wte|word_embeddings|embed_in|embeddings\.word_embeddings)\.weight'
FINAL = r'(?:norm|final_layernorm|ln_f|norm_f|final_norm)\.weight'
HEAD = r'(?:lm_head|output|embed_out)\.weight'
SIDE = re.compile(r'vision|visual|audio|encoder|mm_proj|multi_modal|speech|image|mtp')

def cfg_get(cfg, *keys, d=None):
    # з1
    for k in keys:
        if cfg.get(k) is not None: return cfg[k]
    return d

def roles_of(T, lp, l):
    # и1
    pre = lp + f'{l}.'; suf = [k[len(pre):] for k in T if k.startswith(pre)]
    out = {}
    for r, pat in ROLES.items():
        rx = re.compile(r'^' + pat + r'(?:\.weight)?$')
        hit = sorted(s for s in suf if rx.match(s) and not s.endswith('.bias'))
        if hit: out[r] = hit[0]
        bx = re.compile(r'^' + pat + r'\.bias$'); bh = [s for s in suf if bx.match(s)]
        if bh: out[r + '_bias'] = bh[0]
    return out

def layer_prefix(T):
    # к1
    cnt = {}
    for k in T:
        if SIDE.search(k): continue
        m = re.match(r'^(.*?\b(?:layers|h|blocks|layer))\.(\d+)\.', k)
        if m: cnt.setdefault(m.group(1) + '.', set()).add(int(m.group(2)))
    if not cnt: return None, 0
    lp = max(cnt, key=lambda p: (len(cnt[p]), 'language_model' in p, -len(p)))
    return lp, 1 + max(cnt[lp])

def describe(mdir, tens=None):
    # л1
    cfg0 = load_json(os.path.join(mdir, 'config.json'), {}); cfg = text_cfg(cfg0)
    gen = load_json(os.path.join(mdir, 'generation_config.json'), {})
    T, files = tens if tens else tensors(mdir); why = []
    mt = str(cfg.get('model_type') or cfg0.get('model_type') or '').lower()
    lp, nl = layer_prefix(T)
    if not lp: return None, ['Нет пронумерованных слоёв трансформера.']
    pre = lp[:lp.rindex('.', 0, len(lp) - 1) + 1] if lp.count('.') > 1 else ''
    R0 = roles_of(T, lp, 0)
    if any('experts' in k for k in T) or cfg_get(cfg, 'num_experts', 'num_local_experts', 'n_routed_experts', 'enable_moe_block'):
        return None, ['MoE (эксперты) пока не поддерживается.']
    if 'q' not in R0 and 'qkv' not in R0:
        return None, ['Слои без внимания вида q/k/v (линейное внимание, SSM) пока не поддерживаются.']
    need = ('q', 'k', 'o') if 'qkv' not in R0 else ('qkv', 'o')
    RN = {'q': 'q (запрос)', 'k': 'k (ключ)', 'o': 'выход внимания', 'qkv': 'слитые q/k/v'}
    miss0 = [RN[x] for x in need if x not in R0]
    if miss0: return None, [f'В слоях не найдено: {", ".join(miss0)} (тензоры слоя 0: {", ".join(sorted(k[len(lp) + 2:] for k in T if k.startswith(lp + "0."))[:8])}).']
    odd = [l for l in range(nl) if not all(x in roles_of(T, lp, l) for x in need)]
    if odd: return None, [f'Слои {", ".join(map(str, odd[:6]))}{"…" if len(odd) > 6 else ""} устроены иначе, чем слой 0 (смешанные блоки): пока не поддерживаются.']
    has = lambda l, r: r in roles_of(T, lp, l)
    def tn(l, r): return lp + f'{l}.' + roles_of(T, lp, l)[r]
    shp = lambda l, r: T[tn(l, r)][1]
    emb = next((k for k in T if k.startswith(pre) and re.fullmatch(re.escape(pre) + EMBED, k)), None)
    if not emb: return None, ['Нет матрицы эмбеддингов.']
    vocab, hidden = T[emb][1]
    fin = next((k for k in T if k.startswith(pre) and re.fullmatch(re.escape(pre) + FINAL, k)), None)
    if not fin: why.append('Нет финальной нормировки.')
    head = next((k for k in T if re.fullmatch(r'(?:.*\.)?' + HEAD, k) and not SIDE.search(k) and k != emb), '')
    heads = int(cfg_get(cfg, 'num_attention_heads', 'n_head', 'num_heads', d=0) or 0)
    # л2
    qkv_split = None
    if 'qkv' in R0:
        if re.sub(r'\.weight$', '', R0['qkv']).split('.')[-1] not in ('qkv_proj', 'Wqkv'):
            why.append('Слитые q/k/v с перемешанными по головам строками пока не поддерживаются.')
        else:
            rows = shp(0, 'qkv')[0]; hd0 = int(cfg.get('head_dim') or (hidden // heads if heads else 0))
            nkv0 = (rows // hd0 - heads) // 2 if hd0 else 0
            if hd0 and nkv0 > 0 and (heads + 2 * nkv0) * hd0 == rows: qkv_split = (heads * hd0, nkv0 * hd0)
            else: why.append(f'Слитые q/k/v: {rows} строк не раскладываются на {heads} голов q и одинаковые k, v.')
    if 'gate' not in R0 and 'gate_up' not in R0:
        why.append('MLP без затвора (fc1 → активация → fc2) пока не поддерживается.' if 'fc1' in R0 else 'Не найдены матрицы MLP.')
    if 'down' not in R0: why.append('Не найдена выходная матрица MLP.')
    inter = shp(0, 'gate')[0] if 'gate' in R0 else (shp(0, 'gate_up')[0] // 2 if 'gate_up' in R0 else 0)   # л3
    if 'gate' in R0 and any(has(l, 'gate') and shp(l, 'gate')[0] != inter for l in range(nl)): why.append('Разная ширина MLP по слоям пока не поддерживается.')
    # л4
    if any(r + '_bias' in R0 for r in ('iln', 'pal', 'pfln', 'pofln')) or str(cfg_get(cfg, 'norm_type', 'normalization', d='')).lower() == 'layernorm' \
            or (cfg.get('layer_norm_eps') is not None and cfg.get('rms_norm_eps') is None and mt not in ('exaone',)):
        why.append('LayerNorm (со сдвигом) пока не поддерживается, движок считает RMSNorm.')
    iln = R0.get('iln', '-')
    if 'pfln' in R0: pre_mlp, attn_out = R0['pfln'], R0.get('pal', R0.get('psal', '-'))     # л5
    elif iln == '-' and 'pal' in R0 and 'pofln' in R0: pre_mlp, attn_out = '-', R0['pal']    # л6
    else: pre_mlp, attn_out = R0.get('pal', '-'), R0.get('psal', '-')                       # л7
    pofln = R0.get('pofln', '-')
    dt = sorted({d for d, _ in T.values()})
    if any(d not in ('BF16',) for d in dt if d in ('F16', 'F32', 'F64')): why.append(f'Веса {"/".join(dt)}: требуется объединение в один файл BF16 (arch.py merge).')
    if len(files) > 1: why.append('Веса в нескольких файлах: требуется объединение частей (arch.py merge).')
    gemma = mt.startswith('gemma'); g4 = mt.startswith('gemma4')
    qk = 'qn' in R0; bias = 'q_bias' in R0; ls = 'ls' in R0
    if cfg.get('num_kv_shared_layers'): why.append('Общий KV между слоями пока не поддерживается.')
    if cfg.get('hidden_size_per_layer_input') and any('per_layer' in k for k in T): why.append('Входы по слоям (per-layer embeddings) пока не поддерживаются.')
    if cfg.get('use_double_wide_mlp'): why.append('Двойной MLP пока не поддерживается.')
    if cfg.get('use_bidirectional_attention') in (True, 'all', 'text'): why.append('Двустороннее внимание не предназначено для генерации.')   # л8
    if cfg.get('parallel_attn') or cfg.get('use_parallel_residual') or cfg.get('parallel_residual'): why.append('Параллельные ветки внимания и MLP пока не поддерживаются.')
    # л9
    emul, amul = cfg.get('embedding_multiplier'), cfg.get('attention_multiplier')
    rmul, ldiv = float(cfg.get('residual_multiplier') or 1), float(cfg.get('logits_scaling') or 1)   # л10
    ltypes = cfg.get('layer_types')
    if not (isinstance(ltypes, list) and len(ltypes) == nl):
        if g4: ltypes = ['full_attention' if not has(l, 'v') or (l + 1) % 6 == 0 else 'sliding_attention' for l in range(nl)]
        elif mt.startswith('gemma3'):
            pat = int(cfg.get('sliding_window_pattern') or 6); ltypes = ['full_attention' if (l + 1) % pat == 0 else 'sliding_attention' for l in range(nl)]
        elif mt.startswith('gemma2'): ltypes = ['sliding_attention' if l % 2 == 0 else 'full_attention' for l in range(nl)]
        elif cfg.get('sliding_window') and (cfg.get('use_sliding_window') or mt.startswith('mistral')): ltypes = ['sliding_attention'] * nl
        else: ltypes = ['full_attention'] * nl
    window = int(cfg.get('sliding_window') or 0)
    act = str(cfg_get(cfg, 'hidden_activation', 'hidden_act', 'activation_function', d='silu'))
    actk = 'gelu_tanh' if 'gelu' in act else 'silu' if act in ('silu', 'swish', 'swiglu') else None
    if not actk: why.append(f'Активация {act} пока не поддерживается.')
    nope = cfg.get('no_rope_layers')   # л11
    if not (isinstance(nope, list) and len(nope) == nl): nope = None
    L, freqs, pairs, mscs, qk_full = [], {}, 'head', set(), False
    for l in range(nl):
        full = ltypes[l] != 'sliding_attention'
        RL = roles_of(T, lp, l)
        if 'q' not in RL and not qkv_split: break      # л12
        qr, kr = qkv_split if qkv_split else (shp(l, 'q')[0], shp(l, 'k')[0])
        qn_dim = shp(l, 'qn')[0] if qk else 0
        # л13
        hd = qn_dim if qk and qn_dim and qn_dim != qr else (int(cfg.get('head_dim') or 0) or (qr // heads if heads else 0))
        if qk and qn_dim == qr and qr != hd: qk_full = True   # л14
        if not heads and hd: heads = qr // hd
        if not hd or qr != heads * hd or kr % hd: why.append(f'Слой {l}: формы q/k ({qr}, {kr}) не согласуются с головами {heads}×{hd}.'); break
        theta, sc, frac = rope_for(cfg, 'full_attention' if full else 'sliding_attention')
        if g4 and full and frac == 1.0 and not cfg.get('rope_parameters'): frac, theta = 0.25, (theta if theta != 10000.0 else 1e6)
        rot = int(round(hd * frac)) // 2 * 2
        if nope is not None and not nope[l]: rot = 0
        # л15
        part = rot not in (hd, 0) and not g4
        if part: pairs = 'rot'
        f, msc, err = inv_freq(theta, sc, hd, rot, rot if part else hd, cfg) if rot else (None, 1.0, None)
        if err: why.append(err); break
        if f: freqs[l] = f
        mscs.add(round(msc, 9))
        L.append(dict(full=int(full), hd=hd, nkv=kr // hd, has_v=int(has(l, 'v') or bool(qkv_split)), window=0 if full else window, theta=theta, rot=rot))
    if len(mscs) > 1: why.append('Разный множитель RoPE по слоям пока не поддерживается.')
    if g4: ascale = 1.0
    elif gemma: ascale = float(cfg.get('query_pre_attn_scalar') or L[0]['hd']) ** -0.5 if L else 1.0
    elif amul is not None: ascale = float(amul)
    else: ascale = L[0]['hd'] ** -0.5 if L else 1.0
    eos = []
    for e in (gen.get('eos_token_id'), cfg.get('eos_token_id'), cfg0.get('eos_token_id')):
        for x in (e if isinstance(e, list) else [e]):
            if isinstance(x, int) and x not in eos: eos.append(x)
    if g4 and 106 not in eos: eos.append(106)
    r = lambda k: R0.get(k, '-')
    W = lambda s: s if s == '-' or s.endswith('.weight') or s.endswith('.bias') else s + '.weight'
    if qkv_split:
        qd, kd = qkv_split; f_ = W(R0['qkv'])
        qn_, kn_, vn_ = f'{f_}[0:{qd}]', f'{f_}[{qd}:{qd + kd}]', f'{f_}[{qd + kd}:{qd + 2 * kd}]'
    else: qn_, kn_, vn_ = W(r('q')), W(r('k')), W(r('v'))
    if 'gate_up' in R0: gu = W(R0['gate_up']); gn_, un_ = f'{gu}[0:{inter}]', f'{gu}[{inter}:{2 * inter}]'
    else: gn_, un_ = W(r('gate')), W(r('up'))
    names = {'iln': W(iln), 'q': qn_, 'k': kn_, 'v': vn_, 'o': W(r('o')), 'gate': gn_, 'up': un_, 'down': W(r('down')),
             'paln': W(attn_out), 'pfln': W(pre_mlp), 'pofln': W(pofln), 'qn': W(r('qn')), 'kn': W(r('kn')), 'ls': r('ls'),
             'qb': r('q_bias'), 'kb': r('k_bias'), 'vb': r('v_bias'),
             'final': fin[len(pre):] if fin else '-', 'embed': emb[len(pre):]}
    escale = 'sqrt' if gemma else (float(emul) if emul is not None else 1)
    A = dict(family=mt or 'unknown', prefix=pre, layer_prefix=lp, hidden=hidden, inter=inter, heads=heads, vocab=vocab, layers=nl,
             eps=float(cfg_get(cfg, 'rms_norm_eps', 'layer_norm_epsilon', 'layer_norm_eps', 'norm_eps', d=1e-6)), softcap=float(cfg.get('final_logit_softcapping') or 0),
             emb_scale=escale, attn_scale=ascale, norm1p=int(gemma and not g4),
             post_attn=int(attn_out != '-'), post_ffn=int(pofln != '-'), qk_norm=int(qk), v_norm=int(g4), layer_scalar=int(ls),
             qkv_bias=int(bias), act=actk or act, tied=int(not head), head=head, eos=eos, names=names, L=L, freqs=freqs,
             rope_pairs=pairs, rope_mscale=(mscs.pop() if len(mscs) == 1 else 1.0), resid_mul=rmul, logit_div=ldiv, qk_norm_full=int(qk_full),
             attn_softcap=float(cfg.get('attn_logit_softcapping') or 0),
             files=files, dtypes=dt, params=sum(math.prod(s) for _, s in T.values()))
    return A, why

def write(A, why, path):
    open(path, 'w').write(text(A, why))

def text(A, why):
    o = ['# Описание модели для движка TierLM. Создаётся arch.py, ручное редактирование не требуется.']
    for k in ('family', 'prefix', 'layer_prefix', 'hidden', 'inter', 'heads', 'vocab', 'eps', 'softcap', 'emb_scale', 'attn_scale', 'norm1p',
              'post_attn', 'post_ffn', 'qk_norm', 'v_norm', 'layer_scalar', 'qkv_bias', 'act', 'tied'): o.append(f'{k} {A[k]}')
    if A['head']: o.append(f'head {A["head"]}')
    if A.get('rope_pairs', 'head') != 'head': o.append(f'rope_pairs {A["rope_pairs"]}')
    if abs(A.get('rope_mscale', 1.0) - 1.0) > 1e-9: o.append(f'rope_mscale {A["rope_mscale"]:.9g}')
    if A.get('qk_norm_full'): o.append('qk_norm_full 1')
    if A.get('resid_mul', 1.0) != 1.0: o.append(f'resid_mul {A["resid_mul"]:.9g}')
    if A.get('logit_div', 1.0) != 1.0: o.append(f'logit_div {A["logit_div"]:.9g}')
    if A.get('attn_softcap'): o.append(f'attn_softcap {A["attn_softcap"]:.9g}')   # н1
    opt = [l.split()[0] for l in o[1:] if l.split()[0] in ('rope_pairs', 'rope_mscale', 'qk_norm_full', 'resid_mul', 'logit_div', 'attn_softcap')]
    if opt: o.append('requires ' + ' '.join(opt))   # н2
    o.append('eos ' + ' '.join(map(str, A['eos'])))
    o += [f'name {k} {v}' for k, v in A['names'].items()]
    o.append(f'layers {A["layers"]}')
    o += [f'layer {i} {s["full"]} {s["hd"]} {s["nkv"]} {s["has_v"]} {s["window"]} {s["theta"]:.9g} {s["rot"]}' for i, s in enumerate(A['L'])]
    o += [f'freq {i} ' + ' '.join(f'{x:.9g}' for x in f) for i, f in A['freqs'].items()]
    o += [f'unsupported {w}' for w in why]
    return '\n'.join(o) + '\n'

def summary(A, why):
    if not A: return 'Не поддерживается. ' + ' '.join(why)
    nf = sum(s['full'] for s in A['L'])
    return (f'{A["family"]}: {A["params"] / 1e9:.2f}B, слоёв {A["layers"]} (полного внимания {nf}), ширина {A["hidden"]}, MLP {A["inter"]}, '
            f'голов {A["heads"]}/{A["L"][0]["nkv"] if A["L"] else "?"}, словарь {A["vocab"]}, активация {A["act"]}'
            + (', смещения q/k/v' if A['qkv_bias'] else '') + (', QK-норм' if A['qk_norm'] else '') + (', нормировки после блоков' if A['post_attn'] else '')
            + ('' if A['tied'] else ', отдельная выходная матрица') + ('.\nНе поддерживается. ' + ' '.join(why) if why else '.'))

def merge(mdir, keep=False):
    # п1
    import numpy as np, zlib, shutil
    idx = load_json(os.path.join(mdir, 'model.safetensors.index.json'))
    files = sorted(set(idx['weight_map'].values())) if idx else ['model.safetensors']
    heads = {fn: st_header(os.path.join(mdir, fn)) for fn in files}
    items = sorted(((k, fn, v) for fn, (h, _) in heads.items() for k, v in h.items()), key=lambda x: x[0])
    conv = sorted({v['dtype'] for _, _, v in items if v['dtype'] in ('F16', 'F32')})
    if not conv and len(files) == 1: print('Файл уже один и в формате BF16, объединение частей не требуется.'); return
    hdr, off = {}, 0
    for k, fn, v in items:
        src = v['data_offsets'][1] - v['data_offsets'][0]
        n = math.prod(v['shape']) * 2 if v['dtype'] in ('F16', 'F32') else src
        hdr[k] = {'dtype': 'BF16' if v['dtype'] in ('F16', 'F32') else v['dtype'], 'shape': v['shape'], 'data_offsets': [off, off + n]}; off += n
    hb = json.dumps(hdr, separators=(',', ':')).encode(); hb += b' ' * ((8 - len(hb) % 8) % 8)
    need, free = 8 + len(hb) + off, shutil.disk_usage(mdir).free
    if free < need + (1 << 30): raise SystemExit(f'Недостаточно места на диске: требуется {need / 2**30:.1f} ГиБ, свободно {free / 2**30:.1f} ГиБ.')
    out = os.path.join(mdir, 'model.safetensors.part'); crc = {}
    CH = 64 << 20
    def bf16(raw, dt):
        a = np.frombuffer(raw, dtype=np.float16 if dt == 'F16' else np.float32).astype(np.float32)
        u = a.view(np.uint32); return ((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype(np.uint16).tobytes()   # п2
    with open(out, 'wb') as fo:
        fo.write(struct.pack('<Q', len(hb))); fo.write(hb)
        for i, (k, fn, v) in enumerate(items):
            h, base = heads[fn]; a, b = v['data_offsets']; es = 2 if v['dtype'] == 'F16' else 4 if v['dtype'] == 'F32' else 1
            c = 0
            with open(os.path.join(mdir, fn), 'rb') as fi:
                fi.seek(base + a); left = b - a
                while left:
                    raw = fi.read(min(left, CH // es * es)); left -= len(raw)
                    if not raw: raise SystemExit(f'{fn}: Файл обрывается на {k}.')
                    if v['dtype'] in ('F16', 'F32'): raw = bf16(raw, v['dtype'])
                    c = zlib.crc32(raw, c); fo.write(raw)
            crc[k] = c
            if i % 50 == 0: print(f'Объединение {i + 1}/{len(items)}', flush=True)
    # п3
    with open(out, 'rb') as f:
        n = struct.unpack('<Q', f.read(8))[0]; h2 = json.loads(f.read(n)); base = 8 + n
        for k, v in h2.items():
            a, b = v['data_offsets']; f.seek(base + a); c, left = 0, b - a
            while left:
                raw = f.read(min(left, CH)); left -= len(raw); c = zlib.crc32(raw, c)
                if not raw: break
            if c != crc[k]: os.remove(out); raise SystemExit(f'Сверка не пройдена на {k}. Исходные файлы не затронуты.')
    os.replace(out, os.path.join(mdir, 'model.safetensors'))
    if not keep:
        for fn in files:
            if fn != 'model.safetensors': os.remove(os.path.join(mdir, fn))
        if idx: os.remove(os.path.join(mdir, 'model.safetensors.index.json'))
    print('Объединение в model.safetensors выполнено, сверка пройдена' + (f'; преобразование в BF16 из {"/".join(conv)}' + (' (у F16 отброшены младшие 3 бита мантиссы)' if 'F16' in conv else '') if conv else '') + '.')

def g4check(mdir):
    # р1
    A, why = describe(mdir)
    if not A: return [f'Описание не получено: {" ".join(why)}']
    want = dict(family='gemma4_text', prefix='model.language_model.', layer_prefix='model.language_model.layers.', hidden=3840, inter=15360,
                heads=16, vocab=262144, layers=48, eps=1e-6, softcap=30.0, emb_scale='sqrt', attn_scale=1.0, norm1p=0, post_attn=1, post_ffn=1,
                qk_norm=1, v_norm=1, layer_scalar=1, qkv_bias=0, act='gelu_tanh', tied=1)
    bad = [f'{k}: в движке {v}, в описании {A.get(k)}.' for k, v in want.items() if (A.get(k) != v and not (k == 'family' and str(A.get(k)).startswith('gemma4')))]
    for l, L in enumerate(A['L']):
        full = (l + 1) % 6 == 0
        w = dict(full=int(full), hd=512 if full else 256, nkv=1 if full else 8, has_v=0 if full else 1, window=0 if full else 1024,
                 theta=1e6 if full else 1e4, rot=128 if full else 256)
        bad += [f'Слой {l} {k}: в движке {v}, в описании {L[k]}.' for k, v in w.items() if L[k] != v]
    if A['freqs']: bad.append(f'Частоты RoPE заданы таблицей у слоёв {sorted(A["freqs"])[:6]}.')
    if sorted(A['eos']) != [1, 106]: bad.append(f'Концы ответа {A["eos"]}, в движке [1, 106].')
    return bad + [f'Не поддерживается: {w}' for w in why]

if __name__ == '__main__':
    a = sys.argv[1:]
    if len(a) >= 2 and a[0] == 'g4check':
        b = g4check(a[1]); print('\n'.join(b) if b else 'Совпадает: движок 2.3 считает эту Gemma 4 12B так же, как 2.2.6.'); sys.exit(1 if b else 0)
    if len(a) >= 2 and a[0] == 'desc':     # р2
        A, why = describe(a[1])
        if not A: print('unsupported ' + ' '.join(why)); sys.exit(2)
        sys.stdout.write(text(A, why)); sys.exit(0)
    if len(a) >= 2 and a[0] == 'merge': merge(a[1], keep='--keep' in a); sys.exit(0)
    if len(a) >= 2 and a[0] == 'info':
        A, why = describe(a[1])
        if '--json' in a:
            d = {k: v for k, v in (A or {}).items() if k not in ('freqs', 'files')}; d['why'] = why; d['ok'] = bool(A) and not why; d['merge'] = bool(A) and bool(why) and all('merge' in w for w in why)
            print(json.dumps(d, ensure_ascii=False))
        else: print(summary(A, why))
        sys.exit(0 if A and not why else 2)
    if len(a) == 1:
        A, why = describe(a[0])
        if A: write(A, why, os.path.join(a[0], 'tier_arch.txt'))
        print(summary(A, why)); sys.exit(0 if A and not why else 2)
    print(open(__file__, encoding='utf-8').read().split('\nimport', 1)[0]); sys.exit(1)
