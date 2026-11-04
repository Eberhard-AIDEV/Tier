#!/usr/bin/env python3
# а1
import glob, json, os, re, shutil, subprocess, sys

ROOT = os.environ.get('HWPROBE_ROOT', '')          # а2
def p(path): return ROOT + path
def rd(path, d=''):
    try:
        with open(p(path), encoding='utf-8', errors='replace') as f: return f.read().strip()
    except OSError: return d
def sh(*cmd, t=8):
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=t)
        return r.stdout if r.returncode == 0 else ''
    except (OSError, subprocess.SubprocessError): return ''
def have(name): return shutil.which(name) is not None
def num(s):
    m = re.match(r'\s*([\d.]+)', s or ''); return float(m.group(1)) if m else 0.0

# ж1
def cpu():
    flags = set(); model = ''
    for line in rd('/proc/cpuinfo').splitlines():
        if line.startswith('flags') and not flags: flags = set(line.split(':', 1)[1].split())
        if line.startswith('model name') and not model: model = line.split(':', 1)[1].strip()
    return {'модель': model, 'ядра': os.cpu_count() or 0,
            'avx2': 'avx2' in flags, 'fma': 'fma' in flags, 'avx512': any(x.startswith('avx512f') for x in flags)}

def ram():
    m = {}
    for line in rd('/proc/meminfo').splitlines():
        k, _, v = line.partition(':')
        m[k] = num(v) * 1024
    return {'всего': int(m.get('MemTotal', 0)), 'доступно': int(m.get('MemAvailable', 0))}

def gpus_drm():
    # и1
    out = []
    for d in sorted(glob.glob(p('/sys/class/drm/card[0-9]*/device'))):
        v = rd(d[len(ROOT):] + '/vendor')
        if v not in ('0x1002', '0x10de'): continue
        base = d[len(ROOT):]
        gen, width = pcie(base)
        g = {'вендор': 'amd' if v == '0x1002' else 'nvidia', 'путь': base,
             'память_всего': int(num(rd(base + '/mem_info_vram_total', '0'))),
             'память_занято': int(num(rd(base + '/mem_info_vram_used', '0'))),
             'gtt_всего': int(num(rd(base + '/mem_info_gtt_total', '0'))),
             'шина': {'поколение': gen, 'линий': width,        # и2
                      'скорость': rd(base + '/current_link_speed'), 'ширина': rd(base + '/current_link_width'),
                      'макс_скорость': rd(base + '/max_link_speed'), 'макс_ширина': rd(base + '/max_link_width')}}
        out.append(g)
    return out

def gpu_dev(pci_or_card=None):
    # к1
    x = pci_or_card or ''
    if x.startswith('/sys/'): return x
    if re.fullmatch(r'card\d+', x): return '/sys/class/drm/' + x + '/device'
    if x: return '/sys/bus/pci/devices/' + x
    best, bm = '', -1
    for d in sorted(glob.glob(p('/sys/class/drm/card[0-9]*/device'))):
        d = d[len(ROOT):]
        if rd(d + '/vendor') not in ('0x1002', '0x10de'): continue
        m = int(num(rd(d + '/mem_info_vram_total', '0')))
        if m > bm: best, bm = d, m
    return best

GT = ((63, 6), (31, 5), (15, 4), (7, 3), (4, 2), (0.1, 1))   # к2
def pcie(pci=None):
    # л1
    d = gpu_dev(pci)
    gen = width = 0
    if d:
        cur = os.path.realpath(p(d))
        while os.path.exists(cur + '/max_link_speed'):
            s = num(rd(cur[len(ROOT):] + '/max_link_speed')); w = int(num(rd(cur[len(ROOT):] + '/max_link_width')))
            g = next((n for t, n in GT if s >= t), 0)
            if g: gen = min(gen, g) if gen else g
            if w: width = min(width, w) if width else w
            cur = os.path.dirname(cur)
    if not (gen and width) and have('nvidia-smi'):
        ln = sh('nvidia-smi', '--query-gpu=pcie.link.gen.max,pcie.link.width.max', '--format=csv,noheader,nounits').strip().splitlines()
        f = [x.strip() for x in ln[0].split(',')] if ln else []
        if len(f) >= 2: gen, width = gen or int(num(f[0])), width or int(num(f[1]))
    return gen, width

def temp_sensors(pci_or_card=None):
    # м1
    out, d = [], gpu_dev(pci_or_card)
    c = lambda s: num(s) / 1000 if s else None
    for h in sorted(glob.glob(p(d + '/hwmon/hwmon*'))) if d else []:
        h = h[len(ROOT):]
        for f in sorted(glob.glob(p(h + '/temp*_input'))):
            k = os.path.basename(f)[:-6]; now = rd(h + f'/{k}_input')
            if not now: continue
            out.append({'label': rd(h + f'/{k}_label') or k, 'now_c': c(now), 'crit_c': c(rd(h + f'/{k}_crit')), 'emerg_c': c(rd(h + f'/{k}_emergency'))})
    if not out and have('nvidia-smi') and (not d or rd(d + '/vendor') in ('', '0x10de')):
        sel = ['-i', os.path.basename(os.path.realpath(p(d)))] if d and rd(d + '/vendor') == '0x10de' else []
        now = sh('nvidia-smi', *sel, '--query-gpu=temperature.gpu', '--format=csv,noheader,nounits').strip().splitlines()[:1]
        q = sh('nvidia-smi', *sel, '-q', '-d', 'TEMPERATURE')
        th = lambda name: (lambda m: float(m.group(1)) if m else None)(re.search(name + r'\s*:\s*([\d.]+)\s*C', q))
        if now and num(now[0]):
            out.append({'label': 'gpu', 'now_c': num(now[0]), 'crit_c': th('GPU Slowdown Temp'), 'emerg_c': th('GPU Shutdown Temp')})
    return out

def kfd_nodes():
    # н1
    out = []
    for n in sorted(glob.glob(p('/sys/class/kfd/kfd/topology/nodes/*'))):
        v = 0
        for line in rd(n[len(ROOT):] + '/properties').splitlines():
            if line.startswith('gfx_target_version'): v = int(num(line.split()[1]))
        if not v: continue
        mem = 0
        for b in glob.glob(n + '/mem_banks/*/properties'):
            h = s = 0
            for line in open(b, errors='replace'):
                k, _, val = line.partition(' ')
                if k == 'heap_type': h = int(val or 0)
                if k == 'size_in_bytes': s = int(val or 0)
            if h in (1, 2): mem += s
        out.append({'arch': 'gfx%d%d%x' % (v // 10000, v // 100 % 100, v % 100), 'память': mem})
    return out

def nvidia():
    if not have('nvidia-smi'): return []
    txt = sh('nvidia-smi', '--query-gpu=name,memory.total,compute_cap,driver_version,pcie.link.gen.current,pcie.link.width.current',
             '--format=csv,noheader,nounits')
    out = []
    for line in txt.strip().splitlines():
        f = [x.strip() for x in line.split(',')]
        if len(f) < 6: continue
        out.append({'имя': f[0], 'память': int(num(f[1]) * 2**20), 'cc': f[2].replace('.', ''), 'драйвер': f[3],
                    'шина': {'поколение': f[4], 'ширина': f[5]}})
    return out

def toolchain():
    rocm = rd('/opt/rocm/.info/version') or rd('/opt/rocm/share/doc/hip/version')
    cuda = ''
    if have('nvcc'):
        m = re.search(r'release ([\d.]+)', sh('nvcc', '--version'))
        cuda = m.group(1) if m else ''
    return {'hipcc': have('hipcc'), 'rocm': rocm, 'nvcc': have('nvcc'), 'cuda': cuda,
            'rocminfo': have('rocminfo'), 'hipblaslt': any(os.path.exists(x + '/include/hipblaslt/hipblaslt.h') for x in (rocm_root(), '/usr'))}   # п1

def python_pkgs():
    miss = []
    for mod, pkg in (('tokenizers', 'tokenizers'), ('numpy', 'numpy'), ('jinja2', 'jinja2')):
        try: __import__(mod)
        except ImportError: miss.append(pkg)
    return miss

def distro():
    d = {}
    for line in rd('/etc/os-release').splitlines():
        k, _, v = line.partition('='); d[k] = v.strip('"')
    ids = (d.get('ID', '') + ' ' + d.get('ID_LIKE', '')).split()
    return 'arch' if 'arch' in ids else 'debian' if ('debian' in ids or 'ubuntu' in ids) else ''

# т1
def gfx_parse(arch):
    # т2
    m = re.fullmatch(r'gfx([0-9]+)([0-9a-f])([0-9a-f])', arch or '')
    return (int(m.group(1)), int(m.group(2), 16), int(m.group(3), 16)) if m else None

def rocm_root():
    r = os.environ.get('ROCM_PATH') or sh('hipconfig', '--rocmpath').strip()
    return r if r and os.path.isdir(r) else '/opt/rocm'

def shipped(lib):
    # ф1
    import glob
    dirs = [os.path.join(rocm_root(), 'lib', lib, 'library')] + glob.glob(f'/usr/lib*/{lib}/library') + glob.glob(f'/usr/lib*/*/{lib}/library')   # ф2
    if lib == 'rocblas' and os.environ.get('ROCBLAS_TENSILE_LIBPATH'): dirs.insert(0, os.environ['ROCBLAS_TENSILE_LIBPATH'])
    out = set()
    for d in dirs:
        try: out |= {m.group(0) for f in os.listdir(d) for m in [re.search(r'gfx[0-9]+[0-9a-f]{2}', f)] if m}
        except OSError: pass
    return out

def shipped_any():
    # х1
    out = set()
    for lib in ('hipblaslt', 'hipsparselt', 'rocsparse', 'rocfft', 'miopen', 'rocsolver'): out |= shipped(lib)
    for top in [os.path.join(rocm_root(), 'lib')] + glob.glob('/usr/lib*/rocm') + glob.glob('/usr/share/miopen'):
        for base, dirs, files in os.walk(top):
            if base.count(os.sep) - top.count(os.sep) >= 3: dirs[:] = []
            out |= {m.group(0) for f in files for m in [re.search(r'gfx[0-9]+[0-9a-f]{2}', f)] if m}
    return out

def amd_group(t):
    # ц1
    ma, mi, st = t
    if ma == 9 and mi == 0: return 'cdna1-2' if st in (8, 10) else 'gcn5'      # ц2
    if ma == 11: return 'rdna3'                                                # ц3
    return f'{ma}.{mi}'                                                        # ц4

def amd_rules(arch, lib=None):
    # ч1
    t = gfx_parse(arch); lib = shipped('rocblas') if lib is None else lib; lt = shipped('hipblaslt')
    r = {'arch': arch, 'сборка_для': arch, 'hsa_override': '', 'rocblas': sorted(lib), 'hipblaslt_ядра': arch in lt}
    if t: r['bf16'] = t[0] >= 11 or (t[0] == 9 and (t[1] >= 4 or t[2] == 10))   # ч2
    if not t or arch in lib: return r                                          # ч3
    guess = not lib
    if guess:   # ч4
        lib = shipped_any()
        r['замечание'] = 'Каталог ядер rocBLAS не найден: донор определён по ядрам других библиотек ROCm.' if lib else 'Каталог ядер rocBLAS не найден, подмена архитектуры не определена.'
        if not lib or arch in lib: return r
    g = amd_group(t)
    donors = sorted((d for d in lib if gfx_parse(d) and amd_group(gfx_parse(d)) == g),
                    key=lambda d: (gfx_parse(d)[:2] != t[:2], gfx_parse(d)[2] != 0, abs(gfx_parse(d)[2] - t[2]), d))
    if not donors and guess: return r                                          # ч5
    if not donors:
        r.update({'сборка_для': '', 'нет_ядер': f'Установленный rocBLAS не содержит ядер для {arch} и для соседних архитектур семейства: требуется ROCm с поддержкой {arch} или старшего в семействе.'})
        return r
    d = gfx_parse(donors[0]); r.update({'сборка_для': donors[0], 'hsa_override': f'{d[0]}.{d[1]}.{d[2]}'})
    return r

def nv_rules(cc):
    c = int(cc or 0)
    if c >= 80: return {'cc': cc, 'bf16': True, 'тензорные_ядра': True, 'нужен_cuda': '11.0'}
    if c >= 70: return {'cc': cc, 'bf16': False, 'тензорные_ядра': True, 'нужен_cuda': '11.0', 'замечание': 'BF16 недоступен, используется путь FP16/FP32.'}
    return {'cc': cc, 'bf16': False, 'тензорные_ядра': False, 'нужен_cuda': '11.8' if c >= 35 else '10.2',
            'замечание': 'Устаревшая архитектура (Kepler/Maxwell/Pascal): BF16 и тензорные ядра недоступны, движку требуется путь FP32.'}

def decide(f):
    # э1
    cand = []
    for g in kfd_nodes():
        cand.append(('amd', g['память'], amd_rules(g['arch'])))
    for g in nvidia():
        cand.append(('nvidia', g['память'], nv_rules(g['cc']) | {'имя': g['имя']}))
    for g in gpus_drm():
        if g['вендор'] == 'amd' and not any(c[0] == 'amd' for c in cand):
            cand.append(('amd', g['память_всего'], {'arch': '', 'замечание': 'Видна в sysfs, архитектура не определена (нет KFD).'}))
    if not cand: return {'вендор': '', 'замечание': 'Видеокарта не найдена.'}
    v, mem, rules = max(cand, key=lambda c: c[1])
    rules = dict(rules); rules.update({'вендор': v, 'память': mem, 'вендоров_видно': sorted({c[0] for c in cand})})
    return rules

def conf(key, d=0):
    # ю1
    path = os.environ.get('TIER_CONF') or os.path.expanduser('~/.config/tier.conf')
    try:
        for line in open(path, encoding='utf-8', errors='replace'):
            w = line.split()
            if len(w) >= 2 and w[0] == key: return num(w[1]) if isinstance(d, (int, float)) else w[1]
    except OSError: pass
    return d

def ram_reserve(total_bytes, avail_bytes, cap_pct=98, other_bytes=1 << 30):
    # я1
    c = min(99, max(50, int(cap_pct or 98)))
    return int(max(0, total_bytes - avail_bytes) + total_bytes * (100 - c) // 100 + other_bytes)

def ram_reserve_now():
    # аа1
    m = ram(); kw = {}
    try: kw['other_bytes'] = int(float(os.environ['TIER_RAM_OTHER_GIB']) * 2**30)
    except (KeyError, ValueError): pass
    return ram_reserve(m['всего'], m['доступно'], int(conf('mem_cap_pct', 98)), **kw)

# аб1
def act_bytes(pack, hidden, inter, heads, layers):
    # аб2
    P = int(pack)
    return P * (8 * hidden + 3 * inter) * 4 + 2 * heads * P * P * 4 + layers * P * hidden * 4

def arch_dims(path):
    # ав1
    txt = ''
    if path and os.path.isdir(path):
        f = os.path.join(path, 'tier_arch.txt')
        if os.path.isfile(f): path = f
        else:
            try:
                sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import arch
                A, _why = arch.describe(path); txt = arch.text(A, []) if A else ''
            except Exception: txt = ''
    if not txt and path and os.path.isfile(path):
        try: txt = open(path, encoding='utf-8', errors='replace').read()
        except OSError: txt = ''
    d, kv = {}, 0
    for line in txt.splitlines():
        w = line.split()
        if len(w) >= 2 and w[0] in ('hidden', 'inter', 'heads', 'layers', 'vocab'):
            try: d[w[0]] = int(w[1])
            except ValueError: pass
        elif len(w) >= 5 and w[0] == 'layer':       # ав2
            try: kv += 2 * int(w[3]) * int(w[4]) * 4
            except ValueError: pass
    if kv: d['kv_per_pos'] = kv
    return d if all(k in d for k in ('hidden', 'inter', 'heads', 'layers')) else {}

PACK_LO = 512                       # ав3
PACK_CAP = (64 << 10) // 4 - 64     # ав4
TRAIN_ACT_SHARE = 8                 # ав5
EVAL_ACT_SHARE = 2                  # ав6

def vram_free_mib(pci_or_card=None):
    # аг1
    d = gpu_dev(pci_or_card)
    if d and rd(d + '/vendor') == '0x1002':
        t, u = num(rd(d + '/mem_info_vram_total', '0')), num(rd(d + '/mem_info_vram_used', '0'))
        if t: return int(max(0, t - u)) >> 20
    if have('nvidia-smi'):
        sel = ['-i', os.path.basename(os.path.realpath(p(d)))] if d and rd(d + '/vendor') == '0x10de' else []
        ln = sh('nvidia-smi', *sel, '--query-gpu=memory.free', '--format=csv,noheader,nounits').strip().splitlines()
        if ln: return int(num(ln[0]))
    return 0

def pack_fit(room_bytes, dims, lo=PACK_LO, cap=PACK_CAP):
    # ад1
    best, P = 0, lo
    while P <= cap and act_bytes(P, dims['hidden'], dims['inter'], dims['heads'], dims['layers']) <= room_bytes:
        best, P = P, P * 2
    return best

def pack_choice(p99, n_examples, vram_free_mib, dims):
    # ае1
    p99, n = int(p99 or 0), int(n_examples or 0)
    if n <= 256: return 0
    if dims and vram_free_mib:
        pmax = pack_fit((int(vram_free_mib) << 20) // TRAIN_ACT_SHARE, dims)
    else:
        pmax = PACK_LO
    want = PACK_LO
    while want * 2 < 5 * p99 and want * 2 <= PACK_CAP: want *= 2
    if 5 * p99 <= 2 * want and want <= pmax: return want
    return pmax if pmax and 32 * p99 <= 25 * pmax else 0

# аж1
def _log_speed():
    # аж2
    work = os.environ.get('WORK') or os.environ.get('D') or ''
    if not work:
        data = os.environ.get('TIER_DATA') or os.path.join(os.environ.get('XDG_DATA_HOME') or os.path.expanduser('~/.local/share'), 'tierlm')
        try: work = json.load(open(os.path.join(data, 'settings.json'), encoding='utf-8')).get('WORK') or ''
        except (OSError, ValueError, AttributeError): work = ''
        work = work or os.path.join(data, 'work')
    try: fs = sorted((os.path.join(work, f) for f in os.listdir(work) if f.endswith(('_log.tsv', '_console.txt', '_infer.log'))), key=os.path.getmtime, reverse=True)
    except OSError: return 0.0
    rx = re.compile(r'Кэш весов: ([\d.]+) ГиБ за ([\d.]+) с \((\d+) МБ/с')
    for f in fs[:20]:
        try:
            with open(f, 'rb') as fo:
                fo.seek(max(0, os.path.getsize(f) - (8 << 20))); txt = fo.read().decode('utf-8', 'replace')
        except OSError: continue
        for m in reversed(list(rx.finditer(txt))):
            if float(m.group(1)) >= 0.5 and float(m.group(2)) >= 0.2:    # аж3
                return float(m.group(3)) * 1048576
    return 0.0

def _block_of(path):
    # аз1
    def base(name):
        sysd = os.path.realpath('/sys/class/block/' + name)
        if os.path.exists(sysd + '/partition'): sysd = os.path.dirname(sysd)
        sl = glob.glob(sysd + '/slaves/*')
        return base(os.path.basename(sl[0])) if sl else os.path.basename(sysd)
    try:
        st = os.stat(path); d = os.path.realpath(f'/sys/dev/block/{os.major(st.st_dev)}:{os.minor(st.st_dev)}')
        if os.path.isdir(d): return base(os.path.basename(d))
    except OSError: pass
    best, src = '', ''
    ap = os.path.realpath(path)
    for line in rd('/proc/self/mountinfo').splitlines():
        a, _, b = line.partition(' - ')
        mp, bw = a.split()[4] if len(a.split()) > 4 else '', b.split()
        if len(bw) >= 2 and bw[1].startswith('/dev/') and (ap == mp or ap.startswith(mp.rstrip('/') + '/')) and len(mp) > len(best):
            best, src = mp, bw[1]
    return base(os.path.basename(os.path.realpath(src))) if src else ''

def disk_read_bps(path=None):
    # аи1
    v = _log_speed()
    if v: return v
    b = _block_of(path or os.getcwd())
    if b.startswith('nvme'): return 2.5e9
    if rd(f'/sys/block/{b}/queue/rotational') == '1': return 150e6
    return 500e6

def facts():
    d = decide({})
    f = {'процессор': cpu(), 'ОЗУ': ram(), 'видеокарты_drm': gpus_drm(), 'kfd': kfd_nodes(), 'nvidia': nvidia(),
         'инструменты': toolchain(), 'дистрибутив': distro(), 'главная': d}
    f['резерв_ОЗУ'] = ram_reserve_now()
    f['нехватает'] = missing(f)
    return f

def missing(f):
    # ал1
    m = {'python': python_pkgs(), 'компилятор': [], 'системное': []}
    d, t = f['главная'], f['инструменты']
    if d.get('вендор') == 'amd' and not t['hipcc']: m['компилятор'].append('rocm-hip-sdk')
    if d.get('вендор') == 'amd' and not t['hipblaslt']: m['системное'].append('hipblaslt')
    if d.get('вендор') == 'nvidia' and not t['nvcc']: m['компилятор'].append('cuda')
    return m

PYPKG = {'arch': {'tokenizers': 'python-tokenizers', 'numpy': 'python-numpy', 'jinja2': 'python-jinja'},
         'debian': {'numpy': 'python3-numpy', 'jinja2': 'python3-jinja2'}}   # ал2
# ал3
PKG = {'arch': {'rocm-hip-sdk': ['rocm-hip-sdk'], 'hipblaslt': ['hipblaslt'], 'cuda': ['cuda']},
       'debian': {'rocm-hip-sdk': None, 'hipblaslt': None, 'cuda': ['nvidia-cuda-toolkit']}}

def install(f, log=print):
    m = missing(f)
    # ам1
    sys_pkgs = [PYPKG.get(f['дистрибутив'], {})[x] for x in m['python'] if x in PYPKG.get(f['дистрибутив'], {})]
    pip_left = [x for x in m['python'] if x not in PYPKG.get(f['дистрибутив'], {})]
    if pip_left:
        log('Установка пакетов Python: ' + ' '.join(pip_left))
        for cmd in ([sys.executable, '-m', 'pip', 'install', '--user'], [sys.executable, '-m', 'pip', 'install', '--user', '--break-system-packages']):
            if subprocess.run(cmd + pip_left, capture_output=True).returncode == 0: break
        else: log('Не удалось установить пакеты через pip: ' + ' '.join(pip_left))
    for need in m['компилятор'] + m['системное']:
        if f['дистрибутив'] not in PKG: log(f'Дистрибутив не распознан ({f["дистрибутив"] or "?"}): пакет «{need}» установить невозможно, требуется пакетный менеджер Arch или Debian/Ubuntu.'); continue
        names = PKG[f['дистрибутив']].get(need)
        if names is None: log(f'Пакет «{need}» отсутствует в репозиториях {f["дистрибутив"]}: требуется репозиторий AMD ROCm, автоматическая установка невозможна.'); continue
        sys_pkgs += names
    if not sys_pkgs: return m
    dist = f['дистрибутив']
    if dist == 'arch': cmd = ['pacman', '-S', '--needed', '--noconfirm'] + sys_pkgs
    elif dist == 'debian': cmd = ['apt-get', 'install', '-y'] + sys_pkgs
    else: log('Дистрибутив не распознан. Требуется установить вручную: ' + ' '.join(sys_pkgs)); return m
    if os.geteuid() != 0:
        pre = ['pkexec'] if have('pkexec') else ['sudo', '-n'] if have('sudo') else None
        if not pre: log('Требуются права администратора: ' + ' '.join(cmd)); return m
        cmd = pre + cmd
    log('Установка системных пакетов: ' + ' '.join(sys_pkgs))
    r = subprocess.run(cmd)
    if r.returncode and dist == 'debian': subprocess.run(['apt-get', 'update'])
    return m

def fingerprint():
    # ан1
    import hashlib
    d = decide({}); t = toolchain()
    return '|'.join([d.get('вендор', ''), d.get('arch', '') or str(d.get('cc', '')), t.get('rocm', '') or t.get('cuda', ''),
                     hashlib.sha1(' '.join(d.get('rocblas', [])).encode()).hexdigest()[:12]])

# ао1
def env_lines(f):
    d = f['главная']
    out = {}
    if d.get('вендор') == 'amd' and d.get('arch'):
        out['TIER_VENDOR'] = 'amd'; out['TIER_ARCH'] = d.get('сборка_для') or d['arch']; out['TIER_GPU_ARCH'] = d['arch']
        if d.get('hsa_override'): out['HSA_OVERRIDE_GFX_VERSION'] = d['hsa_override']
        if d.get('нет_ядер'): out['TIER_ARCH_ERR'] = "'" + d['нет_ядер'].replace("'", '') + "'"
    if d.get('вендор') == 'nvidia' and d.get('cc'):
        out['TIER_VENDOR'] = 'nvidia'; out['TIER_ARCH'] = 'sm_' + d['cc']
    out['TIER_BF16'] = '1' if d.get('bf16') else '0'
    out['TIER_DT'] = 'bf16' if d.get('bf16') else 'fp16'      # ао2
    out['TIER_RESERVE_RAM'] = str(f['резерв_ОЗУ'])
    return out

if __name__ == '__main__':
    args = sys.argv[1:]
    if args[:1] == ['--pack']:   # ао3
        if len(args) < 3: sys.exit('Формат вызова: hwprobe.py --pack P99 ПРИМЕРОВ [tier_arch.txt|папка модели]')
        print(pack_choice(int(num(args[1])), int(num(args[2])), vram_free_mib(), arch_dims(args[3] if len(args) > 3 else ''))); sys.exit(0)
    f = facts()
    if '--deps' in args:
        install(f); f = facts()
        if '--env' not in args and '--json' not in args: sys.exit(0 if not any(missing(f).values()) else 1)
    if '--fingerprint' in args:
        print(fingerprint()); sys.exit(0)
    if '--env' in args:
        for k, v in env_lines(f).items(): print(f'{k}={v}')
    else:
        print(json.dumps({'факты': f, 'конфигурация': env_lines(f)}, ensure_ascii=False, indent=1))
