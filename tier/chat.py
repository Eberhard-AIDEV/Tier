# а1
import re, struct, subprocess, sys, os, threading, time
try: import readline  # а2
except ImportError: pass
from tokenizers import Tokenizer
E = os.environ.get
# а3
NAME  = E('NAME',  'run')                                        # а4
TITLE = E('TITLE', 'Обучение')                                        # а5
DATA  = E('DATA',  './dataset.jsonl')  # а6
M     = E('MODEL', './model')   # а7
D     = E('WORK',  './work')                                 # а8
BIN   = E('TIER_BIN', './tier')                                   # а9
CKPT  = E('CKPT',  f'{D}/{NAME}_tier1.bin')                       # а10
TIER  = E('TIER',  f'{D}/tier_assignment_{NAME}_act.txt')         # а11
PROMPT= f'/tmp/{NAME}_prompt.bin'                                # а12
LOG   = E('LOG',   f'{D}/{NAME}_infer.log')  # а13
# а14
a = sys.argv[1:]; opt = lambda k, d: a[a.index(k) + 1] if k in a else d
flags = {'--base', '--time', '--ab'}; skip = {a.index(k) + 1 for k in ('--temp', '--max', '--system', '--sweep', '--ctx', '--top-k', '--top-p', '--min-p', '--rep', '--seed') if k in a}
qs = [x for i, x in enumerate(a) if x not in flags and not x.startswith('--') and i not in skip]
tok = Tokenizer.from_file(M + '/tokenizer.json'); sp = tok.token_to_id; enc = lambda s: tok.encode(s, add_special_tokens=False).ids
C = sys.stdout.isatty(); col = lambda c, s: f'\033[{c}m{s}\033[0m' if C else s
DIM, CY, LI, RED = '38;5;244', '1;96', '1;95', '1;91'
def ctx_auto(d=1024):
    # б1
    try:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import hwprobe, arch
        kv, free = hwprobe.arch_dims(M).get('kv_per_pos'), hwprobe.vram_free_mib()
        if not kv or not free: return d
        n = (free << 20) // 4 // kv
        mp = int(arch.text_cfg(arch.load_json(os.path.join(M, 'config.json'), {})).get('max_position_embeddings') or 0)
        if mp: n = min(n, mp)
        return max(256, n // 256 * 256)
    except Exception: return d
temp = opt('--temp', '0.5'); maxn = int(opt('--max', '200')); ctx = int(opt('--ctx', E('CTX', '0')) or 0) or ctx_auto()
# б2
SM = {'top_k': int(opt('--top-k', '0')), 'top_p': float(opt('--top-p', '1')), 'min_p': float(opt('--min-p', '0')),
      'повтор': float(opt('--rep', '1')), 'окно': 256, 'зерно': int(opt('--seed', '-1')), 'присутствие': 0.0, 'частота': 0.0}
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tmpl import Fmt
F = Fmt(M, tok)   # б3
scene = opt('--system', ''); base = '--base' in a
if not base and 'CKPT' not in os.environ and '--serve' not in a and not os.path.exists(f'{D}/{NAME}_step.txt'):
    # б4
    sys.exit(f'У запуска «{NAME}» нет снимка весов: обучение не дошло до первого сохранения.')
QB = E('QBASE', '')                                                # б5
p = subprocess.Popen([BIN, '--chat', M + '/model.safetensors', '-' if base else CKPT, TIER, str(ctx)] + (['--qbase', QB] if QB else []),
                     stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, cwd=os.path.dirname(os.path.abspath(__file__)))
SERVE = '--serve' in a
def jout(o):   # в1
    import json as _j
    sys.stdout.write(_j.dumps(o, ensure_ascii=False) + '\n'); sys.stdout.flush()
log = open(LOG, 'a'); log.write(f'\n===== {time.strftime("%F %T")} Запуск: {"Исходная модель" if base else TITLE}, контекст {ctx}, температура {temp}{", база " + os.path.basename(QB) if QB else ""}\n'); log.flush()
ready = threading.Event(); last = {'t': ''}
def err_reader():  # г1
    buf = ''
    for ch in iter(lambda: p.stderr.read(1), ''):
        if ch not in '\r\n': buf += ch; continue
        if buf:
            log.write(time.strftime('%T ') + buf + '\n'); log.flush()
            if not ready.is_set():
                if SERVE: jout({'load': buf[:200]})
                else: print('\r\033[K' + col(DIM, buf[:110]), end='', flush=True)
            if buf.lower().startswith('запрос '): last['t'] = buf
            if 'HIP err' in buf or 'hipBLAS err' in buf or 'CUDA err' in buf:
                if SERVE: jout({'ошибка': 'Ошибка движка: ' + buf[:300]})
                else: print('\n' + col(RED, 'Ошибка движка: ' + buf + f'  (подробности: {LOG})'), flush=True)
        buf = ''
threading.Thread(target=err_reader, daemon=True).start()
t0 = time.time()
for line in p.stdout:
    if line.strip() == 'ready': break
else:
    time.sleep(0.4)   # г2
    try:   # г3
        txt = open(LOG, encoding='utf-8', errors='replace').read(); txt = txt[txt.rfind('=====') + 1:] if '=====' in txt else txt
        ls = [re.sub(r'^\d\d:\d\d:\d\d ', '', x.strip()) for x in txt.replace('\r', '\n').split('\n')]
        why = [x for x in ls if x and not x.startswith(('Загрузка', '===='))][-1][:500]
    except Exception: why = ''
    try: rc = p.wait(timeout=5)
    except subprocess.TimeoutExpired: rc = None
    if rc is not None and rc < 0: why = f'Процесс движка остановлен сигналом {-rc}' + (' (нехватка памяти: процесс снят системой).' if rc == -9 else '.')
    if SERVE: jout({'ошибка': f'Движок завершился при загрузке. {why} Подробности: {LOG}'.replace('  ', ' ')})
    else: print(col(RED, f'\nДвижок завершился при загрузке. {why}\nПодробности: {LOG}'))
    sys.exit(1)
ready.set()
if not SERVE: print('\r\033[K' + col(DIM, f'Загружено за {time.time() - t0:.0f} с.'), flush=True)
EMIT = None   # г4
def gen(ids, label, lc):
    open(PROMPT, 'wb').write(struct.pack('<%di' % len(ids), *ids)); last['t'] = ''
    p.stdin.write(f"{PROMPT} {len(ids)} {maxn} {temp} {SM['top_k']} {SM['top_p']:g} {SM['min_p']:g} {SM['повтор']:g} {SM['окно']} {SM['зерно']} {SM['присутствие']:g} {SM['частота']:g}\n"); p.stdin.flush()
    if not EMIT: print(col(lc, label + ' › '), end='', flush=True)
    out, shown = [], ''
    def emit(piece):
        if not piece: return
        if EMIT: EMIT(piece)
        else: print(piece, end='', flush=True)
    for line in p.stdout:
        if line.startswith('gen_end'): break
        t = int(line.split()[1])
        if t in F.stop: continue
        out.append(t); s = tok.decode(out)
        if s.endswith('\ufffd'): continue        # д1
        if not s.startswith(shown): shown = shown[:len(os.path.commonprefix([s, shown]))]
        emit(s[len(shown):]); shown = s
    s = tok.decode(out) if out else ''
    if len(s) > len(shown): emit(s[len(shown):])
    for _ in range(20):  # д2
        if last['t']: break
        time.sleep(0.02)
    if not EMIT: print('\n' + col(DIM, '   ' + last['t'].split('|')[-1].strip()) if last['t'] else '')
    return out
def prompt(hist, q):  # е1
    h = list(hist)            # е2
    ids = F.prompt(scene, h + [(q, None)])
    while h and len(ids) + maxn > ctx: h = h[1:]; ids = F.prompt(scene, h + [(q, None)])
    return ids, h
def mode(tr):
    p.stdin.write('mode ' + ('trained' if tr else 'base') + '\n'); p.stdin.flush()
    while not p.stdout.readline().startswith('mode_ok'): pass
name = lambda: 'Исходная модель' if base else TITLE
if SERVE:
    # ж1
    hist = []
    def set_scale(k):
        p.stdin.write('scale %g\n' % k); p.stdin.flush()
        while True:
            r = p.stdout.readline()
            if not r or r.startswith('scale_ok'): break
    jout({'ready': True, 'имя': name()})
    for line in sys.stdin:
        line = line.strip()
        if not line: continue
        try: req = __import__('json').loads(line)
        except Exception: continue
        if req.get('new'): hist = []
        if req.get('base') is not None and bool(req['base']) != base:
            base = bool(req['base']); mode(not base); hist = []
        try:
            if req.get('temp') is not None: temp = '%g' % max(0.0, min(2.0, float(req['temp'])))
            if req.get('max') is not None: maxn = max(1, min(2048, int(req['max'])))
            if req.get('сила') is not None: set_scale(max(0.0, min(3.0, float(req['сила'])))); hist = []
            for k, lo, hi, ty in (('top_k', 0, 1000, int), ('top_p', 0.01, 1, float), ('min_p', 0, 1, float), ('повтор', 1, 2, float),
                                  ('окно', 1, 4096, int), ('зерно', -1, 2**31 - 1, int), ('присутствие', 0, 2, float), ('частота', 0, 2, float)):
                if req.get(k) is not None: SM[k] = ty(max(lo, min(hi, ty(req[k]))))
        except (TypeError, ValueError): pass
        if req.get('сцена') is not None and str(req['сцена']) != scene: scene = str(req['сцена']); hist = []
        if req.get('думать') is not None and bool(req['думать']) != F.think: F.think = bool(req['думать']); hist = []
        if isinstance(req.get('история'), list):    # ж2
            hist = [(str(x[0]), str(x[1])) for x in req['история'] if isinstance(x, (list, tuple)) and len(x) == 2]
        q = (req.get('q') or '').strip()
        if not q:
            jout({'end': True, 'имя': name()}); continue
        EMIT = lambda piece: jout({'tok': piece})
        t1 = time.time(); ids, hist = prompt(hist, q); out = gen(ids, name(), LI)
        EMIT = None
        hist += [(q, tok.decode(out))]
        jout({'end': True, 'имя': name(), 'токенов': len(out), 'сек': round(time.time() - t1, 2),
              'скорость': last['t'].split('|')[-1].strip() if last['t'] else ''})
    p.stdin.close(); p.terminate(); sys.exit(0)
if '--sweep' in a:
    for f in opt('--sweep', '').split(','):
        p.stdin.write('cpu ' + f + '\n'); p.stdin.flush()
        while not p.stdout.readline().startswith('cpu_ok'): pass
        gen(prompt([], qs[0] if qs else 'Привет')[0], f'Доля {f}', LI)
elif qs:
    for q in qs:
        print(col(CY, 'Вопрос › ') + q)
        if '--ab' in a and not base: gen(prompt([], q)[0], TITLE, LI); mode(False); gen(prompt([], q)[0], 'Исходная модель', DIM); mode(True)
        else: gen(prompt([], q)[0], name(), LI)
        print()
else:
    print(col(DIM, f'{name()} · температура {temp} · контекст {ctx} · Команды: /новый /исходная /обученная /инструкция /сила /temp /выход'))
    hist = []
    while True:
        try: q = input(col(CY, 'Вопрос › ') if C else 'Вопрос › ').strip()
        except (EOFError, KeyboardInterrupt): print(); break
        if not q: continue
        if q in ('/выход', '/exit', '/q'): break
        if q == '/новый': hist = []; print(col(DIM, '   Разговор сброшен.')); continue
        if q.startswith('/сила'):
            try: k = float(q.split()[1])
            except Exception: print(col(DIM, '   Формат: /сила 1.0. Значение 1.0 соответствует результату обучения, 1.5 усиливает его, 0 соответствует исходной модели.')); continue
            p.stdin.write('scale %g\n' % k); p.stdin.flush()
            while True:
                a = p.stdout.readline()
                if not a or a.startswith('scale_ok'): break
            hist = []; print(col(DIM, f'   Сила обучения: {k}. Разговор сброшен.')); continue
        if q in ('/исходная', '/обученная'):
            base = q == '/исходная'; mode(not base); hist = []; print(col(DIM, f'   Выбрана модель: {name()}. Разговор сброшен.')); continue
        if q.startswith('/temp'): temp = q.split()[1] if len(q.split()) > 1 else temp; print(col(DIM, f'   Температура: {temp}.')); continue
        if q.startswith(('/инструкция', '/сцена')):
            scene = q.split(None, 1)[1].strip() if len(q.split(None, 1)) > 1 else ''; hist = []; print(col(DIM, f'   Системная инструкция: {scene or "нет"}. Разговор сброшен.')); continue
        ids, hist = prompt(hist, q); out = gen(ids, name(), LI)
        hist += [(q, tok.decode(out))]
p.stdin.close(); p.terminate(); log.write(f'{time.strftime("%T")} Выход\n'); log.close()
