#!/usr/bin/env python3
# а1
import glob, hashlib, http.server, json, math, os, queue, re, shutil, signal, socketserver, struct, subprocess, sys, threading, time, urllib.parse

VERSION = '2.4'
APP = os.path.dirname(os.path.abspath(__file__))
DATA = os.environ.get('TIER_DATA') or os.path.join(
    os.environ.get('XDG_DATA_HOME') or os.path.join(os.path.expanduser('~'), '.local', 'share'), 'tierlm')
os.makedirs(DATA, exist_ok=True)
SETTINGS = os.path.join(DATA, 'settings.json')
PIDFILE = os.path.join(DATA, 'train.pid')
SETUP_LOG = os.path.join(DATA, 'setup.log')
DEFAULTS = {'MODEL': os.path.join(DATA, 'model'), 'WORK': os.path.join(DATA, 'work'),
            'HF_REPO': '', 'THEME': '', 'MODE': 'авто', 'APPEAR': '', 'BG': '', 'LANG': 'ru', 'UI': '',
            'TAB': 'tier', 'LORA': '', 'QBASE': '', 'HF_TOKEN': '', 'INFER': '', 'MODELS': ''}
NOTIFY = None        # а2
UI_PID = os.path.join(DATA, 'ui.pid')


def S():
    d = dict(DEFAULTS)
    try:
        d.update({k: v for k, v in json.load(open(SETTINGS, encoding='utf-8')).items() if k in DEFAULTS})
    except (OSError, ValueError):
        pass
    for k in ('MODEL', 'WORK'):
        d[k] = os.path.expanduser(d[k] or DEFAULTS[k])
    d['MODE'] = mode_key(d['MODE'])                  # б1
    return d


def ui_pref(k, d=None):
    try:
        return json.loads(S()['UI'] or '{}').get(k, d)
    except ValueError:
        return d


def S_save(changes):
    cur = S()
    cur.update({k: str(v).strip() for k, v in changes.items() if k in DEFAULTS})
    tmp = SETTINGS + '.part'
    with open(os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600), 'w', encoding='utf-8') as f:   # г1
        json.dump(cur, f, ensure_ascii=False, indent=1)
    os.chmod(tmp, 0o600)
    os.replace(tmp, SETTINGS)


def engine():
    return os.path.join(DATA, 'tier')


def engine_env():
    # е1
    out = {}
    for line in read_text(engine() + '.env').splitlines():
        k, _, v = line.partition('=')
        if k.strip() and k.strip() not in os.environ:
            out[k.strip()] = v.strip()
    return out


ENGINE_SRC = ('tier.cpp', 'gpu_compat.h', 'tier_wmma.h', 'quant.h', 'quant_tables.h')
_HW_NOW = None   # е2


def engine_fresh():
    # ж1
    if not os.path.exists(engine()):
        return False
    try:
        import hashlib
        src = b''.join(open(os.path.join(APP, f), 'rb').read() for f in ENGINE_SRC)
        if read_text(engine() + '.src').strip() != hashlib.sha1(src).hexdigest():
            return False
    except OSError:
        return False
    # ж2
    hw = read_text(engine() + '.hw').strip()
    if not hw:
        return True     # ж3
    global _HW_NOW
    if _HW_NOW is None:
        try:
            import hwprobe
            _HW_NOW = hwprobe.fingerprint()
        except Exception:
            _HW_NOW = ''
    return not _HW_NOW or _HW_NOW == hw


def child_env(name=None):
    # з1
    s = S()
    os.makedirs(s['WORK'], exist_ok=True)
    e = dict(os.environ)
    e.update(engine_env())
    e.update(TIER_APP=APP, TIER_DATA=DATA, WORK=s['WORK'], D=s['WORK'], MODEL=s['MODEL'],
             M=os.path.join(s['MODEL'], 'model.safetensors'), B=engine(), TIER_BIN=engine(),
             TIER_OUT=engine(), HF_REPO=s['HF_REPO'], TIER_PY=sys.executable,
             TIER_NOTIFY='1' if ui_pref('notify', 1) else '0', TIER_LANG=s['LANG'] or 'ru', TIER_UI_PID=UI_PID)
    if s.get('HF_TOKEN') and not os.environ.get('HF_TOKEN'):
        e['HF_TOKEN'] = s['HF_TOKEN']
    ap = model_info().get('описание')
    if ap:
        e["TIER_MODEL_DESC"] = ap
    if name:
        e['NAME'] = name
    return e


# и1
def train_state():
    try:
        pid, name = open(PIDFILE, encoding='utf-8').read().split('\n')[:2]
        os.killpg(int(pid), 0)
        return int(pid), name
    except (OSError, ValueError):
        return None, None


def run_file(name, suffix):
    return os.path.join(S()['WORK'], f'{name}_{suffix}')


def name_ok(name):
    if not name or len(name) > 40 or not re.fullmatch(r'[\w.-]+', name) or name.startswith('.'):
        return False, 'Имя может содержать буквы, цифры, точку, дефис и подчёркивание, не более 40 знаков.'
    return True, ''


def run_choice(name):
    # м1
    try:
        d = json.load(open(run_file(name, 'run.json'), encoding='utf-8'))
    except (OSError, ValueError):
        return None
    return d


def train_start(name, data='', extra=None):
    # н1
    if train_state()[0]:
        return False, 'Обучение уже выполняется.'
    if not os.path.exists(engine()):
        return False, 'Движок не собран.'
    if not engine_fresh():
        return False, 'Движок собран из прежней версии, требуется пересборка.'
    chat_stop()                                      # н2
    env = child_env(name)
    env.update(extra or {})
    out = open(run_file(name, 'console.txt'), 'ab')
    out.write(f'\n=== Запуск {time.strftime("%F %T")} ===\n'.encode())
    out.flush()
    p = subprocess.Popen(['bash', os.path.join(APP, 'auto.sh')] + ([data] if data else []), cwd=DATA,
                         env=env, stdout=out, stderr=out, start_new_session=True)
    open(PIDFILE, 'w', encoding='utf-8').write(f'{p.pid}\n{name}\n')
    threading.Thread(target=p.wait, daemon=True).start()      # н3
    return True, ''


def resume_env(name):
    # о1
    txt = ''
    try:
        for line in open(log_path(name), encoding='utf-8', errors='replace'):
            r = line.rstrip('\n').split('\t')
            if len(r) > 2 and r[1] == 'Запуск' and 'шагов на эпоху' in r[-1]:
                txt = r[-1]
    except OSError:
        return None
    m = re.search(r'LR ([^,\s]+), до (\d+) токенов в шаге, \d+ шагов на эпоху × (\d+) = \d+ шагов, '
                  r'валидация каждые (\d+) шагов, снимок весов каждые (\d+) шагов', txt)
    if not m:
        return None
    env = {'LR': m.group(1), 'PACK': m.group(2), 'EPOCHS': m.group(3), 'EV': m.group(4), 'FL': m.group(5)}
    ch = run_choice(name) or {}
    env.update({k: v for k, v in (ch.get('env') or {}).items() if k in ('STOP_AFTER', 'X')})
    env['REPLAY'] = read_text(run_file(name, 'replay.txt')).strip() or '0'   # о2
    for line in read_text(run_file(name, 'mode.txt')).splitlines():          # о3
        k, _, v = line.partition('=')
        if k in ('MATS', 'NSGD', 'POSDW', 'WHITEN', 'WHITEN_FRAC', 'LORA'):
            env[k] = v.strip()
            if k == 'MATS':
                env['MATS_AUTO'] = '0'      # о4
    return env


def train_stop():
    pid, name = train_state()
    if not pid:
        return False
    try:
        os.killpg(pid, signal.SIGTERM)
    except OSError:
        pass
    try:            # п1
        open(run_file(name, 'console.txt'), 'a', encoding='utf-8').write('\n=== Остановлено вручную ===\n')
    except OSError:
        pass
    for i in range(40):                 # п2
        if not train_state()[0]:
            return True
        if i == 25:                     # п3
            try:
                os.killpg(pid, signal.SIGKILL)
            except OSError:
                pass
        time.sleep(0.1)
    return not train_state()[0]


# р1
def log_path(name):
    return os.path.join(S()['WORK'], f'{name}_log.tsv')


def runs():
    try:
        return sorted(f[:-8] for f in os.listdir(S()['WORK']) if f.endswith('_log.tsv'))
    except OSError:
        return []


def num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def stamp(p):
    try:
        st = os.stat(p)
        return (st.st_mtime, st.st_size)
    except OSError:
        return None


def log_stamp(name):
    try:
        st = os.stat(log_path(name))
        return (st.st_mtime, st.st_size)
    except OSError:
        return None


def read_text(p):
    try:
        return open(p, encoding='utf-8', errors='replace').read()
    except OSError:
        return ''


ANSI = re.compile(r'\x1b\[[0-9;?]*[A-Za-z]')


def console_tail(name, n=80):
    # ц1
    try:
        with open(run_file(name, 'console.txt'), 'rb') as f:
            f.seek(0, 2)
            f.seek(max(0, f.tell() - 262144))
            raw = f.read().decode('utf-8', 'replace')
    except OSError:
        return []
    raw = re.sub(r'\x1b\[\?25l.*?(\x1b\[\?25h|(?=\n=== )|$)', '\n', raw, flags=re.S)
    lines = [ANSI.sub('', x).strip() for x in raw.replace('\r', '\n').split('\n')]
    lines = [x for x in lines if x]
    start = max((i for i, x in enumerate(lines) if x.startswith('=== Запуск ')), default=-1)
    return lines[start + 1:][-n:]


ERR = re.compile(r'упал|ошибк|отказ|не собран|нет весов|нет токенов|нет датасета|нет сводки|матрицы не найдены|Traceback|'
                 r'HIP err|hipBLAS err|CUDA err|out of memory|не хватило|недостаточно видеопамяти', re.I)


def run_status(live, stage, steps, msgs, out):
    # ч1
    if live:
        return 'идёт', ''
    if stage == 'Завершено':
        return 'завершено', next((m['текст'] for m in reversed(msgs) if m['текст'].startswith('Остановка:')), '')
    if out and out[-1].startswith('=== Остановлено вручную'):
        return 'остановлено', ''
    bad = next((x for x in reversed(out) if ERR.search(x)), '') or \
        next((m['текст'] for m in reversed(msgs) if ERR.search(m['текст'])), '')
    if bad:
        return 'ошибка', bad
    return ('остановлено', '') if steps or stage else ('нет', '')


_logc = {}
_after = lambda n: f'После {n} ' + ('валидации' if n % 10 == 1 and n % 100 != 11 else 'валидаций')      # ч2


def parse_log(name):
    # ш1
    st = log_stamp(name)
    c = _logc.get(name)
    if c and c[0] == st:
        return c[1]
    steps, evals, msgs, last, plan = [], [], [], {}, {}
    ts = lambda x: time.mktime(time.strptime(x, '%Y-%m-%d %H:%M:%S'))
    prev = None                                      # ш2
    try:
        f = open(log_path(name), encoding='utf-8', errors='replace')
    except OSError:
        f = None
    if f:
        with f:
            idx = {k: i for i, k in enumerate(f.readline().rstrip('\n').split('\t'))}
            g = lambda r, k: num(r[idx[k]]) if k in idx and idx[k] < len(r) else None
            t = lambda r, k: r[idx[k]] if k in idx and idx[k] < len(r) else ''
            for line in f:
                r = line.rstrip('\n').split('\t')
                if len(r) < 3:
                    continue
                ev = t(r, 'Событие')
                if ev == 'Шаг':
                    s0 = g(r, 'Шаг')
                    if steps and s0 is not None and steps[-1]['шаг'] is not None and s0 <= steps[-1]['шаг']:
                        # ш3
                        steps = [x for x in steps if x['шаг'] is not None and x['шаг'] < s0]
                        evals = [x for x in evals if x['шаг'] is None or x['шаг'] < s0]
                    steps.append({
                        'шаг': g(r, 'Шаг'), 'эпоха': g(r, 'Эпоха'), 'loss': g(r, 'Loss'), 'ср': g(r, 'Loss_ср50'),
                        'ppl': g(r, 'PPL'), 'acc': g(r, 'Acc_%'), 'acc_ср': g(r, 'Acc_ср50_%'),
                        'сек': g(r, 'Сек_шаг'), 'ток': g(r, 'Ток_в_с'), 'прямой': g(r, 'Прямой_с'),
                        'головка': g(r, 'Головка_с'), 'обратный': g(r, 'Обратный_Adam_с'),
                        'токенов': g(r, 'Токенов'), 'в_loss': g(r, 'Токенов_в_loss'), 'примеров': g(r, 'Примеров'),
                        'vram': g(r, 'VRAM_пик_МиБ'), 'vram_всего': g(r, 'VRAM_всего_МиБ'),
                        'живых': g(r, 'Живых_МиБ'), 'блоков': g(r, 'Блоков'), 'область': g(r, 'Область_МиБ'),
                        'мимо': g(r, 'Мимо_области'), 'озу': g(r, 'ОЗУ_кэш_ГиБ'), 'tier1': g(r, 'Обучаемая_часть_в_VRAM'),
                        'край': g(r, 'T_край_°C'), 'горячая': g(r, 'T_горячая_°C'), 'память': g(r, 'T_память_°C'),
                        'вент': g(r, 'Вент_об_мин'), 'гн': g(r, '|g|'), 'гн_макс': g(r, '|g|_макс'),
                        'lr': g(r, 'LR')})
                    last = {'шаг': g(r, 'Шаг'), 'эпоха': g(r, 'Эпоха'), 'lr': t(r, 'LR'),
                            'прошло': t(r, 'Прошло'), 'осталось': t(r, 'Осталось'), 'время': r[0]}
                    prev = (g(r, 'Шаг'), r[0])
                elif ev == 'Оценка':
                    dur, pk = None, re.match(r'(\d+) упаков', t(r, 'Примеров'))
                    if prev and prev[0] == g(r, 'Шаг'):   # ш4
                        try:
                            dur = ts(r[0]) - ts(prev[1])
                        except ValueError:
                            dur = None
                    evals.append({'шаг': g(r, 'Шаг'), 'loss': g(r, 'Loss_отложенные'),
                                  'лучший': g(r, 'Лучший_отложенный'), 'acc': g(r, 'Acc_%'),
                                  'с': dur if dur is not None and 0 < dur < 3600 else None, 'упаковок': int(pk.group(1)) if pk else None})
                elif ev in ('Сообщение', 'Запуск', 'Завершено', 'Лучший снимок'):
                    txt = t(r, 'Сообщение')
                    msgs.append({'время': r[0], 'событие': ev, 'текст': txt})
                    m = re.search(r'(\d+) шагов на эпоху × (\d+) = (\d+) шагов', txt) if ev == 'Запуск' else None
                    if m:       # ш5
                        plan = {'на_эпоху': int(m.group(1)), 'эпох': int(m.group(2)), 'всего': int(m.group(3))}
                        for k, rx in (('lr', r'LR ([^,\s]+)'), ('упаковка', r'до (\d+) токенов'),
                                      ('оценка', r'валидация каждые (\d+)'), ('сброс', r'снимок весов каждые (\d+)')):
                            mm = re.search(rx, txt)
                            if mm:
                                plan[k] = mm.group(1)
    r = (steps, evals, msgs, last, plan)
    _logc[name] = (st, r)
    return r


def read_log(name):
    steps, evals, msgs, last, plan = parse_log(name)
    stage = read_text(run_file(name, 'stage.txt')).strip()
    sweep = []
    for line in read_text(run_file(name, 'lrfind.txt')).splitlines():
        p = line.split()
        if len(p) >= 4 and p[0] == 'lrfind':
            try:
                sweep.append([int(p[1]), float(p[2]), float(p[3])])
            except ValueError:
                pass
    pid, running = train_state()
    live = bool(pid) and running == name
    out = console_tail(name)
    status, why = run_status(live, stage, steps, msgs, out)
    cont = read_text(run_file(name, 'step.txt')).strip()
    return {'имя': name, 'шаги': steps, 'оценки': evals, 'сообщения': msgs[-40:], 'последний': last, 'план': plan,
            'идёт': live, 'идёт_имя': running or '', 'запуски': runs(), 'этап': stage, 'ход_этапа': read_text(run_file(name, 'stagep.txt')).strip() if live else '', 'развёртка': sweep,
            'загрузка': read_text(run_file(name, 'load.txt')).strip() if live else '',
            'выбор': run_choice(name), 'состояние': status, 'почему': why, 'вывод': out[-60:],
            'продолжение': int(cont) if cont.isdigit() else 0, 'сводка': stats(name),
            'повтор': read_text(run_file(name, 'replay.txt')).strip(),
            'способ': dict(l.partition('=')[::2] for l in read_text(run_file(name, 'mode.txt')).splitlines() if '=' in l)}


# э1
_gk = [0, '']


def gpu_kind():
    # ю1
    if os.environ.get('TIER_VENDOR') in ('amd', 'nvidia'):
        return os.environ['TIER_VENDOR']
    if time.time() - _gk[0] < 60:
        return _gk[1]
    amd = nv = 0
    for d in glob.glob('/sys/class/drm/card[0-9]*/device'):
        v = read_text(d + '/vendor').strip()
        if v == '0x1002':
            amd = max(amd, int(num(read_text(d + '/mem_info_vram_total').strip()) or 1))
        elif v == '0x10de':
            nv = 1
    if nv and shutil.which('nvidia-smi'):
        try:
            r = subprocess.run(['nvidia-smi', '--query-gpu=memory.total', '--format=csv,noheader,nounits'],
                               capture_output=True, text=True, timeout=4)
            nv = max([int(x) << 20 for x in r.stdout.split() if x.isdigit()] or [1])
        except (OSError, subprocess.SubprocessError):
            pass
    k = ('nvidia' if nv > amd else 'amd') if amd or nv else \
        'amd' if shutil.which('rocminfo') or shutil.which('rocm-smi') or os.path.isdir('/sys/module/amdgpu') else \
        'nvidia' if shutil.which('nvidia-smi') or os.path.isdir('/proc/driver/nvidia') else ''
    _gk[:] = [time.time(), k]
    return k


_mods = {}


def has_module(name):
    # я1
    t, ok = _mods.get(name, (0, False))
    if ok or time.time() - t < 3:
        return ok
    ok = subprocess.run([sys.executable, '-c', f'import {name}'], stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL).returncode == 0
    _mods[name] = (time.time(), ok)
    return ok


def health():
    s = S()
    comp = 'hipcc' if shutil.which('hipcc') else ('nvcc' if shutil.which('nvcc') else '')
    tok = has_module('tokenizers') and (has_module('jinja2') or tmpl_mod().manual(model_info().get('семья', '')))   # аа1
    mi, gi = model_info(), gpu_info()
    return {'версия': VERSION, 'карта': gpu_kind(), 'компилятор': comp,
            'движок': engine_fresh(),
            'движок_есть': os.path.exists(engine()),
            'веса': os.path.exists(os.path.join(s['MODEL'], 'model.safetensors')),
            'словарь': os.path.exists(os.path.join(s['MODEL'], 'tokenizer.json')),
            'tokenizers': tok, 'numpy': has_module('numpy'), 'настройка': setup_alive(),
            'настройка_что': setup_what if setup_alive() else '',
            'модель': mi, 'гпу': gi, 'озу': meminfo(), 'резерв_озу': ram_reserve(), 'видеопамять': vram_info(),
            'запуски': runs_summary(), 'режимы': [{'к': k, 'имя': t, 'суть': d, 'знач': v} for k, t, d, v in MODES],
            'цели': [{'к': k, 'имя': v[2], 'суть': GOAL_SHORT[k], 'слои': goal_layers(k), 'вид': v[0]} for k, v in GOALS.items()],
            'лора_матрицы': [list(x) for x in LORA_MATS], 'лора_исходно': LORA_DEF, 'геометрия': list(lora_geom()),
            'железо': conf_read(), 'идёт': train_state()[1] or '', 'диск': disk_info(),
            'разговор': {'имя': chat_name if chat_alive() else None, 'база': os.path.basename(chat_base) if chat_alive() else '', **chat_info},
            'базы': base_list(), 'база': S()['QBASE'], 'база_папка': QDIR, 'база_bf16': layer_bf16_bytes(),
            'рецепты': [{'к': r, 'байт': recipe_bytes(r), 'суть': RECIPE_TIP[r]} for r in RECIPES]}


def disk_info():
    try:
        u = shutil.disk_usage(S()['WORK'] if os.path.isdir(S()['WORK']) else DATA)
        return {'всего': u.total, 'свободно': u.free}
    except OSError:
        return {}


def ram_reserve():
    # ав1
    mem = meminfo()
    tot, av = mem.get('всего') or 0, mem.get('доступно') or 0
    if chat_alive():        # ав2
        todo, seen = [chat_proc.pid], set()
        while todo:
            pid = todo.pop()
            if pid in seen:
                continue
            seen.add(pid)
            m = re.search(r'VmRSS:\s*(\d+)', read_text(f'/proc/{pid}/status'))
            av += int(m.group(1)) * 1024 if m else 0
            for c in glob.glob(f'/proc/{pid}/task/*/children'):
                todo += [int(x) for x in read_text(c).split()]
        av = min(tot, av)
    h = hwp()
    try:
        if h and hasattr(h, 'ram_reserve') and tot and av:
            return int(h.ram_reserve(tot, av, int(num(conf_read().get('mem_cap_pct')) or 98)))
    except Exception:
        pass
    return int(min(8 * 2**30, max(5 * 2**30, tot / 4)))


# ав3
_minfo = {}
QDT = {'I8', 'U8', 'I4', 'U4', 'F8_E4M3', 'F8_E5M2', 'F8_E4M3FN', 'F4', 'F6_E2M3', 'F6_E3M2', 'E8M0'}


def amod():
    if APP not in sys.path:
        sys.path.insert(0, APP)
    import arch
    return arch


def tmpl_mod():
    if APP not in sys.path:
        sys.path.insert(0, APP)
    import tmpl
    return tmpl


FAMILY = {'gemma4': 'Gemma 4', 'gemma3': 'Gemma 3', 'gemma2': 'Gemma 2', 'gemma': 'Gemma', 'qwen3': 'Qwen 3', 'qwen2': 'Qwen 2',
          'llama': 'Llama', 'mistral': 'Mistral', 'ministral': 'Ministral', 'phi3': 'Phi 3', 'olmo2': 'OLMo 2', 'granite': 'Granite'}


def fam_name(fam, d=''):
    f = re.sub(r'_(text|causal_lm)$', '', fam or '')
    return FAMILY.get(f) or f.replace('_', ' ').title() or os.path.basename((d or '').rstrip('/')) or '?'


def pretty_size(n):
    b = n / 1e9
    return f'{b:.0f}B' if b >= 9.5 else f'{b:.1f}B'.replace('.0B', 'B') if b >= 0.95 else f'{n / 1e6:.0f}M'


def model_files(mdir):
    one = os.path.join(mdir, 'model.safetensors')
    if os.path.exists(one):
        return [one]
    try:
        idx = json.load(open(os.path.join(mdir, 'model.safetensors.index.json'), encoding='utf-8'))
        fs = [os.path.join(mdir, f) for f in sorted(set(idx['weight_map'].values()))]
        return fs if all(os.path.exists(f) for f in fs) else []
    except (OSError, ValueError, KeyError, AttributeError):
        return []


def model_info():
    # аи1
    s = S()
    files = model_files(s['MODEL'])
    if not files:
        _arch.clear(); _minfo.clear()
        return {'есть': False, 'папка': s['MODEL']}
    try:
        key = tuple((f, os.stat(f).st_mtime, os.stat(f).st_size) for f in files) + tuple(
            os.stat(os.path.join(s['MODEL'], n)).st_mtime for n in ('config.json',) if os.path.exists(os.path.join(s['MODEL'], n)))
    except OSError:
        return {'есть': False, 'папка': s['MODEL']}
    if _minfo.get('key') == key:
        return _minfo['v']
    out = {'есть': True, 'папка': s['MODEL'], 'байт': sum(k[2] for k in key if isinstance(k, tuple)), 'файлов': len(files)}
    ar = amod()
    try:
        T, _ = ar.tensors(s['MODEL'])
        A, why = ar.describe(s['MODEL'])
    except (OSError, ValueError, KeyError, IndexError, TypeError, struct.error, SystemExit) as e:
        out.update(вид='?', движок=False, почему=f'Не удалось прочитать заголовок файла ({e}).')
        _minfo.update(key=key, v=out)
        return out
    lp = (A or {}).get('layer_prefix') or ''
    pre = (A or {}).get('prefix') or ''
    params = total = lay = experts = 0
    dts, groups, quant = {}, {'mlp': 0, 'внимание': 0, 'прочее': 0}, False      # аи2
    for k, (dt, shp) in T.items():
        c = math.prod(shp) if shp else 1
        total += c
        if pre and not k.startswith(pre) and k != (A or {}).get('head'):
            continue                                       # аи3
        params += c
        dts[dt] = dts.get(dt, 0) + c
        if lp and k.startswith(lp):
            lay += c
            groups['mlp' if '.mlp.' in k or '.experts.' in k else 'внимание' if '.self_attn.' in k else 'прочее'] += c
        kl = k.lower()
        if re.search(r'\.experts?\.|block_sparse_moe|\.moe\.|shared_expert', kl):
            experts += 1
        if dt in QDT or kl.endswith(('.qweight', '.qzeros', '.scales', '.weight_scale', '.absmax', '.quant_map')):
            quant = True
    try:
        cfg = json.load(open(os.path.join(s['MODEL'], 'config.json'), encoding='utf-8'))
    except (OSError, ValueError):
        cfg = {}
    tc = cfg.get('text_config') or cfg
    kind = ('Quantized MoE' if experts else 'Quantized') if quant else ('MoE' if experts else 'Dense')
    main = max(dts, key=dts.get) if dts else '?'
    merge = bool(A) and bool(why) and all('merge' in w for w in why) and not quant and not experts
    if kind == 'MoE':
        why_t = 'Модель использует смесь экспертов (MoE), а движок пока поддерживает только плотные модели.'
    elif kind.startswith('Quantized'):
        why_t = 'Веса в файле уже квантованы, а движок обучает поверх BF16, поэтому требуется исходная модель.'
    elif merge:
        why_t = (f'Веса в {len(files)} файлах' if len(files) > 1 else '') + (' и ' if len(files) > 1 and main != 'BF16' else '') + \
                (f'в {main}' if main != 'BF16' and len(files) > 1 else f'Веса в {main}' if main != 'BF16' else '') + ': требуется объединение частей в один файл BF16.'
    else:
        why_t = ' '.join(why)
    fam = (A or {}).get('family') or str(tc.get('model_type') or '')
    label = fam_name(fam, s['MODEL']) + ' ' + pretty_size(params)
    out.update(параметров=params, эмбеддинги=params - lay, всего_в_файле=total, типы=dts, основной=main,
               слоёв=(A or {}).get('layers') or 0, экспертов=experts, вид=kind, группы=groups, семья=fam,
               архитектура=(cfg.get('architectures') or [''])[0], модель=label, склеить=merge,
               скрытый=(A or {}).get('hidden') or tc.get('hidden_size'), словарь=(A or {}).get('vocab') or tc.get('vocab_size'),
               слой_полный=[s_['full'] for s_ in (A or {}).get('L', [])], сводка=ar.summary(A, why) if A else '',
               движок=not why_t, почему=why_t)
    if not why_t:
        try:       # аи4
            d = os.path.join(DATA, 'arch'); os.makedirs(d, exist_ok=True)
            ap = os.path.join(d, hashlib.sha1(repr((key, VERSION)).encode()).hexdigest()[:16] + '.txt')
            if not os.path.exists(ap):
                ar.write(A, why, ap + '.part'); os.replace(ap + '.part', ap)
            out['описание'] = ap
        except OSError as e:
            out.update(движок=False, почему=f'Не удалось записать описание модели ({e}).')
    _minfo.update(key=key, v=out)
    _arch.update(A=A if not why_t else None, any=A)
    return out


_arch = {}


def arch_now():
    # ак1
    model_info()
    return _arch.get('A')


# ак2
MODELS_DIR = os.path.join(DATA, 'models')


def hmod():
    if APP not in sys.path:
        sys.path.insert(0, APP)
    import hub
    return hub


def hwp():
    # ам1
    if APP not in sys.path:
        sys.path.insert(0, APP)
    try:
        import hwprobe
        return hwprobe
    except Exception:
        return None


def known_dirs():
    # ан1
    s = S()
    try:
        L = [x for x in json.loads(s['MODELS'] or '[]') if isinstance(x, str)]
    except ValueError:
        L = []
    if os.path.isdir(MODELS_DIR):
        L += [os.path.join(MODELS_DIR, d) for d in sorted(os.listdir(MODELS_DIR))]
    out = []
    for d in [s['MODEL']] + L:
        d = os.path.realpath(os.path.expanduser(d))
        if d not in out and (os.path.isdir(d) or d == os.path.realpath(s['MODEL'])):
            out.append(d)
    return out


def remember_dir(d):
    s = S()
    try:
        L = [x for x in json.loads(s['MODELS'] or '[]') if isinstance(x, str)]
    except ValueError:
        L = []
    d = os.path.realpath(os.path.expanduser(d))
    if d not in L:
        S_save({'MODELS': json.dumps((L + [d])[-30:], ensure_ascii=False)})


_local = {}


def local_models():
    # ап1
    cur, ar, out = os.path.realpath(S()['MODEL']), amod(), []
    for d in known_dirs():
        fs = model_files(d)
        try:
            key = tuple((f, os.stat(f).st_mtime, os.stat(f).st_size) for f in fs)
        except OSError:
            key = ()
        if _local.get(d, (None,))[0] != key:
            e = {'папка': d, 'байт': sum(k[2] for k in key), 'есть': bool(fs)}
            if fs:
                try:
                    A, why = ar.describe(d)
                    P = A['params'] if A else 0
                    fam = (A or {}).get('family', '')
                    e.update(модель=fam_name(fam, d) + (' ' + pretty_size(P) if P else ''),
                             годна=bool(A) and not why, склеить=bool(A) and bool(why) and all('merge' in w for w in why))
                    e['почему'] = 'Требуется объединение частей в один файл BF16.' if e['склеить'] else ' '.join(why)
                except (OSError, ValueError, KeyError, IndexError, TypeError, struct.error, SystemExit) as x:
                    e.update(модель='?', годна=False, почему=f'Не удалось прочитать заголовок ({x}).')
            src = read_text(os.path.join(d, '.tier_repo')).strip()
            e['repo'] = src
            if not e.get('модель'):
                e['модель'] = src or os.path.basename(d)
            _local[d] = (key, e)
        out.append(dict(_local[d][1], текущая=d == cur))
    return out


def model_use(d):
    # ар1
    d = os.path.realpath(os.path.expanduser(d or ''))
    if not model_files(d):
        return False, f'В папке {d} нет файла model.safetensors.'
    if train_state()[0]:
        return False, 'Выполняется обучение: модель можно сменить после его завершения.'
    remember_dir(S()['MODEL'])
    remember_dir(d)
    chat_stop()
    S_save({'MODEL': d})
    _minfo.clear(); _arch.clear(); _lmats.clear(); _bases.clear()
    return True, ''


def hw_now():
    # ас1
    c, mem = conf_read(), meminfo()
    v = vram_info()
    vr = v.get('всего') or (int(num(c.get('vram_mib')) or 0) << 20)
    os.makedirs(MODELS_DIR, exist_ok=True)
    try:
        free = shutil.disk_usage(MODELS_DIR).free
    except OSError:
        free = None
    bus = gpu_info().get('шина') or {}      # ас2
    hw = dict(ram=mem.get('всего') or 0, vram=vr, disk_free=free, pcie_gen=bus.get('поколение'), pcie_width=bus.get('линий'),
              tier1=int(num(c.get('tier1_params')) or 0))
    a = step_anchor()
    if a:
        hw['anchor_step'] = a
    t = tok_anchor()
    if t:
        hw['anchor_tok'] = t
    return hw


SPEED = os.path.join(DATA, 'speed.json')


def speed_note(line):
    # ат1
    m = re.search(r'Ответ (\d+) токенов: [\d.]+ с \(([\d.]+) с/токен\)', line, re.I)
    if not m or int(m.group(1)) < 8 or chat_base:
        return
    mi = model_info()
    lay = 2 * ((mi.get('параметров') or 0) - (mi.get('эмбеддинги') or 0))
    if not lay:
        return
    try:
        d = json.load(open(SPEED, encoding='utf-8'))
    except (OSError, ValueError):
        d = {}
    k = mi.get('модель') or '?'
    old = (d.get('разговор') or {}).get(k)
    v = float(m.group(2)) if not old else 0.7 * old['с'] + 0.3 * float(m.group(2))
    d.setdefault('разговор', {})[k] = {'с': v, 'байт': lay, 'когда': time.time()}
    try:
        json.dump(d, open(SPEED + '.part', 'w', encoding='utf-8'), ensure_ascii=False)
        os.replace(SPEED + '.part', SPEED)
    except OSError:
        pass


def tok_anchor():
    try:
        d = json.load(open(SPEED, encoding='utf-8')).get('разговор') or {}
    except (OSError, ValueError, AttributeError):
        return None
    if not d:
        return None
    k, v = max(d.items(), key=lambda kv: kv[1].get('когда', 0))
    return (v['с'], v['байт'], k)


def step_anchor():
    # аф1
    best = None
    for n in runs():
        try:
            mt = os.stat(log_path(n)).st_mtime
        except OSError:
            continue
        if best and mt <= best[0]:
            continue
        mm = (run_choice(n) or {}).get('модель') or {}
        if not mm.get('параметров'):
            continue
        best = (mt, n, mm)
    if not best:
        return None
    _, n, mm = best
    vals = []
    try:
        rows = open(log_path(n), encoding='utf-8').read().splitlines()
        hd = rows[0].split('\t'); i = hd.index('Сек_шаг')
        for r in rows[1:]:
            c = r.split('\t')
            if len(c) > i and c[i]:
                try:
                    vals.append(float(c[i]))
                except ValueError:
                    pass
    except (OSError, ValueError, IndexError):
        return None
    if len(vals) < 5:
        return None
    vals = sorted(vals[-60:])
    lay = 2 * (mm['параметров'] - (mm.get('эмбеддинги') or 0))
    if not mm.get('эмбеддинги') and mm.get('папка') and os.path.realpath(mm['папка']) == os.path.realpath(S()['MODEL']):
        lay = 2 * (model_info().get('параметров', 0) - model_info().get('эмбеддинги', 0)) or lay
    return (vals[len(vals) // 2], lay, n)


def hub_token():
    return S()['HF_TOKEN'] or os.environ.get('HF_TOKEN')


def hub_mark(rows):
    # ац1
    h = hmod()
    loc = {e.get('repo', '').lower(): e for e in local_models() if e.get('repo')}
    for r in h.mark(rows):
        r['на_диске'] = bool(loc.get(r['repo'].lower(), {}).get('есть'))
    return rows


def hub_catalog(q='', sort='downloads', limit=30, fresh=False, offline=False):
    # ач1
    res = hmod().catalog(q, sort, limit, token=hub_token(), offline=offline, fresh=fresh)
    res['rows'] = hub_mark(res['rows'])
    return res


def hub_check(repos, cb=None, cancel=None):
    # аш1
    return hmod().check(repos, hub_token(), cb=cb, cancel=cancel)


def hub_info(q, fresh=False):
    # аэ1
    h = hmod()
    m = h.info(q, hub_token(), fresh)
    if m.get('ошибка'):
        return m
    d = hub_dir(m['repo'])
    hw = hw_now()
    hw['есть_байт'] = sum(os.path.getsize(os.path.join(d, f)) for f in (os.listdir(d) if os.path.isdir(d) else []) if os.path.isfile(os.path.join(d, f)))
    m['прогноз'] = [dict(что=a, знач=b, как=c, вид=k) for a, b, c, k in h.forecast(m, hw)]
    m['папка'] = d
    m['на_диске'] = bool(model_files(d))
    m['проверка'] = h.known(m['repo'])['проверка']
    return m


def hub_dir(repo):
    return os.path.join(MODELS_DIR, repo.replace('/', '__'))


def hub_get(q):
    # ая1
    h = hmod()
    repo = h.repo_of(q)
    if not repo:
        return False, 'Требуется указать «организация/модель» или ссылку с huggingface.co.'
    if train_state()[0]:
        return False, 'Выполняется обучение: загрузка новой модели заменит текущую, поэтому её можно начать после завершения обучения.'
    if setup_alive():
        return False, 'Уже выполняется другая подготовка.'
    d = hub_dir(repo)
    os.makedirs(d, exist_ok=True)
    open(os.path.join(d, '.tier_repo'), 'w').write(repo + '\n')
    remember_dir(S()['MODEL'])
    remember_dir(d)
    chat_stop()
    S_save({'MODEL': d, 'HF_REPO': repo})
    _minfo.clear(); _arch.clear(); _lmats.clear(); _bases.clear()
    return setup_start('веса')


# ба1
def kfd_gfx():
    # ба2
    for p in sorted(glob.glob('/sys/class/kfd/kfd/topology/nodes/*/properties')):
        m = re.search(r'gfx_target_version (\d+)', read_text(p))
        if m and int(m.group(1)):
            v = int(m.group(1))
            return f'gfx{v // 10000}{(v // 100) % 100:x}{v % 100:x}'
    return ''


def nvidia_query(fields):
    # бб1
    if not shutil.which('nvidia-smi'):
        return []
    try:
        r = subprocess.run(['nvidia-smi', f'--query-gpu=memory.total,{fields}', '--format=csv,noheader,nounits'],
                           capture_output=True, text=True, timeout=4)
        rows = [[x.strip() for x in line.split(',')] for line in r.stdout.splitlines() if line.strip()] if r.returncode == 0 else []
        if not rows:
            return []
        i = os.environ.get('TIER_GPU_INDEX', '').strip()
        row = rows[max(0, min(len(rows) - 1, int(i)))] if re.fullmatch(r'-?\d+', i) else max(rows, key=lambda a: num(a[0]) or 0)
        return row[1:]
    except (OSError, subprocess.SubprocessError, IndexError):
        return []


def main_pci(nv_bus=''):
    # бв1
    if nv_bus:      # бв2
        d, _, rest = nv_bus.strip().lower().partition(':')
        try:
            return f'{int(d, 16):04x}:{rest}'
        except ValueError:
            return ''
    best, slot = 0, ''
    for p in glob.glob('/sys/class/drm/card*/device/mem_info_vram_total'):
        t = num(read_text(p).strip()) or 0
        if t > best:
            best, slot = t, os.path.basename(os.path.realpath(os.path.dirname(p)))
    return slot


def gpu_limits(slot):
    # бг1
    h = hwp()
    out = {}
    try:
        for s in (h.temp_sensors(slot) if h and hasattr(h, 'temp_sensors') else []) or []:
            lab, crit = str(s.get('label') or '').lower(), num(s.get('crit_c'))
            if not crit:
                continue
            k = 'память' if 'mem' in lab else 'край' if 'edge' in lab else 'горячая' if ('junction' in lab or 'hotspot' in lab or 'горячая' not in out) else ''
            if k:
                out[k] = crit
    except Exception:
        return {}
    return out


_ginfo = {}


def gpu_info():
    if _ginfo:
        return _ginfo
    c = conf_read()
    name = c.get('gpu', '')
    if not name or name == '?':
        name = kfd_gfx()
    vendor, cc, title, slot = '', '', '', ''
    if name.startswith('gfx'):
        slot = main_pci()
        vendor = 'amd'
        try:     # бд1
            r = subprocess.run(['lspci', '-mm', '-d', '1002:'], capture_output=True, text=True, timeout=3)
            m = re.search(r'"(?:VGA compatible|Display) controller" "[^"]*" "[^"]*\[([^\]]+)\]"', r.stdout)
            title = m.group(1) if m else ''
        except (OSError, subprocess.SubprocessError):
            pass
    else:
        q = nvidia_query('name,compute_cap,pci.bus_id')
        if q:
            vendor, title, cc = 'nvidia', q[0], q[1] if len(q) > 1 else ''
            slot = main_pci(q[2]) if len(q) > 2 else ''
        elif name:
            vendor, title = ('nvidia', name) if 'nvidia' in name.lower() or 'geforce' in name.lower() else ('', name)
    gen, wid = num(c.get('pcie_gen')), num(c.get('pcie_width'))
    if not gen or not wid:      # бд2
        h = hwp()
        try:
            g2, w2 = h.pcie(slot) if h and hasattr(h, 'pcie') else (0, 0)
            gen, wid = (num(g2), num(w2)) if g2 and w2 else (gen, wid)
        except Exception:
            pass
    _ginfo.update(архитектура=name if name.startswith('gfx') else (f'sm_{cc.replace(".", "")}' if cc else ''),
                  название=title, производитель=vendor or gpu_kind(),
                  видеопамять=num(c.get('vram_mib')),
                  шина={'поколение': gen, 'линий': wid}, пределы=gpu_limits(slot))
    return _ginfo


# бе1
def meminfo():
    m = {}
    for line in read_text('/proc/meminfo').splitlines():
        k, _, v = line.partition(':')
        try:
            m[k] = int(v.split()[0]) * 1024
        except (ValueError, IndexError):
            pass
    return {'всего': m.get('MemTotal', 0), 'доступно': m.get('MemAvailable', 0)}


def nvidia_mem():
    # бж1
    best = (0, 0)
    if not shutil.which('nvidia-smi'):
        return best
    try:
        r = subprocess.run(['nvidia-smi', '--query-gpu=memory.total,memory.used', '--format=csv,noheader,nounits'],
                           capture_output=True, text=True, timeout=4)
        for line in r.stdout.splitlines() if r.returncode == 0 else []:
            a = [x.strip() for x in line.split(',')]
            try:
                t, u = int(float(a[0])) << 20, int(float(a[1])) << 20
            except (ValueError, IndexError):
                continue
            best = max(best, (t, u))
    except (OSError, subprocess.SubprocessError):
        pass
    return best


def vram_info():
    tot, used, users = 0, 0, {}
    for p in glob.glob('/sys/class/drm/card*/device/mem_info_vram_total'):
        t, u = num(read_text(p).strip()) or 0, num(read_text(p.replace('_total', '_used')).strip()) or 0
        if t > tot:
            tot, used = int(t), int(u)
    nv = nvidia_mem()   # бз1
    if nv[0] > tot:
        tot = 0
    if tot:     # бз2
        seen = set()
        for fd in glob.glob('/proc/[0-9]*/fdinfo/*'):
            txt = read_text(fd)
            if 'vram' not in txt:
                continue
            cid = re.search(r'drm-client-id:\s*(\d+)', txt)
            # бз3
            v = re.search(r'drm-total-vram:\s*(\d+)\s*([KMG]i)B', txt) or re.search(r'drm-memory-vram:\s*(\d+)\s*([KMG]i)B', txt)
            if not cid or not v or cid.group(1) in seen:
                continue
            seen.add(cid.group(1))
            pid = fd.split('/')[2]
            who = read_text(f'/proc/{pid}/comm').strip() or pid
            b = int(v.group(1)) << {'Ki': 10, 'Mi': 20, 'Gi': 30}[v.group(2)]
            users[who] = min(tot, users.get(who, 0) + b)
    else:
        tot, used = nv
        if tot:
            try:
                r = subprocess.run(['nvidia-smi', '--query-compute-apps=process_name,used_memory',
                                    '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=4)
                for line in r.stdout.splitlines():
                    a, _, b = line.rpartition(',')
                    users[os.path.basename(a.strip())] = users.get(os.path.basename(a.strip()), 0) + (int(float(b)) << 20)
            except (OSError, subprocess.SubprocessError, ValueError):
                pass
    top = sorted(users.items(), key=lambda x: -x[1])[:5]
    return {'всего': tot, 'занято': used, 'кем': [{'кто': k, 'байт': v} for k, v in top if v > 0]}


# бз4
MODES = [
    ('авто', 'Авто', 'Всё подбирается само: шаг обучения, эпохи, проверки и остановка по данным, умножение и подкачка по железу.',
     dict(lr='', оценок=10, каждые=120, терпение=3, эпохи=1.0, умножение='железо', подкачка='железо', повтор=1.0)),
    ('эксперт', 'Режим эксперта', 'Все параметры плана доступны для изменения.', None),
]
OLD_MODES = ('быстрый', 'баланс', 'качество')        # бз5


def mode_key(k):
    return k if k in ('авто', 'эксперт') else 'авто'


MODE = {k: (t, d, v) for k, t, d, v in MODES}
GEMM = {'bf3': ('Режим ×3 (FP16/BF16)', ['--bf3']), 'fp32': ('FP32, собственные ядра', ['--no-bf3']),
        'lt': ('FP32, библиотека производителя (Lt)', ['--no-bf3', '--no-own']),
        'blas': ('FP32, простая библиотека производителя', ['--no-bf3', '--hipblas']), 'железо': ('', [])}


# би1
GOALS = {
    'срез': ('', '', 'Срез строк по всей сети', 'Часть строк каждой матрицы и все нормировки, оптимизатор Adam. Подходит для тона и знаний, но затрагивает остальное.'),
    'точно': ('down_proj', '16,24,32', 'Точность: факты', 'Целые матрицы down_proj трёх слоёв средней трети сети, NSGD, защита общих знаний, градиент только по ответу. '
              'В испытании (30 правок) перенос на другие формулировки составил 96–99 %, ухудшение остальных фактов — от −0,1 до −0,2 %, у среза строк — −3 %; на 100 правках — от −1 до −2 % против −7 %.'),
    'характер': ('attn', '8,16,24', 'Характер: голос и манера', 'Внимание ранней половины сети, NSGD, защита общих знаний, подмешивание. В испытании (150 персонажей в данных) '
                 'манера у новых персонажей составила 96 %, вместе с верным фактом — 87 %, без метки манера не проявляется (0,0 %), ухудшение остальных фактов — −0,3 %; у среза строк — 87 %, 72 %, 4,8 % и −6,4 % соответственно.'),
}
GOAL_SHORT = {'срез': 'Часть строк каждой матрицы. Подходит для тона и знаний, но затрагивает остальное.',
              'точно': 'Факты и правила. Правка переносится на другие формулировки и почти не затрагивает остальное.',
              'характер': 'Голос и манера персонажа. Не проявляется там, где персонаж не задан.'}
GOAL_FRAC = {'точно': (1 / 3, 1 / 2, 2 / 3), 'характер': (1 / 6, 1 / 3, 1 / 2)}   # би2


def n_layers():
    return len(lora_geom()[3])


def goal_layers(goal):
    # бл1
    if goal not in GOAL_FRAC:
        return GOALS.get(goal, ('', ''))[1]
    n = n_layers()
    if not n:
        return ''
    return ','.join(str(x) for x in sorted({min(n - 1, round(n * f)) for f in GOAL_FRAC[goal]}))


def v_goal(goal, lay):
    return {'срез': 'Срез строк по всей сети', 'точно': f'Матрицы down_proj слоёв {lay} целиком', 'характер': f'Внимание слоёв {lay} целиком'}[goal]


# бм1
LORA_MATS = (('q', 'q_proj'), ('k', 'k_proj'), ('v', 'v_proj'), ('o', 'o_proj'), ('g', 'gate_proj'), ('u', 'up_proj'), ('d', 'down_proj'))
LORA_DEF = {'r': 16, 'alpha': 32, 'tg': 'qkvogud', 'drop': 0.05}
LORA_LR = '2e-4'                 # бм2


def lora_norm(lp):
    # бн1
    if isinstance(lp, str):
        lp = dict(zip(('r', 'alpha', 'tg', 'drop'), lp.split(','))) if lp.strip() else {}
    lp = lp if isinstance(lp, dict) else {}
    d = dict(LORA_DEF)
    try:
        d['r'] = max(1, min(1024, int(float(lp.get('r', d['r'])))))
    except (TypeError, ValueError):
        pass
    try:
        a = float(str(lp.get('alpha', d['alpha'])).replace(',', '.'))
        if 0 < a <= 65536:
            d['alpha'] = int(a) if a == int(a) else round(a, 4)
    except (TypeError, ValueError):
        pass
    if 'tg' in lp:
        d['tg'] = ''.join(c for c, _ in LORA_MATS if c in str(lp['tg']))
    try:
        d['drop'] = round(max(0.0, min(0.9, float(str(lp.get('drop', d['drop'])).replace(',', '.')))), 4)
    except (TypeError, ValueError):
        pass
    return d


def lora_arg(lp):
    return f'{lp["r"]},{lp["alpha"]:g},{lp["tg"]},{lp["drop"]:g}'


def lora_geom():
    # бп1
    model_info()
    A = _arch.get('any')
    if A and A.get('L'):
        return A['hidden'], A['inter'], A['heads'], [(L['hd'], L['nkv'], bool(L['has_v'])) for L in A['L']]
    return 0, 0, 0, []          # бп2


def lora_size(lp):
    # бр1
    H, I, NH, LL = lora_geom()
    n = p = 0
    for hd, kv, hv in LL:
        io = {'q': H + NH * hd, 'k': H + kv * hd, 'v': H + kv * hd, 'o': NH * hd + H, 'g': H + I, 'u': H + I, 'd': I + H}
        for c, _ in LORA_MATS:
            if c in lp['tg'] and not (c == 'v' and not hv):
                n, p = n + 1, p + lp['r'] * io[c]
    return n, p


def lora_desc(lp):
    return f'LoRA r {lp["r"]}, α {lp["alpha"]:g} · ' + ' '.join(nm.replace('_proj', '') for c, nm in LORA_MATS if c in lp['tg'])


def lora_view(lp):
    # бт1
    lp = lora_norm(lp)
    n, p = lora_size(lp)
    t1 = num(conf_read().get('tier1_params'))
    return dict(lp, адаптеров=n, чисел=p, байт=p * 12, s=lp['alpha'] / lp['r'], бюджет=int(t1) * 12 if t1 else None,
                аргумент=lora_arg(lp))

def mats_params(goal, lay):
    # бу1
    H, I, NH, LL = lora_geom()
    ls = [int(x) for x in lay.split(',') if x.strip().isdigit()]
    return sum(H * I if goal == 'точно' else 2 * H * NH * LL[l][0] + (2 if LL[l][2] else 1) * H * LL[l][1] * LL[l][0] for l in ls if l < len(LL))


def whiten_bytes(goal, lay, frac):
    # бф1
    H, I, NH, LL = lora_geom()
    ls = [int(x) for x in lay.split(',') if x.strip().isdigit()]
    return sum(sum(int(frac * d) * d * 4 for d in ([I] if goal == 'точно' else [H, NH * LL[l][0]])) for l in ls if l < len(LL))


def stats(name):
    out = {}
    for line in read_text(run_file(name, 'stats.txt')).splitlines():
        k, _, v = line.partition(' ')
        if v.strip().isdigit():
            out[k] = int(v)
    if 'examples' not in out:
        return None
    for k, suf in (('обучение', 'tokens.bin'), ('отложено', 'eval.bin')):
        try:
            with open(run_file(name, suf), 'rb') as f:
                out[k] = struct.unpack('<i', f.read(4))[0]
        except (OSError, struct.error):
            out[k] = None
    return out


def packs(path, pack):
    # бц1
    cnt = ct = cur = 0
    with open(path, 'rb') as f:
        n = struct.unpack('<i', f.read(4))[0]
        for _ in range(n):
            L, _p = struct.unpack('<ii', f.read(8))
            f.seek(4 * L * (2 if _p < 0 else 1), 1)      # бц2
            if pack <= 0:
                cnt += 1
                continue
            if cur and ct + L > pack:
                cnt, ct, cur = cnt + 1, 0, 0
            ct, cur = ct + L, cur + 1
    return cnt + (1 if cur else 0)


def pack_room(hw):
    # бч1
    vr, t1 = num(hw.get('vram_mib')) or 0, num(hw.get('tier1_params')) or 0
    if not vr or not t1:
        return None
    cap = min(99, max(50, int(num(hw.get('mem_cap_pct')) or 98)))
    mi, n = model_info(), n_layers()
    head = (num(mi.get('словарь')) or 0) * (num(mi.get('скрытый')) or 0) * 2
    lay = layer_bf16_bytes() / n if n else 0
    return vr - t1 * 12 / 2**20 - vr * (100 - cap) / 100 - (head + lay) / 2**20


def pack_dims(h):
    # бш1
    p = model_info().get('описание')
    try:
        d = h.arch_dims(p) if p and hasattr(h, 'arch_dims') else None
    except Exception:
        d = None
    if not d:
        H, I, NH, LL = lora_geom()
        d = dict(hidden=H, inter=I, heads=NH, layers=len(LL))
    return d


def pack_auto(hw, P99, EX):
    # бэ1
    why = f'99-й процентиль длины — {P99} токенов'
    h, room = hwp(), pack_room(hw)
    try:
        if h and hasattr(h, 'pack_choice') and room is not None:
            dims = pack_dims(h)
            pk = int(h.pack_choice(P99, EX, max(0, room), dims))
            nxt = pk * 2 if pk else 512
            need = h.act_bytes(nxt, dims['hidden'], dims['inter'], dims['heads'], dims['layers']) / 2**20
            if need > room and (P99 * 16 > pk * 5 if pk else True):      # бэ2
                why += f'; {nxt} токенов в шаге не помещаются: буферам шага требуется ≈ {need:.0f} МиБ, вне среза остаётся ≈ {max(0, room):.0f} МиБ'
            return pk, why + '.'
    except Exception:
        pass
    big = num(hw.get('vram_mib')) or 0              # бэ3
    if 160 < P99 <= 400 and big < 12000:
        why += '; 1024 токена в шаге — только на картах от 12 ГБ'
    return (512 if P99 <= 160 or (P99 <= 400 and big < 12000) else 1024 if P99 <= 400 else 0), why + '.'


def mix_packs(path, rp, pack):
    # бю1
    try:
        R = [json.loads(l) for l in open(os.path.join(APP, 'replay.jsonl'), encoding='utf-8') if l.strip()]
        q = sum(len(d.get('input') or '') for d in R) / len(R); a = sum(len(d.get('output') or '') for d in R) / len(R)
    except (OSError, ValueError, ZeroDivisionError):
        q, a = 28, 160
    RL, RA = max(1, round(q / 3.2 + 16 + a / 3.2)), max(1, round(0.7 * a / 3.2))
    cnt = ct = cur = 0
    with open(path, 'rb') as f:
        n = struct.unpack('<i', f.read(4))[0]
        for _ in range(n):
            L, p = struct.unpack('<ii', f.read(8))
            f.seek(4 * L * (2 if p < 0 else 1), 1)
            for x in [L] + [RL] * max(1, -(-int(rp * (L - abs(p))) // RA)):
                if cur and ct + x > pack:
                    cnt, ct, cur = cnt + 1, 0, 0
                ct, cur = ct + x, cur + 1
    return cnt + (1 if cur else 0)


def step_time():
    # бя1
    best = None
    for n in runs():
        st = log_stamp(n)
        if st and (not best or st[0] > best[0]):
            best = (st[0], n)
    if not best:
        return None, '', None, None
    steps, evals, _m, _l, pl = parse_log(best[1])
    T = [x['сек'] for x in steps if x['сек']]
    if len(T) < 3:
        return None, '', None, None
    pk, sec = pl.get('упаковка'), sum(T) / len(T)
    K = sorted(e['с'] / (e['упаковок'] * sec) for e in evals if e.get('с') and e.get('упаковок'))
    k = min(1.5, max(0.15, K[len(K) // 2])) if K else None
    return sec, best[1], int(pk) if pk and pk.isdigit() else None, k


def hms(s):
    s = int(round(s))
    return f'{s // 3600} ч {s % 3600 // 60:02d} мин' if s >= 3600 else f'{s // 60} мин {s % 60:02d} с' if s >= 60 else f'{s} с'


def plan(name, mode, over=None, goal=None, lora=None):
    # вб1
    st = stats(name)
    if not st or not st.get('обучение'):
        return {'ок': False, 'почему': 'Датасет ещё не токенизирован.'}
    mode = mode_key(mode)
    m = dict(MODE['авто'][2])                        # вб2
    o = {k: str(v).strip() for k, v in (over or {}).items() if str(v).strip()} if mode == 'эксперт' else {}
    hw = conf_read()
    rows, x, bad = [], [], []
    goal = o.get('что учить', goal or 'срез')
    if goal not in GOALS and goal != 'lora':
        bad.append('что учить'); goal = 'срез'
    lo = goal == 'lora'
    lp = lora_norm(lora) if lo else None

    def iv(k, lo=0):
        try:
            v = int(o[k])
            if v < lo:
                raise ValueError
            return v
        except (KeyError, ValueError):
            if k in o:
                bad.append(k)
            return None

    EX, P99 = st['examples'], st.get('p99', 0)
    pack = iv('PACK')
    why_pack = 'Задано вручную.' if pack is not None else 'По одному примеру на шаг: набор небольшой.' if EX <= 256 else ''
    if pack is None:
        pack, why_pack = (0, why_pack) if EX <= 256 else pack_auto(hw, P99, EX)
    ep = iv('EPOCHS', 1)
    if ep is None:
        base = 12 if EX <= 64 else 6 if EX <= 512 else 3 if EX <= 4096 else 2
        ep = max(1, round(base * m['эпохи']))
        why_ep = f'Примеров: {EX}' + (f', базовое число эпох {base} × {m["эпохи"]:g}' if m['эпохи'] != 1 else '') + '.'
    else:
        why_ep = 'Задано вручную.'
    per = packs(run_file(name, 'tokens.bin'), pack)
    rp = o.get('повтор', '0' if lo else str(m['повтор']))      # вб3
    try:
        rp = float(rp)
        if not 0 <= rp <= 4:
            raise ValueError
    except ValueError:
        bad.append('повтор'); rp = 0.0
    has_rp = rp > 0 and os.path.isfile(os.path.join(APP, 'replay.jsonl'))
    if has_rp and pack:                              # вб4
        per = mix_packs(run_file(name, 'tokens.bin'), rp, pack)
    total = per * ep
    sec, src, src_pack, k_ev = step_time()
    sp = sec or 3.0                                  # вб5
    # вб6
    pe = packs(run_file(name, 'eval.bin'), pack) if os.path.exists(run_file(name, 'eval.bin')) else 0
    k_ev = k_ev or 0.45
    ev = iv('EV', 1)
    if ev is None:
        if mode_key(mode) == 'авто':                   # вб7
            evn = max(1, per // 2); ev = evn
        else:
            evn = max(3, total // m['оценок'])
            ev = evn if not m['каждые'] else min(evn, max(1, int(m['каждые'] // sp)))
            ev = min(ev, 100 if m['оценок'] <= 10 else 50)
        cost = -(-int(pe * k_ev * 100) // 15)        # вб8
        why_ev = (f'не реже {m["оценок"]} раз за обучение' +
                  (f' и примерно раз в {m["каждые"] // 60} мин' if m['каждые'] and cost <= ev else ''))
        if cost > ev:
            ev = min(evn, cost)
            why_ev += f'; упаковок в валидационной выборке — {pe}, на валидацию уходит не более 15 % времени'
    else:
        why_ev = 'задано вручную'
    fl = iv('FL', 1)
    why_fl = 'Задано вручную.' if fl is not None else 'На шаг позже валидации, чтобы запись не совпадала с чтением.'
    if fl is None:
        fl = ev + 1
    stop = iv('STOP_AFTER', 0)
    why_stop = 'Задано вручную.' if stop is not None else ''
    if stop is None:                                 # вб9
        stop = max(2, -(-per // max(1, ev)) * max(1, (ep + 1) // 2)) if mode_key(mode) == 'авто' else m['терпение']
    # вб10
    lr = o.get('LR', (LORA_LR if m['lr'] else '') if lo else m['lr'] if goal == 'срез' else '')
    if lr:
        try:
            if not 0 < float(lr) < 1:
                raise ValueError
        except ValueError:
            bad.append('LR')
    n_ev = 1 + -(-total // ev)                       # вб11
    rows += [
        dict(к='LR', сырое=lr, что='Шаг обучения (LR)', знач=lr or 'Подбор',
             как='Задано вручную.' if 'LR' in o else ('Без подбора: обучение начинается сразу.' if lr else
                                               'Подбор по 40 коротким шагам: берётся десятая часть значения LR, при котором сглаженная потеря (loss) минимальна.')),
        dict(к='EPOCHS', сырое=ep, что='Число эпох', знач=str(ep), как=why_ep),
        dict(к='PACK', сырое=pack, что='Токенов в шаге', знач=str(pack) if pack else 'По одному примеру', как=why_pack),
        dict(к='', что='Шагов всего', знач=(f'Около {total}' if has_rp else f'{total}'), как=f'Шагов в эпохе: {per}, эпох: {ep}' + (', с учётом подмешивания' if has_rp else '') + '.'),
    ]
    if st.get('instr_rows'):                         # вб12
        try:
            sw = scene_norm(json.load(open(run_file(name, 'source.json'), encoding='utf-8')).get('сцена'))['всегда']
        except (OSError, ValueError, AttributeError):
            sw = ''
        rows.append(dict(к='', что='Системная инструкция', знач=f'{st.get("with_instr", 0)} из {EX}',
                         как='Поле instruction передаётся как системная инструкция' + (f'; всегда — при словах: {sw}' if sw else '') + '; доля задаётся в карточке «Данные».'))
    rows += [
        dict(к='повтор', сырое=f'{rp:g}', что='Подмешивание',
             знач=(f'×{rp:g} к токенам данных' if has_rp else 'Выключено'),
             как=('К каждому примеру добавляются обычные ответы исходной модели: модель сохраняет их при обучении и меньше искажает то, '
                  'чего нет в данных. Для персонажа, который проявляется всегда, подмешивание не требуется (значение 0).'
                  if has_rp else 'Задано вручную.' if 'повтор' in o else 'Классическая LoRA обучается только на данных; подмешивание включается в режиме эксперта.'
                  if lo else 'Отсутствует набор replay.jsonl.')),
        dict(к='EV', сырое=ev, что='Валидация каждые', знач=f'{ev}', как=f'Валидаций за обучение: {n_ev}; ' + why_ev + '.'),
        dict(к='FL', сырое=fl, что='Снимок весов каждые', знач=f'{fl}', как=why_fl),
        dict(к='STOP_AFTER', сырое=stop, что='Ранняя остановка', знач=(_after(stop) + ' без улучшения' if stop else 'Выключена'),
             как=why_stop or 'Обучение прекращается, когда потеря (loss) на валидации перестаёт снижаться.'),
    ]
    g = o.get('умножение', m['умножение'])
    if g not in GEMM:
        bad.append('умножение')
        g = 'железо'
    hw_bf3 = hw.get('bf16x3') == '1'
    if g == 'bf3' and hw.get('bf16x3') == '0' and 'умножение' not in o:
        g = 'железо'                                 # вб13
    x += GEMM[g][1]
    rows.append(dict(к='умножение', сырое=g, варианты=[[k, v[0] or 'По оборудованию'] for k, v in GEMM.items()],
                     что='Умножение матриц',
                     знач=GEMM[g][0] or ('Режим ×3 (FP16/BF16)' if hw_bf3 else 'FP32'),
                     как={'bf3': 'Три умножения с накоплением в FP32: сначала FP16×3, при неудачной пробе BF16×3; если обе пробы не проходят, используется FP32.',
                          'fp32': 'Эталонная точность; шаг дольше примерно на треть.'}.get(g, 'Определяется автоматически по оборудованию.' if g == 'железо' else 'Задано вручную.')))
    pf = o.get('подкачка', m['подкачка'])
    if pf not in ('вкл', 'выкл', 'железо'):
        bad.append('подкачка')
        pf = 'железо'
    x += {'вкл': ['--prefetch'], 'выкл': ['--no-prefetch'], 'железо': []}[pf]
    rows.append(dict(к='подкачка', сырое=pf, варианты=[['железо', 'По оборудованию'], ['вкл', 'Включена'], ['выкл', 'Выключена']],
                     что='Опережающая подкачка', знач={'вкл': 'Включена', 'выкл': 'Выключена'}.get(pf, 'Включена' if hw.get('prefetch', '1') == '1' else 'Выключена'),
                     как='Следующий слой передаётся по шине, пока считается текущий.' if pf != 'выкл' else 'Задано вручную.'))
    w1, damp, cpu = iv('первый', 1), o.get('придержка', ''), iv('процессор', 0)
    if w1 and w1 > 1:
        x += ['--first-weight', str(w1)]
    if damp:
        try:
            if float(damp) < 0:
                raise ValueError
            if float(damp) > 0:
                x += ['--damp', damp]
        except ValueError:
            bad.append('придержка')
    t1 = num(hw.get('tier1_params'))
    mats, wfrac, lim = '', '', []                    # вб14
    if lo:                                           # вб15
        n_ad, n_par = lora_size(lp)
        cap = int(t1) * 12 if t1 else None
        if not lp['tg']:
            lim.append('Не выбрано ни одной матрицы LoRA.')
        rows.append(dict(к='', сырое=goal, что='Что учится', знач=lora_desc(lp),
                         как=f'Адаптеры поверх замороженных весов: W + α/r · B·A, s = α/r = {lp["alpha"] / lp["r"]:g}; A — равномерно в ±1/√вход, '
                             'B — нули, AdamW FP32. Исходные веса не меняются, адаптеры сохраняются отдельным файлом.'))
        rows.append(dict(к='', что='Адаптеры', знач=f'{n_ad} · {n_par:,}'.replace(',', ' ') + ' чисел',
                         как=f'≈ {n_par * 12 / 2**30:.2f} ГиБ видеопамяти: A, B и два момента AdamW в FP32; в слоях полного внимания v_proj отсутствует.'))
        if cap and n_par * 12 > cap:
            lim.append(f'Адаптеры LoRA ({n_par * 12 / 2**30:.2f} ГиБ) превышают бюджет видеопамяти ({cap / 2**30:.2f} ГиБ): требуется уменьшить ранг или число матриц.')
        rows.append(dict(к='', что='Dropout', знач=f'{lp["drop"]:g}' if lp['drop'] else 'Выключено', как='Применяется на входе адаптера, только при обучении.'))
    else:
        lay = o.get('слои правки', goal_layers(goal))
        if goal != 'срез' and (not re.fullmatch(r'\d+(,\d+)*', lay or '') or max(map(int, lay.split(','))) >= n_layers()):
            bad.append('слои правки'); lay = goal_layers(goal)
        mats = f'{GOALS[goal][0]}:{lay}' if goal != 'срез' else ''
        rows.append(dict(к='', сырое=goal, что='Что учится', знач=v_goal(goal, lay), как=GOALS[goal][3]))
        if goal != 'срез' and 'слои правки' not in o:
            rows.append(dict(к='', что='Слои правки', знач='Замером перед обучением', как=f'Шесть шагов на данных запуска: выбираются три слоя, на которые данные влияют сильнее всего; без замера — {lay}.'))   # вб16
    qp, qe, qwhy = qbase()                           # вб17
    if qe:
        if qp:
            x += ['--qbase', qp]
        else:
            lim.append(f'База {qe["имя"]}: {qwhy}')
        tys = ', '.join(f'{t} ×{n}' for t, n in qe.get('типы') or [])
        rows.append(dict(к='', что='База', знач=f'{qe["имя"][:-3]} · {qe["байт"] / 2**30:.1f} ГиБ',
                         как=f'{tys}. Замороженные матрицы слоёв читаются из квантованной базы и разворачиваются на карте; '
                             'обучаемая часть берётся из исходного BF16; нормировки и эмбеддинги конвертируются в FP16 при загрузке' +
                             (f'; ошибка весов {100 * qe["ошибка"]:.1f} %' if qe.get('ошибка') is not None else '') + '.'))
    elif S()['QBASE']:
        lim.append(qwhy)
    else:
        rows.append(dict(к='', что='База', знач='BF16', как='Замороженные веса читаются из исходного BF16; внутреннее хранение — FP16.'))
    if not lo and goal != 'срез':
        need = mats_params(goal, lay); t1 = num(hw.get('tier1_params')); cap = int(t1) * 12 if t1 else None
        # вб18
        wb = lambda f: whiten_bytes(goal, lay, f)
        wfrac = next((f for f in (0.25, 0.1) if not cap or need * 6 + wb(f) <= cap), '0.1')
        tot = need * 6 + wb(float(wfrac))
        rows.append(dict(к='слои правки', сырое=lay, что='Слои правки', знач=lay,
                         как=f'≈ {tot / 2**30:.1f} ГиБ видеопамяти: вес и момент — {need * 6 / 2**30:.1f} ГиБ, защита общих знаний ({float(wfrac):g} главных направлений) — {wb(float(wfrac)) / 2**30:.2f} ГиБ' + (
                             f'. Превышен бюджет {cap / 2**30:.1f} ГиБ: слоёв слишком много' if cap and tot > cap else '') + '.'))
        if cap and tot > cap:
            bad.append('слои правки')
    sl = '' if lo else o.get('слои', '')            # вб19
    if sl:
        if re.fullmatch(r'\d+:\d+', sl) and int(sl.split(':')[0]) <= int(sl.split(':')[1]) < n_layers():
            x += ['--slice-range', sl]
        else:
            bad.append('слои')
    if cpu:
        x += ['--cpu-share', str(cpu)]
    rows += [dict(к='первый', сырое=w1 or 1, что='Вес первого токена', знач=f'× {w1 or 1}', как='Первый токен определяет, каким будет ответ.'),
             dict(к='придержка', сырое=damp or 0, что='Ограничение градиента', знач=(f'Выше {damp} средних' if damp and damp != '0' else 'Выключено'),
                  как='Шаг уменьшается, если норма градиента резко возросла.')]
    if not lo:
        rows.append(dict(к='слои', сырое=sl, что='Обучаемые слои', знач=(sl or f'Все · 0:{n_layers() - 1}') if goal == 'срез' else 'Как в строке «Слои правки»',
                         как='Остальные слои заморожены.' if goal == 'срез' else 'В этом режиме обучаются только целые матрицы выбранных слоёв.'))
    rows.append(dict(к='процессор', сырое=cpu or 0, что='Потоков процессора', знач=str(cpu or 0),
                     как='0 — выключено (обычно быстрее всего). Иначе процессор считает часть замороженных строк из ОЗУ; если шаг от этого замедляется, движок возвращает строки карте.'))
    if t1 and not lo:
        rows.append(dict(к='', что='Бюджет среза строк', знач=f'{int(t1):,}'.replace(',', ' ') + ' чисел',
                         как=f'{t1 * 12 / 2**30:.2f} ГиБ видеопамяти на вес и Adam; значение берётся из настроек оборудования.'))
    env = {'LR': lr, 'EPOCHS': str(ep), 'PACK': str(pack), 'EV': str(ev), 'FL': str(fl), 'STOP_AFTER': str(stop),
           'X': ' '.join(x), 'REPLAY': f'{rp:g}' if has_rp else '0', 'MATS': mats,
           'NSGD': '1' if mats else '0', 'POSDW': '1' if goal == 'точно' else '0', 'WHITEN_FRAC': str(wfrac),
           'LORA': lora_arg(lp) if lo else '', 'MATS_AUTO': '1' if mats and 'слои правки' not in o else '0'}
    est = ''
    if sec:
        est = f'≈ {hms(total * sec + n_ev * pe * k_ev * sec)} — по скорости запуска «{src}», с учётом валидации'
        if src_pack is not None and src_pack != pack:
            pk = lambda p: f'{p} токенов в шаге' if p else 'по одному примеру в шаге'
            est += ', упаковка другая, поэтому время приблизительное'
        est += '.'
    return {'ок': not bad and not lim, 'почему': '; '.join(([('неверно: ' + ', '.join(bad))] if bad else []) + ([' '.join(lim)] if lim else [])), 'строки': rows, 'env': env,
            'шагов': total, 'оценок': n_ev, 'длительность': est, 'режим': mode, 'что': goal, 'лора': lp,
            # вб20
            'запись': int((n_par * 12 if lo else tot if mats else (t1 or 0) * 12) * 2)}


RUN_SUFFIXES = ('log.tsv', 'console.txt', 'stage.txt', 'stagep.txt', 'tokens.bin', 'eval.bin', 'stats.txt', 'source.json',
                'prepare.txt', 'run.json', 'tier1.bin', 'tier1_best.bin', 'step.txt', 'step_base.txt', 'elapsed.txt', 'mix.bin',
                'replay.txt', 'mode.txt', 'lrfind.txt', 'load.txt', 'infer.log')


def run_files(name):
    # вв1
    own = {f'{name}_{s}' for s in RUN_SUFFIXES} | {f'adam_state_{name}.bin', f'tier_assignment_{name}_act.txt',
                                                   f'tier_assignment_{name}_act.txt.t'}
    out = []
    try:
        for e in os.scandir(S()['WORK']):
            if e.is_file(follow_symlinks=False) and (e.name in own or re.fullmatch(re.escape(name) + r'_bench_\d{8}_\d{6}\.tsv', e.name)):
                out.append(e)
    except OSError:
        pass
    return out


def run_bytes(name):
    t = 0
    for e in run_files(name):
        try:
            st = e.stat(follow_symlinks=False)
            t += min(st.st_size, st.st_blocks * 512) if st.st_blocks else st.st_size   # вг1
        except OSError:
            pass
    return t


def goal_of(choice, mode_txt):
    env, mt = (choice or {}).get('env') or {}, dict(l.partition('=')[::2] for l in mode_txt.splitlines() if '=' in l)
    if env.get('LORA') or mt.get('LORA'):
        return 'lora'
    mats = env.get('MATS') or mt.get('MATS', '')
    return 'точно' if mats.startswith('down_proj') else 'характер' if mats.startswith('attn') else 'срез'


_rsum = {}


def runs_summary():
    out, live = [], train_state()[1]
    for n in runs():
        key = (log_stamp(n), stamp(run_file(n, 'console.txt')), stamp(run_file(n, 'stage.txt')), stamp(run_file(n, 'step.txt')),
               stamp(run_file(n, 'run.json')), live == n)
        c = _rsum.get(n)
        if not c or c[0] != key or live == n:
            d = read_log(n)
            best = [e['loss'] for e in d['оценки'] if e['loss'] is not None]
            ch = d['выбор'] or {}
            row = {'имя': n, 'состояние': d['состояние'], 'шаг': d['последний'].get('шаг'), 'всего': d['план'].get('всего'),
                   'лучшая': min(best) if best else None, 'режим': ch.get('режим', ''),
                   'цель': goal_of(ch, read_text(run_file(n, 'mode.txt'))), 'датасет': os.path.basename(ch.get('датасет') or ''),
                   'начало': ch.get('начало', ''), 'прошло': d['последний'].get('прошло', ''),
                   'веса': os.path.exists(run_file(n, 'step.txt')), 'продолжение': d['продолжение'],
                   'когда': d['последний'].get('время') or (d['сообщения'][-1]['время'] if d['сообщения'] else '')}
            c = (key, row, time.time() - 99)
        if time.time() - c[2] > 30:        # ве1
            c = (c[0], dict(c[1], размер=run_bytes(n)), time.time())
        _rsum[n] = c
        out.append(c[1])
    return sorted(out, key=lambda r: r['когда'], reverse=True)


def run_delete(name):
    ok, why = name_ok(name)
    if ok and train_state()[1] == name:
        ok, why = False, 'Выполняется обучение.'
    files = run_files(name) if ok else []
    if ok and not files:
        ok, why = False, 'Такого запуска нет.'
    freed = 0
    if ok:
        if chat_name == name:
            chat_stop()
        for e in files:
            try:
                st = e.stat(follow_symlinks=False)
                freed += min(st.st_size, st.st_blocks * 512) if st.st_blocks else st.st_size
                os.remove(e.path)
            except OSError:
                pass
        _logc.pop(name, None); _rsum.pop(name, None)
    return {'ок': ok, 'почему': why, 'освобождено': freed}


# вз1
def read_examples(path, start=0, n=5, cap=4000):
    # вз2
    out, total = [], 0
    try:
        f = open(path, encoding='utf-8', errors='replace')
    except OSError as e:
        return {'ок': False, 'почему': str(e)}
    jsonl = path.lower().endswith(('.jsonl', '.json'))
    with f:
        for line in f:
            if not line.strip():
                continue
            if start <= total < start + n:
                if jsonl:
                    try:
                        r = json.loads(line)
                    except ValueError:
                        r = {'_': line.strip()[:cap]}
                    if not isinstance(r, dict):
                        r = {'_': str(r)[:cap]}
                    out.append({k: (v if isinstance(v, str) else json.dumps(v, ensure_ascii=False))[:cap] for k, v in r.items()})
                else:
                    out.append({'instruction': line.strip()[:cap]})
            total += 1
    return {'ок': True, 'всего': total, 'с': start, 'примеры': out}


def questions(path):
    r = read_examples(path, 0, 500)
    if not r['ок']:
        return r
    qs = []
    for x in r['примеры']:     # ви1
        sc = (x.get('instruction') or '').strip()
        q = (x.get('input') or x.get('question') or x.get('q') or x.get('_') or '').strip()
        if not q:
            q, sc = sc, ''
        if q:
            qs.append({'q': q, 'сцена': sc, 'ждём': (x.get('output') or x.get('answer') or '').strip()})
    return {'ок': True, 'вопросы': qs, 'всего': r['всего']}


def bench_save(name, head, rows):
    ok, why = name_ok(name or 'исходная')
    if not ok:
        return {'ок': False, 'почему': why}
    p = os.path.join(S()['WORK'], f'{name or "исходная"}_bench_{time.strftime("%Y%m%d_%H%M%S")}.tsv')
    clean = lambda v: str(v).replace('\t', ' ').replace('\n', ' ⏎ ')
    try:
        with open(p, 'w', encoding='utf-8') as f:
            f.write('\t'.join(map(clean, head)) + '\n')
            for r in rows:
                f.write('\t'.join(map(clean, r)) + '\n')
    except OSError as e:
        return {'ок': False, 'почему': str(e)}
    return {'ок': True, 'путь': p}


# вк1
DRAFT = '_черновик'
DRAFT_FILES = ('tokens.bin', 'eval.bin', 'stats.txt', 'source.json', 'prepare.txt')
prep_jobs = {}


def scene_norm(sc):
    # вл1
    sc = sc or {}
    try:
        d = min(1.0, max(0.0, float(sc.get('доля', 1))))
    except (TypeError, ValueError):
        d = 1.0
    return {'доля': d, 'всегда': ', '.join(x.strip() for x in str(sc.get('всегда') or '').split(',') if x.strip())}


def prep_start(name, data, scene=None):
    ok, why = name_ok(name)
    if not ok:
        return False, why
    if name != DRAFT and os.path.exists(log_path(name)):
        return False, f'Запуск «{name}» уже есть.'
    if not os.path.isfile(data):
        return False, 'Файл не найден.'
    j = prep_jobs.get(name)
    if j and j['p'].poll() is None:                 # вм1
        j['p'].kill()
        try:
            j['p'].wait(timeout=5)
        except subprocess.TimeoutExpired:
            return False, 'Не удалось остановить прежнюю токенизацию.'
    for suf in ('stats.txt', 'tokens.bin', 'eval.bin'):
        try:
            os.remove(run_file(name, suf))
        except OSError:
            pass
    sc = scene_norm(scene)
    env = child_env(name)
    env.update(INSTR=f'{sc["доля"]:g}', INSTR_ALWAYS=sc['всегда'])
    out = open(run_file(name, 'prepare.txt'), 'wb')
    p = subprocess.Popen([sys.executable, os.path.join(APP, 'prepare.py'), data], cwd=DATA, env=env,
                         stdout=out, stderr=subprocess.STDOUT)
    out.close()
    st = os.stat(data)
    json.dump({'путь': data, 'размер': st.st_size, 'время': st.st_mtime, 'сцена': sc},
              open(run_file(name, 'source.json'), 'w', encoding='utf-8'), ensure_ascii=False)
    prep_jobs[name] = {'p': p}
    threading.Thread(target=p.wait, daemon=True).start()
    return True, ''


def prep_state(name):
    j = prep_jobs.get(name)
    try:
        src = json.load(open(run_file(name, 'source.json'), encoding='utf-8'))
    except (OSError, ValueError):
        src = None
    busy = bool(j and j['p'].poll() is None)
    out = [x for x in read_text(run_file(name, 'prepare.txt')).splitlines() if x.strip()][-12:]
    st = None if busy else stats(name)
    fresh = False
    if src:
        try:
            s2 = os.stat(src['путь'])
            fresh = s2.st_size == src['размер'] and abs(s2.st_mtime - src['время']) < 1
        except OSError:
            pass
    return {'идёт': busy, 'код': None if busy or not j else j['p'].returncode, 'сводка': st, 'источник': src,
            'свежий': fresh, 'вывод': out, 'запуск_есть': os.path.exists(log_path(name)), 'сцена': scene_norm((src or {}).get('сцена'))}


# вн1
setup_proc = None
SETUP_TITLE = {'движок': 'Сборка движка', 'веса': 'Загрузка весов', 'склеить': 'Объединение частей весов', 'пакеты': 'Установка пакетов',
               'квант': 'Квантование', 'импорт': 'Перенос квантованной базы'}
setup_what = ''          # вн2


def setup_alive():
    return setup_proc is not None and setup_proc.poll() is None


def setup_start(what):
    # вп1
    global setup_proc, setup_what
    if setup_alive():
        return False, 'Подготовка уже выполняется.'
    kind, _, arg = what.partition(':')
    pre = None
    if kind in ('квант', 'импорт'):
        cmd, why = quant_job(kind, arg)
        if not cmd:
            return False, why
        pre = lambda: os.nice(10)                    # вп2
    else:
        cmd = {'движок': ['bash', os.path.join(APP, 'build.sh')],
               'веса': ['bash', os.path.join(APP, 'fetch_model.sh')] if S()['HF_REPO'] else None,
               'склеить': [sys.executable, os.path.join(APP, 'arch.py'), 'merge', S()['MODEL']],
               'пакеты': ['bash', os.path.join(APP, 'pydeps.sh')]}.get(what)
    if not cmd:
        return False, 'Модель не выбрана: в разделе «Модели» требуется указать имя с Hugging Face или ссылку.' if what == 'веса' else 'Неизвестное действие.'
    out = open(SETUP_LOG, 'wb')
    out.write(f'=== {SETUP_TITLE.get(kind, "Подготовка")}{" " + arg if arg else ""} ===\n'.encode())
    out.flush()
    setup_proc = subprocess.Popen(cmd, cwd=DATA, env=child_env(), stdout=out, stderr=out, preexec_fn=pre)
    setup_what = what
    _bases.clear()
    return True, ''


# вп3
QDIR = os.path.join(DATA, 'quant')
RECIPES = ('Q8_0', 'Q6_K', 'Q5_K_M', 'Q5_K_S', 'Q4_K_M', 'Q4_K_S')
RECIPE_TIP = {'Q8_0': '8,5 бита на вес: от BF16 почти не отличается.',
              'Q6_K': '6,6 бита на вес: потери малы.',
              'Q5_K_M': '5,5 бита на вес; v_proj и часть down_proj — Q6_K.',
              'Q5_K_S': '5,5 бита на вес.',
              'Q4_K_M': '4,5 бита на вес; v_proj и часть down_proj — Q6_K. Обычный выбор для разговора.',
              'Q4_K_S': '4,5 бита на вес; первые слои — Q5_K. Самый лёгкий вариант.'}
QB = {'Q8_0': (32, 34), 'Q6_K': (256, 210), 'Q5_K': (256, 176), 'Q4_K': (256, 144)}
_lmats, _bases = {}, {}
LAYER_MAT_TAIL = r'(\d+)\.(self_attn\.[qkvo]_proj|mlp\.(?:gate|up|down)_proj)\.weight$'
MAT_ORDER = ('self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj', 'self_attn.o_proj', 'mlp.gate_proj', 'mlp.up_proj', 'mlp.down_proj')


def model_sig():
    try:
        st = os.stat(os.path.join(S()['MODEL'], 'model.safetensors'))
        return f'{st.st_size} {int(st.st_mtime)}'
    except OSError:
        return ''


def layer_mats():
    # вс1
    path, sig = os.path.join(S()['MODEL'], 'model.safetensors'), model_sig()
    if _lmats.get('sig') == (path, sig):
        return _lmats['v']
    out = []
    try:
        with open(path, 'rb') as f:
            n = struct.unpack('<Q', f.read(8))[0]
            hdr = json.loads(f.read(n)) if 2 <= n <= 200 << 20 else {}
        A = arch_now()
        rx = re.compile('^' + re.escape(A['layer_prefix'] if A else 'layers.') + LAYER_MAT_TAIL)
        for k, v in hdr.items():
            m = rx.match(k)
            if m and isinstance(v, dict) and len(v.get('shape', [])) == 2:
                out.append((int(m.group(1)), m.group(2), v['shape'][0], v['shape'][1]))
    except (OSError, ValueError, struct.error):
        out = []
    out.sort(key=lambda x: (x[0], MAT_ORDER.index(x[1])))
    _lmats.update(sig=(path, sig), v=out)
    return out


def recipe_types(rec, mats):
    # вт1
    base = 'Q8_0' if rec == 'Q8_0' else 'Q6_K' if rec == 'Q6_K' else rec[:4]
    mm, ms = rec.endswith('_M'), rec.endswith('_S')
    nl = 1 + max((l for l, k, *_ in mats if k == 'mlp.down_proj'), default=-1)
    nv = sum(1 for _, k, *_ in mats if k == 'self_attn.v_proj')
    more = lambda i, n: i < n // 8 or i >= 7 * n // 8 or (i - n // 8) % 3 == 2
    out, iv = [], 0
    for l, k, r, c in mats:
        t = base
        if k == 'self_attn.v_proj':
            t = 'Q6_K' if mm and more(iv, nv) else 'Q5_K' if ms and base == 'Q4_K' and iv < 4 else base
            iv += 1
        if k == 'mlp.down_proj':
            t = 'Q6_K' if mm and more(l, nl) else 'Q5_K' if ms and base == 'Q4_K' and l < nl // 8 else base
        out.append(t)
    return out


def recipe_bytes(rec):
    mats = layer_mats()
    if not mats:
        return 0
    tot = 0
    for (l, k, r, c), t in zip(mats, recipe_types(rec, mats)):
        b, by = QB[t]
        tot += -(-(r * (c // b) * by) // 4096) * 4096
    return tot + -(-(len(mats) * 200 + 4096) // 4096) * 4096


def layer_bf16_bytes():
    return sum(r * c * 2 for _, _, r, c in layer_mats())


def qmod():
    if APP not in sys.path:
        sys.path.insert(0, APP)
    import quant
    return quant


def base_list():
    # вц1
    try:
        names = sorted(f for f in os.listdir(QDIR) if f.endswith('.tq'))
    except OSError:
        names = []
    sig, out = model_sig(), []
    for fn in names:
        p = os.path.join(QDIR, fn)
        try:
            st = os.stat(p)
            cst = os.stat(p + '.check').st_mtime if os.path.exists(p + '.check') else 0
        except OSError:
            continue
        key = (st.st_size, st.st_mtime, cst, sig)
        if _bases.get(fn, (None,))[0] == key:
            out.append(_bases[fn][1])
            continue
        e = {'имя': fn, 'байт': st.st_size}
        try:
            inf = qmod().info(p)
        except (OSError, ValueError, SystemExit):
            inf = None
        if not inf:
            e.update(годна=False, почему='Файл не читается.')
        else:
            meta = inf['мета']
            fr = meta.get('from', '').split(' ', 1)
            try:
                chk = json.load(open(p + '.check', encoding='utf-8'))
            except (OSError, ValueError):
                chk = {}
            own = fr[0] == 'tierlm'
            err = float(meta['err']) if own and meta.get('err') else chk.get('ошибка')
            why = ''
            if own and meta.get('src') != sig:
                why = 'Модель изменилась после квантования.'
            elif not own and chk.get('модель') != sig:
                why = 'Не сверена с текущей моделью.' if chk else 'Не сверена с моделью.'
            tys = sorted(inf['типы'].items(), key=lambda kv: -kv[1])
            e.update(откуда=('Собственный квант ' + fr[1]) if own else ('GGUF' if fr[0] == 'gguf' else 'bitsandbytes') + (f' · {fr[1]}' if len(fr) > 1 else ''),
                     рецепт=fr[1] if own and len(fr) > 1 else '', типы=tys, матриц=inf['матриц'], ошибка=err,
                     худшая=chk.get('худшая'), годна=not why, почему=why)
        _bases[fn] = (key, e)
        out.append(e)
    return out


def qbase():
    # вч1
    nm = S()['QBASE']
    if not nm:
        return None, None, ''
    e = next((b for b in base_list() if b['имя'] == nm), None)
    if not e:
        return None, None, f'Базы {nm} больше нет.'
    return (os.path.join(QDIR, nm) if e['годна'] else None), e, e.get('почему', '')


def qbase_of_x(x):
    m = re.search(r'--qbase\s+(\S+)', x or '')
    return m.group(1) if m else ''


def quant_job(kind, arg):
    # вэ1
    mp = os.path.join(S()['MODEL'], 'model.safetensors')
    if not os.path.exists(mp):
        return None, 'Нет исходной модели: база строится и сверяется по ней.'
    if not engine_fresh():
        return None, 'Движок не собран или собран из прежней версии.'
    if not model_info().get('движок'):
        return None, 'Движок не поддерживает эту модель.'
    if re.search(r'\s', QDIR):
        return None, f'В пути к папке баз есть пробел ({QDIR}).'
    os.makedirs(QDIR, exist_ok=True)
    free = shutil.disk_usage(QDIR).free
    if kind == 'квант':
        if arg not in RECIPES:
            return None, f'Неизвестный рецепт: {arg}.'
        need = recipe_bytes(arg)
        if not need:
            return None, 'В модели нет матриц слоёв.'
        if need + (1 << 30) > free:
            return None, f'На диске свободно {free / 2**30:.1f} ГиБ, требуется {need / 2**30:.1f} ГиБ.'
        return [engine(), '--quantize', mp, os.path.join(QDIR, arg + '.tq'), arg], ''
    src = os.path.abspath(os.path.expanduser(arg or ''))
    if not os.path.exists(src):
        return None, f'Файл не найден ({arg}).'
    if os.path.isfile(src) and not src.endswith(('.gguf', '.safetensors')):
        return None, 'Требуется файл .gguf, .safetensors или папка модели bitsandbytes.'
    if os.path.isfile(src) and src.endswith('.safetensors') and os.path.exists(os.path.join(os.path.dirname(src), 'model.safetensors.index.json')):
        src = os.path.dirname(src)                   # вэ2
    if os.path.realpath(src) == os.path.realpath(mp) or os.path.realpath(src) == os.path.realpath(S()['MODEL']):
        return None, 'Это исходная модель, а не квантованная база.'
    stem = re.sub(r'-\d{5}-of-\d{5}$', '', os.path.basename(src.rstrip('/')).rsplit('.', 1)[0] if os.path.isfile(src) else os.path.basename(src.rstrip('/')))
    out = os.path.join(QDIR, (re.sub(r'[^\w.+-]+', '_', stem) or 'база') + '.tq')
    try:
        size = os.path.getsize(src) if os.path.isfile(src) else sum(os.path.getsize(os.path.join(src, f)) for f in os.listdir(src) if f.endswith('.safetensors'))
    except OSError:
        size = 0
    if size + (1 << 30) > free:
        return None, f'На диске свободно {free / 2**30:.1f} ГиБ, требуется до {size / 2**30:.1f} ГиБ.'
    return [sys.executable, os.path.join(APP, 'quant.py'), 'import', src, out, mp, engine()], ''


def base_pick(nm):
    # вю1
    if nm and not next((b for b in base_list() if b['имя'] == nm and b['годна']), None):
        return False, 'База не подходит к текущей модели.'
    S_save({'QBASE': nm})
    return True, ''


def base_delete(nm):
    p = os.path.join(QDIR, os.path.basename(nm or ''))
    if not nm or not os.path.exists(p):
        return False, 'Такой базы нет.'
    if setup_alive() and setup_what.startswith(('квант', 'импорт')):
        return False, 'Выполняется квантование.'
    live = train_state()[1]
    if live and qbase_of_x(((run_choice(live) or {}).get('env') or {}).get('X')) == p:
        return False, f'На этой базе выполняется обучение «{live}».'
    if chat_alive() and chat_base == p:
        chat_stop()
    for f in (p, p + '.check'):
        try:
            os.remove(f)
        except OSError:
            pass
    if S()['QBASE'] == os.path.basename(p):
        S_save({'QBASE': ''})
    _bases.pop(os.path.basename(p), None)
    return True, ''


# вя1
chat_proc = None
chat_name = None         # вя2
chat_base = ''           # вя3
chat_info = {'готов': False, 'загрузка': '', 'ошибка': ''}
chat_ctx = 0            # вя4


def ctx_max():
    # га1
    try:
        c = json.load(open(os.path.join(S()['MODEL'], 'config.json'), encoding='utf-8'))
        c = dict(c, **(c.get('text_config') or {}))
        return max(256, min(32768, int(c.get('max_position_embeddings') or 8192)))
    except (OSError, ValueError, TypeError):
        return 8192


def kv_per_token():
    # гб1
    n = 0
    try:
        for line in open(model_info().get('описание') or '', encoding='utf-8'):
            a = line.split()
            if len(a) > 4 and a[0] == 'layer':
                n += int(a[3]) * int(a[4])          # гб2
    except (OSError, ValueError):
        n = 0
    if not n:
        n = sum(hd * kv for hd, kv, _ in lora_geom()[3])
    return 2 * n * 4


def ctx_default():
    # гв1
    kv, v = kv_per_token(), vram_info()
    free = (v.get('всего') or 0) - (v.get('занято') or 0)
    if not kv or free <= 0:
        return min(1024, ctx_max())
    return max(256, min(ctx_max(), free // 10 // kv // 256 * 256))


def gen_defaults():
    # гг1
    try:
        g = json.load(open(os.path.join(S()['MODEL'], 'generation_config.json'), encoding='utf-8'))
    except (OSError, ValueError):
        return {}
    out = {}
    if g.get('do_sample') is False:
        out['temp'] = 0.0
    elif isinstance(g.get('temperature'), (int, float)):
        out['temp'] = float(g['temperature'])
    for k, d in (('top_k', 'top_k'), ('top_p', 'top_p'), ('min_p', 'min_p'), ('repetition_penalty', 'повтор')):
        if isinstance(g.get(k), (int, float)):
            out[d] = g[k]
    return out
chat_subs = []
chat_lock = threading.Lock()


def chat_alive():
    return chat_proc is not None and chat_proc.poll() is None


def _chat_kill():
    # ге1
    global chat_proc, chat_name
    p, chat_proc, chat_name = chat_proc, None, None
    chat_info.update(готов=False, загрузка='', ошибка='')
    if not p or p.poll() is not None:
        return False
    try:
        os.killpg(p.pid, signal.SIGKILL)
    except OSError:
        try:
            p.kill()
        except OSError:
            pass
    return True


def chat_stop():
    # гж1
    with chat_lock:
        was = _chat_kill()
    if was:
        for q in list(chat_subs):
            q.put({'end': True, 'закрыт': True, 'остановлено': True})
    return was


def run_model_why(name):
    # гз1
    if not name:
        return ''
    mm = (run_choice(name) or {}).get('модель') or {}
    cur = model_info()
    if mm.get('параметров') and cur.get('параметров') and (mm['параметров'] != cur['параметров'] or
                                                           os.path.realpath(mm.get('папка') or '') != os.path.realpath(cur.get('папка') or '')):
        return f'Запуск «{name}» обучался на модели {mm.get("модель") or "?"} ({mm.get("папка")}), а выбрана {cur.get("модель")}: требуется вернуть папку модели в настройках.'
    return ''


def chat_blocked(name=''):
    # ги1
    why = run_model_why(name)
    if why:
        return why
    if chat_alive() or not train_state()[0]:
        return ''
    return 'Выполняется обучение: видеопамять занята им, поэтому разговор возможен после завершения или остановки обучения.'


def chat_start(name, ctx=None, base=None):
    # гк1
    global chat_proc, chat_name, chat_base, chat_ctx
    if base is None:
        qb = qbase()[0] or ''                        # гк2
    elif not base:
        qb = ''                                      # гк3
    else:
        e = next((b for b in base_list() if b['имя'] == base), None)
        if not e or not e.get('годна'):
            return False
        qb = os.path.join(QDIR, base)
    try:
        ctx = max(256, min(ctx_max(), int(ctx))) if ctx else (chat_ctx or ctx_default())
    except (TypeError, ValueError):
        ctx = chat_ctx or ctx_default()
    with chat_lock:
        if chat_alive() and chat_base == qb and chat_ctx == ctx and (chat_name == name or (name == '' and chat_name is not None)):
            return True
        _chat_kill()
        env = child_env(name or 'run')
        env['QBASE'] = qb
        env['CTX'] = str(ctx)
        chat_ctx = ctx
        try:
            p = subprocess.Popen([sys.executable, os.path.join(APP, 'chat.py'), '--serve'] + ([] if name else ['--base']), cwd=DATA,
                                 env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                 stderr=subprocess.DEVNULL, text=True, bufsize=1, start_new_session=True)
        except OSError:
            return False
        chat_proc, chat_name, chat_base = p, name, qb
        chat_info.update(готов=False, загрузка='', ошибка='')
        threading.Thread(target=chat_reader, args=(p,), daemon=True).start()
        return True


def chat_reader(p):
    for line in p.stdout:
        try:
            ev = json.loads(line)
        except ValueError:
            continue
        if p is not chat_proc:
            continue
        if ev.get('ready'):
            chat_info.update(готов=True, загрузка='')
        if ev.get('load'):
            chat_info['загрузка'] = ev['load']
        if ev.get('ошибка'):
            chat_info['ошибка'] = ev['ошибка']
        if ev.get('end') and ev.get('скорость'):
            speed_note(ev['скорость'])
        for q in list(chat_subs):
            q.put(ev)
    if p is chat_proc:                       # гл1
        for q in list(chat_subs):
            q.put({'end': True, 'закрыт': True, 'ошибка': chat_info.get('ошибка', '')})


# гм1
def conf_path():
    return os.environ.get('TIER_CONF') or os.path.join(os.path.expanduser('~'), '.config', 'tier.conf')


def conf_read():
    out = {}
    try:
        for line in open(conf_path(), encoding='utf-8'):
            k, _, v = line.partition(' ')
            if k and v:
                out[k.strip()] = v.strip()
    except OSError:
        pass
    return out


def conf_write(changes):
    # го1
    changes = dict(changes)
    if 'mem_cap_pct' in changes:   # го2
        try:
            changes['mem_cap_pct'] = str(min(99, max(50, int(float(str(changes['mem_cap_pct']).replace(',', '.'))))))
        except ValueError:
            changes.pop('mem_cap_pct')
    try:
        lines = open(conf_path(), encoding='utf-8').read().splitlines()
    except OSError:
        return False
    seen, out = set(), []
    for line in lines:
        k = line.split(' ', 1)[0].strip()
        out.append(f'{k} {changes[k]}' if k in changes else line)
        seen.add(k)
    out += [f'{k} {v}' for k, v in changes.items() if k not in seen]
    try:
        open(conf_path(), 'w', encoding='utf-8').write('\n'.join(out) + '\n')
        return True
    except OSError:
        return False


class Handler(http.server.BaseHTTPRequestHandler):
    protocol_version = 'HTTP/1.1'

    def log_message(self, *a):
        pass

    def send_bytes(self, b, ctype, code=200):
        self.send_response(code)
        self.send_header('Content-Type', ctype)
        self.send_header('Content-Length', str(len(b)))
        self.send_header('Cache-Control', 'no-store')
        self.end_headers()
        self.wfile.write(b)

    def send_json(self, obj, code=200):
        self.send_bytes(json.dumps(obj, ensure_ascii=False).encode(), 'application/json; charset=utf-8', code)

    def open_stream(self):
        self.send_response(200)
        self.send_header('Content-Type', 'text/event-stream; charset=utf-8')
        self.send_header('Cache-Control', 'no-store')
        self.send_header('Connection', 'close')
        self.end_headers()

    def push(self, obj):
        self.wfile.write(b'data: ' + json.dumps(obj, ensure_ascii=False).encode() + b'\n\n')
        self.wfile.flush()

    def stream_state(self, name):
        # гп1
        self.open_stream()
        seen, beat = None, 0.0
        try:
            while True:
                now = (log_stamp(name), train_state(), stamp(run_file(name, 'stage.txt')), stamp(run_file(name, 'lrfind.txt')),
                       stamp(run_file(name, 'load.txt')), stamp(run_file(name, 'stagep.txt')))   # гп2
                if now != seen or time.time() - beat > 10:
                    seen, beat = now, time.time()
                    self.push(read_log(name))
                time.sleep(0.3)
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass

    def stream_setup(self):
        # гп3
        self.open_stream()
        pos = 0
        try:
            while True:
                try:
                    with open(SETUP_LOG, 'rb') as f:
                        f.seek(pos)
                        chunk = f.read()
                        pos = f.tell()
                except OSError:
                    chunk = b''
                if chunk:
                    self.push({'текст': chunk.decode('utf-8', 'replace')})
                if not setup_alive():
                    _mods.clear()
                    self.push({'конец': True, 'код': setup_proc.returncode if setup_proc else 0,
                               'готовность': health()})
                    return
                time.sleep(0.3)
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass

    def stream_chat(self):
        q = queue.Queue()
        chat_subs.append(q)
        self.open_stream()
        try:
            while True:
                try:
                    self.push(q.get(timeout=10))
                except queue.Empty:
                    self.wfile.write(': ждём\n\n'.encode())
                    self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass
        finally:
            if q in chat_subs:
                chat_subs.remove(q)

    def do_GET(self):
        u = urllib.parse.urlparse(self.path)
        qs = urllib.parse.parse_qs(u.query)
        run = qs.get('run', [''])[0] or train_state()[1] or (runs() or ['run'])[-1]
        if u.path in ('/', '/index.html'):
            # гп4
            page = PAGE.replace('<html lang="ru">', '<html lang="ru" data-theme="light">', 1) \
                if S()['THEME'] == 'light' else PAGE
            try:                                       # гп5
                ap = json.dumps(json.loads(S()['APPEAR'] or '{}'), ensure_ascii=False)
            except ValueError:
                ap = '{}'
            page = page.replace('/*APPEAR*/{}', ap.replace('</', '<\\/'), 1)
            try:
                ui = json.dumps(json.loads(S()['UI'] or '{}'), ensure_ascii=False)
            except ValueError:
                ui = '{}'
            page = page.replace('/*UI*/{}', ui.replace('</', '<\\/'), 1)
            page = page.replace("/*LANG*/'ru'", json.dumps('en' if S()['LANG'] == 'en' else 'ru'), 1)
            self.send_bytes(page.encode(), 'text/html; charset=utf-8')
        elif u.path == '/bg':                          # гп6
            p = S()['BG']; mt = {'.png': 'image/png', '.jpg': 'image/jpeg', '.jpeg': 'image/jpeg', '.webp': 'image/webp',
                                 '.avif': 'image/avif', '.gif': 'image/gif'}.get(os.path.splitext(p)[1].lower())
            if p and mt and os.path.isfile(p):
                self.send_bytes(open(p, 'rb').read(), mt)
            else:
                self.send_error(404)
        elif u.path == '/logo.svg':
            self.send_bytes(open(os.path.join(APP, 'logo.svg'), 'rb').read(), 'image/svg+xml')
        elif u.path.startswith('/font/'):
            p = os.path.join(APP, 'fonts', os.path.basename(u.path))
            if p.endswith('.woff2') and os.path.isfile(p):
                self.send_bytes(open(p, 'rb').read(), 'font/woff2')
            else:
                self.send_error(404)
        elif u.path == '/api/stream':
            self.stream_state(run)
        elif u.path == '/api/state':
            self.send_json(read_log(run))
        elif u.path == '/api/health':
            self.send_json(health())
        elif u.path == '/api/setup/stream':
            self.stream_setup()
        elif u.path == '/api/chat/stream':
            self.stream_chat()
        elif u.path == '/api/settings':
            self.send_json({'свои': S(), 'железо': conf_read(),
                            'пути': {'программа': APP, 'данные программы': DATA, 'настройка оборудования': conf_path()}})
        else:
            self.send_error(404)

    def do_POST(self):
        n = int(self.headers.get('Content-Length', 0))
        try:
            body = json.loads(self.rfile.read(n) or b'{}')
        except ValueError:
            body = {}
        if self.path == '/api/start':
            self.send_json(dict(zip(('ок', 'почему'), start_from_window(body))))
        elif self.path == '/api/plan':
            self.send_json({'план': plan(DRAFT, mode_key(body.get('режим')), body.get('свои'), body.get('что'), body.get('лора')),
                            'подготовка': prep_state(DRAFT), 'имя': name_state(body.get('имя') or ''),
                            'лора': lora_view(body.get('лора'))})
        elif self.path == '/api/prepare':
            self.send_json(dict(zip(('ок', 'почему'), prep_start(DRAFT, body.get('датасет') or '', body.get('сцена')))))
        elif self.path == '/api/stop':
            self.send_json({'ок': train_stop()})
        elif self.path == '/api/kill':                 # гп7
            ok = train_stop() if train_state()[0] else True
            chat_stop()
            for pr in [setup_proc] + [j['p'] for j in prep_jobs.values()]:
                try:
                    if pr and pr.poll() is None:
                        pr.kill()
                except OSError:
                    pass
            self.send_json({'ок': ok})
            threading.Timer(.3, lambda: os._exit(0)).start()
        elif self.path == '/api/setup':
            ok, why = setup_start(body.get('что') or '')
            self.send_json({'ок': ok, 'почему': why})
        elif self.path == '/api/base':
            ok, why = (base_delete if body.get('удалить') else base_pick)(body.get('имя') or '')
            self.send_json({'ок': ok, 'почему': why, 'готовность': health()})
        elif self.path == '/api/settings':
            S_save(body.get('свои') or {})
            hw = {k: v for k, v in (body.get('железо') or {}).items()
                  if k in ('headroom_mib', 'tier1_params', 'bf16x3', 'prefetch', 'mem_cap_pct') and str(v).strip()}
            self.send_json({'ок': conf_write(hw) if hw else True, 'готовность': health()})
        elif self.path == '/api/chat':
            nm, base = body.get('имя') or '', bool(body.get('base'))
            if nm and not os.path.exists(run_file(nm, 'step.txt')):   # гп8
                self.send_json({'ок': False, 'почему': f'У запуска «{nm}» нет снимка весов: обучение не дошло до первого сохранения.'})
                return
            if not nm and not base:
                self.send_json({'ок': False, 'почему': 'Модель не выбрана.'})
                return
            if not os.path.exists(engine()) or not os.path.exists(os.path.join(S()['MODEL'], 'model.safetensors')):
                self.send_json({'ок': False, 'почему': 'Нет движка или весов модели.'})
                return
            why = chat_blocked(nm)
            if why or not chat_start(nm, body.get('ctx')):
                self.send_json({'ок': False, 'почему': why or 'Не удалось открыть разговор.'})
                return
            try:
                chat_proc.stdin.write(json.dumps({k: body[k] for k in ('q', 'new', 'base', 'temp', 'max', 'сцена', 'сила', 'top_k', 'top_p', 'min_p', 'повтор', 'зерно') if k in body},
                                                 ensure_ascii=False) + '\n')
                chat_proc.stdin.flush()
                self.send_json({'ок': True, 'имя': chat_name})
            except (BrokenPipeError, AttributeError, OSError):
                self.send_json({'ок': False, 'почему': 'Разговор закрыт.'})
        elif self.path == '/api/chat/stop':
            chat_stop()
            self.send_json({'ок': True})
        elif self.path == '/api/runs/delete':
            self.send_json(run_delete(body.get('имя') or ''))
        elif self.path == '/api/peek':
            self.send_json(read_examples(body.get('путь') or '', max(0, int(body.get('с') or 0)), min(50, max(1, int(body.get('n') or 5)))))
        elif self.path == '/api/questions':
            self.send_json(questions(body.get('путь') or ''))
        elif self.path == '/api/bench/save':
            self.send_json(bench_save(body.get('имя') or '', body.get('заголовок') or [], body.get('строки') or []))
        else:
            self.send_error(404)


def name_state(name):
    ok, why = name_ok(name)
    if ok and os.path.exists(log_path(name)):
        ok, why = False, f'Запуск «{name}» уже есть.'
    return {'ок': ok, 'почему': why}


def start_from_window(body):
    name, mode = body.get('имя') or '', mode_key(body.get('режим'))
    if body.get('продолжить'):
        env = resume_env(name)
        if env is None:
            return False, 'Нет записи о запуске: продолжать нечего.'
        if run_model_why(name):
            return False, run_model_why(name)
        qp = qbase_of_x(env.get('X'))
        if qp and not os.path.exists(qp):
            return False, f'Запуск обучался на базе {os.path.basename(qp)}, но этой базы больше нет.'
        return train_start(name, '', env)
    ns = name_state(name)
    if not ns['ок']:
        return False, ns['почему']
    if train_state()[0]:
        return False, 'Уже выполняется другой запуск.'
    ps = prep_state(DRAFT)
    if ps['идёт'] or not ps['сводка']:
        return False, 'Датасет ещё не токенизирован.'
    if not ps['свежий']:
        return False, 'Файл датасета изменился после токенизации, требуется повторная токенизация.'
    pl = plan(DRAFT, mode, body.get('свои'), body.get('что'), body.get('лора'))
    if not pl['ок']:
        return False, pl['почему']
    for suf in DRAFT_FILES:                      # гс1
        try:
            os.replace(run_file(DRAFT, suf), run_file(name, suf))
        except OSError:
            pass
    pl = plan(name, mode, body.get('свои'), body.get('что'), body.get('лора'))
    if not pl['ок']:
        return False, pl['почему']
    ok, why = train_start(name, '', pl['env'])
    if ok:
        mi = model_info()
        json.dump({'режим': pl['режим'], 'режим_имя': MODE[pl['режим']][0], 'свои': body.get('свои') or {},
                   'что': pl['что'], 'лора': pl['лора'], 'база': os.path.basename(qbase_of_x(pl['env'].get('X'))),
                   'env': pl['env'], 'строки': pl['строки'], 'шагов': pl['шагов'], 'начало': time.strftime('%F %T'),
                   'датасет': (ps['источник'] or {}).get('путь', ''),
                   'модель': {k: mi.get(k) for k in ('модель', 'вид', 'основной', 'параметров', 'эмбеддинги', 'папка', 'семья')}},
                  open(run_file(name, 'run.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    return ok, why


def watch_runs():
    # гт1
    was = train_state()[1]
    while True:
        time.sleep(2)
        now = train_state()[1]
        if was and now != was and ui_pref('notify', 1):
            d = read_log(was)
            best = [e['loss'] for e in d['оценки'] if e['loss'] is not None]
            st = d['состояние']
            if st != 'остановлено':
                t = ('Завершено' + (f' · лучшая потеря (loss) {min(best):.4f}' if best else '')) if st == 'завершено' else 'Ошибка: ' + (d['почему'] or 'причина указана в журнале')[:160]
                notify(f'Запуск «{was}»: {t}', st == 'ошибка')
        was = now


def notify(text, urgent=False):
    if NOTIFY:
        try:
            NOTIFY('TierLM', text, urgent)
            return
        except Exception:
            pass
    if shutil.which('notify-send'):
        subprocess.Popen(['notify-send', '-a', 'TierLM', '-i', os.path.join(APP, 'logo.svg')] + (['-u', 'critical'] if urgent else []) +
                         ['TierLM', text], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def serve(port=0):
    # гф1
    threading.Thread(target=watch_runs, daemon=True).start()
    socketserver.ThreadingTCPServer.allow_reuse_address = True
    socketserver.ThreadingTCPServer.daemon_threads = True
    srv = socketserver.ThreadingTCPServer(('127.0.0.1', port), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, f'http://127.0.0.1:{srv.server_address[1]}/'



def main():
    a = sys.argv[1:]
    cmd = a[0] if a else 'окно'
    if cmd in ('--version', '-v', 'версия'):
        print('TierLM', VERSION)
    elif cmd in ('учить', 'train'):
        # гх1
        name = os.environ.get('NAME', 'run')
        mode = 'авто'                                # гх2
        goal = next((x for x in a[1:] if x in GOALS or x == 'lora'), 'lora' if os.environ.get('LORA') else None)
        files = [x for x in a[1:] if x not in MODE and x not in OLD_MODES and x not in GOALS and x != 'lora']
        env = child_env(name)
        if os.path.exists(log_path(name)):
            if files:
                sys.exit(f'Запуск «{name}» уже есть. Без указания файла он продолжается; для новых данных требуется другое имя (NAME=...).')
            extra = resume_env(name)
            if extra is None:
                sys.exit('Нет записи о запуске: продолжать нечего.')
        else:
            if files and subprocess.run([sys.executable, os.path.join(APP, 'prepare.py'), os.path.abspath(files[0])],
                                        cwd=DATA, env=env).returncode:
                sys.exit(1)
            pl = plan(name, mode, goal=goal, lora=os.environ.get('LORA'))
            if not pl['ок']:
                sys.exit(pl['почему'] + (' Требуется указать файл датасета.' if not files else ''))
            print(f'Режим «{MODE[mode][0]}»:')
            for row in pl['строки']:
                if os.environ.get('MATS') and (row['что'] == 'Что учится' or row['к'] == 'слои правки'):
                    continue
                print(f"  {row['что']}: {row['знач']}" + (f" — {row['как']}" if row['как'] else ''))
            if os.environ.get('MATS'):
                print(f"  Что учится: целиком {os.environ['MATS']}, остальное заморожено")
            extra = pl['env']
        extra.update({k: os.environ[k] for k in ('LR', 'EPOCHS', 'PACK', 'EV', 'FL', 'STOP_AFTER', 'X', 'REPLAY', 'MATS', 'NSGD', 'POSDW', 'WHITEN', 'WHITEN_FRAC') if k in os.environ})
        env.update(extra)
        os.execvpe('bash', ['bash', os.path.join(APP, 'auto.sh')], env)
    elif cmd in ('говорить', 'chat'):
        os.chdir(DATA)
        env = child_env(os.environ.get('NAME', 'run'))
        env.setdefault('QBASE', qbase()[0] or '')      # гх3
        os.execvpe(sys.executable, [sys.executable, os.path.join(APP, 'chat.py')] + a[1:], env)
    elif cmd in ('квант', 'quantize'):
        # гх4
        if len(a) < 2:
            sys.exit('Формат команды: tierlm квант <' + '|'.join(RECIPES) + ' | файл.gguf | папка bitsandbytes>')
        cmd_, why = quant_job('квант' if a[1] in RECIPES else 'импорт', a[1])
        if not cmd_:
            sys.exit(why)
        os.chdir(DATA)
        os.nice(10)
        os.execvpe(cmd_[0], cmd_, child_env())
    elif cmd in ('база', 'base'):
        # гх5
        if len(a) > 1:
            nm = '' if a[1].upper() == 'BF16' else a[1] if a[1].endswith('.tq') else a[1] + '.tq'
            ok, why = base_pick(nm)
            if not ok:
                sys.exit(why)
        cur = S()['QBASE']
        print(('* ' if not cur else '  ') + f'BF16  как в файле, {layer_bf16_bytes() / 2**30:.1f} ГиБ')
        for b in base_list():
            err = f', ошибка весов {100 * b["ошибка"]:.1f} %' if b.get('ошибка') is not None else ''
            print(('* ' if b['имя'] == cur else '  ') + f'{b["имя"][:-3]}  {b.get("откуда", "")}, {b["байт"] / 2**30:.1f} ГиБ{err}' +
                  (f'  — не подходит: {b["почему"]}' if not b.get('годна') else ''))
    elif cmd in ('собрать', 'build'):
        os.chdir(DATA)
        os.execvpe('bash', ['bash', os.path.join(APP, 'build.sh')], child_env())
    elif cmd in ('сервер', 'server'):
        srv, url = serve(int(os.environ.get('PORT', '8800')))
        print('Панель:', url)
        try:
            threading.Event().wait()
        except KeyboardInterrupt:
            pass
    elif cmd in ('окно', 'window'):
        # гх6
        try:
            import ui
        except ImportError as e:
            sys.exit(f'Для окна требуется PySide6 (Qt 6 с Qt Quick): {e.name or e}.\n'
                     '  Установка в Arch: sudo pacman -S pyside6 qt6-declarative qt6-svg\n'
                     '  Установка в Debian/Ubuntu: sudo apt install python3-pyside6.qtquick python3-pyside6.qtsvg qml6-module-qtquick-layouts '
                     'qml6-module-qtquick-window\n'
                     '  Режим без окна: tierlm сервер (панель открывается в браузере).')
        import atexit
        try:
            open(UI_PID, 'w').write(str(os.getpid()))
            atexit.register(lambda: os.path.exists(UI_PID) and open(UI_PID).read().strip() == str(os.getpid()) and os.remove(UI_PID))
        except OSError:
            pass
        if not ui.run(sys.modules[__name__]):
            sys.exit('Окно не открылось: Qt не загрузил ui.qml, подробности выше. Режим без окна: tierlm сервер.')
    else:
        print('Формат: tierlm [окно | учить <файл> [авто] [срез|точно|характер|lora] | говорить | квант <рецепт|файл> | база [имя|BF16] | '
              'собрать | сервер | --version]')
        sys.exit(1)


PAGE = r"""<!doctype html>
<html lang="ru"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>TierLM</title>
<link rel="icon" href="/logo.svg">
<script>window.LANG=(/*LANG*/'ru');
// Английский: страница пишется по-русски, а перевод накладывается на текст при появлении — словарём кусков, длинные сначала
if(window.LANG==='en'){ document.documentElement.lang='en';
  const D={"Предел кристалла": "Die limit", "Шагов выше": "Steps above", "Выше ": "Above ", "Новое обучение": "New run", "Запуски": "Runs", "Разговор": "Chat", "Проверка": "Evaluation", "Настройки": "Settings", "Обучение": "Training", "Модель": "Model", "Сегодня": "Today", "Вчера": "Yesterday", "КиБ": "KiB", "МиБ": "MiB", "ГиБ": "GiB", "ГБ": "GB", "Б": "B", "вопрос": "question", "Вопрос": "Question", "вопроса": "questions", "Вопроса": "Questions", "вопросов": "questions", "Вопросов": "Questions", "Вопросы": "Questions", "· По одному на строку": "· One per line", "Один вопрос на строку. Ожидаемый ответ — после « || », по нему ищется совпадение": "One question per line. The expected answer goes after « || » and is used for matching", "Вопрос || Ожидаемый ответ": "Question || Expected answer", "Системная строка": "System prompt", "Сообщение": "Message", "Поиск": "Search", "Сцена": "System prompt", "Путь к .jsonl или .txt": "Path to .jsonl or .txt", "/путь/к/папке": "/path/to/folder", "Команда, раздел или запуск": "Command, section or run", "Команды": "Commands", "Команды (Ctrl+K)": "Commands (Ctrl+K)", "Команды и поиск": "Commands and search", "Предыдущий": "Previous", "Следующий": "Next", "Разделы": "Sections", "Разделы настроек": "Settings sections", "Разделы по порядку": "Sections in order", "подробности, клавиша": "details, key", "Подробности, клавиша": "Details, key", "Клавиша": "Key", "Клавиши": "Keys", "Отправить · Enter": "Send · Enter", "Новая строка · Shift+Enter": "New line · Shift+Enter", "Отправить": "Send", "Удалить": "Delete", "Удалить набор": "Delete preset", "Удалить запуски:": "Delete runs:", "Удалено": "Deleted", "Удалено не всё": "Not everything was deleted", "Освобождено": "Freed", "Освободится": "Will free", "не удалось": "failed", "Не удалось": "Failed", "Ниже": "Down", "Выше": "Up", "Выберите 2–6 запусков": "Select 2–6 runs", "Сравнить": "Compare", "Сравнение": "Comparison", "Сравнение с исходной": "Compare with base", "Сравнение с исходной: каждый вопрос — обученной и исходной модели, с чистого листа": "Compare with base: each question goes to the trained and the base model, from a clean context", "Сравнить с исходной моделью": "Compare with the base model", "Исходная модель": "Base model", "Исходная": "Base", "Обученная": "Trained", "Обученная против исходной": "Trained vs base", "У исходной": "Base model", "Без обучения": "Untrained", "Не загружена": "Not loaded", "Загружена": "Loaded", "Загружается": "Loading", "Модель в видеопамяти": "Model in VRAM", "Веса грузятся при первом вопросе": "Weights load on the first question", "Новый разговор": "New chat", "Параметры": "Parameters", "Параметр": "Parameter", "Модели": "Models", "Длина ответа": "Answer length", "Длина ответа, токенов": "Answer length, tokens", "Температура": "Temperature", "Сила обучения": "Training strength", "1 — как обучено, 0 — исходная модель, 1.5 — резче": "1 as trained, 0 base model, 1.5 sharper", "1 — как обучено, 0 — исходная": "1 as trained, 0 base", "0 — всегда самый вероятный ответ": "0 always picks the most likely answer", "Токенов": "Tokens", "Разговор по умолчанию": "Chat defaults", "Проверка вопросами": "Question evaluation", "Запустить проверку": "Run evaluation", "Из данных запуска": "From the run data", "Из данных": "From data of", "Из файла": "From a file", "Свой список": "Custom list", "Свой список пуст": "Custom list is empty", "Нет результатов": "No results", "Итог": "Result", "Копировать": "Copy", "Сохранить таблицу": "Save table", "Таблица скопирована": "Table copied", "Таблица сохранена": "Table saved", "Не сохранилось": "Not saved", "Совпало с ожидаемым": "Matched expected", "Совпало": "Matched", "Ожидается": "Expected", "Ответ изменился": "Answer changed", "✓ Совпало": "✓ Matched", "✗ Не совпало": "✗ No match", "= Как у исходной": "= Same as base", "≠ Изменился": "≠ Changed", "Щелчок — отметить вручную": "Click to mark manually", "Идёт": "Running", "Проверка завершена": "Evaluation finished", "Проверка прервана": "Evaluation interrupted", "Проверка не началась": "Evaluation did not start", "Нет вопросов": "No questions", "не выбран файл с вопросами": "no question file selected", "Не выбран файл с вопросами": "No question file selected", "у запуска нет записи о файле данных — выберите файл": "the run has no record of its data file; choose a file", "У запуска нет записи о файле данных — выберите файл": "The run has no record of its data file; choose a file", "файл не читается": "file is unreadable", "Файл не читается": "File is unreadable", "движок закрылся": "engine closed", "Движок закрылся": "Engine closed", "отменено": "cancelled", "Отменено": "Cancelled", "нет связи с программой": "no connection to the program", "Нет связи с программой": "No connection to the program", "нет связи": "no connection", "Нет связи": "No connection", "Движок завершился при загрузке — подробно в": "The engine exited while loading; details in", "Ошибка": "Error", "Остановить": "Stop", "Остановка…": "Stopping", "Остановить «": "Stop «", "Обучение остановлено": "Training stopped", "Обучение продолжено": "Training resumed", "Обучение запущено": "Training started", "Обучение остановилось с ошибкой": "Training stopped with an error", "Сохранений ещё нет: продолжение начнётся с шага 1.": "No weight snapshots yet: resuming starts from step 1.", "Сохранений нет": "No weight snapshots", "Продолжение с шага": "Resume from step", "Продолжить с шага": "Resume from step", "Начать заново": "Start over", "Веса сохранены до шага": "Weights saved up to step", "Лучшая": "Best", "на шаге": "at step", "На шаге": "At step", "Осталось": "Left", "Прошло": "Elapsed", "Завершается": "Finishing", "Повтор: разметка ответов": "Base-answer replay: labeling answers", "Подбор шага: загрузка весов": "Learning-rate search: loading weights", "Загрузка весов в ОЗУ": "Loading weights into RAM", "Точка": "Point", "Упаковка": "Packing", "По одному примеру": "One example at a time", "Оценка каждые": "Evaluate every", "Сброс каждые": "Weight snapshot every", "Выкл": "Off", "оценок без улучшения": "evaluations without improvement", "Оценок без улучшения": "Evaluations without improvement", "Всё, что задано": "Everything set", "Что было задано": "What was set", "Запуск": "Run", "Режим": "Mode", "Журнал": "Log", "Журнал запуска": "Run log", "Время": "Time", "Событие": "Event", "Что произошло": "What happened", "Вывод программы": "Program output", "Вывода пока нет": "No output yet", "Событий пока нет": "No events yet", "Числа появятся после первых шагов": "Numbers appear after the first steps", "Запуск из терминала: записи о выборе нет": "Started from the terminal: no record of the choices", "Без сглаживания: виден разброс между примерами.": "Unsmoothed: shows the spread between examples.", "Доля угаданных токенов ответа.": "Share of answer tokens guessed right.", "Пик за шаг; пунктир — потолок карты.": "Peak per step; dashed line is the card memory limit.", "Самая горячая точка кристалла, память и край.": "Hottest spot of the die, memory and edge.", "Обороты в минуту; ноль на холодной карте — норма.": "RPM; zero on a cold card is normal.", "Насколько сильно веса просятся измениться.": "A measure of how strongly the weights need to change.", "Разгон в начале, затем плавный спуск.": "Warm-up first, then a smooth decay.", "не показаны": "not shown", "Не показаны": "Not shown", "Длина примеров": "Example length", "Что внутри": "What is inside", "Вентилятор": "Fan", "Шаг обучения": "Learning rate", "Всего": "Total", "Перплексия": "Perplexity", "Падение за 50 шагов": "Drop over 50 steps", "Разница средних": "Difference of means", "Среднее за 50 шагов": "Mean over 50 steps", "Среднее первых 10 шагов": "Mean of the first 10 steps", "Последние 50 шагов": "Last 50 steps", "За весь прогон": "Whole run", "Втрое выше среднего за 50 шагов": "Three times above the 50-step mean", "Упала в": "Dropped by", "Предел кристалла ≈ 110°": "Die limit ≈ 110°", "Меньше 200 — риск падения": "Below 200 risks a crash", "Тензоров постоянно": "Resident tensors", "Кэш в ОЗУ": "Cache in RAM", "На диске": "On disk", "Самый долгий": "Slowest", "Пик": "Peak", "Живых": "Live", "Блоков": "Blocks", "Область": "Area", "Мимо": "Outside", "Память": "Memory", "Край": "Edge", "Норма": "Norm", "Максимум": "Maximum", "На отложенных": "Held-out", "На обучении": "Training", "Замеров": "Measurements", "Готово": "Done", "Нужно действие": "Action needed", "Нет данных": "No data", "Разбор": "Parsing", "Готова": "Ready", "Нет весов": "No weights", "Не поддерживается": "Not supported", "Пока не поддерживается": "Not supported yet", "Вид модели в файле": "Model type in the file", "Только чтение": "Read-only", "Программа только читает этот файл": "The program only reads this file", "Изменить папку": "Change folder", "Указать папку": "Choose folder", "Другая папка": "Another folder", "Скачать ≈ 22 ГБ": "Download ≈ 22 GB", "Модель не найдена": "Model not found", "нет model.safetensors": "no model.safetensors", "Нет model.safetensors": "No model.safetensors", "В папке": "The folder", "Появится, когда найдётся файл весов.": "Appears once the weights file is found.", "параметров": "parameters", "Параметров": "Parameters", "слоёв": "layers", "Слоёв": "Layers", "Словарь": "Vocabulary", "Вид обучения": "Training type", "Учится": "Trained", "С Adam в видеопамяти": "With Adam in VRAM", "Вес и два момента": "Weight and two moments", "Заморожено в ОЗУ": "Frozen in RAM", "BF16, как в файле": "BF16, as in the file", "По шине за шаг": "Over the bus per step", "По шине": "Over the bus", "Срез FP32": "FP32 row slice", "Смесь экспертов: учится срез каждого эксперта, по шине едут только выбранные маршрутизатором.": "Mixture of experts: a row slice of every expert is trained; only router-picked experts travel over the bus.", "Квантованная: замороженная часть хранится сжатой — меньше ОЗУ и шины; обучаемый срез остаётся FP32.": "Quantized: the frozen part is stored compressed, less RAM and bus traffic; the trained row slice stays FP32.", "Движок пока учит только плотные модели": "The engine trains only dense models so far", "Движок пока учит только поверх BF16; таблица — что даст сжатие на этой машине": "The engine trains only on top of BF16 so far; the table shows what compression would give on this machine", "Движок пока не умеет этот вид": "The engine does not support this type yet", "Пока недоступно": "Not available yet", "Учить как Dense": "Train as Dense", "Что сжимать": "What to compress", "Только MLP": "MLP only", "Все слои": "All layers", "Формат": "Format", "Бит": "Bits", "В ОЗУ": "In RAM", "Шина ≥": "Bus ≥", "Теоретический предел": "Theoretical limit", "этой карте": "this card", "Этой карте": "This card", "Нативно": "Native", "Распаковкой": "Unpacked", "Влезет": "Fits", "Мало": "Too little", "сейчас": "now", "Сейчас": "Now", "Есть": "Available", "Режим «Эксперт»: правка каждого числа": "Expert mode: edit every number", "Вернуть числа автопилота": "Restore the values picked by auto-tuning", "Все поля плана": "All plan fields", "Ключи движка": "Engine flags", "Оценок": "Evaluations", "Эпох": "Epochs", "Стоп": "Stop", "Шагов": "Steps", "Шага": "Steps", "Шаг": "Step", "Эпоха": "Epoch", "Эпохи": "Epochs", "Оценка": "Evaluation", "Оценки": "Evaluations", "Что обучаем": "What to train", "Данные": "Data", "Файл с примерами": "Examples file", "Имя запуска": "Run name", "Выбрать…": "Choose", "Неверно": "Invalid", "Примеров": "Examples", "На обучение": "For training", "Отложено": "Held out", "Длина, медиана": "Length, median", "Длина, p99": "Length, p99", "Самый длинный": "Longest", "Размер файла": "File size", "ток.": "tok.", "Ток.": "Tok.", "Примеры из файла": "Examples from the file", "Пример": "Example", "Инструкция": "Instruction", "Ввод": "Input", "Ответ": "Answer", "Проверки": "Checks", "План": "Plan", "Начать обучение": "Start training", "Не задано": "Not set", "Не выбраны": "Not selected", "Повторить": "Retry", "Собран": "Built", "Собран из прежней версии": "Built from the previous version", "Не собран": "Not built", "Нет набора разработки": "No development kit", "Видеокарта не найдена": "No GPU found", "Сборка под установленную карту, несколько минут": "Build for the installed card, a few minutes", "Собрать": "Build", "Установка пакетов": "Installing packages", "Нет пакета tokenizers": "No tokenizers package", "Поставить": "Install", "Нет tokenizer.json": "No tokenizer.json", "Скачать": "Download", "Защита общих знаний": "Knowledge protection", "Нет пакета numpy": "No numpy package", "Карта": "Card", "Занята": "Busy", "Перейти": "Go", "ОЗУ": "RAM", "нужных": "needed", "Нужных": "Needed", "Замороженное BF16": "Frozen BF16", "эмбеддинги": "embeddings", "Эмбеддинги": "Embeddings", "система ≈ 6 ГиБ": "system ≈ 6 GiB", "Система ≈ 6 ГиБ": "System ≈ 6 GiB", "Видеопамять": "VRAM", "Проверка при запуске": "Checked at start", "Свободно": "Free", "Диск": "Disk", "ещё": "more", "Ещё": "More", "Сборка движка": "Building the engine", "Не получилось": "Failed", "Код": "Code", "Ошибка (код": "Error (code", "причина выше": "reason above", "Причина выше": "Reason above", "Запуск…": "Starting", "Папка модели": "Model folder", "Где лежит model.safetensors": "Where model.safetensors is", "Где лежит model.safetensors. Программа только читает его": "Where model.safetensors is. The program only reads it", "Рабочая папка": "Work folder", "Журналы, токены, обученные веса": "Logs, tokens, trained weights", "Откуда скачивать веса": "Where to download weights from", "Программа": "Program", "Сохранить": "Save", "Сохранено": "Saved", "Действует со следующего запуска движка": "Applies from the next engine start", "Не записалось": "Not written", "Загрузка…": "Loading", "Машина": "Machine", "Доступно": "Available", "Шина": "Bus", "Свободно из": "Free of", "Запас видеопамяти, МиБ": "VRAM reserve, MiB", "Видеопамять под промежуточные данные шага": "VRAM for intermediate step data", "Бюджет среза": "Row slice budget", "Обучаемых чисел в видеопамяти: вес и два момента Adam — 12 байт на число": "Trained numbers in VRAM: weight and two Adam moments, 12 bytes per number", "Три умножения BF16 вместо одного FP32: быстрее на картах без быстрого FP32": "Three BF16 multiplies instead of one FP32: faster on cards without fast FP32", "Следующий слой едет по шине, пока считается текущий": "The next layer travels over the bus while the current one computes", "Движок": "Engine", "Состояние": "State", "Пересобрать": "Rebuild", "Нет компилятора": "No compiler", "Несколько минут": "A few minutes", "Браузер": "Browser", "Связь с автором": "Contact the author", "ВКонтакте": "VK", "Шрифты": "Fonts", "Оформление": "Appearance", "Графики": "Charts", "Поведение": "Behavior", "Оборудование": "Hardware", "Пути": "Paths", "О программе": "About", "Наборы": "Presets", "Сохранить текущее": "Save current", "Экспорт": "Export", "Импорт": "Import", "Цвет": "Color", "Тема": "Theme", "Тёмная": "Dark", "Светлая": "Light", "Акцент": "Accent", "Свой цвет": "Custom color", "Фон": "Background", "Картинка": "Image", "Другой файл": "Another file", "Выбрать файл": "Choose file", "Убрать": "Remove", "Затемнение картинки": "Image dimming", "Размытие картинки": "Image blur", "Поверхности": "Surfaces", "Непрозрачность": "Opacity", "Карточки и панели": "Cards and panels", "Размытие под стеклом": "Glass blur", "Тени": "Shadows", "Нет": "None", "Мягкие": "Soft", "Глубокие": "Deep", "Линии": "Lines", "Тонкие": "Thin", "Обычные": "Regular", "Чёткие": "Crisp", "Скругление": "Corner radius", "Текст": "Text", "Шрифт": "Font", "Размер": "Size", "Начертание": "Weight", "Лёгкое": "Light", "Обычное": "Regular", "Плотное": "Medium", "Раскладка": "Layout", "Плотность": "Density", "Плотно": "Compact", "Просторно": "Spacious", "Панель": "Sidebar", "Слева": "Left", "Справа": "Right", "Ширина панели": "Sidebar width", "Ширина содержимого": "Content width", "Ограничена": "Limited", "Во всю ширину": "Full width", "С зазором": "With a gap", "Вплотную": "Edge to edge", "Окно, развёрнутое на весь экран": "A window maximized to the full screen", "Анимации": "Animations", "Свой CSS": "Custom CSS", "Очистить": "Clear", "Переменные:": "Variables:", "Вернуть оформление по умолчанию": "Restore default appearance", "Сбросить": "Reset", "Сбросить оформление?": "Reset appearance?", "Свои наборы останутся.": "Custom presets are kept.", "Сохранить набор": "Save preset", "Название": "Name", "Набор сохранён": "Preset saved", "Оформление скопировано": "Appearance copied", "Импорт оформления": "Import appearance", "Вставьте JSON, скопированный кнопкой «Экспорт»": "Paste the JSON copied with “Export”", "Применить": "Apply", "Оформление применено": "Appearance applied", "Не JSON": "Not JSON", "Нужна картинка": "An image is needed", "PNG, JPG, WebP, AVIF или GIF": "PNG, JPG, WebP, AVIF or GIF", "свой": "custom", "Свой": "Custom", "Свой набор": "Custom preset", "Набор": "Preset", "Ночь": "Night", "Океан": "Ocean", "Графит": "Graphite", "Угли": "Embers", "Терминал": "Terminal", "Контраст": "Contrast", "Пудра": "Powder", "Бумага": "Paper", "Туман": "Mist", "Лаванда": "Lavender", "Сияние": "Aurora", "Рассвет": "Dawn", "Тушь": "Ink", "Ровный": "Plain", "Системный": "System", "Моноширинный": "Monospace", "Плитки на «Ходе обучения»": "Tiles on “Training progress”", "Плитки графиков на «Ходе обучения»": "Chart tiles on “Training progress”", "Колонок": "Columns", "Сами": "Auto", "Высота графика": "Chart height", "Толщина линии": "Line width", "Заливка под кривой": "Fill under the curve", "Сетка": "Grid", "Стартовый экран": "Start screen", "Сам": "Auto", "«Сам» — обучение, если идёт": "“Auto”: training if a run is going", "Уведомление по окончании": "Notify when finished", "На рабочий стол — и когда окно закрыто": "To the desktop, also when the window is closed", "В трей при закрытии": "Close to tray", "Пока идёт обучение": "While training runs", "Подтверждать остановку": "Confirm stop", "Десятичный знак": "Decimal separator", "Сменить тему": "Toggle theme", "Поиск по запускам": "Search runs", "Закрыть окно или диалог": "Close a window or dialog", "Завершить всё": "Quit everything", "Завершить": "Quit", "Завершить всё?": "Quit everything?", "Программа закроется полностью.": "The program will close completely.", "будет остановлено, программа закроется полностью.": "will be stopped and the program will close completely.", "Будет остановлено, программа закроется полностью.": "Will be stopped and the program will close completely.", "TierLM завершён": "TierLM has quit", "Действия": "Actions", "Светлая тема": "Light theme", "Тёмная тема": "Dark theme", "Русский язык": "Russian", "Ничего не найдено": "Nothing found", "Запусков пока нет": "No runs yet", "Все": "All", "Идут": "Running", "Завершены": "Finished", "Остановлены": "Stopped", "Ошибки": "Errors", "Запусков": "Runs", "Занимают": "Take up", "Имя": "Name", "Способ": "Method", "Когда": "When", "Открыть": "Open", "Выбрать все": "Select all", "Удалить «": "Delete «", "Журнал, токены и обученные веса": "Log, tokens and trained weights", "Без возврата.": "Cannot be undone.", "Исходная модель не затрагивается.": "The base model is not touched.", "Остальное — вопрос": "The rest is the question", "Потеря на отложенных": "Held-out loss", "Секунд на шаг": "Seconds per step", "Потеря": "Loss", "Точность на ответе": "Answer accuracy", "Длительность шага": "Step duration", "Норма градиента": "Gradient norm", "Потолок": "Ceiling", "Из": "Of", "Не начинался": "Not started", "Завершено": "Finished", "Остановлено": "Stopped", "Ошибка запуска": "Start error", "Ошибка на шаге": "Error at step", "Остановлено на шаге": "Stopped at step", "Параметры запуска": "Run parameters", "Срез по всей сети": "Row slice across the network", "Точно: факты": "Precise: facts", "Характер: голос и манера": "Character: voice and manner", "Срез": "Row slice", "Точно": "Precise", "Характер": "Character", "Токены": "Tokens", "Повтор": "Base-answer replay", "Не было": "Skipped", "Задан": "Set", "Выключен": "Off", "Не нужно": "Not needed", "Срез по видеопамяти": "Row slice by VRAM", "к данным": "to data", "К данным": "To data", "Целиком": "Whole", "Целиком:": "Whole:", "40 шагов": "40 steps", "Что учится": "What is trained", "Строю раскладку среза": "Building the slice layout", "раскладка записана": "layout written", "Раскладка записана": "Layout written", "строк каждой матрицы": "rows of every matrix", "Строк каждой матрицы": "Rows of every matrix", "матриц": "matrices", "Матриц": "Matrices", "Разговор с": "Chat with", "Нужно": "Needed", "Кривая": "Curve", "Свободно 18.2 ГиБ из 252 ГиБ": "Free 18.2 GiB of 252 GiB", "Сброс оформления": "Reset appearance", "Вид графиков": "Chart style", "Скачивание весов": "Downloading weights", "Имя занято": "Name taken", "на стенде (30 правок) перенос на другие формулировки 96–99 %, порча чужих фактов −0,1…−0,2 %, у среза — −3 %; на 100 правках −1…−2 % против −7 %": "in the test (30 edits): transfer to other phrasings 96–99 %, damage to unrelated facts −0.1…−0.2 %, row slice −3 %; at 100 edits −1…−2 % vs −7 %", "На стенде (30 правок) перенос на другие формулировки 96–99 %, порча чужих фактов −0,1…−0,2 %, у среза — −3 %; на 100 правках −1…−2 % против −7 %": "On the test bench (30 edits): transfer to other phrasings 96–99 %, damage to unrelated facts −0.1…−0.2 %, row slice −3 %; at 100 edits −1…−2 % vs −7 %", "манера у новых людей 96 %, вместе с верным фактом 87 %, без метки не просачивается (0,0 %), чужое −0,3 %; у среза — 87 %, 72 %, 4,8 % и −6,4 %": "manner on new people 96 %, with the correct fact 87 %, no leakage without the tag (0.0 %), unrelated −0.3 %; row slice: 87 %, 72 %, 4.8 % and −6.4 %", "Манера у новых людей 96 %, вместе с верным фактом 87 %, без метки не просачивается (0,0 %), чужое −0,3 %; у среза — 87 %, 72 %, 4,8 % и −6,4 %": "Manner on new people 96 %, with the correct fact 87 %, no leakage without the tag (0.0 %), unrelated −0.3 %; row slice: 87 %, 72 %, 4.8 % and −6.4 %", "к каждому примеру добавляются обычные ответы исходной модели: она сохраняет их при обучении и меньше искажает то": "ordinary answers of the original model are mixed into the examples: it keeps them while training and distorts less of what", "К каждому примеру добавляются обычные ответы исходной модели: она сохраняет их при обучении и меньше искажает то": "Ordinary answers of the original model are mixed into the examples: it keeps them while training and distorts less of what", "память под главные направления защиты общих знаний (FP32): на down_proj вход 15360, на внимание — 3840 (q/k/v) и 4096 (o": "memory for the principal directions of knowledge protection (FP32): down_proj input 15360, attention 3840 (q/k/v) and 4096 (o", "Память под главные направления защиты общих знаний (FP32): на down_proj вход 15360, на внимание — 3840 (q/k/v) и 4096 (o": "Memory for the principal directions of knowledge protection (FP32): down_proj input 15360, attention 3840 (q/k/v) and 4096 (o", "маршрутизатор отправляет каждый токен к немногим экспертам из многих. TierLM учил бы срез каждого эксперта": "the router sends each token to a few experts out of many. TierLM would train a row slice of every expert", "Маршрутизатор отправляет каждый токен к немногим экспертам из многих. TierLM учил бы срез каждого эксперта": "The router sends each token to a few experts out of many. TierLM would train a row slice of every expert", "Веса и обученное могут жить где угодно — например на отдельном диске. Веса открываются только на чтение": "Weights and results can live anywhere, e.g. on a separate disk. Weights are opened read-only", "часть строк каждой матрицы и все нормировки, Adam: подходит для тона и знаний, но затрагивает остальное": "some rows of every matrix and all norms, Adam: suits tone and knowledge, but affects everything else", "Часть строк каждой матрицы и все нормировки, Adam: подходит для тона и знаний, но затрагивает остальное": "Some rows of every matrix and all norms, Adam: suits tone and knowledge, but affects everything else", "замороженная часть хранится сжатой — меньше ОЗУ и меньше везти по шине. Обучаемый срез остаётся в FP32": "the frozen part is stored compressed: less RAM and less bus traffic. The trained row slice stays in FP32", "Замороженная часть хранится сжатой — меньше ОЗУ и меньше везти по шине. Обучаемый срез остаётся в FP32": "The frozen part is stored compressed: less RAM and less bus traffic. The trained row slice stays in FP32", "приближёнными становятся только замороженные веса, через которые идут прямой и обратный проход": "only the frozen weights become approximate; forward and backward passes go through them", "Приближёнными становятся только замороженные веса, через которые идут прямой и обратный проход": "Only the frozen weights become approximate; forward and backward passes go through them", "сколько чисел учится в выбранных целиком матрицах — по config.json, без него как у Gemma 4 12B": "how many numbers are trained in the fully selected matrices, from config.json (Gemma 4 12B if absent)", "Сколько чисел учится в выбранных целиком матрицах — по config.json, без него как у Gemma 4 12B": "How many numbers are trained in the fully selected matrices, from config.json (Gemma 4 12B if absent)", "Факты и правила. Правка переносится на другие формулировки и почти не затрагивает остальное": "Facts and rules. The edit transfers to other phrasings and barely affects the rest", "целые down_proj трёх слоёв средней трети, nsgd, защита общих знаний, градиент только с ответа": "whole down_proj of three middle-third layers, NSGD, knowledge protection, gradient from the answer only", "Целые down_proj трёх слоёв средней трети, nsgd, защита общих знаний, градиент только с ответа": "Whole down_proj of three middle-third layers, NSGD, knowledge protection, gradient from the answer only", "быстрый": "fast", "Быстрый": "Fast", "баланс": "balanced", "Баланс": "Balanced", "качество": "quality", "Качество": "Quality", "а по шине возил бы только тех, кого маршрутизатор выбрал хотя бы для одного токена шага": "and carry over the bus only those the router picked for at least one token of the step", "А по шине возил бы только тех, кого маршрутизатор выбрал хотя бы для одного токена шага": "And carry over the bus only those the router picked for at least one token of the step", "внимание ранней половины сети, nsgd, защита общих знаний, повтор: на стенде (150 людей в данных": "attention of the early half, NSGD, knowledge protection, base-answer replay: on the bench (150 people in data", "Внимание ранней половины сети, nsgd, защита общих знаний, повтор: на стенде (150 людей в данных": "Attention of the early half, NSGD, knowledge protection, base-answer replay: on the bench (150 people in data", "Из осмотра машины. Запас — сколько видеопамяти оставить под промежуточные данные": "From the machine survey. Reserve is how much VRAM to keep for intermediate data", "Часть строк каждой матрицы. Подходит для тона и знаний, но затрагивает остальное": "Some rows of every matrix. Suits tone and knowledge, but affects everything else", "уже есть: продолжить его можно на экране обучения, для нового нужно другое имя": "already exists: continue it on the training screen; a new run needs another name", "Уже есть: продолжить его можно на экране обучения, для нового нужно другое имя": "Already exists: continue it on the training screen; a new run needs another name", "три умножения BF16 вместо одного FP32: быстрее при погрешности на уровне FP32": "three BF16 multiplies instead of one FP32: faster, with FP32-level error", "Три умножения BF16 вместо одного FP32: быстрее при погрешности на уровне FP32": "Three BF16 multiplies instead of one FP32: faster, with FP32-level error", "Сколько займёт, станет видно по первым шагам: на этой машине ещё не учились": "Duration will show after the first steps: nothing has been trained on this machine yet", "числа появятся, когда будут выбраны данные: шаги считаются по самим токенам": "numbers appear once data is selected: steps are counted from the tokens", "Числа появятся, когда будут выбраны данные: шаги считаются по самим токенам": "Numbers appear once data is selected: steps are counted from the tokens", "уже есть: без файла — продолжить его, для новых данных — другое имя (NAME": "already exists: without a file, continue it; for new data, another name (NAME", "Уже есть: без файла — продолжить его, для новых данных — другое имя (NAME": "Already exists: without a file, continue it; for new data, another name (NAME", "обороты в минуту; ноль — остановлен, это нормально на холодной карте": "revolutions per minute; zero means stopped, normal for a normally cooled card", "Обороты в минуту; ноль — остановлен, это нормально на холодной карте": "Revolutions per minute; zero means stopped, normal for a normally cooled card", "уже есть — его можно продолжить, а для новых данных нужно другое имя": "already exists: it can be continued; new data needs another name", "Уже есть — его можно продолжить, а для новых данных нужно другое имя": "Already exists: it can be continued; new data needs another name", "Голос и манера персонажа. Не проявляется там, где персонаж не задан": "A character's voice and manner. Absent where the character is not set", "в модели смесь экспертов, а движок пока умеет только плотные модели": "the model is a mixture of experts; the engine handles only dense models so far", "В модели смесь экспертов, а движок пока умеет только плотные модели": "The model is a mixture of experts; the engine handles only dense models so far", "бюджет обучаемого среза пересчитается при следующем запуске движка": "the budget of the trained row slice is recalculated at the next engine start", "Бюджет обучаемого среза пересчитается при следующем запуске движка": "The budget of the trained row slice is recalculated at the next engine start", "веса в файле уже квантованы, а движок пока учит только поверх BF16": "the weights are already quantized; the engine trains only on top of BF16 so far", "Веса в файле уже квантованы, а движок пока учит только поверх BF16": "The weights are already quantized; the engine trains only on top of BF16 so far", "до первого сброса на диск не дошло — продолжение начнётся с шага 1": "no weight snapshot was reached; continuing starts from step 1", "До первого сброса на диск не дошло — продолжение начнётся с шага 1": "No weight snapshot was reached; continuing starts from step 1", "процессор считает часть замороженных строк непосредственно из ОЗУ": "the CPU computes part of the frozen rows directly from RAM", "Процессор считает часть замороженных строк непосредственно из ОЗУ": "The CPU computes part of the frozen rows directly from RAM", "чего нет в данных. Для персонажа, который проявляется всегда, — 0": "what is not in the data. For a character that is always present: 0", "Чего нет в данных. Для персонажа, который проявляется всегда, — 0": "What is not in the data. For a character that is always present: 0", "События запуска, свежие сверху; ниже — вывод программы как есть": "Run events, newest first; below is the raw program output", "выбран для просмотра; обучение пока доступно только как Dense": "selected for viewing; training is available only as Dense so far", "Выбран для просмотра; обучение пока доступно только как Dense": "Selected for viewing; training is available only as Dense so far", "имя: буквы, цифры, точка, дефис и подчёркивание, до 40 знаков": "name: letters, digits, dot, hyphen and underscore, up to 40 characters", "Имя: буквы, цифры, точка, дефис и подчёркивание, до 40 знаков": "Name: letters, digits, dot, hyphen and underscore, up to 40 characters", "в этом режиме обучаются только целые матрицы выбранных слоёв": "in this mode only whole matrices of the chosen layers are trained", "В этом режиме обучаются только целые матрицы выбранных слоёв": "In this mode only whole matrices of the chosen layers are trained", "сколько токенов приходит на шаг и сколько из них под loss": "tokens per step and how many of them are in the loss", "Сколько токенов приходит на шаг и сколько из них под loss": "Tokens per step and how many of them are in the loss", "следующий слой передаётся по шине, пока считается текущий": "the next layer is transferred while the current one computes", "Следующий слой передаётся по шине, пока считается текущий": "The next layer is transferred while the current one computes", "Защита общих знаний: поиск главных направлений обычного текста": "Knowledge protection: finding principal directions of ordinary text", "недоступна для проверки — движок проверит при запуске": "cannot be checked: the engine will check at start", "Недоступна для проверки — движок проверит при запуске": "Cannot be checked: the engine will check at start", "ГиБ видеопамяти на вес и Adam — из настройки оборудования": "GiB of VRAM for weights and Adam, from hardware settings", "веса грузятся один раз — первый ответ ждать дольше": "weights load once: the first answer takes longer", "Веса грузятся один раз — первый ответ ждать дольше": "Weights load once: the first answer takes longer", "каждый токен проходит через все веса. Учится срез": "every token goes through all weights. A row slice is trained", "Каждый токен проходит через все веса. Учится срез": "Every token goes through all weights. A row slice is trained", "нет пакета numpy: без него защита общих знаний недоступна": "the numpy package is missing: knowledge protection is unavailable without it", "Нет пакета numpy: без него защита общих знаний недоступна": "The numpy package is missing: knowledge protection is unavailable without it", "Повтор: исходная модель размечает обычные ответы": "Base-answer replay: the original model labels ordinary answers", "перейти в режим «Эксперт»: каждое число правится": "switch to Expert mode: every number is editable", "Перейти в режим «Эксперт»: каждое число правится": "Switch to Expert mode: every number is editable", "уровни под нормальное распределение, как в QLoRA": "levels fitted to a normal distribution, as in QLoRA", "Уровни под нормальное распределение, как в QLoRA": "Levels fitted to a normal distribution, as in QLoRA", "числа обучения — в плане запуска, режим «Эксперт": "training numbers are in the run plan, mode “Expert", "Числа обучения — в плане запуска, режим «Эксперт": "Training numbers are in the run plan, mode “Expert", "шаг уменьшается, когда норма градиента выскочила": "the step shrinks when the gradient norm spikes", "Шаг уменьшается, когда норма градиента выскочила": "The step shrinks when the gradient norm spikes", "эталонная точность; шаг дольше примерно на треть": "reference precision; steps about a third longer", "Эталонная точность; шаг дольше примерно на треть": "Reference precision; steps about a third longer", "tokenizers есть, словарь придёт вместе с весами": "tokenizers present; the vocabulary comes with the weights", "Tokenizers есть, словарь придёт вместе с весами": "Tokenizers present; the vocabulary comes with the weights", "Модель, режим и данные — затем «Начать обучение": "Model, mode and data, then “Start training", "без сглаживания — видно разброс между примерами": "unsmoothed: spread between examples is visible", "Без сглаживания — видно разброс между примерами": "Unsmoothed: spread between examples is visible", "сборка под установленную карту, несколько минут": "build for the installed card, a few minutes", "этот запуск из терминала — записи о выборе нет": "this run was started from a terminal; no record of the choice", "Этот запуск из терминала — записи о выборе нет": "This run was started from a terminal; no record of the choice", "GTK с WebKit не найдены — открываю в браузере": "GTK with WebKit not found, opening in the browser", "продолжение доступно после устранения причины": "continuing is possible after fixing the cause", "Продолжение доступно после устранения причины": "Continuing is possible after fixing the cause", "срезу может не хватить места: закройте лишнее": "the row slice may not fit: close other programs", "Срезу может не хватить места: закройте лишнее": "The row slice may not fit: close other programs", "Самая горячая точка кристалла, память и край": "Hottest die point, memory and edge", "втрое выше среднего за полсотни шагов до них": "three times the average of the 50 steps before", "Втрое выше среднего за полсотни шагов до них": "Three times the average of the 50 steps before", "нужна картинка: png, jpg, webp, avif или gif": "an image is required: png, jpg, webp, avif or gif", "Нужна картинка: png, jpg, webp, avif или gif": "An image is required: png, jpg, webp, avif or gif", "числа появятся, как только данные разберутся": "numbers appear once data is parsed", "Числа появятся, как только данные разберутся": "Numbers appear once data is parsed", "Вид станет виден, когда найдётся файл весов": "Type is shown once the weights file is found", "Все параметры плана доступны для изменения": "All plan parameters are editable", "Окно можно закрыть: обучение идёт отдельно": "The window can be closed: training runs separately", "ROCm для AMD либо CUDA Toolkit для nvidia": "ROCm for AMD or CUDA Toolkit for nvidia", "Насколько сильно веса просятся измениться": "A measure of how strongly the weights need to change", "Раскладка обучаемого среза по видеопамяти": "Trained slice layout in VRAM", "вернуть числа, которые подобрал автопилот": "restore the values picked by auto-tuning", "Вернуть числа, которые подобрал автопилот": "Restore the values picked by auto-tuning", "сборка под установленную карту — ход ниже": "build for the installed card; progress below", "Сборка под установленную карту — ход ниже": "Build for the installed card; progress below", "движок знает только Gemma 4 12B, а здесь": "the engine knows only Gemma 4 12B, but here", "Движок знает только Gemma 4 12B, а здесь": "The engine knows only Gemma 4 12B, but here", "нет записи о старте — продолжать нечего": "no start record: nothing to continue", "Нет записи о старте — продолжать нечего": "No start record: nothing to continue", "FP32, простая библиотека производителя": "FP32, basic vendor library", "без подбора: обучение начинается сразу": "no search: training starts right away", "Без подбора: обучение начинается сразу": "No search: training starts right away", "поговорить с обученной моделью, или «р": "talk to the trained model, or “р", "Поговорить с обученной моделью, или «р": "Talk to the trained model, or “р", "Отправная точка для большинства задач": "A starting point for most tasks", "разгон в начале, дальше плавный спуск": "warmup first, then a smooth decay", "Разгон в начале, дальше плавный спуск": "Warmup first, then a smooth decay", "не удалось записать настройку оборудования": "failed to write hardware settings", "Не удалось записать настройку оборудования": "Failed to write hardware settings", "Пик за шаг; пунктир — потолок карты": "Peak per step; dashed line is the card limit", "замороженные веса, BF16 как в файле": "frozen weights, BF16 as in the file", "Замороженные веса, BF16 как в файле": "Frozen weights, BF16 as in the file", "определяет, какие матрицы обучаются": "decides which matrices are trained", "Определяет, какие матрицы обучаются": "Decides which matrices are trained", "остальное — вопрос, по нему не учим": "the rest is the question, not trained on", "Остальное — вопрос, по нему не учим": "The rest is the question, not trained on", "FP32, библиотека производителя (Lt": "FP32, vendor library (Lt", "Папка, где лежит model.safetensors": "Folder containing model.safetensors", "по одному примеру: набор небольшой": "one example at a time: the set is small", "По одному примеру: набор небольшой": "One example at a time: the set is small", "пустое поле — подбор автоматически": "empty field: automatic search", "Пустое поле — подбор автоматически": "Empty field: automatic search", "во столько раз модель «колеблется": "how many times the model “hesitates", "Во столько раз модель «колеблется": "How many times the model “hesitates", "не найдена: нет model.safetensors": "not found: no model.safetensors", "Не найдена: нет model.safetensors": "Not found: no model.safetensors", "нет tokenizer.json рядом с весами": "no tokenizer.json next to the weights", "Нет tokenizer.json рядом с весами": "No tokenizer.json next to the weights", "числа появятся после первых шагов": "numbers appear after the first steps", "он определяет, каким будет ответ": "it decides what the answer will be", "Он определяет, каким будет ответ": "It decides what the answer will be", "разговор закрылся — движок вышел": "chat closed: the engine exited", "Разговор закрылся — движок вышел": "Chat closed: the engine exited", "ГиБ видеопамяти: вес и инерция": "GiB of VRAM: weights and momentum", "Остановилось с ошибкой на шаге": "Stopped with an error at step", "не установлен пакет tokenizers": "tokenizers package not installed", "Не установлен пакет tokenizers": "Tokenizers package not installed", "отсутствует набор replay.jsonl": "the replay.jsonl set is missing", "Отсутствует набор replay.jsonl": "The replay.jsonl set is missing", "числа режима — в плане запуска": "mode numbers are in the run plan", "Числа режима — в плане запуска": "Mode numbers are in the run plan", "Доля угаданных токенов ответа": "Share of correctly predicted answer tokens", "длина заголовка вне разумного": "header length out of range", "Длина заголовка вне разумного": "Header length out of range", "по последним пятидесяти шагам": "over the last fifty steps", "По последним пятидесяти шагам": "Over the last fifty steps", "потоков процессора на матрицы": "CPU threads for matrices", "Потоков процессора на матрицы": "CPU threads for matrices", "установка пакетным менеджером": "install via package manager", "Установка пакетным менеджером": "Install via package manager", "tokenizers и словарь найдены": "tokenizers and vocabulary found", "Tokenizers и словарь найдены": "Tokenizers and vocabulary found", "каждое число плана — вручную": "every plan number by hand", "Каждое число плана — вручную": "Every plan number by hand", "открыть подробности, клавиша": "open details, key", "Открыть подробности, клавиша": "Open details, key", "тензоров живёт там постоянно": "tensors stay there permanently", "Тензоров живёт там постоянно": "Tensors stay there permanently", "установка пакетов — ход ниже": "installing packages; progress below", "Установка пакетов — ход ниже": "Installing packages; progress below", "Остановлено до первых шагов": "Stopped before the first steps", "до завершения или остановки": "until finished or stopped", "До завершения или остановки": "Until finished or stopped", "заголовок файла не читается": "file header unreadable", "Заголовок файла не читается": "File header unreadable", "медиана: всплески не в счёт": "median: spikes excluded", "Медиана: всплески не в счёт": "Median: spikes excluded", "не удалось открыть разговор": "failed to open chat", "Не удалось открыть разговор": "Failed to open chat", "нет файла model.safetensors": "no model.safetensors file", "Нет файла model.safetensors": "No model.safetensors file", "общая степень на блок из 32": "shared exponent per 32-block", "Общая степень на блок из 32": "Shared exponent per 32-block", "оно общее для всех запусков": "shared by all runs", "Оно общее для всех запусков": "Shared by all runs", "остальное заморожено в BF16": "the rest is frozen in BF16", "Остальное заморожено в BF16": "The rest is frozen in BF16", "предел кристалла около 110°": "die limit about 110°", "Предел кристалла около 110°": "Die limit about 110°", "среднее первых десяти шагов": "average of the first ten steps", "Среднее первых десяти шагов": "Average of the first ten steps", "Обучение ещё не начиналось": "Training has not started yet", "максимум за прогон, об/мин": "maximum per run, rpm", "Максимум за прогон, об/мин": "Maximum per run, rpm", "не выбран файл с примерами": "no examples file selected", "Не выбран файл с примерами": "No examples file selected", "полная точность, срез FP32": "full precision, FP32 row slice", "Полная точность, срез FP32": "Full precision, FP32 row slice", "ровно это уйдёт автопилоту": "exactly this goes to auto-tuning", "Ровно это уйдёт автопилоту": "Exactly this goes to auto-tuning", "вес первого токена ответа": "weight of the first answer token", "Вес первого токена ответа": "Weight of the first answer token", "загрузка весов — ход ниже": "loading weights; progress below", "Загрузка весов — ход ниже": "Loading weights; progress below", "поиск главных направлений": "finding principal directions", "Поиск главных направлений": "Finding principal directions", "продолжение пойдёт с шага": "continuing from step", "Продолжение пойдёт с шага": "Continuing from step", "шагов длиннее двух медиан": "steps longer than two medians", "Шагов длиннее двух медиан": "Steps longer than two medians", "ГиБ: слоёв слишком много": "GiB: too many layers", "карта сбрасывает частоты": "the card is throttling", "Карта сбрасывает частоты": "The card is throttling", "меньше 200 — жди падения": "below 200: expect a crash", "Меньше 200 — жди падения": "Below 200: expect a crash", "по одному примеру в шаге": "one example per step", "По одному примеру в шаге": "One example per step", "повтор: беречь остальное": "base-answer replay: protect the rest", "Повтор: беречь остальное": "Base-answer replay: protect the rest", "характер: голос и манера": "character: voice and manner", "4,4 % строк всех матриц": "4.4 % of rows of all matrices", "бюджет обучаемого среза": "trained row slice budget", "Бюджет обучаемого среза": "Trained row slice budget", "журнал, открыть целиком": "log, open in full", "Журнал, открыть целиком": "Log, open in full", "какой запуск показывать": "which run to show", "Какой запуск показывать": "Which run to show", "масштаб на группу из 64": "scale per group of 64", "Масштаб на группу из 64": "Scale per group of 64", "передайте файл датасета": "pass the dataset file", "Передайте файл датасета": "Pass the dataset file", "Минимальное время шага": "Minimal step time", "Не запустилось: ошибка": "Failed to start: error", "веса сохранены до шага": "weights saved up to step", "как в настройке оборудования": "as in hardware settings", "Как в настройке оборудования": "As in hardware settings", "не удалось скопировать": "copy failed", "Не удалось скопировать": "Copy failed", "открыть или продолжить": "open or continue", "Открыть или продолжить": "Open or continue", "причина в строках выше": "the reason is in the lines above", "Причина в строках выше": "The reason is in the lines above", "уже идёт другой запуск": "another run is in progress", "Уже идёт другой запуск": "Another run is in progress", "видеокарта не найдена": "no GPU found", "внимание ранних слоёв": "attention of early layers", "Внимание ранних слоёв": "Attention of early layers", "за последние 50 шагов": "over the last 50 steps", "За последние 50 шагов": "Over the last 50 steps", "модель, режим, данные": "model, mode, data", "Модель, режим, данные": "Model, mode, data", "не задано имя запуска": "no run name set", "Не задано имя запуска": "No run name set", "нет набора разработки": "no development kit", "сохранённых весов нет": "no saved weights", "Сохранённых весов нет": "No saved weights", "что сейчас происходит": "what is happening now", "Что сейчас происходит": "What is happening now", "Обучения ещё не было": "No training yet", "Поговорить с моделью": "Talk to the model", "Строка — один пример": "One line is one example", "Файл только читается": "The file is read-only", "а движок читает BF16": "but the engine reads BF16", "А движок читает BF16": "But the engine reads BF16", "выбрать другую папку": "choose another folder", "Выбрать другую папку": "Choose another folder", "выполняется один раз": "runs once", "Выполняется один раз": "Runs once", "движок ещё не собран": "engine not built yet", "Движок ещё не собран": "Engine not built yet", "движок пока не умеет": "the engine cannot do this yet", "Движок пока не умеет": "The engine cannot do this yet", "остальное заморожено": "the rest is frozen", "Остальное заморожено": "The rest is frozen", "остальные заморожены": "the others are frozen", "Остальные заморожены": "The others are frozen", "оценка на отложенных": "held-out evaluation", "Оценка на отложенных": "Held-out evaluation", "повтор держит знания": "base-answer replay keeps knowledge", "Повтор держит знания": "Base-answer replay keeps knowledge", "примеры для обучения": "training examples", "Примеры для обучения": "Training examples", "среднее по полусотне": "average over fifty", "Среднее по полусотне": "Average over fifty", "упала во столько раз": "dropped by this factor", "Упала во столько раз": "Dropped by this factor", "Tier1 в видеопамяти": "Trainable part in VRAM", "вернуть подобранное": "restore picked values", "Вернуть подобранное": "Restore picked values", "выбрать другой файл": "choose another file", "Выбрать другой файл": "Choose another file", "запуск из терминала": "started from terminal", "Запуск из терминала": "Started from terminal", "остановлено вручную": "stopped manually", "Остановлено вручную": "Stopped manually", "от начала до сейчас": "from start until now", "От начала до сейчас": "From start until now", "по скорости запуска": "by the speed of run", "По скорости запуска": "By the speed of run", "путь к файлу .jsonl": "path to the .jsonl file", "Путь к файлу .jsonl": "Path to the .jsonl file", "самая горячая точка": "hottest point", "Самая горячая точка": "Hottest point", "сброс на диск — шаг": "weight snapshot: step", "Сброс на диск — шаг": "Weight snapshot: step", "среднее на обучении": "training average", "Среднее на обучении": "Training average", "срез по видеопамяти": "row slice by VRAM", "умножение по оборудованию": "multiply by hardware", "Умножение по оборудованию": "Multiply by hardware", "что учится: целиком": "trained: whole", "Что учится: целиком": "Trained: whole", "nsgd + защита общих знаний": "NSGD + knowledge protection", "Nsgd + защита общих знаний": "NSGD + knowledge protection", "Подготовка токенов": "Preparing tokens", "вернуться к выбору": "back to selection", "Вернуться к выбору": "Back to selection", "максимум за прогон": "maximum per run", "Максимум за прогон": "Maximum per run", "перейти к обучению": "go to training", "Перейти к обучению": "Go to training", "подготовка токенов": "preparing tokens", "разговор с моделью": "chat with the model", "Разговор с моделью": "Chat with the model", "точность на ответе": "answer accuracy", "3 целые down_proj": "3 whole down_proj", "Запуск автопилота": "Starting auto-tuning", "Максимум качества": "Maximum quality", "градиент с ответа": "gradient from the answer", "Градиент с ответа": "Gradient from the answer", "длительность шага": "step duration", "замороженное BF16": "frozen BF16", "кэш в оперативной": "cache in RAM", "Кэш в оперативной": "Cache in RAM", "масштаб на строку": "scale per row", "Масштаб на строку": "Scale per row", "на что уходит шаг": "where the step goes", "На что уходит шаг": "Where the step goes", "обучать как Dense": "train as Dense", "Обучать как Dense": "Train as Dense", "по одному примеру": "one example at a time", "подбор развёрткой": "sweep search", "Подбор развёрткой": "Sweep search", "разговор закрылся": "chat closed", "Разговор закрылся": "Chat closed", "срез по всей сети": "row slice across the network", "токенов в секунду": "tokens per second", "Токенов в секунду": "Tokens per second", "numpy установлен": "numpy installed", "Numpy установлен": "Numpy installed", "Сбалансированный": "Balanced", "Что будет задано": "What will be set", "готово к запуску": "ready to start", "Готово к запуску": "Ready to start", "закрыть, или Esc": "close, or Esc", "Закрыть, или Esc": "Close, or Esc", "запас до потолка": "headroom to the limit", "Запас до потолка": "Headroom to the limit", "и примерно раз в": "and roughly once per", "И примерно раз в": "And roughly once per", "к токенам данных": "to data tokens", "К токенам данных": "To data tokens", "настройка оборудования": "hardware settings", "Настройка оборудования": "Hardware settings", "нет такого файла": "no such file", "Нет такого файла": "No such file", "падение среднего": "average drop", "Падение среднего": "Average drop", "разметка повтора": "base-answer replay labeling", "Разметка повтора": "Base-answer replay labeling", "ранняя остановка": "early stopping", "Ранняя остановка": "Early stopping", "скачать (≈ 22 ГБ": "download (≈ 22 GB", "Скачать (≈ 22 ГБ": "Download (≈ 22 GB", "слои правки выше": "edit layers above", "Слои правки выше": "Edit layers above", "событий пока нет": "no events yet", "токенов под loss": "tokens in loss", "Токенов под loss": "Tokens in loss", "умножение FP16×3": "FP16×3 multiply", "Умножение FP16×3": "FP16×3 multiply", "умножение матриц": "matrix multiply", "Умножение матриц": "Matrix multiply", "файл с примерами": "examples file", "FP32, свои ядра": "FP32, own kernels", "down_proj слоёв": "down_proj of layers", "Down_proj слоёв": "Down_proj of layers", "Проверки и план": "Checks and plan", "Прошлые запуски": "Previous runs", "Смесь экспертов": "Mixture of experts", "всего за прогон": "total per run", "Всего за прогон": "Total per run", "всё, что задано": "everything set", "вывод программы": "program output", "вывода пока нет": "no output yet", "норма градиента": "gradient norm", "обучаемых чисел": "trained numbers", "Обучаемых чисел": "Trained numbers", "примеров в шаге": "examples per step", "Примеров в шаге": "Examples per step", "проверки и план": "checks and plan", "раз за обучение": "times per training", "Раз за обучение": "Times per training", "раскладка среза": "slice layout", "Раскладка среза": "Slice layout", "связь с автором": "contact the author", "спросить модель": "ask the model", "Спросить модель": "Ask the model", "ушло автопилоту": "passed to auto-tuning", "Ушло автопилоту": "Passed to auto-tuning", "что было задано": "what was set", "экспертов и нет": "no experts", "Экспертов и нет": "No experts", "Не запустилось": "Failed to start", "больше бюджета": "over budget", "Больше бюджета": "Over budget", "внимание слоёв": "attention of layers", "Внимание слоёв": "Attention of layers", "выбор и запуск": "selection and start", "Выбор и запуск": "Selection and start", "длина примеров": "example length", "длина, медиана": "length, median", "доля всплесков": "share of spikes", "Доля всплесков": "Share of spikes", "загрузка весов": "loading weights", "Загрузка весов": "Loading weights", "изменить папку": "change folder", "медиана на шаг": "median per step", "Медиана на шаг": "Median per step", "начало к концу": "start to end", "Начало к концу": "Start to end", "не запустилось": "failed to start", "не знаю такого": "unknown", "Не знаю такого": "Unknown", "норма по шагам": "norm per step", "Норма по шагам": "Norm per step", "обучаемые слои": "trained layers", "Обучаемые слои": "Trained layers", "защита общих знаний по": "knowledge protection over", "Защита общих знаний по": "Knowledge protection over", "по имени файла": "by file name", "По имени файла": "By file name", "по шине за шаг": "over the bus per step", "порядок работы": "workflow", "Порядок работы": "Workflow", "рост за прогон": "growth per run", "Рост за прогон": "Growth per run", "связь потеряна": "connection lost", "Связь потеряна": "Connection lost", "система ~6 ГиБ": "system ~6 GiB", "Система ~6 ГиБ": "System ~6 GiB", "сумма по шагам": "sum over steps", "Сумма по шагам": "Sum over steps", "токенов в loss": "tokens in loss", "Токенов в loss": "Tokens in loss", "токенов в шаге": "tokens per step", "Токенов в шаге": "Tokens per step", "умножение FP32": "FP32 multiply", "Умножение FP32": "FP32 multiply", "шагов выше 95°": "steps above 95°", "Шагов выше 95°": "Steps above 95°", "шагов на эпоху": "steps per epoch", "Шагов на эпоху": "Steps per epoch", "Быстрый старт": "Quick start", "Выбрать папку": "Choose folder", "Обучение: шаг": "Training: step", "без улучшения": "no improvement", "Без улучшения": "No improvement", "во весь экран": "full screen", "Во весь экран": "Full screen", "всего времени": "total time", "Всего времени": "Total time", "где что лежит": "where things are", "Где что лежит": "Where things are", "доля под loss": "share in loss", "Доля под loss": "Share in loss", "ещё не собран": "not built yet", "Ещё не собран": "Not built yet", "забыть беседу": "forget the chat", "Забыть беседу": "Forget the chat", "задан заранее": "set in advance", "Задан заранее": "Set in advance", "исходные веса": "original weights", "Исходные веса": "Original weights", "на отложенных": "held-out", "начать заново": "start over", "пик за прогон": "peak per run", "Пик за прогон": "Peak per run", "почти потолок": "near the limit", "Почти потолок": "Near the limit", "рабочая папка": "work folder", "размер текста": "text size", "Размер текста": "Text size", "самый длинный": "longest", "сброс на диск": "weight snapshot", "Сброс на диск": "Weight snapshot", "сбросить свои": "reset custom", "Сбросить свои": "Reset custom", "сделано шагов": "steps done", "Сделано шагов": "Steps done", "секунд на шаг": "seconds per step", "со всплесками": "with spikes", "Со всплесками": "With spikes", "токенов всего": "tokens total", "Токенов всего": "Tokens total", "только чтение": "read-only", "указать папку": "set folder", "что произошло": "what happened", "Квантованная": "Quantized", "длительность": "duration", "Длительность": "Duration", "занята: идёт": "busy: running", "Занята: идёт": "Busy: running", "занято: идёт": "busy: running", "Занято: идёт": "Busy: running", "ключи движка": "engine flags", "не начинался": "not started", "не требуется": "not required", "Не требуется": "Not required", "папка модели": "model folder", "после выбора": "after selection", "После выбора": "After selection", "прозрачность": "transparency", "Прозрачность": "Transparency", "размер файла": "file size", "самый долгий": "slowest", "сменить тему": "toggle theme", "точно: факты": "precise: facts", "уже разбираю": "already parsing", "Уже разбираю": "Already parsing", "ход обучения": "training progress", "Ход обучения": "Training progress", "шаг обучения": "learning rate", "Подбор шага": "Rate search", "видеопамять": "VRAM", "выполняется": "running", "Выполняется": "Running", "ещё не было": "none yet", "Ещё не было": "None yet", "завершается": "finishing", "имя запуска": "run name", "на обучение": "for training", "на обучении": "on training", "направлений": "directions", "Направлений": "Directions", "обычный шаг": "regular step", "Обычный шаг": "Regular step", "остановлено": "stopped", "защита общих знаний": "knowledge protection", "откуда веса": "where weights come from", "Откуда веса": "Where weights come from", "ошибка (код": "error (code", "пересобрать": "rebuild", "подбор шага": "rate search", "продолжение": "continuation", "Продолжение": "Continuation", "скопировано": "copied", "Скопировано": "Copied", "скопировать": "copy", "Скопировать": "Copy", "слои правки": "edit layers", "Слои правки": "Edit layers", "температура": "temperature", "температуры": "temperatures", "Температуры": "Temperatures", "что сжимать": "what to compress", "шагов всего": "steps total", "Шагов всего": "Steps total", "автопилоту": "to auto-tuning", "Автопилоту": "To auto-tuning", "вентилятор": "fan", "вид модели": "model type", "Вид модели": "Model type", "готовность": "readiness", "Готовность": "Readiness", "длина, p99": "length, p99", "запас, МиБ": "reserve, MiB", "Запас, МиБ": "Reserve, MiB", "затемнение": "dimming", "Затемнение": "Dimming", "компилятор": "compiler", "Компилятор": "Compiler", "на эпоху ×": "per epoch ×", "На эпоху ×": "Per epoch ×", "настройках": "settings", "Настройках": "Settings", "начертание": "weight", "нормировки": "norms", "Нормировки": "Norms", "осталось ≈": "left ≈", "Осталось ≈": "Left ≈", "отложенные": "held-out", "Отложенные": "Held-out", "оформление": "appearance", "пик за шаг": "peak per step", "Пик за шаг": "Peak per step", "подготовка": "preparation", "Подготовка": "Preparation", "продолжить": "continue", "Продолжить": "Continue", "с повтором": "with base-answer replay", "С повтором": "With base-answer replay", "скругление": "corner radius", "только MLP": "MLP only", "что внутри": "what is inside", "что учится": "what is trained", "p99 длины": "length p99", "P99 длины": "Length p99", "Подробнее": "Details", "влезет ли": "will it fit", "Влезет ли": "Will it fit", "выключена": "off", "Выключена": "Off", "готовится": "preparing", "Готовится": "Preparing", "завершено": "finished", "настройка": "setting", "Настройка": "Setting", "настройки": "settings", "не собран": "not built", "ни одного": "none", "Ни одного": "None", "остановка": "stop", "Остановка": "Stop", "отправить": "send", "плотность": "density", "повторить": "retry", "поколение": "generation", "Поколение": "Generation", "последний": "last", "Последний": "Last", "поставить": "install", "придержка": "clipping", "Придержка": "Clipping", "программа": "program", "просторно": "spacious", "процессор": "CPU", "Процессор": "CPU", "развёртка": "sweep", "Развёртка": "Sweep", "раскладка": "layout", "свой цвет": "custom color", "сложность": "difficulty", "Сложность": "Difficulty", "сообщение": "message", "сообщения": "messages", "Сообщения": "Messages", "состояние": "state", "сохранить": "save", "умножение": "multiply", "Умножение": "Multiply", "что учить": "what to train", "Что учить": "What to train", "в начале": "at start", "В начале": "At start", "внимание": "attention", "Внимание": "Attention", "все слои": "all layers", "выбросов": "outliers", "Выбросов": "Outliers", "выключен": "off", "говорить": "talk", "Говорить": "Talk", "доступно": "available", "загрузка": "loading", "Загрузка": "Loading", "запущено": "started", "Запущено": "Started", "значение": "value", "Значение": "Value", "изменить": "change", "Изменить": "Change", "источник": "source", "Источник": "Source", "картинка": "image", "максимум": "maximum", "мимо неё": "outside it", "Мимо неё": "Outside it", "нужно от": "needs from", "Нужно от": "Needs from", "обратный": "backward", "Обратный": "Backward", "обучение": "training", "обучения": "training", "Обучения": "Training", "основной": "main", "Основной": "Main", "осталось": "left", "отложено": "held out", "под loss": "in loss", "Под loss": "In loss", "подкачка": "prefetch", "Подкачка": "Prefetch", "примеров": "examples", "проверки": "checks", "разговор": "chat", "свернуть": "minimize", "Свернуть": "Minimize", "свободно": "free", "терпение": "patience", "Терпение": "Patience", "точность": "accuracy", "Точность": "Accuracy", "уже идёт": "already running", "Уже идёт": "Already running", "упаковка": "packing", "характер": "character", "Выбрать": "Choose", "Плотная": "Dense", "Причина": "Reason", "Сначала": "First", "Эксперт": "Expert", "в ОЗУ и": "in RAM and", "В ОЗУ и": "In RAM and", "в папке": "in folder", "выбрать": "choose", "головка": "head", "Головка": "Head", "горячая": "hot", "Горячая": "Hot", "датасет": "dataset", "Датасет": "Dataset", "закрыть": "close", "Закрыть": "Close", "замеров": "measurements", "запуски": "runs", "медиана": "median", "Медиана": "Median", "нативно": "native", "не было": "none", "не реже": "at least every", "Не реже": "At least every", "неверно": "invalid", "область": "area", "обычное": "regular", "обычные": "regular", "открыть": "open", "плотное": "bold", "потолок": "limit", "ресурсы": "resources", "Ресурсы": "Resources", "светлая": "light", "скачать": "download", "словарь": "vocabulary", "сначала": "first", "собрать": "build", "событие": "event", "средний": "average", "Средний": "Average", "средних": "average", "Средних": "Average", "токенов": "tokens", "форматы": "formats", "Форматы": "Formats", "целиком": "whole", "эксперт": "expert", "Отмена": "Cancel", "акцент": "accent", "блоков": "blocks", "в шаге": "per step", "В шаге": "Per step", "версия": "version", "Версия": "Version", "веса в": "weights in", "Веса в": "Weights in", "выбран": "selected", "Выбран": "Selected", "готово": "ready", "данные": "data", "движок": "engine", "оборудование": "hardware", "журнал": "log", "задано": "set", "Задано": "Set", "закрыт": "closed", "Закрыт": "Closed", "заняло": "took", "Заняло": "Took", "занято": "used", "Занято": "Used", "запуск": "run", "каждые": "every", "Каждые": "Every", "лучшая": "best", "лучший": "best", "Лучший": "Best", "лёгкое": "light", "модель": "model", "начало": "start", "Начало": "Start", "об/мин": "rpm", "Об/мин": "Rpm", "обычно": "regular", "Обычно": "Regular", "откуда": "source", "Откуда": "Source", "оценка": "evaluation", "оценки": "evaluations", "оценок": "evaluations", "ошибка": "error", "пакеты": "packages", "Пакеты": "Packages", "память": "memory", "панель": "panel", "первый": "first", "Первый": "First", "плотно": "compact", "повтор": "base-answer replay", "потеря": "loss", "почему": "why", "Почему": "Why", "прочее": "other", "Прочее": "Other", "прошло": "elapsed", "прямой": "forward", "Прямой": "Forward", "разбор": "parsing", "размер": "size", "режимы": "modes", "Режимы": "Modes", "свежий": "latest", "Свежий": "Latest", "сводка": "summary", "Сводка": "Summary", "секунд": "seconds", "Секунд": "Seconds", "сервер": "server", "Сервер": "Server", "собран": "built", "способ": "method", "строки": "rows", "Строки": "Rows", "токены": "tokens", "тонкие": "thin", "тёмная": "dark", "формат": "format", "чёткие": "crisp", "шина ≥": "bus ≥", "эпох ×": "epochs ×", "Эпох ×": "Epochs ×", "Выбор": "Select", "Сразу": "At once", "Форма": "Shape", "в ОЗУ": "in RAM", "время": "time", "всего": "total", "выбор": "selection", "вывод": "output", "Вывод": "Output", "живых": "live", "задан": "set", "из 40": "of 40", "Из 40": "Of 40", "карта": "card", "ключи": "flags", "Ключи": "Flags", "когда": "when", "конец": "end", "Конец": "End", "линии": "lines", "линий": "lines", "Линий": "Lines", "норма": "norm", "нужно": "needed", "около": "about", "Около": "About", "папка": "folder", "Папка": "Folder", "после": "after", "После": "After", "пущен": "started", "Пущен": "Started", "режим": "mode", "сброс": "weight snapshot", "Сброс": "Weight snapshot", "старт": "start", "Старт": "Start", "текст": "text", "ток/с": "tok/s", "Ток/с": "Tok/s", "точка": "point", "точно": "precise", "учить": "train", "Учить": "Train", "через": "in", "Через": "In", "чисел": "numbers", "Чисел": "Numbers", "шагов": "steps", "шага": "steps", "эпоха": "epoch", "эпохи": "epochs", "этапы": "stages", "Этапы": "Stages", "МБ/с": "MB/s", "байт": "bytes", "Байт": "Bytes", "вент": "fan", "Вент": "Fan", "веса": "weights", "Веса": "Weights", "выкл": "off", "выше": "above", "есть": "present", "ждём": "waiting", "Ждём": "Waiting", "знач": "value", "Знач": "Value", "идёт": "running", "край": "edge", "мимо": "outside", "окно": "window", "Окно": "Window", "план": "plan", "пути": "paths", "путь": "path", "Путь": "Path", "свои": "custom", "Свои": "Custom", "своё": "custom", "Своё": "Custom", "слои": "layers", "Слои": "Layers", "срез": "row slice", "стоп": "stop", "суть": "essence", "Суть": "Essence", "тема": "theme", "файл": "file", "Файл": "File", "цели": "goals", "Цели": "Goals", "шаги": "steps", "Шаги": "Steps", "шина": "bus", "эпох": "epochs", "этап": "stage", "Этап": "Stage", "язык": "language", "Язык": "Language", "бит": "bits", "вид": "type", "Вид": "Type", "вкл": "on", "Вкл": "On", "всё": "all", "Всё": "All", "гпу": "GPU", "Гпу": "GPU", "имя": "name", "код": "code", "мин": "min", "Мин": "Min", "нет": "no", "озу": "RAM", "Озу": "RAM", "пик": "peak", "под": "under", "Под": "Under", "раз": "times", "Раз": "Times", "сек": "s", "Сек": "S", "сис": "sys", "Сис": "Sys", "ток": "tok", "Ток": "Tok", "что": "what", "Что": "What", "шаг": "step", "ч": "h", "Ч": "H", "из": "of", "на": "on", "На": "On", "ок": "ok", "Ок": "Ok", "ср": "avg", "Ср": "Avg", "и": "and", "И": "And", "в": "in", "В": "In", "по": "by", "По": "By", "с": "s", "С": "S", "у": "at", "У": "At", "за": "per", "За": "Per", "для": "for", "Для": "For", "без": "without", "Без": "Without", "до": "to", "До": "To", "от": "from", "От": "From", "или": "or", "Или": "Or", "не": "not", "Не": "Not", "это": "this", "Это": "This", "при": "at", "При": "At", "только": "only", "Только": "Only", "готово:": "ready:", "Готово:": "Ready:", "эпоху": "epoch", "Эпоху": "Epoch", "в файле": "in the file", "В файле": "In the file", "пока нет": "none yet", "Пока нет": "None yet", "вместе с Adam": "together with Adam", "Вместе с Adam": "Together with Adam", "видеопамяти": "VRAM", "Видеопамяти": "VRAM", "не меньше": "at least", "Не меньше": "At least", "мало": "too little", "все": "all", "запас видеопамяти": "VRAM reserve", "Запас видеопамяти": "VRAM reserve", "живёт в": "lives in", "Живёт в": "Lives in", "потеря на отложенных не улучшилась": "held-out loss has not improved for", "Потеря на отложенных не улучшилась": "Held-out loss has not improved for", "не улучшилась": "has not improved for", "Не улучшилась": "Has not improved for", "оценок подряд": "evaluations in a row", "Оценок подряд": "Evaluations in a row", "подряд": "in a row", "Подряд": "In a row", "дальше учить было незачем": "no point training further", "Дальше учить было незачем": "No point training further", "лучший снимок": "best snapshot", "Лучший снимок": "Best snapshot", "снимок": "snapshot", "Снимок": "Snapshot", "тг": "tg", "Тг": "Tg", "вк": "vk", "Вк": "Vk", "дс": "ds", "Дс": "Ds", "развернуть": "maximize", "Развернуть": "Maximize", "с зазором": "with a gap", "вплотную": "edge to edge", "завершить всё": "quit everything", "Обучение будет остановлено, программа закроется полностью. Завершить": "Training will be stopped and the program will close completely. Quit", "Закрыть программу полностью": "Close the program completely", "программа завершена": "program closed", "Программа завершена": "Program closed", "собран из прежней версии": "built from the previous version", "движок собран из прежней версии — пересобрать": "the engine was built from the previous version: rebuild", "Движок собран из прежней версии — пересобрать": "The engine was built from the previous version: rebuild", "нет сохранённых весов: обучение не дошло до первого сброса на диск": "has no saved weights: training never reached the first weight snapshot", "Нет сохранённых весов: обучение не дошло до первого сброса на диск": "Has no saved weights: training never reached the first weight snapshot", "не поддерживается": "not supported", "не выбраны": "not selected", "не задано": "not set", "сборка": "building", "Сборка": "Building", "готов": "ready", "Готов": "Ready", "уже есть": "already exists", "Уже есть": "Already exists", "занята": "busy", "проверка при запуске": "checked at start", "нет данных": "no data", "пусто — автоматически": "empty: automatic", "Пусто — автоматически": "Empty: automatic", "время: после первых шагов": "time: after the first steps", "Время: после первых шагов": "Time: after the first steps", "словарь — вместе с весами": "vocabulary comes with the weights", "Словарь — вместе с весами": "Vocabulary comes with the weights", "нет пакета": "missing package", "Нет пакета": "Missing package", "Остановлено · шаг": "Stopped · step", "оценка каждые": "evaluate every", "сброс каждые": "weight snapshot every", "вес первого токена": "first token weight", "Вес первого токена": "First token weight", "придержка градиента": "gradient clipping", "Придержка градиента": "Gradient clipping", "потоков процессора": "CPU threads", "Потоков процессора": "CPU threads", "бюджет среза": "row slice budget", "по оборудованию": "by hardware", "По оборудованию": "By hardware", "Замороженные веса, BF16": "Frozen weights, BF16", "Как в «Слои правки»": "Same as “Edit layers”", "Одна строка — один пример": "One line, one example", "обучается": "trains", "Обучается": "Trains", "Сцена:": "System prompt:", "Выйти": "Quit", "Показать окно": "Show window", "Данные программы": "Program data", "Каждый вопрос — обученной и исходной модели, с чистого листа": "Each question goes to the trained and the base model, from a clean slate", "Обучение продолжается. Окно — в трее.": "Training continues. The window is in the tray.", "Обычный размер": "Restore", "Память при обучении": "Memory during training", "По одному на строку": "One per line", "Появится с первыми шагами": "Appears with the first steps", "не выбрана модель": "no model selected", "Не выбрана модель": "No model selected", "нет движка или весов модели": "no engine or model weights", "Нет движка или весов модели": "No engine or model weights", "шине": "the bus", "Шине": "The bus", "у «": "«", "У «": "«", "Сохранений ещё нет: продолжать будет не с чего": "No weight snapshots yet: there will be nothing to resume from", "lossа": "loss", "Lossа": "Loss", "lossе": "loss", "Lossе": "Loss", "lossу": "loss", "Lossу": "Loss", "Loss на валидации": "Validation loss", "loss на валидации": "validation loss", "Loss на обучении": "Training loss", "Сырой loss по шагам": "Raw loss per step", "Лучший loss": "Best loss", "лучший loss": "best loss", "Лучший на валидации": "Best on validation", "Последний на валидации": "Last on validation", "На валидации": "Validation", "на валидации": "on validation", "На валидацию": "For validation", "на валидацию": "for validation", "Валидация": "Validation", "валидация": "validation", "Валидации": "Validation", "валидации": "validation", "Валидаций": "Validations", "валидаций": "validations", "Валидация каждые": "Validate every", "валидация каждые": "validate every", "Точность на валидации": "Validation accuracy", "Точки валидации": "Validation dots", "В loss": "In loss", "Доля в loss": "Share in loss", "Токенов на шаг и сколько из них в loss.": "Tokens per step and how many of them count toward the loss.", "Ниже — лучше. Обучение сглажено по 50 шагам, валидация — отдельные замеры.": "Lower is better. Training is smoothed over 50 steps; validation points are separate measurements.", "40 шагов с растущим LR; берётся десятая часть LR у минимума сглаженного loss": "40 steps with a growing LR; a tenth of the LR at the smoothed-loss minimum is taken", "40 коротких шагов; берётся десятая часть LR, на котором сглаженный loss минимален": "40 short steps; a tenth of the LR at which the smoothed loss is lowest is taken", "Подбор LR": "LR search", "подбор LR": "LR search", "Подбор LR: loss при растущем LR": "LR search: loss as the LR grows", "Ниже — подбор LR.": "The LR search is below.", "LR подбором": "LR by search", "подбор": "search", "Подбор": "Search", "Подобран": "Found by search", "Checkpoint нет": "No weight snapshots", "Checkpoint каждые": "Weight snapshot every", "checkpoint каждые": "weight snapshot every", "Checkpoint на шаге": "Weight snapshot at step", "Checkpoint ещё нет: продолжать будет не с чего": "No weight snapshots yet: there will be nothing to resume from", "Лучший checkpoint": "Best weight snapshot", "Рабочая папка: checkpoint и состояние оптимизатора": "Work folder: weight snapshots and optimizer state", "нет checkpoint: обучение не дошло до первого": "has no weight snapshot: training never reached the first one", "Нет checkpoint: обучение не дошло до первого": "Has no weight snapshot: training never reached the first one", "Всплески — валидация и запись checkpoint.": "Spikes are validation and weight snapshot writes.", "на шаг позже валидации, чтобы запись не совпадала с чтением": "one step after validation, so writing does not collide with reading", "На шаг позже валидации, чтобы запись не совпадала с чтением": "One step after validation, so writing does not collide with reading", "Токенизатор": "Tokenizer", "Токенизация": "Tokenizing", "Токенизация данных": "Tokenizing data", "Токенизировать заново": "Tokenize again", "Ошибка токенизации": "Tokenization error", "Файл изменён после токенизации": "File changed after tokenization", "Файл изменился после токенизации": "File changed after tokenization", "Файл не токенизирован — причина ниже": "File not tokenized — reason below", "датасет ещё не токенизирован": "the dataset is not tokenized yet", "Датасет ещё не токенизирован": "The dataset is not tokenized yet", "файл датасета изменился после токенизации — токенизируйте его заново": "the dataset file changed after tokenization — tokenize it again", "Файл датасета изменился после токенизации — токенизируйте его заново": "The dataset file changed after tokenization — tokenize it again", "Приходит вместе с весами": "Comes with the weights", "Арена": "VRAM arena", "Мимо арены": "Outside the VRAM arena", "Живая память и арена промежуточных данных слоя; растущее «мимо арены» — память фрагментируется.": "Live memory and the VRAM arena for the intermediate data of a layer; a growing “Outside the arena” value means memory fragmentation.", "Прямой проход, LM head, обратный проход с шагом оптимизатора.": "Forward pass, LM head, backward pass with the optimizer step.", "Prefetch следующего слоя": "Next-layer prefetch", "Запас видеопамяти, бюджет среза, FP16×3, prefetch": "VRAM reserve, row slice budget, FP16×3, prefetch", "Троттлинг": "Throttling", "троттлинг": "throttling", "Был троттлинг": "Throttled", "Выше 95° — троттлинг": "Above 95° — throttling", "Скрытая размерность": "Hidden size", "Больше валидаций и эпох, точное умножение.": "More validations and epochs, exact multiplication.", "после 1 валидации без улучшения": "after 1 validation without improvement", "После 1 валидации без улучшения": "After 1 validation without improvement", "валидаций без улучшения": "validations without improvement", "Валидаций без улучшения": "Validations without improvement", "валидации без улучшения": "validation without improvement", "Валидации без улучшения": "Validation without improvement", "валидация без улучшения": "validation without improvement", "Валидация без улучшения": "Validation without improvement", "loss на валидации перестаёт снижаться": "the validation loss stops falling", "Loss на валидации перестаёт снижаться": "The validation loss stops falling", "обучение прекращается, когда loss на валидации перестаёт снижаться": "training stops when the validation loss stops falling", "Обучение прекращается, когда loss на валидации перестаёт снижаться": "Training stops when the validation loss stops falling", "остановка: loss на валидации не улучшился": "stop: the validation loss has not improved", "Остановка: loss на валидации не улучшился": "Stop: the validation loss has not improved", "не улучшился": "has not improved", "Не улучшился": "Has not improved", "раза подряд": "times in a row", "Раза подряд": "Times in a row", "раз подряд": "times in a row", "Раз подряд": "Times in a row", "время приблизительное": "the time is approximate", "Время приблизительное": "The time is approximate", "Подбор LR ·": "LR search ·", "упаковка другая — время приблизительное": "the packing differs — the time is approximate", "Упаковка другая — время приблизительное": "The packing differs — the time is approximate", "берётся десятая часть LR, на котором сглаженный loss минимален": "a tenth of the LR at which the smoothed loss is lowest is taken", "Берётся десятая часть LR, на котором сглаженный loss минимален": "A tenth of the LR at which the smoothed loss is lowest is taken", "Путь к файлу — в карточке «Данные»": "File path goes in the “Data” card", "Выбирается в окне программы": "Chosen in the program window", "нет такого запуска": "no such run", "Нет такого запуска": "No such run", "причина в журнале": "reason in the log", "Причина в журнале": "Reason in the log", "строк": "rows", "Строк": "Rows", "Со сценой, %": "With system prompt, %", "Доля примеров, где поле instruction идёт system-репликой. Остальные — без неё: так манера держится и в обычном разговоре.": "Share of examples where the instruction field goes in as the system prompt. The rest go without it, so the manner holds in plain conversation too.", "Всегда со сценой": "Always with system prompt", "Пример идёт со сценой всегда, если в его instruction есть одно из слов.": "An example always gets the system prompt if its instruction contains one of the words.", "Слова из instruction через запятую": "Words from instruction, comma-separated", "Со сценой": "With system prompt", "прежняя токенизация не остановилась": "the previous tokenization did not stop", "Прежняя токенизация не остановилась": "The previous tokenization did not stop", "Выгрузить": "Unload", "Освобождает ОЗУ и видеопамять. Ответ обрывается; следующий вопрос загрузит модель заново": "Frees RAM and VRAM. The answer is cut off; the next question loads the model again", "Модель выгружена": "Model unloaded", "идёт обучение: видеопамять занята им, разговор — после окончания или остановки": "training is running and holds the VRAM: chat after it finishes or is stopped", "Идёт обучение: видеопамять занята им, разговор — после окончания или остановки": "Training is running and holds the VRAM: chat after it finishes or is stopped", "идёт обучение": "training is running", "Идёт обучение": "Training is running", "Модель разговора выгружена": "Chat model unloaded", "Открыть в браузере": "Open in browser", "instruction идёт system-репликой": "instruction goes in as the system prompt", "Instruction идёт system-репликой": "Instruction goes in as the system prompt", "; всегда — при словах:": "; always for the words:", "; доля — в карточке «Данные»": "; the share is set in the “Data” card", "В памяти": "In memory", "Выгрузить модель": "Unload model", "примера": "examples", "Примера": "Examples", "пример": "example", "Кэш весов целиком": "Full weight cache", "Кэш весов": "Weight cache", "системе остаётся": "left for the system", "Системе остаётся": "Left for the system", "Остальные": "The other", "читаются с диска дважды за шаг: обучение медленнее. Разговору нужно": "are read from disk twice per step: training is slower. Chat needs", "Читаются с диска дважды за шаг: обучение медленнее. Разговору нужно": "Are read from disk twice per step: training is slower. Chat needs", "Ранг r": "Rank r", "Матрицы": "Matrices", "ранг слишком велик": "rank too high", "Ранг слишком велик": "Rank too high", "По умолчанию": "Defaults", "r 16 · α 32 · все матрицы · dropout 0,05": "r 16 · α 32 · all matrices · dropout 0.05", "R 16 · α 32 · все матрицы · dropout 0,05": "R 16 · α 32 · all matrices · dropout 0.05", "Классическая LoRA (Hu и др., 2021), как в PEFT: W + α/r · B·A поверх замороженных весов. A — равномерно в ±1/√вход, B — нули, AdamW FP32. Исходные веса не меняются, адаптеры — отдельным файлом": "Classic LoRA (Hu et al., 2021), as in PEFT: W + α/r · B·A on top of frozen weights. A is uniform in ±1/√in, B is zeros, AdamW FP32. The original weights stay unchanged; adapters are a separate file", "Сколько направлений поправки в каждой матрице. Больше — больше обучаемых чисел и видеопамяти": "How many update directions each matrix gets. Higher means more trainable numbers and more VRAM", "Масштаб поправки: s = α / r. Обычно α = 2r": "Update scale: s = α / r. Usually α = 2r", "Доля входа адаптера, обнуляемая при обучении. На валидации и в разговоре не действует": "Share of the adapter input zeroed during training. Not applied in validation or chat", "Где стоят адаптеры: q k v o — внимание, gate up down — MLP; во всех слоях. В слоях полного внимания своей v_proj нет: V там берётся из K": "Where the adapters go: q k v o — attention, gate up down — MLP; in all 48 layers. Full-attention layers have no v_proj of their own: V is taken from K there", "Хотя бы одна матрица остаётся": "At least one matrix stays on", "Адаптеров": "Adapters", "Пар A и B": "A and B pairs", "модели": "of the model", "A, B и два момента": "A, B and two moments", "Множитель s": "Scale s", "Адаптеры FP32": "FP32 adapters", "С AdamW в видеопамяти": "With AdamW in VRAM", "С nsgd в видеопамяти": "With NSGD in VRAM", "Вес и инерция": "Weight and momentum", "Адаптеры": "Adapters", "адаптеры поверх замороженных весов: W + α/r · B·A, s = α/r =": "adapters on top of frozen weights: W + α/r · B·A, s = α/r =", "Адаптеры поверх замороженных весов: W + α/r · B·A, s = α/r =": "Adapters on top of frozen weights: W + α/r · B·A, s = α/r =", "; A — равномерно в ±1/√вход, B — нули, AdamW FP32. Исходные веса не меняются, адаптеры — отдельным файлом": "; A is uniform in ±1/√in, B is zeros, AdamW FP32. The original weights stay unchanged; adapters are a separate file", "ГиБ видеопамяти: A, B и два момента AdamW в FP32; в слоях полного внимания v_proj нет": "GiB of VRAM: A, B and two AdamW moments in FP32; full-attention layers have no v_proj", "на входе адаптера, только при обучении": "on the adapter input, training only", "На входе адаптера, только при обучении": "On the adapter input, training only", "классическая LoRA учится только на данных; включается в «Эксперте»": "classic LoRA learns from the data only; can be turned on in “Expert”", "Классическая LoRA учится только на данных; включается в «Эксперте»": "Classic LoRA learns from the data only; can be turned on in “Expert”", "адаптеры LoRA": "LoRA adapters", "Адаптеры LoRA": "LoRA adapters", "больше бюджета видеопамяти": "exceed the VRAM budget of", "Больше бюджета видеопамяти": "Exceed the VRAM budget of", "меньше ранг или матриц": "lower the rank or the matrix count", "Меньше ранг или матриц": "Lower the rank or the matrix count", "не выбрано ни одной матрицы LoRA": "no LoRA matrix selected", "Не выбрано ни одной матрицы LoRA": "No LoRA matrix selected", "Не нужна": "Not needed", "База": "Base", "Квантовать": "Quantize", "Как в файле": "As in the file", "Не годится": "Unusable", "Идёт другое задание": "Another job is running", "Нужен собранный движок": "The engine must be built", "Нужна поддерживаемая модель": "A supported model is needed", "Удалить файл базы": "Delete the base file", "Файл базы удаляется с диска. Исходная модель не затрагивается.": "The base file is deleted from disk. The original model is not touched.", "Замороженные матрицы слоёв — из исходного файла, без потерь": "Frozen layer matrices come from the original file, lossless", "Ошибка весов — среднеквадратичная, относительно BF16": "Weight error is RMS, relative to BF16", "Свой квант из BF16 на процессоре, по правилам llama.cpp. Исходный файл только читается": "Own quant from BF16 on the CPU, by the llama.cpp rules. The original file is only read", "Готовый квант этой же модели: GGUF любого типа, в том числе динамические Unsloth, или bitsandbytes 4 бита. Блоки переносятся без пересчёта и сверяются с исходной моделью": "A ready quant of this same model: GGUF of any type, Unsloth dynamic ones included, or 4-bit bitsandbytes. Blocks are copied without requantizing and checked against the original model", "Замороженные матрицы хранятся в FP16 или в выбранном кванте. Обучаемые веса читаются из исходного BF16 и хранятся в FP32; замороженные нормировки и эмбеддинги конвертируются в FP16": "Frozen matrices are stored in FP16 or the selected quant format. Trainable weights are read from the original BF16 and stored in FP32; frozen norms and embeddings are converted to FP16", "Свой квант": "Own quant", "Квантование": "Quantization", "Перенос кванта": "Quant import", "Перенос": "Import", "Базы": "Base", "больше нет": "no longer exists", "Больше нет": "No longer exists", "модель изменилась после квантования": "the model changed after quantization", "Модель изменилась после квантования": "The model changed after quantization", "не сверена с нынешней моделью": "not checked against the current model", "Не сверена с нынешней моделью": "Not checked against the current model", "не сверена с моделью": "not checked against the model", "Не сверена с моделью": "Not checked against the model", "база не годится к нынешней модели": "the base does not fit the current model", "База не годится к нынешней модели": "The base does not fit the current model", "нет такой базы": "no such base", "Нет такой базы": "No such base", "идёт квантование": "quantization is running", "Идёт квантование": "Quantization is running", "на ней идёт обучение": "training is running on it", "На ней идёт обучение": "Training is running on it", "базы": "base", "Замороженные матрицы слоёв читаются из кванта и разворачиваются в FP16 на карте; обучаемое — из исходного BF16, нормировки и эмбеддинги — FP16": "Frozen layer matrices are read from the quant and unpacked as FP16 on the GPU; trainable weights come from the original BF16; norms and embeddings are stored in FP16", "ошибка весов": "weight error", "Ошибка весов": "Weight error", "замороженные веса — как в файле модели": "frozen weights as in the model file", "Замороженные веса — как в файле модели": "Frozen weights as in the model file", "Замороженное": "Frozen", "8,5 бита на вес: от BF16 почти не отличить": "8.5 bits per weight: nearly indistinguishable from BF16", "6,6 бита на вес: потери малы": "6.6 bits per weight: small loss", "5,5 бита, v_proj и часть down_proj — Q6_K": "5.5 bits, v_proj and part of down_proj in Q6_K", "5,5 бита на вес": "5.5 bits per weight", "4,5 бита, v_proj и часть down_proj — Q6_K: обычный выбор для разговора": "4.5 bits, v_proj and part of down_proj in Q6_K: the usual pick for chat", "4,5 бита, первые слои — Q5_K: самый лёгкий": "4.5 bits, first layers in Q5_K: the lightest", "нет исходной модели: база строится и сверяется по ней": "no original model: a base is built and checked against it", "Нет исходной модели: база строится и сверяется по ней": "No original model: a base is built and checked against it", "движок не собран или собран из прежней версии": "the engine is not built or is from an older version", "Движок не собран или собран из прежней версии": "The engine is not built or is from an older version", "движок не умеет эту модель": "the engine does not support this model", "Движок не умеет эту модель": "The engine does not support this model", "в пути к папке баз пробел": "the base folder path contains a space", "В пути к папке баз пробел": "The base folder path contains a space", "рецепт": "recipe", "Рецепт": "Recipe", "не знаю": "is unknown", "Не знаю": "Is unknown", "в модели нет матриц слоёв": "the model has no layer matrices", "В модели нет матриц слоёв": "The model has no layer matrices", "на диске свободно": "free on disk:", "На диске свободно": "Free on disk:", ", нужно до": ", needed up to", ", нужно": ", needed", "нужен файл .gguf, .safetensors или папка модели bitsandbytes": "a .gguf or .safetensors file or a bitsandbytes model folder is needed", "Нужен файл .gguf, .safetensors или папка модели bitsandbytes": "A .gguf or .safetensors file or a bitsandbytes model folder is needed", "это исходная модель, а не квант": "this is the original model, not a quant", "Это исходная модель, а не квант": "This is the original model, not a quant", "запуск учился на базе": "the run trained on base", "Запуск учился на базе": "The run trained on base", ", а её больше нет": ", which no longer exists", "Обучению не мешает: задание идёт с низким приоритетом": "Does not block training: the job runs at low priority", "Сверка": "Check", "Дообучение моделей, которые не помещаются в видеопамять": "Fine-tuning models that do not fit in VRAM", "Веса лежат в ОЗУ и подаются на карту по слою с опережением, поэтому 12B в BF16 учится на 8 ГБ. Меняются собственные веса модели, а не только адаптер. AMD и nvidia.": "Weights are stored in RAM and passed to the GPU layer by layer ahead of use, so a 12B model in BF16 trains on 8 GB. The model's own weights change, not just an adapter. AMD and nvidia.", "Завершить работу программы": "Quit the program", "Объект обучения": "Training target", "Вернуть значения, подобранные автоподбором.": "Restore the values picked by auto-tuning.", "Классическая LoRA (Hu и др., 2021), как в PEFT: W + α/r · B·A поверх замороженных весов. A инициализируется равномерно в ±1/√вход, B — нулями, оптимизатор AdamW в FP32. Исходные веса не изменяются, адаптеры сохраняются отдельным файлом.": "Classic LoRA (Hu et al., 2021), as in PEFT: W + α/r · B·A on top of frozen weights. A is initialized uniformly in ±1/√in, B with zeros, optimizer AdamW in FP32. The original weights are not modified; adapters are saved as a separate file.", "Значения по умолчанию: r 16, α 32, все матрицы, dropout 0,05.": "Default values: r 16, α 32, all matrices, dropout 0.05.", "Число направлений поправки в каждой матрице. Чем больше ранг, тем больше обучаемых чисел и расход видеопамяти.": "The number of update directions in each matrix. The higher the rank, the more trainable numbers and the more VRAM is used.", "Масштаб поправки: s = α / r. Обычно α = 2r.": "Update scale: s = α / r. Usually α = 2r.", "Доля входа адаптера, обнуляемая при обучении. На валидации и в разговоре не применяется.": "Share of the adapter input zeroed during training. Not applied in validation or chat.", "Матрицы, на которых устанавливаются адаптеры: q, k, v, o — внимание; gate, up, down — MLP; во всех слоях. В слоях полного внимания отдельной v_proj нет: V берётся из K.": "Matrices that receive adapters: q, k, v, o — attention; gate, up, down — MLP; in all layers. Full-attention layers have no separate v_proj: V is taken from K.", "Один пример на строку: {\"instruction\": \"…\", \"input\": \"…\", \"output\": \"…\"}. Файл используется только для чтения.": "One example per line: {\"instruction\": \"…\", \"input\": \"…\", \"output\": \"…\"}. The file is used for reading only.", "С системной инструкцией, %": "With system instruction, %", "Доля примеров, в которых поле instruction передаётся как системная инструкция. В остальных примерах она отсутствует, благодаря чему манера ответов сохраняется и в обычном разговоре.": "Share of examples in which the instruction field is passed as the system instruction. The remaining examples have none, so the style of answers also holds in ordinary conversation.", "Всегда с системной инструкцией": "Always with system instruction", "Системная инструкция добавляется к примеру всегда, если в его поле instruction встречается одно из слов.": "The system instruction is always added to an example if its instruction field contains one of the words.", "Запас видеопамяти, бюджет среза строк, режим ×3 (FP16/BF16), опережающая подкачка.": "VRAM reserve, row slice budget, ×3 mode (FP16/BF16), prefetch.", "Выбор запуска для просмотра.": "Selection of the run to view.", "Первые точки появятся через несколько секунд.": "The first points will appear in a few seconds.", "Обучений пока нет": "No training runs yet", "Освобождает ОЗУ и видеопамять. Текущий ответ прерывается, при следующем вопросе модель загружается заново.": "Frees RAM and VRAM. The current answer is interrupted; the model is loaded again at the next question.", "Каждый вопрос направляется обученной и исходной моделям, диалог начинается с чистого листа.": "Each question is sent to both the trained and the base model, and the conversation starts from a clean slate.", "1 — результат обучения без изменений, 0 — исходная модель, 1.5 — усиленный эффект.": "1 is the training result unchanged, 0 is the base model, 1.5 is an amplified effect.", "Системная инструкция": "System instruction", "Текст системной инструкции": "System instruction text", "Собственный список": "Custom list", "· Один на строку": "· One per line", "Один на строку": "One per line", "Один вопрос на строку. Ожидаемый ответ указывается после « || » и используется для поиска совпадения.": "One question per line. The expected answer is given after « || » and is used to find a match.", "Не показано выше": "Not shown above", "40 шагов с растущим LR; берётся десятая часть LR у минимума сглаженного loss.": "40 steps with a growing LR; a tenth of the LR at the smoothed-loss minimum is taken.", "Loss по шагам без сглаживания": "Loss per step, unsmoothed", "Доля верно предсказанных токенов ответа.": "Share of correctly predicted answer tokens.", "Число токенов на шаг и число токенов в loss.": "Number of tokens per step and number of tokens in the loss.", "Остальное — токены вопроса": "The rest are question tokens", "Состав памяти": "Memory breakdown", "Живая память и область видеопамяти для промежуточных данных слоя; рост значения «Мимо области» означает фрагментацию памяти.": "Live memory and the VRAM arena for the intermediate data of a layer; a growing “Outside the arena” value means memory fragmentation.", "Менее 200 МиБ — риск сбоя": "Below 200 MiB risks a failure", "Обучаемая часть в видеопамяти": "Trainable part in VRAM", "Тензоров, постоянно находящихся в памяти": "Tensors permanently held in memory", "Мимо области": "Outside the arena", "Не зафиксировано": "None recorded", "Троттлинг зафиксирован": "Throttling recorded", "От начала запуска до текущего момента": "From the start of the run until now", "Всплески соответствуют валидации и записи снимка весов.": "Spikes correspond to validation and weight snapshot writes.", "Распределение времени шага": "Breakdown of step time", "Мера того, насколько сильно требуется изменить веса.": "A measure of how strongly the weights need to change.", "Разогрев в начале, затем плавное снижение.": "Warm-up first, then a smooth decay.", "Обучаемая часть": "Trainable part", "обучаемая часть": "trainable part", "Срез строк": "Row slice", "срез строк": "row slice", "среза строк": "row slice", "Среза строк": "Row slice", "срезе строк": "row slice", "Срезе строк": "Row slice", "системной инструкции": "system instruction", "системную инструкцию": "system instruction", "системная инструкция": "system instruction", "С системной инструкцией": "With system instruction", "Подмешивание": "Base-answer replay", "подмешивание": "base-answer replay", "подмешиванием": "base-answer replay", "подмешивания": "base-answer replay", "Защита общих знаний": "Knowledge protection", "защита общих знаний": "knowledge protection", "Подкачка весов": "Weight streaming", "подкачка весов": "weight streaming", "Опережающая подкачка": "Prefetch", "опережающая подкачка": "prefetch", "Опережающая подкачка следующего слоя": "Next-layer prefetch", "Область видеопамяти": "VRAM arena", "область видеопамяти": "VRAM arena", "Снимок весов": "Weight snapshot", "снимок весов": "weight snapshot", "снимка весов": "weight snapshot", "Снимка весов": "Weight snapshot", "Снимок весов каждые": "Weight snapshot every", "снимок весов каждые": "weight snapshot every", "Режим ×3": "×3 mode", "режим ×3": "×3 mode", "Автоподбор": "Auto-tuning", "автоподбор": "auto-tuning", "автоподбором": "auto-tuning", "Режим эксперта": "Expert mode", "режим эксперта": "expert mode", "Сохранение снимка": "Snapshot saving", "Потеря (loss)": "Loss", "Перплексия (ppl)": "Perplexity (ppl)"}, ks=Object.keys(D).sort((a,b)=>b.length-a.length), L='а-яёА-ЯЁ', C=/[а-яё]/i, E=s=>s.replace(/[.*+?^${}()|[\]\\]/g,'\\$&');
  // граница слова — только перед буквой: кусок «; доля — …» стоит сразу за словом и с ней не нашёлся бы
  const re=ks.length? new RegExp('('+ks.filter(s=>!C.test(s[0])).map(s=>E(s)+(C.test(s[s.length-1])?'(?!['+L+'])':'')+'|').join('')
    +'(?<!['+L+'])(?:'+ks.filter(s=>C.test(s[0])).map(E).join('|')+')(?!['+L+']))','g') : null;
  const tr=window.T=s=>re&&/[а-яё]/i.test(s)? s.replace(re,m=>D[m]) : s;
  const AT=['title','placeholder','aria-label','data-empty','data-tip'], SKIP='pre,textarea,script,style,.msg-text,.raw,code';
  const el=n=>{ for(const a of AT){ const v=n.getAttribute(a); if(v){ const t=tr(v); if(t!==v) n.setAttribute(a,t); } } };
  const walk=n=>{ if(n.nodeType===3){ const p=n.parentElement; if(p&&!p.closest(SKIP)){ const t=tr(n.data); if(t!==n.data) n.data=t; } return; }
    if(n.nodeType!==1) return; el(n); if(n.matches(SKIP)) return; for(const c of n.childNodes) walk(c); };
  new MutationObserver(ms=>{ for(const m of ms){ if(m.type==='childList') m.addedNodes.forEach(walk); else if(m.type==='characterData') walk(m.target); else el(m.target); } })
    .observe(document.documentElement,{subtree:true,childList:true,characterData:true,attributes:true,attributeFilter:AT});
} else window.T=s=>s;
</script>
<script>window.APPEAR=/*APPEAR*/{}; window.UI=/*UI*/{};
// Оформление сводится к переменным CSS и ставится до первой отрисовки — без мигания при запуске.
window.AP_DEF={theme:'dark',bg:'aurora',solid:'#0e0f13',img:0,dim:.35,iblur:0,accent:'#0a84ff',radius:16,line:.5,glass:.62,blur:28,
  shadow:1,ff:'inter',fs:13.5,weight:400,density:1,side:'left',sidew:248,wide:0,full:'gap',motion:1,lw:1.75,cfill:1,cgrid:1,cdots:1,css:''};
window.AP_BG={
 aurora:{n:'Сияние',d:'radial-gradient(1200px 700px at 12% 0%,#1d2b5a 0%,transparent 60%),radial-gradient(900px 600px at 95% 15%,#3a1f4f 0%,transparent 55%),radial-gradient(1000px 800px at 60% 110%,#0f3b3a 0%,transparent 60%),#0a0b10',
   l:'radial-gradient(1200px 700px at 12% 0%,#d8e3ff 0%,transparent 60%),radial-gradient(900px 600px at 95% 15%,#f1dcf2 0%,transparent 55%),radial-gradient(1000px 800px at 60% 110%,#d5f0ea 0%,transparent 60%),#f5f5f7'},
 ocean:{n:'Океан',d:'radial-gradient(1100px 800px at 0% 100%,#0b3d5c 0%,transparent 60%),radial-gradient(900px 700px at 100% 0%,#123a6b 0%,transparent 60%),#070b12',
   l:'radial-gradient(1100px 800px at 0% 100%,#cfe9f7 0%,transparent 60%),radial-gradient(900px 700px at 100% 0%,#d9e4fb 0%,transparent 60%),#f3f6fa'},
 dawn:{n:'Рассвет',d:'radial-gradient(1100px 700px at 20% 0%,#4a2a3a 0%,transparent 60%),radial-gradient(900px 700px at 90% 100%,#3b2c22 0%,transparent 60%),#120d10',
   l:'radial-gradient(1100px 700px at 20% 0%,#fbe1e6 0%,transparent 60%),radial-gradient(900px 700px at 90% 100%,#fdeedd 0%,transparent 60%),#faf6f4'},
 ember:{n:'Угли',d:'radial-gradient(1000px 700px at 85% 110%,#4d1d0e 0%,transparent 60%),radial-gradient(800px 600px at 10% 0%,#2a1430 0%,transparent 60%),#0d0909',
   l:'radial-gradient(1000px 700px at 85% 110%,#fde3d3 0%,transparent 60%),radial-gradient(800px 600px at 10% 0%,#efe0f5 0%,transparent 60%),#faf7f5'},
 lavender:{n:'Лаванда',d:'radial-gradient(1200px 800px at 50% -10%,#30295c 0%,transparent 65%),#0c0b14',
   l:'radial-gradient(1200px 800px at 50% -10%,#e6e1fb 0%,transparent 65%),#f7f6fc'},
 mist:{n:'Туман',d:'linear-gradient(180deg,#1a1d22 0%,#0e1013 100%)',l:'linear-gradient(180deg,#eef1f5 0%,#e3e8ee 100%)'},
 ink:{n:'Тушь',d:'radial-gradient(900px 600px at 50% -10%,#24262b 0%,transparent 70%),#050506',l:'radial-gradient(900px 600px at 50% -10%,#ffffff 0%,transparent 70%),#e9e9eb'},
 plain:{n:'Ровный',d:'#0b0c10',l:'#f5f5f7'}};
window.AP_FONTS={inter:['Inter','"Inter"'],golos:['Golos','"Golos Text"'],onest:['Onest','"Onest"'],manrope:['Manrope','"Manrope"'],
  jost:['Jost','"Jost"'],geologica:['Geologica','"Geologica"'],system:['Системный','system-ui'],mono:['Моноширинный','ui-monospace,"JetBrains Mono","DejaVu Sans Mono"']};
window.AP_PRE=[
 ['Ночь',{theme:'dark',bg:'aurora',accent:'#0a84ff',radius:16,line:.5,glass:.62,blur:28,ff:'inter',fs:13.5,weight:400,density:1}],
 ['Океан',{theme:'dark',bg:'ocean',accent:'#64d2ff',radius:18,line:.5,glass:.55,blur:34,ff:'onest',fs:13.5,weight:400,density:1}],
 ['Графит',{theme:'dark',bg:'ink',accent:'#e5e5ea',radius:12,line:1,glass:.88,blur:20,ff:'inter',fs:13.5,weight:400,density:1}],
 ['Угли',{theme:'dark',bg:'ember',accent:'#ff9f0a',radius:16,line:.5,glass:.6,blur:30,ff:'manrope',fs:13.5,weight:400,density:1}],
 ['Терминал',{theme:'dark',bg:'plain',accent:'#30d158',radius:6,line:1,glass:1,blur:0,ff:'mono',fs:13,weight:400,density:.85,shadow:0}],
 ['Контраст',{theme:'dark',bg:'plain',accent:'#ffd60a',radius:10,line:1.5,glass:1,blur:0,ff:'inter',fs:14.5,weight:500,density:1,shadow:0}],
 ['Пудра',{theme:'light',bg:'dawn',accent:'#c06c84',radius:20,line:.5,glass:.72,blur:30,ff:'golos',fs:13.5,weight:400,density:1.1}],
 ['Бумага',{theme:'light',bg:'solid',solid:'#f5f2ec',accent:'#1d1d1f',radius:10,line:.5,glass:.92,blur:12,ff:'geologica',fs:14,weight:400,density:1}],
 ['Туман',{theme:'light',bg:'mist',accent:'#007aff',radius:16,line:.5,glass:.6,blur:28,ff:'inter',fs:13.5,weight:400,density:1}],
 ['Лаванда',{theme:'light',bg:'lavender',accent:'#5e5ce6',radius:18,line:.5,glass:.66,blur:30,ff:'onest',fs:13.5,weight:400,density:1.05}]];
window.UI_DEF={start:'auto',notify:1,tray:1,confirmStop:1,dec:'dot',charts:['loss','acc','vram','temp','time','grad'],ccols:0,ch:150,
  chatTemp:.5,chatMax:200,chatScene:'',chatScale:1,presets:[]};
window.apApply=function(a){
  a=Object.assign({},AP_DEF,a||{}); if(typeof a.font==='number'&&a.fs===AP_DEF.fs) a.fs=a.font; delete a.font;
  if(!AP_FONTS[a.ff]) a.ff='inter'; if(!AP_BG[a.bg]&&a.bg!=='solid'&&a.bg!=='image') a.bg='aurora';
  const R=document.documentElement, st=R.style, light=a.theme==='light';
  if(light) R.dataset.theme='light'; else delete R.dataset.theme;
  const hx=String(a.accent||'#0a84ff').replace('#',''), n=parseInt(hx.length==3?hx.replace(/./g,'$&$&'):hx,16)||0x0a84ff, r=n>>16&255,g=n>>8&255,b=n&255;
  const lum=(.2126*r+.7152*g+.0722*b)/255, set=(k,v)=>st.setProperty(k,v);
  set('--accent',a.accent); set('--s1',a.accent);
  set('--accent-2',`color-mix(in srgb,${a.accent} 86%,${light?'#000':'#fff'})`);
  set('--accent-ink',`color-mix(in srgb,${a.accent} ${light?82:80}%,${light?'#000':'#fff'})`);
  set('--on-accent',lum>.6?'#111':'#fff');
  set('--wash',`rgba(${r},${g},${b},${light?.1:.16})`); set('--wash-2',`rgba(${r},${g},${b},${light?.17:.26})`); set('--ring-c',`rgba(${r},${g},${b},.45)`);
  let wall;
  if(a.bg==='image'&&a.img){ const k=light?'255,255,255':'0,0,0'; wall=`linear-gradient(rgba(${k},${a.dim}),rgba(${k},${a.dim})),url(/bg?t=${a.img}) center/cover no-repeat`; }
  else if(a.bg==='solid') wall=a.solid; else { const B=AP_BG[a.bg]||AP_BG.aurora; wall=light?B.l:B.d; }
  set('--wall',wall); set('--iblur',(a.bg==='image'?a.iblur:0)+'px');
  const rr=Math.max(0,+a.radius||0); set('--r',rr+'px'); set('--r2',Math.round(rr*.62)+'px'); set('--rc',Math.max(4,Math.min(12,Math.round(rr*.5)))+'px'); set('--rs',Math.max(3,Math.round(rr*.36))+'px');
  set('--hair',a.line+'px');
  const al=Math.min(1,.3+.68*a.glass);
  set('--card',light?`rgba(255,255,255,${al.toFixed(3)})`:`rgba(40,40,46,${al.toFixed(3)})`);
  set('--mat',light?`rgba(246,246,249,${Math.min(1,al*.92).toFixed(3)})`:`rgba(26,26,31,${Math.min(1,al*.95).toFixed(3)})`);
  set('--pop',light?'rgba(252,252,253,.94)':'rgba(46,46,52,.94)');
  set('--blur',Math.max(0,+a.blur||0)+'px');
  const sh=+a.shadow||0;
  set('--shadow', sh===0? 'none' : light? (sh>1? '0 1px 2px rgba(0,0,0,.06),0 18px 44px -14px rgba(0,0,0,.22)' : '0 1px 2px rgba(0,0,0,.04),0 10px 30px -14px rgba(0,0,0,.14)')
    : (sh>1? '0 1px 0 rgba(255,255,255,.05) inset,0 24px 50px -16px rgba(0,0,0,.7)' : '0 1px 0 rgba(255,255,255,.04) inset,0 14px 34px -16px rgba(0,0,0,.55)'));
  set('--ff',AP_FONTS[a.ff][1]+',"Inter",system-ui,sans-serif'); set('--fs',a.fs+'px'); set('--wt',a.weight); set('--wth',Math.min(+a.weight+190,700)); set('--wtb',Math.min(+a.weight+240,760));
  set('--d',a.density); set('--sidew',Math.max(200,Math.min(340,+a.sidew||248))+'px'); set('--maxw',a.wide? '100%' : '1560px');
  set('--lw',a.lw); R.dataset.side=a.side==='right'?'right':'left';
  R.classList.toggle('nofill',!+a.cfill); R.classList.toggle('nogrid',!+a.cgrid);
  R.classList.toggle('nodots',!+a.cdots); R.classList.toggle('calm',!+a.motion);
  const css=()=>{ let s=document.getElementById('user-css'); if(!s){ s=document.createElement('style'); s.id='user-css'; document.head.appendChild(s); } s.textContent=a.css||''; };
  document.head? css() : addEventListener('DOMContentLoaded',css);
  { const ink=light?'#1d1d1f':'#f5f5f7', svg=`<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64"><g stroke-linejoin="round" stroke-width="2.4"><polygon points="7,18 32,5.5 57,18 32,30.5" fill="${ink}" stroke="${ink}" opacity=".46"/><polygon points="15.5,29.75 32,38 57,25.5 57,31 32,43.5 15.5,35.25" fill="${a.accent}" stroke="${a.accent}"/><polygon points="7,38 32,50.5 49,42 55,45 32,56.5 7,45" fill="${ink}" stroke="${ink}" opacity=".3"/></g></svg>`;
    const fav=()=>{ let l=document.querySelector('link[rel=icon]'); if(!l){ l=document.createElement('link'); l.rel='icon'; document.head.appendChild(l); } l.href='data:image/svg+xml,'+encodeURIComponent(svg); };
    document.head? fav() : addEventListener('DOMContentLoaded',fav); }
  window.AP=a; return a; };
window.UI=Object.assign({},UI_DEF,window.UI||{});
if(!('theme' in window.APPEAR)) window.APPEAR.theme=document.documentElement.dataset.theme==='light'?'light':'dark';
apApply(window.APPEAR);
</script>
<style>
/* ===================== шрифты: все под OFL, лежат рядом с программой ===================== */
@font-face{font-family:"Inter";font-weight:100 900;font-display:swap;src:url("/font/inter-latin.woff2") format("woff2");unicode-range:U+0000-00FF,U+0131,U+0152-0153,U+02BB-02BC,U+02C6,U+02DA,U+02DC,U+0304,U+0308,U+0329,U+2000-206F,U+20AC,U+2122,U+2191,U+2193,U+2212,U+2215,U+FEFF,U+FFFD}
@font-face{font-family:"Inter";font-weight:100 900;font-display:swap;src:url("/font/inter-cyrillic.woff2") format("woff2");unicode-range:U+0301,U+0400-045F,U+0490-0491,U+04B0-04B1,U+2116}
@font-face{font-family:"Golos Text";font-weight:400 900;font-display:swap;src:url("/font/golos-text-latin.woff2") format("woff2");unicode-range:U+0000-00FF,U+0131,U+0152-0153,U+02BB-02BC,U+02C6,U+02DA,U+02DC,U+0304,U+0308,U+0329,U+2000-206F,U+20AC,U+2122,U+2191,U+2193,U+2212,U+2215,U+FEFF,U+FFFD}
@font-face{font-family:"Golos Text";font-weight:400 900;font-display:swap;src:url("/font/golos-text-cyrillic.woff2") format("woff2");unicode-range:U+0301,U+0400-045F,U+0490-0491,U+04B0-04B1,U+2116}
@font-face{font-family:"Onest";font-weight:100 900;font-display:swap;src:url("/font/onest-latin.woff2") format("woff2");unicode-range:U+0000-00FF,U+0131,U+0152-0153,U+02BB-02BC,U+02C6,U+02DA,U+02DC,U+0304,U+0308,U+0329,U+2000-206F,U+20AC,U+2122,U+2191,U+2193,U+2212,U+2215,U+FEFF,U+FFFD}
@font-face{font-family:"Onest";font-weight:100 900;font-display:swap;src:url("/font/onest-cyrillic.woff2") format("woff2");unicode-range:U+0301,U+0400-045F,U+0490-0491,U+04B0-04B1,U+2116}
@font-face{font-family:"Manrope";font-weight:200 800;font-display:swap;src:url("/font/manrope-latin.woff2") format("woff2");unicode-range:U+0000-00FF,U+0131,U+0152-0153,U+02BB-02BC,U+02C6,U+02DA,U+02DC,U+0304,U+0308,U+0329,U+2000-206F,U+20AC,U+2122,U+2191,U+2193,U+2212,U+2215,U+FEFF,U+FFFD}
@font-face{font-family:"Manrope";font-weight:200 800;font-display:swap;src:url("/font/manrope-cyrillic.woff2") format("woff2");unicode-range:U+0301,U+0400-045F,U+0490-0491,U+04B0-04B1,U+2116}
@font-face{font-family:"Jost";font-weight:100 900;font-display:swap;src:url("/font/jost-latin.woff2") format("woff2");unicode-range:U+0000-00FF,U+0131,U+0152-0153,U+02BB-02BC,U+02C6,U+02DA,U+02DC,U+0304,U+0308,U+0329,U+2000-206F,U+20AC,U+2122,U+2191,U+2193,U+2212,U+2215,U+FEFF,U+FFFD}
@font-face{font-family:"Jost";font-weight:100 900;font-display:swap;src:url("/font/jost-cyrillic.woff2") format("woff2");unicode-range:U+0301,U+0400-045F,U+0490-0491,U+04B0-04B1,U+2116}
@font-face{font-family:"Geologica";font-weight:100 900;font-display:swap;src:url("/font/geologica-latin.woff2") format("woff2");unicode-range:U+0000-00FF,U+0131,U+0152-0153,U+02BB-02BC,U+02C6,U+02DA,U+02DC,U+0304,U+0308,U+0329,U+2000-206F,U+20AC,U+2122,U+2191,U+2193,U+2212,U+2215,U+FEFF,U+FFFD}
@font-face{font-family:"Geologica";font-weight:100 900;font-display:swap;src:url("/font/geologica-cyrillic.woff2") format("woff2");unicode-range:U+0301,U+0400-045F,U+0490-0491,U+04B0-04B1,U+2116}

/* ===================== постоянные цвета темы; изменяемые ставит apApply ===================== */
:root{
  color-scheme:dark;
  --ink:rgba(255,255,255,.93); --ink-2:rgba(235,235,245,.64); --ink-3:rgba(235,235,245,.42); --ink-4:rgba(235,235,245,.22);
  --sep:rgba(255,255,255,.075); --edge:rgba(255,255,255,.085); --edge-2:rgba(255,255,255,.11);
  --fill:rgba(255,255,255,.065); --fill-2:rgba(255,255,255,.1); --fill-3:rgba(255,255,255,.15); --field:rgba(0,0,0,.18);
  --seg-on:rgba(255,255,255,.17); --solid:#2b2b31; --scrim:rgba(0,0,0,.42);
  --ok:#32d74b; --warn:#ffd60a; --bad:#ff453a; --info:#64d2ff;
  --ok-w:rgba(50,215,75,.14); --warn-w:rgba(255,214,10,.14); --bad-w:rgba(255,69,58,.14);
  --s2:#ff9f0a; --s3:#ff6482; --s4:#64d2ff; --s5:#bf5af2; --s6:#30d158;
  --mono:ui-monospace,"JetBrains Mono","DejaVu Sans Mono","Liberation Mono",monospace;
  --pad:calc(12px*var(--d)); --g:calc(12px*var(--d)); --ch:calc(26px + 4px*var(--d)); --wr:12px;
  --ease:cubic-bezier(.2,.75,.25,1);
}
:root[data-theme="light"]{
  color-scheme:light;
  --ink:#1d1d1f; --ink-2:rgba(60,60,67,.74); --ink-3:rgba(60,60,67,.52); --ink-4:rgba(60,60,67,.28);
  --sep:rgba(0,0,0,.075); --edge:rgba(0,0,0,.075); --edge-2:rgba(0,0,0,.1);
  --fill:rgba(0,0,0,.045); --fill-2:rgba(0,0,0,.075); --fill-3:rgba(0,0,0,.11); --field:rgba(255,255,255,.75);
  --seg-on:#fff; --solid:#ffffff; --scrim:rgba(20,20,25,.22);
  --ok:#1f9d3a; --warn:#b97400; --bad:#e0352b; --info:#0a7cc2;
  --ok-w:rgba(31,157,58,.11); --warn-w:rgba(185,116,0,.11); --bad-w:rgba(224,53,43,.1);
  --s2:#d67a00; --s3:#d63864; --s4:#0a8fc8; --s5:#9a3cd0; --s6:#1f9d3a;
}

/* ===================== основа ===================== */
*{box-sizing:border-box}
[hidden]{display:none!important}
html,body{margin:0;height:100%;overflow:hidden}
body{background:#0b0c10;color:var(--ink);font-family:var(--ff);font-size:var(--fs);font-weight:var(--wt);line-height:1.45;
  -webkit-font-smoothing:antialiased;letter-spacing:-.003em;font-feature-settings:"cv11","ss03"}
:root[data-theme="light"] body{background:#f5f5f7}
b,strong{font-weight:var(--wtb)}
h1,h2,h3,h4{margin:0;font-weight:var(--wth);letter-spacing:-.012em}
p{margin:0}
a{color:var(--accent-ink);text-decoration:none}
a:hover{text-decoration:underline}
::selection{background:var(--wash-2)}
:focus{outline:none}
:focus-visible{outline:2px solid var(--ring-c);outline-offset:1px;border-radius:var(--rc)}
.num,td,.tv,.big{font-variant-numeric:tabular-nums}
code,.mono,pre{font-family:var(--mono);font-size:calc(var(--fs) - 1.5px)}
pre{margin:0;white-space:pre-wrap;word-break:break-word}
.nw{white-space:nowrap}
.grow{flex:1}
.muted{color:var(--ink-2)} .faint{color:var(--ink-3)}
.ok-c{color:var(--ok)} .bad-c{color:var(--bad)} .warn-c{color:var(--warn)}
*::-webkit-scrollbar{width:11px;height:11px}
*::-webkit-scrollbar-thumb{background:var(--fill-3);border-radius:8px;border:3px solid transparent;background-clip:padding-box}
*::-webkit-scrollbar-thumb:hover{background:var(--ink-4);background-clip:padding-box;border:3px solid transparent}
*::-webkit-scrollbar-track,*::-webkit-scrollbar-corner{background:transparent}
.ic{width:16px;height:16px;flex:none;fill:none;stroke:currentColor;stroke-width:1.5;stroke-linecap:round;stroke-linejoin:round;display:block}

/* ===================== рама окна, фон, раскладка ===================== */
.frame{position:fixed;inset:0;overflow:hidden;isolation:isolate}
.wall{position:absolute;inset:calc(-2*var(--iblur));background:var(--wall);filter:blur(var(--iblur));z-index:-1}
.app{position:absolute;inset:0;display:flex;gap:var(--pad);padding:var(--pad)}
:root[data-side="right"] .app{flex-direction:row-reverse}
.glass{background:var(--card);-webkit-backdrop-filter:blur(var(--blur)) saturate(175%);backdrop-filter:blur(var(--blur)) saturate(175%);
  box-shadow:inset 0 0 0 var(--hair) var(--edge),var(--shadow)}
.mat{background:var(--mat);-webkit-backdrop-filter:blur(calc(var(--blur) * 1.2)) saturate(180%);backdrop-filter:blur(calc(var(--blur) * 1.2)) saturate(180%);
  box-shadow:inset 0 0 0 var(--hair) var(--edge),var(--shadow)}

/* ----- боковая панель ----- */
.side{width:var(--sidew);flex:none;display:flex;flex-direction:column;gap:2px;padding:calc(14px*var(--d)) calc(10px*var(--d)) calc(10px*var(--d));
  border-radius:var(--r);overflow:auto;user-select:none;-webkit-user-select:none}
.brand{display:flex;align-items:center;gap:10px;padding:2px 8px calc(12px*var(--d))}
.brand .mark{width:26px;height:26px}
.brand .bn{font-size:calc(var(--fs) + 2.5px);font-weight:var(--wth);letter-spacing:-.02em}
.brand .bv{margin-left:auto;font-size:calc(var(--fs) - 2.5px);color:var(--ink-3);padding:1px 7px;border-radius:999px;box-shadow:inset 0 0 0 var(--hair) var(--edge-2)}
.mark .l1{fill:color-mix(in srgb,var(--ink) 46%,transparent);stroke:color-mix(in srgb,var(--ink) 46%,transparent)}
.mark .l2{fill:var(--accent);stroke:var(--accent)}
.mark .l3{fill:color-mix(in srgb,var(--ink) 28%,transparent);stroke:color-mix(in srgb,var(--ink) 28%,transparent)}
.lcard{margin:2px 2px calc(10px*var(--d));padding:10px 12px;border-radius:var(--r2);background:var(--wash);box-shadow:inset 0 0 0 var(--hair) var(--wash-2);cursor:pointer;text-align:left;
  color:var(--ink);display:block;width:auto;font:inherit}
.lcard:hover{background:var(--wash-2)}
.lcard .lh{display:flex;align-items:center;gap:7px;font-weight:var(--wth)}
.lcard .lh b{overflow:hidden;text-overflow:ellipsis;white-space:nowrap;min-width:0;font-weight:var(--wth)}
.lcard .lh span{margin-left:auto;color:var(--accent-ink);font-variant-numeric:tabular-nums;font-size:calc(var(--fs) - 1px)}
.lcard .bar{margin:8px 0 6px}
.lcard .ls{font-size:calc(var(--fs) - 2px);color:var(--ink-2);white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.dot{width:8px;height:8px;border-radius:50%;background:currentColor;flex:none}
.dot.pulse{color:var(--accent);animation:breathe 2.2s ease-in-out infinite}
@keyframes breathe{0%,100%{opacity:1;transform:scale(1)}50%{opacity:.45;transform:scale(.8)}}
.navh{padding:calc(12px*var(--d)) 10px calc(5px*var(--d));font-size:calc(var(--fs) - 2.5px);font-weight:var(--wth);color:var(--ink-3);letter-spacing:.01em}
.ni{display:flex;align-items:center;gap:10px;width:100%;height:calc(30px*var(--d) + 2px);padding:0 10px;border:0;border-radius:var(--rc);background:transparent;
  color:var(--ink-2);font:inherit;font-weight:calc(var(--wt) + 60);text-align:left;cursor:pointer;position:relative}
.ni .ic{width:17px;height:17px;color:var(--ink-3)}
.ni:hover{background:var(--fill);color:var(--ink)}
.ni[aria-current="page"]{background:var(--fill-2);color:var(--ink)}
.ni[aria-current="page"] .ic{color:var(--accent)}
.ni span{flex:1;min-width:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.ni em{font-style:normal;font-size:calc(var(--fs) - 2.5px);color:var(--ink-3);font-variant-numeric:tabular-nums}
.ni .st{width:7px;height:7px;border-radius:50%;flex:none}
.st.ok{background:var(--ok)} .st.need{background:var(--warn)} .st.bad{background:var(--bad)} .st.live{background:var(--accent);animation:breathe 2.2s ease-in-out infinite}
.sfoot{display:flex;align-items:center;gap:4px;padding:10px 4px 0;margin-top:6px;box-shadow:inset 0 var(--hair) 0 var(--sep)}
.sfoot .seg{margin-right:auto}
.disk{padding:6px 10px 2px;font-size:calc(var(--fs) - 2.5px);color:var(--ink-3)}
.disk .bar{margin-top:5px;height:3px}

/* ----- правая часть: полоса заголовка и виды ----- */
.main{flex:1;min-width:0;display:flex;flex-direction:column;gap:var(--pad)}
.top{flex:none;display:flex;align-items:center;gap:8px;min-height:calc(44px*var(--d) + 6px);padding:0 8px 0 calc(16px*var(--d));border-radius:var(--r);position:relative;
  user-select:none;-webkit-user-select:none}
.top .vt{font-size:calc(var(--fs) + 3px);font-weight:var(--wth);letter-spacing:-.02em;white-space:nowrap}
.top .vs{color:var(--ink-3);font-size:calc(var(--fs) - 1px);white-space:nowrap;overflow:hidden;text-overflow:ellipsis;min-width:0}
.top .tact{display:flex;align-items:center;gap:6px;margin-left:auto}
.top .sepv{width:var(--hair);align-self:stretch;margin:12px 4px;background:var(--sep)}
.edge{position:absolute;left:calc(var(--r) * .7);right:calc(var(--r) * .7);bottom:0;height:2px;border-radius:2px;overflow:hidden;pointer-events:none}
.edge i{position:absolute;inset:0 auto 0 0;width:0;background:var(--accent);border-radius:2px}
.views{flex:1;min-height:0;overflow:auto;margin-right:calc(-1*var(--pad));padding-right:max(0px,calc(var(--pad) - 11px));scrollbar-gutter:stable}
.view{max-width:var(--maxw);margin:0 auto;padding-bottom:var(--pad);display:flex;flex-direction:column;gap:var(--g)}

/* ===================== элементы ===================== */
.card{border-radius:var(--r);padding:calc(18px*var(--d)) calc(20px*var(--d));min-width:0;position:relative}
.ch{display:flex;align-items:center;gap:10px;margin-bottom:calc(14px*var(--d));min-height:var(--ch)}
.ch h2{font-size:calc(var(--fs) + 2px)}
.ch .cs{color:var(--ink-3);font-size:calc(var(--fs) - 1px)}
.ch .ca{margin-left:auto;display:flex;gap:6px;align-items:center}
.sec{margin-top:calc(16px*var(--d));padding-top:calc(14px*var(--d));box-shadow:inset 0 var(--hair) 0 var(--sep)}
.sec>h3,.h3{font-size:calc(var(--fs) - .5px);margin-bottom:calc(10px*var(--d));display:flex;align-items:center;gap:8px}
.sec>h3 .ca{margin-left:auto}
.hint{color:var(--ink-3);font-size:calc(var(--fs) - 1.5px)}
.cols{display:grid;gap:var(--g);align-items:start}

.btn{display:inline-flex;align-items:center;justify-content:center;gap:6px;height:var(--ch);padding:0 calc(12px*var(--d) + 1px);border:0;border-radius:var(--rc);
  background:var(--fill);color:var(--ink);font:inherit;font-weight:calc(var(--wt) + 100);white-space:nowrap;cursor:pointer;transition:background .12s,color .12s,box-shadow .12s}
.btn:hover{background:var(--fill-2)} .btn:active{background:var(--fill-3)}
.btn:disabled{opacity:.42;cursor:default;pointer-events:none}
.btn.primary{background:var(--accent);color:var(--on-accent)} .btn.primary:hover{background:var(--accent-2)}
.btn.tint{background:var(--wash);color:var(--accent-ink)} .btn.tint:hover{background:var(--wash-2)}
.btn.quiet{background:transparent;color:var(--ink-2)} .btn.quiet:hover{background:var(--fill);color:var(--ink)}
.btn.danger{color:var(--bad)} .btn.danger:hover{background:var(--bad-w)}
.btn.danger.solid{background:var(--bad);color:#fff}
.btn.icon{width:var(--ch);padding:0}
.btn.sm{height:calc(var(--ch) - 6px);padding:0 9px;font-size:calc(var(--fs) - 1px)} .btn.sm.icon{width:calc(var(--ch) - 6px);padding:0}
.btn.lg{height:calc(var(--ch) + 10px);padding:0 20px;font-size:calc(var(--fs) + .5px);border-radius:calc(var(--rc) + 2px)}
.btn.wide{width:100%}
.btn .kbd{margin-left:4px}
.kbd,kbd{display:inline-flex;align-items:center;justify-content:center;min-width:20px;height:19px;padding:0 5px;border-radius:5px;font:500 calc(var(--fs) - 3px)/1 var(--ff);
  color:var(--ink-3);box-shadow:inset 0 0 0 var(--hair) var(--edge-2),0 1px 0 var(--edge);background:var(--fill)}
.btn.primary .kbd{color:inherit;opacity:.75;background:rgba(255,255,255,.14);box-shadow:none}

input.inp,select.inp,textarea.inp{width:100%;min-width:0;height:var(--ch);padding:0 10px;border:0;border-radius:var(--rc);background:var(--field);color:var(--ink);font:inherit;
  box-shadow:inset 0 0 0 var(--hair) var(--edge-2);transition:box-shadow .12s}
textarea.inp{height:auto;padding:8px 10px;resize:vertical;line-height:1.45}
.inp:focus{box-shadow:inset 0 0 0 1px var(--accent),0 0 0 3px var(--wash)}
.inp::placeholder{color:var(--ink-3)}
.inp[aria-invalid="true"]{box-shadow:inset 0 0 0 1px var(--bad),0 0 0 3px var(--bad-w)}
select.inp{appearance:none;-webkit-appearance:none;padding-right:28px;cursor:pointer;
  background-image:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 16 16'%3E%3Cpath d='M5 6.5 8 9.5 11 6.5' fill='none' stroke='%238e8e93' stroke-width='1.6' stroke-linecap='round' stroke-linejoin='round'/%3E%3C/svg%3E");
  background-repeat:no-repeat;background-position:right 8px center;background-size:14px}
select.inp option{background:var(--solid);color:var(--ink)}
input[type=color].sw-c{width:30px;height:30px;padding:0;border:0;border-radius:50%;background:conic-gradient(red,yellow,lime,cyan,blue,magenta,red);cursor:pointer;overflow:hidden}
input[type=color].sw-c::-webkit-color-swatch-wrapper{padding:3px} input[type=color].sw-c::-webkit-color-swatch{border:0;border-radius:50%}

.seg{display:inline-flex;align-items:center;padding:2px;gap:2px;border-radius:calc(var(--rc) + 2px);background:var(--fill);flex-wrap:nowrap;max-width:100%}
.seg button{height:calc(var(--ch) - 4px);padding:0 calc(11px*var(--d) + 1px);border:0;border-radius:var(--rc);background:transparent;color:var(--ink-2);font:inherit;
  font-weight:calc(var(--wt) + 60);cursor:pointer;white-space:nowrap;transition:background .12s,color .12s}
.seg button:hover{color:var(--ink)}
.seg button[aria-checked="true"]{background:var(--seg-on);color:var(--ink);box-shadow:0 1px 2px rgba(0,0,0,.18),inset 0 0 0 var(--hair) var(--edge)}
.seg.mini button{height:20px;padding:0 8px;font-size:calc(var(--fs) - 2.5px)}
.seg.full{display:flex} .seg.full button{flex:1}

input.sw{appearance:none;-webkit-appearance:none;width:34px;height:20px;margin:0;border-radius:10px;background:var(--fill-3);position:relative;cursor:pointer;flex:none;transition:background .18s}
input.sw::before{content:"";position:absolute;top:2px;left:2px;width:16px;height:16px;border-radius:50%;background:#fff;box-shadow:0 1px 3px rgba(0,0,0,.3);transition:transform .18s var(--ease)}
input.sw:checked{background:var(--accent)} input.sw:checked::before{transform:translateX(14px)}
input.sw:focus-visible{outline:2px solid var(--ring-c);outline-offset:2px}

input[type=range].rng{appearance:none;-webkit-appearance:none;width:100%;height:4px;margin:8px 0;border-radius:2px;cursor:pointer;
  background:linear-gradient(var(--accent),var(--accent)) 0/var(--v,50%) 100% no-repeat,var(--fill-3)}
input[type=range].rng::-webkit-slider-thumb{appearance:none;-webkit-appearance:none;width:16px;height:16px;border-radius:50%;background:#fff;box-shadow:0 0 0 .5px rgba(0,0,0,.2),0 1px 4px rgba(0,0,0,.35)}
input[type=range].rng:disabled{opacity:.4}

.chips{display:flex;flex-wrap:wrap;gap:5px}
.chip{display:inline-flex;align-items:center;gap:4px;height:22px;padding:0 8px;border-radius:999px;background:var(--fill);color:var(--ink-2);font-size:calc(var(--fs) - 2px);white-space:nowrap}
.chip b{color:var(--ink);font-weight:calc(var(--wt) + 160)}
.pill{display:inline-flex;align-items:center;gap:6px;height:22px;padding:0 9px;border-radius:999px;background:var(--fill);color:var(--ink-2);
  font-size:calc(var(--fs) - 2px);font-weight:calc(var(--wt) + 120);white-space:nowrap}
.pill i{width:6px;height:6px;border-radius:50%;background:currentColor}
.pill.ok{color:var(--ok);background:var(--ok-w)} .pill.bad{color:var(--bad);background:var(--bad-w)} .pill.warn{color:var(--warn);background:var(--warn-w)}
.pill.live{color:var(--accent-ink);background:var(--wash)} .pill.live i{animation:breathe 2.2s ease-in-out infinite}

.bar{position:relative;height:4px;border-radius:2px;background:var(--fill-2);overflow:hidden}
.bar>i{position:absolute;inset:0 auto 0 0;width:0;border-radius:inherit;background:var(--accent);transition:width .55s var(--ease)}
.bar.ok>i{background:var(--ok)} .bar.bad>i{background:var(--bad)} .bar.off{opacity:.35}
.bar.ind>i{width:32%!important;animation:slide 1.5s ease-in-out infinite}
@keyframes slide{0%{transform:translateX(-110%)}100%{transform:translateX(330%)}}

.tiles{display:grid;grid-template-columns:repeat(auto-fill,minmax(calc(146px*var(--d) + 12px),1fr));gap:calc(8px*var(--d))}
.tile{padding:calc(10px*var(--d)) 12px;border-radius:var(--r2);background:var(--fill);min-width:0}
.tile .tl{display:block;font-size:calc(var(--fs) - 2px);color:var(--ink-3);white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.tile .tv{display:block;margin-top:2px;font-size:calc(var(--fs) + 4px);font-weight:var(--wth);letter-spacing:-.02em;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.tile .tv small{font-size:calc(var(--fs) - 2px);font-weight:var(--wt);color:var(--ink-3);margin-left:2px;letter-spacing:0}
.tile .tn{display:block;font-size:calc(var(--fs) - 2.5px);color:var(--ink-3);margin-top:1px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}

.rows{display:flex;flex-direction:column}
.kv{display:grid;grid-template-columns:18px minmax(96px,34%) minmax(0,1fr);align-items:center;column-gap:10px;row-gap:4px;min-height:calc(34px*var(--d));padding:calc(5px*var(--d)) 0;
  box-shadow:inset 0 calc(-1*var(--hair)) 0 var(--sep)}
.kv:last-child{box-shadow:none}
.kv>.lamp{color:var(--ink-3);display:flex} .kv>.lamp .ic{width:15px;height:15px}
.kv.ok>.lamp{color:var(--ok)} .kv.need>.lamp{color:var(--warn)} .kv.bad>.lamp{color:var(--bad)} .kv.warn>.lamp{color:var(--warn)} .kv.wait>.lamp{color:var(--accent)}
.kv.wait>.lamp .ic{animation:spin 1.4s linear infinite} @keyframes spin{to{transform:rotate(360deg)}}
.kv>.k{color:var(--ink-2)} .kv>.v{min-width:0;overflow-wrap:anywhere}
.kv>.acts{grid-column:3;display:flex;gap:6px;flex-wrap:wrap;margin-bottom:2px}
.kv.nolamp{grid-template-columns:minmax(110px,40%) minmax(0,1fr)} .kv.nolamp>.acts{grid-column:2}

.tbl{width:100%;border-collapse:separate;border-spacing:0}
.tbl th{position:sticky;top:0;z-index:1;text-align:left;font-weight:calc(var(--wt) + 100);font-size:calc(var(--fs) - 2px);color:var(--ink-3);padding:8px 10px;white-space:nowrap;
  background:var(--pop);box-shadow:inset 0 calc(-1*var(--hair)) 0 var(--sep)}
.tbl td{padding:8px 10px;box-shadow:inset 0 calc(-1*var(--hair)) 0 var(--sep);vertical-align:top}
.tbl tr:last-child td{box-shadow:none}
.tbl.hover tbody tr{cursor:pointer} .tbl.hover tbody tr:hover td{background:var(--fill)}
.tbl th.r,.tbl td.r{text-align:right}
.tbl th.sortable{cursor:pointer} .tbl th.sortable:hover{color:var(--ink)} .tbl th[aria-sort] {color:var(--ink)}
.tbl th[aria-sort="ascending"]::after{content:" ↑"} .tbl th[aria-sort="descending"]::after{content:" ↓"}
.tscroll{overflow:auto;border-radius:var(--r2);box-shadow:inset 0 0 0 var(--hair) var(--sep)}

details.dis>summary{list-style:none;cursor:pointer;display:inline-flex;align-items:center;gap:6px;color:var(--ink-2);font-weight:calc(var(--wt) + 100);user-select:none}
details.dis>summary::-webkit-details-marker{display:none}
details.dis>summary:hover{color:var(--ink)}
details.dis>summary::after{content:"";width:6px;height:6px;border-right:1.4px solid currentColor;border-bottom:1.4px solid currentColor;transform:rotate(45deg) translate(-1px,-1px);transition:transform .2s}
details.dis[open]>summary::after{transform:rotate(-135deg) translate(-1px,-1px)}
details.dis[open]>summary{margin-bottom:calc(10px*var(--d))}

.empty{display:flex;flex-direction:column;align-items:center;justify-content:center;gap:10px;text-align:center;padding:calc(36px*var(--d)) 16px;color:var(--ink-3)}
.empty>.ic{width:34px;height:34px;stroke-width:1.2;color:var(--ink-4)}
.empty h3{color:var(--ink);font-size:calc(var(--fs) + 3px)}
.note{display:flex;gap:8px;align-items:flex-start;padding:9px 12px;border-radius:var(--r2);background:var(--fill);color:var(--ink-2);font-size:calc(var(--fs) - 1px)}
.note .ic{margin-top:1px}
.note.bad{background:var(--bad-w);color:var(--bad)} .note.warn{background:var(--warn-w);color:var(--warn)} .note.ok{background:var(--ok-w);color:var(--ok)}
.out{max-height:260px;overflow:auto;padding:10px 12px;border-radius:var(--r2);background:var(--field);box-shadow:inset 0 0 0 var(--hair) var(--sep);color:var(--ink-2);font-size:calc(var(--fs) - 2px);line-height:1.5}
.field{display:flex;flex-direction:column;gap:6px;min-width:0}
.field>label,.flab{font-size:calc(var(--fs) - 1px);color:var(--ink-2)}
.frow{display:flex;gap:8px;align-items:center}
.frow .inp{flex:1}

/* ===================== Новое обучение ===================== */
.setup{display:grid;grid-template-columns:minmax(0,1fr) minmax(360px,420px);gap:var(--g);align-items:start}
.setup .col{display:flex;flex-direction:column;gap:var(--g);min-width:0}
.launch{position:sticky;top:0}
.mname{font-size:calc(var(--fs) + 9px);font-weight:var(--wth);letter-spacing:-.025em;line-height:1.2}
.mline{margin-top:2px;color:var(--ink-2)}
.mpath{display:flex;align-items:center;gap:6px;margin-top:10px;color:var(--ink-3);font-size:calc(var(--fs) - 2px);min-width:0}
.mpath bdi{overflow:hidden;text-overflow:ellipsis;white-space:nowrap;min-width:0}
.mpath .ro{flex:none;margin-left:4px;padding:1px 7px;border-radius:99px;background:var(--fill);color:var(--ink-2);font-size:calc(var(--fs) - 2.5px)}
.facts{display:flex;flex-wrap:wrap;gap:5px;margin-top:12px}
.opts{display:grid;grid-template-columns:repeat(auto-fit,minmax(calc(170px*var(--d) + 20px),1fr));gap:calc(8px*var(--d))}
#m-modes.opts{grid-template-columns:repeat(auto-fit,minmax(calc(230px*var(--d) + 20px),1fr))}
.opt{position:relative;display:flex;flex-direction:column;align-items:flex-start;gap:5px;padding:calc(12px*var(--d)) 14px calc(12px*var(--d)) 14px;border:0;border-radius:var(--r2);
  background:var(--fill);color:var(--ink);font:inherit;text-align:left;cursor:pointer;min-width:0;transition:background .12s,box-shadow .12s}
.opt:hover{background:var(--fill-2)}
.opt[aria-checked="true"]{background:var(--wash);box-shadow:inset 0 0 0 1.5px var(--accent)}
.opt>b{font-size:calc(var(--fs) + .5px);font-weight:var(--wth);padding-right:22px}
.opt>span{font-size:calc(var(--fs) - 1.5px);color:var(--ink-2);line-height:1.4}
.opt .tick{position:absolute;top:12px;right:12px;width:16px;height:16px;border-radius:50%;box-shadow:inset 0 0 0 1.5px var(--ink-4);transition:box-shadow .15s}
.opt[aria-checked="true"] .tick{box-shadow:inset 0 0 0 5px var(--accent)}
.opt .chips{margin-top:3px}
.opt .chip{background:var(--fill);height:20px;font-size:calc(var(--fs) - 2.5px)}
.opt[aria-checked="true"] .chip{background:var(--wash)}
.dstate{margin-top:calc(14px*var(--d))}
.scene{display:flex;flex-wrap:wrap;gap:calc(10px*var(--d));margin-top:calc(12px*var(--d))}.scene .grow{flex:1 1 220px}
.snote{margin:calc(8px*var(--d)) 0 0;color:var(--ink-3);font-size:calc(var(--fs) - 1.5px)}
/* LoRA: ранг, alpha, dropout в ряд; матрицы — плашками-переключателями */
#tab{margin:0 4px 0 2px;flex:none}
.lora{display:flex;flex-wrap:wrap;gap:calc(10px*var(--d)) calc(22px*var(--d));align-items:flex-start}
.lmh{display:flex;align-items:center;justify-content:space-between;gap:8px;margin-top:calc(14px*var(--d));font-size:calc(var(--fs) - 1px);color:var(--ink-2)}
.togs{display:flex;flex-wrap:wrap;gap:6px;margin:calc(8px*var(--d)) 0 calc(12px*var(--d))}
#lo-tiles{grid-template-columns:repeat(auto-fit,minmax(calc(118px*var(--d) + 12px),1fr))}   /* четыре плитки — в один ряд */
.tog{height:calc(26px + 2px*var(--d));padding:0 12px;border:0;border-radius:var(--rc);background:var(--fill);color:var(--ink-2);font:inherit;font-size:calc(var(--fs) - 1px);cursor:pointer}
.tog:hover{background:var(--fill-2)}
.tog[aria-pressed="true"]{background:var(--wash);box-shadow:inset 0 0 0 1.5px var(--accent);color:var(--accent-ink);font-weight:calc(var(--wt) + 150)}
.ex{display:flex;flex-direction:column;gap:8px}
.ex .exh{display:flex;align-items:center;gap:8px}
.ex .exh .pg{margin-left:auto;display:flex;align-items:center;gap:4px;color:var(--ink-3);font-variant-numeric:tabular-nums}
.ex .exb{display:grid;gap:8px}
.ex .exf{padding:9px 11px;border-radius:var(--r2);background:var(--field);box-shadow:inset 0 0 0 var(--hair) var(--sep)}
.ex .exf>span{display:block;font-size:calc(var(--fs) - 2.5px);color:var(--ink-3);margin-bottom:3px}
.ex .exf>div{white-space:pre-wrap;word-break:break-word;max-height:180px;overflow:auto;font-size:calc(var(--fs) - .5px)}
.bigs{display:flex;gap:calc(18px*var(--d));flex-wrap:wrap;margin:2px 0 calc(14px*var(--d))}
.bigs>div{display:flex;flex-direction:column;min-width:0}
.bigs b{font-size:calc(var(--fs) + 9px);font-weight:var(--wth);letter-spacing:-.025em;line-height:1.15;font-variant-numeric:tabular-nums}
.bigs span{font-size:calc(var(--fs) - 2px);color:var(--ink-3)}
.why:empty{display:none}
.why{margin-top:10px;font-size:calc(var(--fs) - 1px);color:var(--ink-2)} .why.bad{color:var(--bad)}
.prow{display:grid;grid-template-columns:minmax(0,1fr) minmax(120px,46%);align-items:center;gap:10px;min-height:calc(34px*var(--d));padding:calc(4px*var(--d)) 0;
  box-shadow:inset 0 calc(-1*var(--hair)) 0 var(--sep)}
.prow:last-child{box-shadow:none}
.prow>.pk{color:var(--ink-2);min-width:0}
.prow>.pv{text-align:right;font-weight:calc(var(--wt) + 100);overflow-wrap:anywhere}
.prow .inp{height:calc(var(--ch) - 2px);text-align:right}
.prow select.inp{text-align:left}

/* ===================== Ход обучения ===================== */
.now .nhead{display:flex;align-items:center;gap:8px;flex-wrap:wrap}
.now .nhead .ca{margin-left:auto;display:flex;gap:6px;flex-wrap:wrap}
.ntitle{margin-top:calc(14px*var(--d));font-size:calc(var(--fs) + 11px);font-weight:var(--wth);letter-spacing:-.03em;line-height:1.15}
.nsub{margin-top:5px;color:var(--ink-2)}
.nsub b{color:var(--ink)}
.nerr{margin-top:12px}
.mbar{margin:calc(16px*var(--d)) 0 calc(6px*var(--d))}
.mbar .mcap{display:flex;justify-content:space-between;gap:10px;font-size:calc(var(--fs) - 1.5px);color:var(--ink-2);margin-bottom:7px;font-variant-numeric:tabular-nums}
.mbar .bar{height:5px;border-radius:3px}
.stages{display:grid;grid-template-columns:repeat(7,minmax(0,1fr));gap:calc(6px*var(--d));margin-top:calc(14px*var(--d))}
.stg{padding:calc(9px*var(--d)) 11px calc(10px*var(--d));border-radius:var(--r2);background:var(--fill);min-width:0}
.stg b{display:flex;align-items:center;gap:6px;font-size:calc(var(--fs) - 1px);font-weight:var(--wth);white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.stg b .ic{width:13px;height:13px}
.stg small{display:block;margin-top:2px;font-size:calc(var(--fs) - 2.5px);color:var(--ink-3);white-space:nowrap;overflow:hidden;text-overflow:ellipsis;min-height:1.3em}
.stg .bar{margin-top:8px;height:3px}
.stg.done b .ic{color:var(--ok)} .stg.done .bar>i{background:var(--ok)}
.stg.cur{background:var(--wash);box-shadow:inset 0 0 0 1px var(--wash-2)} .stg.cur b .ic{color:var(--accent)}
.stg.bad{background:var(--bad-w)} .stg.bad b .ic{color:var(--bad)}
.stg.todo b,.stg.skip b{color:var(--ink-3)} .stg.skip .bar{opacity:.3}
.stg.stop b .ic{color:var(--warn)}
.sweep{margin-top:calc(14px*var(--d))}
.sweep .pt{display:flex;justify-content:space-between;gap:12px;font-size:calc(var(--fs) - 1.5px);color:var(--ink-2);margin-bottom:6px}
.sweep .pl{height:170px}
.params{margin-top:calc(12px*var(--d))}
.plots{display:grid;grid-template-columns:repeat(var(--cols,3),minmax(0,1fr));gap:var(--g)}
.plot{padding:calc(14px*var(--d)) calc(16px*var(--d)) calc(8px*var(--d));border-radius:var(--r);cursor:pointer;display:flex;flex-direction:column;min-width:0;transition:transform .18s var(--ease)}
.plot:hover{box-shadow:inset 0 0 0 var(--hair) var(--edge-2),var(--shadow)}
.plot .pt{display:flex;align-items:center;gap:8px;color:var(--ink-2);font-size:calc(var(--fs) - 1px)}
.plot .pt .kbd{margin-left:2px}
.plot .pt .flag{margin-left:auto;display:inline-flex;align-items:center;gap:5px;color:var(--warn);font-size:calc(var(--fs) - 2px)}
.plot .pt .exp{margin-left:auto;color:var(--ink-4);opacity:0;transition:opacity .15s}
.plot:hover .pt .exp{opacity:1}
.plot .pv{display:flex;align-items:baseline;gap:5px;margin-top:5px;min-height:1.2em}
.plot .pv b{font-size:calc(var(--fs) + 14px);font-weight:var(--wth);letter-spacing:-.03em;line-height:1.1;font-variant-numeric:tabular-nums}
.plot .pv small{color:var(--ink-3);font-size:calc(var(--fs) - 1px)}
.keys{display:flex;flex-wrap:wrap;gap:4px 12px;margin-top:4px;font-size:calc(var(--fs) - 2px);color:var(--ink-2);min-height:1.4em}
.keys span{display:inline-flex;align-items:center;gap:5px;white-space:nowrap}
.keys i{width:10px;height:2.5px;border-radius:2px}
.keys i.d{width:7px;height:7px;border-radius:50%}
.keys b{color:var(--ink);font-weight:calc(var(--wt) + 140)}
.pl{position:relative;height:var(--chh,150px);margin-top:6px}
.pl:empty::before{content:attr(data-empty);position:absolute;inset:0;display:flex;align-items:center;justify-content:center;color:var(--ink-3);font-size:calc(var(--fs) - 1.5px)}
.journal{display:grid;grid-template-columns:auto minmax(0,1fr) auto;gap:14px;align-items:center;padding:calc(12px*var(--d)) calc(16px*var(--d));border-radius:var(--r);cursor:pointer}
.journal .jt{font-weight:var(--wth);color:var(--ink-2)}
.journal .jl{display:flex;flex-direction:column;gap:3px;min-width:0}
.journal .jr{display:grid;grid-template-columns:62px 80px minmax(0,1fr);gap:8px;font-size:calc(var(--fs) - 1.5px);min-width:0}
.journal .jr i{font-style:normal;color:var(--ink-3);font-variant-numeric:tabular-nums}
.journal .jr span{white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.journal .exp{color:var(--ink-4)}

/* ===================== Запуски ===================== */
.rtool{display:flex;align-items:center;gap:8px;flex-wrap:wrap}
.rtool .inp{width:220px}
.rsum{display:flex;gap:16px;color:var(--ink-3);font-size:calc(var(--fs) - 1.5px);margin-left:auto}
.rsum b{color:var(--ink-2)}
.runs-t td{white-space:nowrap}
.runs-t td.nm{font-weight:var(--wth);max-width:220px;overflow:hidden;text-overflow:ellipsis}
.runs-t td.cb,.runs-t th.cb{width:34px;padding-right:0}
.runs-t input[type=checkbox]{width:15px;height:15px;accent-color:var(--accent);cursor:pointer;margin:2px 0 0}
.runs-t tr.sel td{background:var(--wash)}
.runs-t .rowact{display:flex;gap:2px;justify-content:flex-end;opacity:.55}
.runs-t tr:hover .rowact{opacity:1}
.cmp .plots{--cols:2}
.cmp .legend{display:flex;flex-wrap:wrap;gap:6px 16px;font-size:calc(var(--fs) - 1px);color:var(--ink-2)}
.cmp .legend span{display:inline-flex;align-items:center;gap:6px}
.cmp .legend i{width:14px;height:3px;border-radius:2px}

/* ===================== Разговор ===================== */
.chatv{display:grid;grid-template-columns:240px minmax(0,1fr);gap:var(--g);height:100%;min-height:0}
.chatv .card{display:flex;flex-direction:column;min-height:0}
#c-msel{display:none;width:auto;max-width:220px}
.mlist{display:flex;flex-direction:column;gap:2px;overflow:auto;margin:0 -6px;padding:0 6px}
.mi{display:flex;flex-direction:column;align-items:flex-start;gap:1px;width:100%;padding:8px 10px;border:0;border-radius:var(--rc);background:transparent;color:var(--ink);font:inherit;text-align:left;cursor:pointer}
.mi:hover{background:var(--fill)} .mi[aria-current="true"]{background:var(--wash)}
.mi b{font-weight:var(--wth);max-width:100%;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.mi small{color:var(--ink-3);font-size:calc(var(--fs) - 2.5px)}
.conv{flex:1;min-height:0;overflow:auto;display:flex;flex-direction:column;gap:14px;padding:4px 4px 12px}
.msg{display:flex;flex-direction:column;gap:4px;max-width:78%}
.msg.me{align-self:flex-end;align-items:flex-end}
.msg .who{font-size:calc(var(--fs) - 2.5px);color:var(--ink-3);padding:0 4px}
.msg .bub{padding:9px 13px;border-radius:calc(var(--r2) + 4px);background:var(--fill-2);white-space:pre-wrap;word-break:break-word;line-height:1.5}
.msg.me .bub{background:var(--accent);color:var(--on-accent)}
.msg.sys{align-self:center;max-width:90%} .msg.sys .bub{background:transparent;color:var(--ink-3);font-size:calc(var(--fs) - 1.5px);padding:2px 8px}
.msg .bub.typing::after{content:"▍";animation:blink 1s steps(2) infinite;color:var(--ink-3)}
@keyframes blink{50%{opacity:0}}
.msg .meta{font-size:calc(var(--fs) - 3px);color:var(--ink-4);padding:0 4px}
.ab{display:grid;grid-template-columns:1fr 1fr;gap:10px;width:100%;max-width:100%}
.ab .msg{max-width:100%}
.ab .msg .bub{background:var(--fill)}
.ab .msg.b .who{color:var(--ink-3)} .ab .msg.a .who{color:var(--accent-ink)}
.composer{display:flex;gap:8px;align-items:flex-end;padding-top:12px;box-shadow:inset 0 var(--hair) 0 var(--sep)}
.composer textarea{min-height:var(--ch);max-height:200px;resize:none}
.copts{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:10px;margin-bottom:12px}
.loadline{display:flex;align-items:center;gap:10px;color:var(--ink-2);font-size:calc(var(--fs) - 1.5px)}
.loadline .bar{flex:1}

/* ===================== Проверка ===================== */
.bench-t td.q{max-width:320px}
.bench-t td.ans{max-width:260px;white-space:pre-wrap;word-break:break-word}
.bench-t td.res{white-space:nowrap}
.bench-t tr.hit td.res{color:var(--ok)} .bench-t tr.miss td.res{color:var(--bad)} .bench-t tr.same td.res{color:var(--ink-3)}
.bench-t td.mark{cursor:pointer}

/* ===================== Настройки ===================== */
.setv{display:grid;grid-template-columns:210px minmax(0,1fr);gap:var(--g);align-items:start}
.snav{position:sticky;top:0;padding:8px;border-radius:var(--r);display:flex;flex-direction:column;gap:2px}
.sgroup{display:flex;flex-direction:column;gap:var(--g)}
.srow{display:grid;grid-template-columns:minmax(150px,240px) minmax(0,1fr);align-items:center;gap:16px;min-height:calc(40px*var(--d));padding:calc(6px*var(--d)) 0;
  box-shadow:inset 0 calc(-1*var(--hair)) 0 var(--sep)}
.srow:last-child{box-shadow:none}
.srow>label,.srow>.sl{color:var(--ink)}
.srow>label small,.srow>.sl small{display:block;color:var(--ink-3);font-size:calc(var(--fs) - 2.5px);margin-top:1px}
.srow .sc{display:flex;align-items:center;gap:10px;min-width:0;flex-wrap:wrap}
.srow .sc .rng{flex:1;min-width:120px}
.srow .sc output{min-width:52px;text-align:right;color:var(--ink-2);font-variant-numeric:tabular-nums;font-size:calc(var(--fs) - 1px)}
.presets{display:grid;grid-template-columns:repeat(auto-fill,minmax(150px,1fr));gap:10px}
.pcard{display:flex;flex-direction:column;gap:7px;padding:6px 6px 9px;border:0;border-radius:calc(var(--r2) + 2px);background:var(--fill);color:var(--ink);font:inherit;cursor:pointer;
  text-align:left;position:relative}
.pcard:hover{background:var(--fill-2)}
.pcard[aria-checked="true"]{box-shadow:inset 0 0 0 2px var(--accent)}
.pcard>i{display:block;height:64px;border-radius:var(--r2);background-size:cover;background-position:center;position:relative;overflow:hidden}
.pcard>span{padding:0 4px;font-size:calc(var(--fs) - 1px);font-weight:calc(var(--wt) + 100);display:flex;align-items:center;gap:6px}
.pcard .del{position:absolute;top:10px;right:10px;width:22px;height:22px;border-radius:50%;display:none;align-items:center;justify-content:center;background:rgba(0,0,0,.55);color:#fff}
.pcard:hover .del{display:flex}
.swatches{display:flex;flex-wrap:wrap;gap:9px;align-items:center}
.swatches button{width:24px;height:24px;padding:0;border:0;border-radius:50%;cursor:pointer;box-shadow:inset 0 0 0 1px rgba(0,0,0,.15)}
.swatches button[aria-checked="true"]{box-shadow:0 0 0 2px var(--solid),0 0 0 4px currentColor}
.clist{display:flex;flex-direction:column}
.citem{display:flex;align-items:center;gap:10px;min-height:38px;box-shadow:inset 0 calc(-1*var(--hair)) 0 var(--sep)}
.citem:last-child{box-shadow:none}
.citem span{flex:1}
textarea.code{font-family:var(--mono);font-size:calc(var(--fs) - 1.5px);min-height:180px}
.about{display:flex;gap:18px;align-items:center}
.about .mark{width:64px;height:64px}
.contacts{display:flex;flex-wrap:wrap;gap:8px}

/* ===================== наложения ===================== */
.scrim{position:fixed;inset:0;z-index:60;display:flex;align-items:center;justify-content:center;padding:24px;background:var(--scrim);animation:fade .16s ease}
.scrim.topal{align-items:flex-start;padding-top:12vh}
@keyframes fade{from{opacity:0}}
.pop{background:var(--pop);-webkit-backdrop-filter:blur(30px) saturate(180%);backdrop-filter:blur(30px) saturate(180%);border-radius:calc(var(--r) + 2px);
  box-shadow:0 0 0 var(--hair) var(--edge-2),0 30px 80px -20px rgba(0,0,0,.55);animation:rise .2s var(--ease)}
@keyframes rise{from{opacity:0;transform:translateY(8px) scale(.985)}}
.dialog{width:min(460px,100%);padding:20px 22px 18px}
.dialog h3{font-size:calc(var(--fs) + 3px);margin-bottom:6px}
.dialog .dt{color:var(--ink-2);white-space:pre-wrap}
.dialog .inp{margin-top:14px}
.dialog .dacts{display:flex;justify-content:flex-end;gap:8px;margin-top:20px}
.winbox{width:min(1180px,100%);height:min(860px,100%);display:flex;flex-direction:column;overflow:hidden}
.winhead{display:flex;align-items:center;gap:12px;padding:16px 16px 14px 22px;box-shadow:inset 0 calc(-1*var(--hair)) 0 var(--sep);flex:none}
.winhead h2{font-size:calc(var(--fs) + 4px)}
.winhead p{color:var(--ink-3);font-size:calc(var(--fs) - 1px);min-width:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.winhead .ca{margin-left:auto;display:flex;gap:6px}
.winbody{flex:1;min-height:0;overflow:auto;padding:18px 22px 22px;display:flex;flex-direction:column;gap:18px}
.wplots{display:grid;grid-template-columns:minmax(0,1fr);gap:18px}
.wplot h3{font-size:calc(var(--fs) + .5px)}
.wplot p{color:var(--ink-3);font-size:calc(var(--fs) - 1.5px);margin-top:2px}
.wplot .pl{height:240px} .wplot.sm .pl{height:170px}
.palette{width:min(640px,100%);overflow:hidden;display:flex;flex-direction:column;max-height:70vh}
.palette .pin{display:flex;align-items:center;gap:10px;padding:14px 16px;box-shadow:inset 0 calc(-1*var(--hair)) 0 var(--sep)}
.palette .pin input{flex:1;border:0;background:transparent;color:var(--ink);font:inherit;font-size:calc(var(--fs) + 3px)}
.palette .pin input:focus{outline:none}
.palette .plist{overflow:auto;padding:6px}
.palette .pg{padding:8px 10px 4px;font-size:calc(var(--fs) - 2.5px);color:var(--ink-3);font-weight:var(--wth)}
.palette .pi{display:flex;align-items:center;gap:10px;width:100%;padding:8px 10px;border:0;border-radius:var(--rc);background:transparent;color:var(--ink);font:inherit;text-align:left;cursor:pointer}
.palette .pi .ic{color:var(--ink-3)}
.palette .pi[aria-selected="true"]{background:var(--accent);color:var(--on-accent)} .palette .pi[aria-selected="true"] .ic{color:inherit}
.palette .pi span{flex:1;min-width:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.palette .pi em{font-style:normal;font-size:calc(var(--fs) - 2px);opacity:.6}
.palette .pnone{padding:22px;text-align:center;color:var(--ink-3)}
.toasts{position:fixed;right:22px;bottom:22px;z-index:90;display:flex;flex-direction:column;gap:8px;align-items:flex-end;pointer-events:none}
.toast{pointer-events:auto;display:flex;align-items:flex-start;gap:10px;min-width:240px;max-width:400px;padding:11px 14px;border-radius:calc(var(--r2) + 2px);animation:rise .22s var(--ease)}
.toast .ic{margin-top:1px}
.toast.ok .ic{color:var(--ok)} .toast.bad .ic{color:var(--bad)} .toast.warn .ic{color:var(--warn)} .toast.info .ic{color:var(--accent)}
.toast b{display:block} .toast span{color:var(--ink-2);font-size:calc(var(--fs) - 1px)}
.toast.out{opacity:0;transform:translateY(6px);transition:opacity .25s,transform .25s}
.tip{position:fixed;left:0;top:0;z-index:95;pointer-events:none;opacity:0;padding:8px 10px;border-radius:9px;background:var(--pop);font-size:calc(var(--fs) - 1.5px);
  box-shadow:0 0 0 var(--hair) var(--edge-2),0 10px 30px -8px rgba(0,0,0,.45);-webkit-backdrop-filter:blur(20px);backdrop-filter:blur(20px);transition:opacity .1s;white-space:nowrap}
.tip .th{color:var(--ink-3);margin-bottom:4px}
.hint{display:inline-flex;vertical-align:-2px;margin-left:6px;color:var(--ink-4);cursor:help}.hint:hover{color:var(--ink-2)}.hint .ic{width:14px;height:14px}
.tip.tt{white-space:pre-line;max-width:min(340px,calc(100vw - 16px));line-height:1.42;color:var(--ink-2);z-index:120;transition:opacity .12s}
.tip .tr{display:flex;align-items:center;gap:7px}
.tip .tr i{width:8px;height:8px;border-radius:50%}
.tip .tr b{font-variant-numeric:tabular-nums}
.tip .tr span{color:var(--ink-2)}

/* ===================== кривые ===================== */
.pl svg{display:block;overflow:visible}
.pl .gl{stroke:var(--sep);stroke-width:1}
.nogrid .pl .gl{display:none}
.pl .ax{fill:var(--ink-3);font-size:10.5px;font-family:var(--ff);font-variant-numeric:tabular-nums}
.pl .ln{fill:none;stroke-width:var(--lw);stroke-linejoin:round;stroke-linecap:round}
.pl .area{stroke:none;opacity:.16}
.nofill .pl .area{display:none}
.pl .ring{stroke:var(--solid);stroke-width:1.5}
.nodots .pl .ring.ev{display:none}
.pl .rule{stroke:var(--ink-4);stroke-dasharray:4 4}
.pl .vr{stroke:var(--accent);stroke-dasharray:3 3}
.pl .vrl{fill:var(--accent-ink);font-size:11px}
.pl .lbl{font-size:11px;fill:var(--ink-2)}
.pl .cross{stroke:var(--ink-3);stroke-dasharray:2 3}

/* ===================== узкие окна и спокойствие ===================== */
@media (max-width:1280px){ .setup{grid-template-columns:minmax(0,1fr) minmax(330px,380px)} .stages{grid-template-columns:repeat(4,minmax(0,1fr))} }
@media (max-width:1240px){ .chatv .ca .bl{display:none} .chatv .ca .btn{padding:0 8px} }
@media (max-width:1180px){ .runs-t .c-mode{display:none} }
@media (max-width:980px){ .runs-t .c-time,.runs-t .c-size{display:none} }
@media (max-width:1100px){ .setup{grid-template-columns:minmax(0,1fr)} .launch{position:static} .plots{--cols:2!important} .chatv{grid-template-columns:200px minmax(0,1fr)} }
@media (max-width:900px){ :root{--sidew:64px!important} .side .bn,.side .bv,.navh,.ni span,.ni em,.lcard,.sfoot .seg,.disk{display:none!important} .ni{justify-content:center;padding:0}
  .sfoot{flex-direction:column} .setv{grid-template-columns:minmax(0,1fr)} .snav{position:static;flex-direction:row;flex-wrap:wrap} .plots{--cols:1!important}
  .stages{grid-template-columns:repeat(2,minmax(0,1fr))} .chatv{grid-template-columns:minmax(0,1fr)} .chatv .mcard{display:none} #c-title{display:none} #c-msel{display:block} .copts{grid-template-columns:1fr 1fr} .srow{grid-template-columns:minmax(0,1fr)} }
.calm *,.calm *::before,.calm *::after{animation:none!important;transition:none!important}
@media (prefers-reduced-motion:reduce){ *,*::before,*::after{animation-duration:.001s!important;transition-duration:.001s!important} }

/* база: BF16 или квант — строками-переключателями, как карточки выбора */
#m-base{margin-top:calc(14px*var(--d))}
.bh{display:flex;align-items:center;gap:4px;margin-bottom:6px}.bh>span{font-weight:var(--wth);font-size:calc(var(--fs) - .5px)}.bh .ca{margin-left:auto;display:flex;gap:2px}
.blist{display:grid;gap:6px}
.brow{display:flex;align-items:center;gap:10px;padding:calc(7px*var(--d)) 8px calc(7px*var(--d)) 12px;border-radius:var(--rc);background:var(--fill);cursor:pointer;transition:background .12s,box-shadow .12s}
.brow:hover{background:var(--fill-2)}
.brow[aria-checked="true"]{background:var(--wash);box-shadow:inset 0 0 0 1.5px var(--accent);cursor:default}
.brow[aria-disabled="true"]{cursor:default}.brow[aria-disabled="true"]:hover{background:var(--fill)}
.brow .tick{flex:none;width:14px;height:14px;border-radius:50%;box-shadow:inset 0 0 0 1.5px var(--ink-4)}
.brow[aria-checked="true"] .tick{box-shadow:inset 0 0 0 4.5px var(--accent)}
.brow .bn{flex:1;min-width:0;display:flex;flex-direction:column;gap:1px}
.brow .bn b{font-weight:var(--wt);overflow:hidden;text-overflow:ellipsis;white-space:nowrap}.brow[aria-checked="true"] .bn b{color:var(--accent-ink);font-weight:calc(var(--wt) + 150)}
.brow .bn span{font-size:calc(var(--fs) - 2px);color:var(--ink-3);overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.brow .btn{flex:none}
#b-recs{margin:8px 0 0}
.bjob{margin-top:8px;display:grid;gap:4px}.bjob .hint{overflow:hidden;text-overflow:ellipsis;white-space:nowrap;margin:0}
.about .mtag{margin:6px 0 2px;color:var(--ink)}
</style></head><body>
<div class="frame" id="frame"><div class="wall"></div>
<div class="app" id="app">

<aside class="side mat" id="side" aria-label="Разделы">
  <div class="brand" id="brand"><svg class="mark" viewBox="0 0 64 64" aria-hidden="true"><g stroke-linejoin="round" stroke-width="2.4"><polygon class="l1" points="7,18 32,5.5 57,18 32,30.5"/><polygon class="l2" points="15.5,29.75 32,38 57,25.5 57,31 32,43.5 15.5,35.25"/><polygon class="l3" points="7,38 32,50.5 49,42 55,45 32,56.5 7,45"/></g></svg><span class="bn">TierLM</span><span class="bv" id="ver"></span></div>
  <button class="lcard" id="live" hidden data-go="train" title="Перейти к обучению">
    <span class="lh"><i class="dot pulse"></i><b id="lv-name"></b><span id="lv-pct"></span></span>
    <span class="bar" style="display:block"><i id="lv-bar"></i></span>
    <span class="ls" id="lv-sub" style="display:block"></span>
  </button>
  <div class="navh">Обучение</div>
  <button class="ni" data-go="setup"><svg class="ic" viewBox="0 0 16 16"><rect x="2.5" y="2.5" width="11" height="11" rx="3"/><path d="M8 5.5v5M5.5 8h5"/></svg><span>Новое обучение</span><i class="st" id="st-setup"></i></button>
  <button class="ni" data-go="train"><svg class="ic" viewBox="0 0 16 16"><path d="M1.8 8.5h2.6l1.8-4.5 3 8 1.8-3.5h3.2"/></svg><span>Ход обучения</span><i class="st" id="st-train"></i></button>
  <button class="ni" data-go="runs"><svg class="ic" viewBox="0 0 16 16"><rect x="2.5" y="3" width="11" height="3" rx="1"/><path d="M3.5 6v6.5h9V6M6.5 8.5h3"/></svg><span>Запуски</span><em id="nb-runs"></em></button>
  <div class="navh">Модель</div>
  <button class="ni" data-go="chat"><svg class="ic" viewBox="0 0 16 16"><path d="M2.8 4.2c0-.9.7-1.6 1.6-1.6h7.2c.9 0 1.6.7 1.6 1.6v5c0 .9-.7 1.6-1.6 1.6H7.2L4.4 13v-2.2c-.9 0-1.6-.7-1.6-1.6z"/></svg><span>Разговор</span><i class="st" id="st-chat"></i></button>
  <button class="ni" data-go="bench"><svg class="ic" viewBox="0 0 16 16"><path d="M2.5 4.5l1.2 1.2L6 3.4M2.5 10l1.2 1.2L6 8.9M8.5 4.8h5M8.5 10.3h5"/></svg><span>Проверка</span><i class="st" id="st-bench"></i></button>
  <span class="grow"></span>
  <div class="disk" id="disk" hidden></div>
  <button class="ni" data-go="settings"><svg class="ic" viewBox="0 0 16 16"><path d="M2.5 5h5M11.5 5h2M2.5 11h2M8.5 11h5"/><circle cx="9.5" cy="5" r="1.8"/><circle cx="6.5" cy="11" r="1.8"/></svg><span>Настройки</span></button>
  <div class="sfoot">
    <span class="seg mini" id="lang" role="radiogroup" aria-label="Язык"><button data-l="ru" role="radio">RU</button><button data-l="en" role="radio">EN</button></span>
    <button class="btn icon quiet sm" id="th" title="Тема" aria-label="Сменить тему"><svg class="ic" viewBox="0 0 16 16"><circle cx="8" cy="8" r="5.5"/><path d="M8 2.5a5.5 5.5 0 0 1 0 11z" fill="currentColor" stroke="none"/></svg></button>
    <button class="btn icon quiet sm danger" id="kill" title="Завершить работу программы (Ctrl+Q)" aria-label="Завершить работу программы"><svg class="ic" viewBox="0 0 16 16"><path d="M8 2.5v5"/><path d="M4.6 4.6a5 5 0 1 0 6.8 0"/></svg></button>
  </div>
</aside>

<div class="main">
  <header class="top mat" id="top">
    <h1 class="vt" id="vt">Новое обучение</h1>
    <span class="seg" id="tab" role="radiogroup" aria-label="Способ"><button role="radio" data-tab="tier">TierLM</button><button role="radio" data-tab="lora">LoRA</button></span>
    <span class="vs" id="vs"></span>
    <div class="tact" id="tact"></div>
    <button class="btn icon quiet" id="cmdk" title="Команды (Ctrl+K)" aria-label="Команды"><svg class="ic" viewBox="0 0 16 16"><circle cx="7" cy="7" r="4.3"/><path d="M10.2 10.2 13.5 13.5"/></svg></button>
    <div class="edge" id="edge" role="progressbar" aria-label="Ход обучения" aria-valuemin="0" aria-valuemax="100"><i id="fill"></i></div>
  </header>

  <div class="views" id="views">

  <!-- ========================= Новое обучение ========================= -->
  <section class="view" id="v-setup" aria-label="Новое обучение">
    <div class="setup">
      <div class="col">
        <section class="card glass" id="c-model" aria-labelledby="h-model">
          <div class="ch"><h2 id="h-model">Модель</h2><span class="pill" id="m-st"></span>
            <div class="ca"><button class="btn sm quiet" id="m-folder"><svg class="ic" viewBox="0 0 16 16"><path d="M2.5 5c0-.8.7-1.5 1.5-1.5h2.4l1.4 1.5H12c.8 0 1.5.7 1.5 1.5v5c0 .8-.7 1.5-1.5 1.5H4c-.8 0-1.5-.7-1.5-1.5z"/></svg>Изменить папку</button></div></div>
          <div id="m-card"></div>
          <div id="m-base" hidden></div>
          <details class="dis sec" id="m-kmore" hidden><summary>Память при обучении</summary>
            <div id="m-kdet" style="margin-top:12px"></div></details>
        </section>
        <section class="card glass" aria-labelledby="h-mode">
          <div class="ch"><h2 id="h-mode">Режим</h2></div>
          <div class="opts" id="m-modes" role="radiogroup" aria-labelledby="h-mode"></div>
        </section>
        <section class="card glass" id="c-goal" aria-labelledby="h-goal">
          <div class="ch"><h2 id="h-goal">Объект обучения</h2></div>
          <div class="opts" id="m-goals" role="radiogroup" aria-labelledby="h-goal"></div>
        </section>
        <section class="card glass" id="c-lora" aria-labelledby="h-lora" hidden>
          <div class="ch"><h2 id="h-lora" title="Классическая LoRA (Hu и др., 2021), как в PEFT: W + α/r · B·A поверх замороженных весов. A инициализируется равномерно в ±1/√вход, B — нулями, оптимизатор AdamW в FP32. Исходные веса не изменяются, адаптеры сохраняются отдельным файлом.">LoRA</h2><span class="pill bad" id="lo-over" hidden></span>
            <div class="ca"><button class="btn sm quiet" id="lo-def" title="Значения по умолчанию: r 16, α 32, все матрицы, dropout 0,05." hidden>По умолчанию</button></div></div>
          <div class="lora">
            <div class="field"><label title="Число направлений поправки в каждой матрице. Чем больше ранг, тем больше обучаемых чисел и расход видеопамяти.">Ранг r</label><span class="seg" id="lo-r" role="radiogroup" aria-label="Ранг r"></span></div>
            <div class="field"><label for="lo-a" title="Масштаб поправки: s = α / r. Обычно α = 2r.">Alpha</label>
              <input class="inp" id="lo-a" inputmode="numeric" maxlength="4" autocomplete="off" style="width:calc(96px*var(--d));text-align:right"></div>
            <div class="field"><label title="Доля входа адаптера, обнуляемая при обучении. На валидации и в разговоре не применяется.">Dropout</label><span class="seg" id="lo-d" role="radiogroup" aria-label="Dropout"></span></div>
          </div>
          <div class="lmh"><span title="Матрицы, на которых устанавливаются адаптеры: q, k, v, o — внимание; gate, up, down — MLP; во всех слоях. В слоях полного внимания отдельной v_proj нет: V берётся из K.">Матрицы</span>
            <span class="seg mini" id="lo-set" role="radiogroup" aria-label="Набор матриц"></span></div>
          <div class="togs" id="lo-m" role="group" aria-label="Матрицы"></div>
          <div class="tiles" id="lo-tiles"></div>
        </section>
        <section class="card glass" id="c-data" aria-labelledby="h-data">
          <div class="ch"><h2 id="h-data">Данные</h2><span class="pill" id="d-st"></span></div>
          <div class="field"><label for="d-file">Файл с примерами</label>
            <div class="frow"><input class="inp" id="d-file" placeholder="Путь к файлу .jsonl" autocomplete="off" spellcheck="false"
              title='Один пример на строку: {"instruction": "…", "input": "…", "output": "…"}. Файл используется только для чтения.'>
</div></div>
          <div class="field" style="margin-top:calc(12px*var(--d))"><label for="d-name">Имя запуска</label>
            <div class="frow"><input class="inp" id="d-name" autocomplete="off" spellcheck="false" placeholder="По имени файла"><span class="pill bad" id="d-nst" hidden>Имя занято</span></div></div>
          <div id="d-state" class="dstate"></div>
          <div class="scene" id="d-scene" hidden>
            <div class="field"><label for="d-share" title="Доля примеров, в которых поле instruction передаётся как системная инструкция. В остальных примерах она отсутствует, благодаря чему манера ответов сохраняется и в обычном разговоре.">С системной инструкцией, %</label>
              <input class="inp" id="d-share" inputmode="numeric" maxlength="3" autocomplete="off" style="width:calc(96px*var(--d));text-align:right"></div>
            <div class="field grow"><label for="d-always" title="Системная инструкция добавляется к примеру всегда, если в его поле instruction встречается одно из слов.">Всегда с системной инструкцией</label>
              <input class="inp" id="d-always" autocomplete="off" spellcheck="false" placeholder="Слова из instruction через запятую"></div>
          </div>
          <p class="snote" id="d-snote" hidden></p>
          <details class="dis sec" id="d-prev" hidden><summary>Примеры из файла</summary><div class="ex" id="d-ex"></div></details>
        </section>
      </div>
      <aside class="card glass launch" id="c-launch" aria-labelledby="h-launch">
        <div class="ch"><h2 id="h-launch">Запуск</h2><span class="pill" id="l-pill"></span></div>
        <div class="bigs" id="l-sum"></div>
        <button class="btn primary lg wide" id="l-go" disabled><svg class="ic" viewBox="0 0 16 16"><path d="M5 3.6v8.8L12.4 8z" fill="currentColor"/></svg><span>Начать обучение</span><span class="kbd">Ctrl ↵</span></button>
        <p class="why" id="l-why"></p>
        <div class="sec"><h3>Проверки</h3><div class="rows" id="l-checks"></div><pre class="out" id="setup-log" hidden style="margin-top:10px"></pre></div>
        <div class="sec"><h3>План<span class="ca"><button class="btn sm quiet" id="l-expert">Изменить</button></span></h3>
          <div class="rows" id="l-plan"></div>
          <div id="l-hw" hidden style="margin-top:10px"><button class="btn sm quiet" id="l-hwbtn" title="Запас видеопамяти, бюджет среза строк, режим ×3 (FP16/BF16), опережающая подкачка."><svg class="ic" viewBox="0 0 16 16"><rect x="4" y="4" width="8" height="8" rx="1.5"/><path d="M6 1.8V4M10 1.8V4M6 12v2.2M10 12v2.2M1.8 6H4M1.8 10H4M12 6h2.2M12 10h2.2"/></svg>Настройка оборудования</button></div></div>
      </aside>
    </div>
  </section>

  <!-- ========================= Ход обучения ========================= -->
  <section class="view" id="v-train" hidden aria-label="Ход обучения">
    <div class="card glass now" id="now">
      <div class="nhead">
        <select class="inp" id="run" style="width:auto;min-width:170px;max-width:280px" title="Выбор запуска для просмотра." aria-label="Запуск"></select>
        <span class="pill" id="n-badge"><i></i><span></span></span>
        <div class="ca" id="n-act"></div>
      </div>
      <h1 class="ntitle" id="n-title"></h1>
      <p class="nsub" id="n-sub"></p>
      <div class="mbar" id="n-bar" hidden><div class="mcap"><span id="n-cap"></span><span id="n-pct"></span></div><div class="bar"><i></i></div></div>
      <div class="note bad nerr" id="n-err" hidden></div>
      <div class="stages" id="n-stages" aria-label="Этапы"></div>
      <div class="sweep" id="n-sweep" hidden>
        <div class="pt"><span>Подбор LR: потеря (loss) при растущем LR</span><span id="n-sweep-v"></span></div>
        <div class="pl" data-empty="Первые точки появятся через несколько секунд."></div>
      </div>
      <details class="dis sec params"><summary>Параметры запуска</summary><div class="chips" id="n-chips"></div></details>
    </div>
    <div class="plots" id="plots"></div>
    <div class="journal glass" id="log" role="button" tabindex="0" aria-label="Журнал, открыть целиком"></div>
  </section>
  <section class="view" id="v-blank" hidden>
    <div class="card glass"><div class="empty"><svg class="ic" viewBox="0 0 16 16"><path d="M1.8 8.5h2.6l1.8-4.5 3 8 1.8-3.5h3.2"/></svg>
      <h3>Обучений пока нет</h3><button class="btn primary" data-go="setup">Новое обучение</button></div></div>
  </section>

  <!-- ========================= Запуски ========================= -->
  <section class="view" id="v-runs" hidden aria-label="Запуски">
    <div class="card glass">
      <div class="rtool">
        <input class="inp" id="r-q" type="search" placeholder="Поиск" autocomplete="off" spellcheck="false">
        <span class="seg" id="r-f" role="radiogroup" aria-label="Состояние"></span>
        <div class="rsum" id="r-sum"></div>
      </div>
      <div class="tscroll" style="margin-top:12px;max-height:calc(100vh - 230px)"><table class="tbl hover runs-t" id="r-t"></table></div>
    </div>
    <div class="card glass cmp" id="r-cmp" hidden>
      <div class="ch"><h2>Сравнение</h2><span class="cs" id="r-cmp-s"></span><div class="ca"><button class="btn sm quiet" id="r-cmp-x">Закрыть</button></div></div>
      <div class="legend" id="r-leg"></div>
      <div class="plots" id="r-plots" style="margin-top:12px"></div>
      <div class="tscroll" style="margin-top:14px"><table class="tbl" id="r-ct"></table></div>
    </div>
  </section>

  <!-- ========================= Разговор ========================= -->
  <section class="view" id="v-chat" hidden aria-label="Разговор" style="height:100%">
    <div class="chatv">
      <div class="card glass mcard"><div class="ch"><h2>Модели</h2></div><div class="mlist" id="c-models"></div></div>
      <div class="card glass">
        <div class="ch"><h2 id="c-title">Разговор</h2><select class="inp" id="c-msel" aria-label="Модель"></select><span class="pill" id="c-state"></span>
          <button class="btn sm quiet" id="c-unload" hidden title="Освобождает ОЗУ и видеопамять. Текущий ответ прерывается, при следующем вопросе модель загружается заново."><svg class="ic" viewBox="0 0 16 16"><path d="M8 2.5v5"/><path d="M4.6 4.6a5 5 0 1 0 6.8 0"/></svg><span class="bl">Выгрузить</span></button>
          <div class="ca">
            <label class="frow" style="gap:8px;cursor:pointer" title="Каждый вопрос направляется обученной и исходной моделям, диалог начинается с чистого листа."><input type="checkbox" class="sw" id="c-ab"><span class="muted bl">Сравнение с исходной</span></label>
            <button class="btn sm quiet" id="c-opt" title="Параметры"><svg class="ic" viewBox="0 0 16 16"><path d="M2.5 5h5M11.5 5h2M2.5 11h2M8.5 11h5"/><circle cx="9.5" cy="5" r="1.8"/><circle cx="6.5" cy="11" r="1.8"/></svg><span class="bl">Параметры</span></button>
            <button class="btn sm quiet" id="c-new" title="Новый разговор"><svg class="ic" viewBox="0 0 16 16"><path d="M10.5 2.8l2.7 2.7L6.4 12.3 3.2 13l.7-3.2z"/></svg><span class="bl">Новый разговор</span></button>
          </div></div>
        <div class="copts" id="c-opts" hidden>
          <div class="field"><label for="c-temp">Температура</label><input class="inp" id="c-temp" type="number" min="0" max="2" step="0.05"></div>
          <div class="field"><label for="c-max">Длина ответа, токенов</label><input class="inp" id="c-max" type="number" min="1" max="2048" step="1"></div>
          <div class="field"><label for="c-scale">Сила обучения</label><input class="inp" id="c-scale" type="number" min="0" max="3" step="0.1" title="1 — результат обучения без изменений, 0 — исходная модель, 1,5 — усиленный эффект."></div>
          <div class="field"><label for="c-scene">Системная инструкция</label><input class="inp" id="c-scene" placeholder="Текст системной инструкции"></div>
        </div>
        <div class="loadline" id="c-load" hidden><span id="c-load-t"></span><span class="bar ind"><i></i></span></div>
        <div class="conv" id="c-log"></div>
        <form class="composer" id="c-form"><textarea class="inp" id="c-in" rows="1" placeholder="Сообщение"></textarea>
          <button class="btn primary icon" id="c-send" aria-label="Отправить" title="Отправить · Enter&#10;Новая строка · Shift+Enter"><svg class="ic" viewBox="0 0 16 16"><path d="M8 13V3.5M4 7.5 8 3.5l4 4"/></svg></button></form>
      </div>
    </div>
  </section>

  <!-- ========================= Проверка ========================= -->
  <section class="view" id="v-bench" hidden aria-label="Проверка">
    <div class="card glass">
      <div class="ch"><h2>Проверка вопросами</h2></div>
      <div class="cols" style="grid-template-columns:repeat(auto-fit,minmax(170px,1fr))">
        <div class="field"><label for="b-run">Модель</label><select class="inp" id="b-run"></select></div>
        <div class="field"><label for="b-src">Вопросы</label><select class="inp" id="b-src">
          <option value="run">Из данных запуска</option><option value="file">Из файла</option><option value="list">Собственный список</option></select></div>
        <div class="field"><label for="b-temp">Температура</label><input class="inp" id="b-temp" type="number" min="0" max="2" step="0.05" value="0"></div>
        <div class="field"><label for="b-max">Длина ответа</label><input class="inp" id="b-max" type="number" min="1" max="1024" step="1" value="48"></div>
      </div>
      <div class="field" id="b-file-f" hidden style="margin-top:12px"><label for="b-file">Файл</label><div class="frow"><input class="inp" id="b-file" placeholder="Путь к .jsonl или .txt" spellcheck="false"></div></div>
      <div class="field" id="b-list-f" hidden style="margin-top:12px"><label for="b-list">Вопросы <span class="faint" title="Один вопрос на строку. Ожидаемый ответ указывается после « || » и используется для поиска совпадения.">· Один на строку</span></label><textarea class="inp" id="b-list" rows="6" placeholder="Вопрос || Ожидаемый ответ"></textarea></div>
      <div class="field" style="margin-top:12px"><label for="b-scene">Системная инструкция</label><input class="inp" id="b-scene" placeholder="Текст системной инструкции"></div>
      <div class="frow" style="margin-top:16px;flex-wrap:wrap">
        <label class="frow" style="gap:8px;cursor:pointer"><input type="checkbox" class="sw" id="b-base" checked><span>Сравнить с исходной моделью</span></label>
        <span class="grow"></span><span class="faint" id="b-count"></span>
        <button class="btn primary" id="b-go"><svg class="ic" viewBox="0 0 16 16"><path d="M5 3.6v8.8L12.4 8z" fill="currentColor"/></svg>Запустить проверку</button>
      </div>
    </div>
    <div class="card glass" id="b-empty"><div class="empty"><svg class="ic" viewBox="0 0 16 16"><path d="M2.5 4.5l1.2 1.2L6 3.4M2.5 10l1.2 1.2L6 8.9M8.5 4.8h5M8.5 10.3h5"/></svg><h3>Нет результатов</h3><span id="b-src-info"></span></div></div>
    <div class="card glass" id="b-res" hidden>
      <div class="ch"><h2>Итог</h2><span class="cs" id="b-prog"></span>
        <div class="ca"><button class="btn sm quiet" id="b-copy">Копировать</button><button class="btn sm quiet" id="b-save">Сохранить таблицу</button><button class="btn sm danger" id="b-stop" hidden>Остановить</button></div></div>
      <div class="tiles" id="b-tiles"></div>
      <div class="bar" id="b-bar" style="margin:14px 0"><i></i></div>
      <div class="tscroll"><table class="tbl bench-t" id="b-t"></table></div>
    </div>
  </section>

  <!-- ========================= Настройки ========================= -->
  <section class="view" id="v-settings" hidden aria-label="Настройки">
    <div class="setv">
      <nav class="snav glass" id="s-nav" aria-label="Разделы настроек"></nav>
      <div class="sgroup" id="s-body"></div>
    </div>
  </section>

  </div>
</div>
</div>
</div>
<div class="tip" id="tip"></div>
<div class="toasts" id="toasts" aria-live="polite"></div>
<div id="layer"></div>
<script>
'use strict';
// ============================================================================
// Основа: состояние, числа, значки, тосты, диалоги, слои
// ============================================================================
const $=s=>document.querySelector(s), $$=s=>[...document.querySelectorAll(s)];
const tip=$('#tip');
let H=null, P=null, state=null, cur='', picked=false, view='setup', winOpen=null;
// пределы температур — из датчиков главной карты (crit); троттлинг ждём на 10° ниже предела, без данных датчика — прежние 95°
function tLim(k){ return +((((H&&H.гпу)||{}).пределы||{})[k])||0; }
function tWarn(k){ const c=tLim(k); return c? Math.round(c-10) : 95; }
// Выбор человека для нового обучения живёт здесь. Сервер его не помнит — кроме режима и черновика данных.
const SCENE0={доля:1, всегда:''};
// LoRA — своя вкладка: ранг, alpha, матрицы буквами (q k v o g u d), dropout; хранится строкой «16,32,qkvogud,0.05», как у движка
const LORA0={r:16, alpha:32, tg:'qkvogud', drop:0.05};
const ch={mode:'авто', goal:'срез', tab:'tier', lora:{...LORA0}, file:'', name:'', nameAuto:true, over:{}, scene:{...SCENE0}};
let sceneSeen={path:'', n:0};

const okn=v=>v!=null&&v!==''&&isFinite(v);
const dec=s=>UI.dec==='comma'? String(s).replace(/(\d)\.(\d)/g,'$1,$2') : String(s);
// Разряды делим узким пробелом, но только у чисел от пяти знаков: «8144» читается и так.
const group=t=>t.replace(/^(-?)(\d+)/,(m,s,i)=>s+(i.length<5?i:i.replace(/\B(?=(\d{3})+$)/g,' ')));
const fmt=(v,d=2)=>okn(v)? dec((+v).toFixed(d)) : '—';
const int=v=>okn(v)? group(String(Math.round(v))) : '—';
const num=(v,d=2)=>okn(v)? dec(group((+v).toFixed(d))) : '—';
const gib=(b,d=2)=>okn(b)? num(b/2**30,d)+' ГиБ' : '—';
const bytes=b=>!okn(b)? '—' : b>=2**30? num(b/2**30,b>=100*2**30?0:1)+' ГиБ' : b>=2**20? num(b/2**20,0)+' МиБ' : b>=1024? num(b/1024,0)+' КиБ' : int(b)+' Б';
const pct=(v,d=1)=>okn(v)? dec((+v).toFixed(d))+' %' : '—';
const deg=v=>okn(v)? Math.round(v)+'°' : '—';
const esc=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const cap=s=>{ s=String(s??''); return /^[а-яё]/.test(s)? s[0].toUpperCase()+s.slice(1) : s; };
const plural=(n,a,b,c)=>{ n=Math.abs(Math.round(n)); const t=n%10, h=n%100; return t===1&&h!==11? a : t>=2&&t<=4&&(h<10||h>=20)? b : c; };
const steps_=n=>plural(n,'шаг','шага','шагов');
const base=p=>String(p||'').split('/').filter(Boolean).pop()||'';
const vals=(a,k)=>(a||[]).map(r=>r[k]).filter(v=>v!=null&&isFinite(v));
const max=a=>a.length? a.reduce((m,x)=>x>m?x:m,-Infinity) : null;
const min=a=>a.length? a.reduce((m,x)=>x<m?x:m,Infinity) : null;
const med=a=>{ if(!a.length) return null; const b=[...a].sort((x,y)=>x-y); return b[b.length>>1]; };
const avg=a=>a.length? a.reduce((s,x)=>s+x,0)/a.length : null;
const lastOf=(a,k)=>{ for(let i=(a||[]).length-1;i>=0;i--) if(a[i][k]!=null) return a[i][k]; return null; };
const calm=()=>document.documentElement.classList.contains('calm')||matchMedia('(prefers-reduced-motion:reduce)').matches;
const hms=s=>{ s=Math.max(0,Math.round(s)); return s>=3600? `${Math.floor(s/3600)} ч ${String(Math.floor(s%3600/60)).padStart(2,'0')} мин`
  : s>=60? `${Math.floor(s/60)} мин ${String(s%60).padStart(2,'0')} с` : `${s} с`; };
const post=(url,body)=>fetch(url,{method:'POST',body:JSON.stringify(body||{})}).then(r=>r.json());
const getj=url=>fetch(url).then(r=>r.json());
const sleep=ms=>new Promise(r=>setTimeout(r,ms));
// «2026-10-11 22:55:21» → «Сегодня 22:55», «Вчера 20:38», «09.10.2026 18:02».
function when(s, sec){
  const m=String(s||'').match(/^(\d{4})-(\d\d)-(\d\d)[ T](\d\d):(\d\d)(?::(\d\d))?/); if(!m) return s||'—';
  const d=new Date(+m[1],m[2]-1,+m[3]), t=new Date(), day=86400000, t0=new Date(t.getFullYear(),t.getMonth(),t.getDate());
  const hm=m[4]+':'+m[5]+(sec&&m[6]? ':'+m[6] : ''), k=Math.round((t0-d)/day);
  return k===0? 'Сегодня '+hm : k===1? 'Вчера '+hm : `${m[3]}.${m[2]}.${m[1]} ${hm}`;
}

const PATHS={
  ok:'<path d="M3 8.5l3.2 3.2L13 4.5"/>', need:'<circle cx="8" cy="8" r="5.5"/><path d="M7 5.6 9.4 8 7 10.4"/>',
  bad:'<circle cx="8" cy="8" r="5.5"/><path d="M6.1 6.1l3.8 3.8M9.9 6.1l-3.8 3.8"/>', warn:'<path d="M8 2.6l5.8 10.2H2.2z"/><path d="M8 6.8v2.8M8 11.3v.1"/>',
  wait:'<path d="M8 2.5a5.5 5.5 0 1 1-5.5 5.5"/>', info:'<circle cx="8" cy="8" r="5.5"/><path d="M8 7.4v3.4M8 5.2v.1"/>', todo:'<circle cx="8" cy="8" r="5"/>',
  dot:'<circle cx="8" cy="8" r="3.2" fill="currentColor" stroke="none"/>', pause:'<path d="M6 4.5v7M10 4.5v7"/>', skip:'<path d="M4 8h8"/>',
  stop:'<rect x="4.5" y="4.5" width="7" height="7" rx="1.2"/>', play:'<path d="M5 3.6v8.8L12.4 8z" fill="currentColor"/>', arrow:'<path d="M3 8h10M9.5 4.5 13 8l-3.5 3.5"/>',
  expand:'<path d="M9.5 3H13v3.5M13 3L9 7M6.5 13H3V9.5M3 13l4-4"/>', trash:'<path d="M3 4.5h10M6.5 4.5V3h3v1.5M4.5 4.5l.6 8.5h5.8l.6-8.5"/>',
  chat:'<path d="M2.8 4.2c0-.9.7-1.6 1.6-1.6h7.2c.9 0 1.6.7 1.6 1.6v5c0 .9-.7 1.6-1.6 1.6H7.2L4.4 13v-2.2c-.9 0-1.6-.7-1.6-1.6z"/>',
  open:'<path d="M6 3.5H3.5v9h9V10M9 3.5h3.5V7M12.5 3.5 7.5 8.5"/>', copy:'<rect x="5.5" y="5.5" width="8" height="8" rx="1.5"/><path d="M10.5 5.5V3.8c0-.7-.6-1.3-1.3-1.3H3.8c-.7 0-1.3.6-1.3 1.3v5.4c0 .7.6 1.3 1.3 1.3h1.7"/>',
  save:'<path d="M8 2.5v8M4.8 7.5 8 10.7l3.2-3.2M3 13.5h10"/>', refresh:'<path d="M13 8a5 5 0 1 1-1.5-3.6M13 2.8v2.7h-2.7"/>',
  compare:'<path d="M2.5 12.5l3.5-4 3 2.5 4.5-6"/><path d="M2.5 9l3.5-2 3 1.5 4.5-4" opacity=".45"/>', folder:'<path d="M2.5 5c0-.8.7-1.5 1.5-1.5h2.4l1.4 1.5H12c.8 0 1.5.7 1.5 1.5v5c0 .8-.7 1.5-1.5 1.5H4c-.8 0-1.5-.7-1.5-1.5z"/>',
  file:'<path d="M4 2.5h5l3 3v8H4z"/><path d="M9 2.5v3h3"/>', lock:'<rect x="3.5" y="7" width="9" height="6.5" rx="1.5"/><path d="M5.5 7V5.3a2.5 2.5 0 0 1 5 0V7"/>',
  left:'<path d="M10 3.5 5.5 8l4.5 4.5"/>', right:'<path d="M6 3.5 10.5 8 6 12.5"/>', up:'<path d="M3.5 10 8 5.5l4.5 4.5"/>', down:'<path d="M3.5 6 8 10.5 12.5 6"/>',
  plus:'<path d="M8 3.5v9M3.5 8h9"/>', x:'<path d="M4 4l8 8M12 4l-8 8"/>', search:'<circle cx="7" cy="7" r="4.3"/><path d="M10.2 10.2 13.5 13.5"/>',
  setup:'<rect x="2.5" y="2.5" width="11" height="11" rx="3"/><path d="M8 5.5v5M5.5 8h5"/>', train:'<path d="M1.8 8.5h2.6l1.8-4.5 3 8 1.8-3.5h3.2"/>',
  runs:'<rect x="2.5" y="3" width="11" height="3" rx="1"/><path d="M3.5 6v6.5h9V6M6.5 8.5h3"/>', bench:'<path d="M2.5 4.5l1.2 1.2L6 3.4M2.5 10l1.2 1.2L6 8.9M8.5 4.8h5M8.5 10.3h5"/>',
  settings:'<path d="M2.5 5h5M11.5 5h2M2.5 11h2M8.5 11h5"/><circle cx="9.5" cy="5" r="1.8"/><circle cx="6.5" cy="11" r="1.8"/>',
  theme:'<circle cx="8" cy="8" r="5.5"/><path d="M8 2.5a5.5 5.5 0 0 1 0 11z" fill="currentColor" stroke="none"/>', power:'<path d="M8 2.5v5"/><path d="M4.6 4.6a5 5 0 1 0 6.8 0"/>',
  palette:'<circle cx="8" cy="8" r="5.6"/><circle cx="6" cy="6.2" r=".9" fill="currentColor" stroke="none"/><circle cx="9.6" cy="5.8" r=".9" fill="currentColor" stroke="none"/><circle cx="10.6" cy="9" r=".9" fill="currentColor" stroke="none"/><path d="M8 13.6c-1-.6-.8-2.2.6-2.2"/>',
  chart:'<path d="M2.5 13.5h11M4 11V8M7 11V5M10 11V7M13 11V3.5"/>', chip:'<rect x="4" y="4" width="8" height="8" rx="1.5"/><path d="M6 1.8V4M10 1.8V4M6 12v2.2M10 12v2.2M1.8 6H4M1.8 10H4M12 6h2.2M12 10h2.2"/>',
  keys:'<rect x="1.5" y="4" width="13" height="8" rx="1.5"/><path d="M4 7h.01M6.5 7h.01M9 7h.01M11.5 7h.01M5 9.5h6"/>', globe:'<circle cx="8" cy="8" r="5.5"/><path d="M2.5 8h11M8 2.5c1.6 1.6 2.4 3.4 2.4 5.5S9.6 11.9 8 13.5C6.4 11.9 5.6 10.1 5.6 8S6.4 4.1 8 2.5z"/>',
  bolt:'<path d="M9 1.8 3.5 9h4l-.5 5.2L12.5 7h-4z"/>', send:'<path d="M8 13V3.5M4 7.5 8 3.5l4 4"/>', edit:'<path d="M10.5 2.8l2.7 2.7L6.4 12.3 3.2 13l.7-3.2z"/>',
  eye:'<path d="M1.8 8S4 3.8 8 3.8 14.2 8 14.2 8 12 12.2 8 12.2 1.8 8 1.8 8z"/><circle cx="8" cy="8" r="2"/>', star:'<path d="M8 2.3l1.7 3.6 3.9.5-2.9 2.7.8 3.9L8 11.1 4.5 13l.8-3.9-2.9-2.7 3.9-.5z"/>',
  build:'<path d="M9.8 2.8a3 3 0 0 0-3.6 3.9L2.5 10.4l3.1 3.1 3.7-3.7a3 3 0 0 0 3.9-3.6l-1.9 1.9-1.6-.5-.5-1.6z"/>',
};
const hint=t=>t? `<span class="hint" title="${esc(t)}">${ic('info')}</span>` : '';
const ic=(n,cls='')=>`<svg class="ic${cls?' '+cls:''}" viewBox="0 0 16 16" aria-hidden="true">${PATHS[n]||''}</svg>`;

// --- слои: Esc закрывает верхний, фокус возвращается туда, откуда открыли ---
const layers=[];
function pushLayer(el, close){ const back=document.activeElement; const L={el,close:()=>{ const i=layers.indexOf(L); if(i>=0) layers.splice(i,1); close(); el.remove(); if(back&&back.focus) back.focus(); }}; layers.push(L); $('#layer').append(el); return L; }
function topLayer(){ return layers[layers.length-1]; }

function toast(title, text='', kind='info', ms=4200){
  const t=document.createElement('div'); t.className='toast pop '+kind; t.setAttribute('role','status');
  t.innerHTML=ic({ok:'ok',bad:'bad',warn:'warn'}[kind]||'info')+`<div><b>${esc(title)}</b>${text?`<span>${esc(text)}</span>`:''}</div>`;
  $('#toasts').append(t); const gone=()=>{ t.classList.add('out'); setTimeout(()=>t.remove(),280); };
  t.onclick=gone; setTimeout(gone, ms);
}
// Диалог вместо системного confirm/prompt: в тон оформлению и с переводом.
function dialog({title, text='', ok='Готово', cancel='Отмена', danger=false, input=null}){
  return new Promise(res=>{
    const s=document.createElement('div'); s.className='scrim';
    const field= !input? '' : input.area? `<textarea class="inp${input.mono?' code':''}" rows="${input.rows||8}" placeholder="${esc(input.ph||'')}" spellcheck="false">${esc(input.value||'')}</textarea>`
      : `<input class="inp" value="${esc(input.value||'')}" placeholder="${esc(input.ph||'')}" spellcheck="false">`;
    s.innerHTML=`<div class="pop dialog" role="dialog" aria-modal="true"><h3>${esc(title)}</h3>${text?`<p class="dt">${esc(text)}</p>`:''}${field}
      <div class="dacts">${cancel?`<button class="btn quiet" data-r="0">${esc(cancel)}</button>`:''}<button class="btn ${danger?'danger solid':'primary'}" data-r="1">${esc(ok)}</button></div></div>`;
    let done=false; const fin=v=>{ if(done) return; done=true; L.close(); res(v); };
    const L=pushLayer(s, ()=>{ if(!done){ done=true; res(false); } });
    const f=s.querySelector('input,textarea');
    s.addEventListener('pointerdown',e=>{ if(e.target===s) fin(false); });
    s.querySelectorAll('[data-r]').forEach(b=>b.onclick=()=>fin(b.dataset.r==='1'? (f? f.value : true) : false));
    s.addEventListener('keydown',e=>{ if(e.key==='Enter'&&!(f&&f.tagName==='TEXTAREA')){ e.preventDefault(); fin(f? f.value : true); } });
    setTimeout(()=>(f||s.querySelector('[data-r="1"]')).focus(), 20);
  });
}
async function copyText(t, what='Скопировано'){
  try{ await navigator.clipboard.writeText(t); toast(what,'','ok',1800); }
  catch(e){ const a=document.createElement('textarea'); a.value=t; document.body.append(a); a.select(); try{ document.execCommand('copy'); toast(what,'','ok',1800); }catch(_){ toast('Не удалось скопировать','','bad'); } a.remove(); }
}
function rangeFill(el){ const lo=+el.min||0, hi=+el.max||1, v=+el.value; el.style.setProperty('--v', (100*(v-lo)/((hi-lo)||1))+'%'); }

// ============================================================================
// Кривые. Одна ось на картинку; рисуем в настоящих пикселях, подписи не тянутся.
// ============================================================================
function nice(lo,hi,n){
  if(!isFinite(lo)||!isFinite(hi)){ lo=0; hi=1; }
  if(lo===hi){ const pad=Math.abs(lo)*0.05||1; lo-=pad; hi+=pad; }
  const raw=(hi-lo)/n, p=Math.pow(10,Math.floor(Math.log10(raw))), f=raw/p, st=(f<1.5?1:f<3?2:f<7?5:10)*p;
  return {lo:Math.floor(lo/st+1e-9)*st, hi:Math.ceil(hi/st-1e-9)*st, st};
}
// Точек больше, чем пикселей, рисовать незачем. Из каждой горсти — низ и верх: всплески не пропадают.
function thin(a,W){
  const n=a.length, capn=Math.max(120, Math.round(W*0.75));
  if(n<=capn*2) return a;
  const out=[];
  for(let b=0;b<capn;b++){ const i0=Math.floor(b*n/capn), i1=Math.floor((b+1)*n/capn); let lo=i0, hi=i0;
    for(let i=i0+1;i<i1;i++){ if(a[i].y<a[lo].y) lo=i; if(a[i].y>a[hi].y) hi=i; }
    out.push(a[Math.min(lo,hi)]); if(lo!==hi) out.push(a[Math.max(lo,hi)]); }
  if(out[out.length-1]!==a[n-1]) out.push(a[n-1]);
  return out;
}
function near(d,x){ let lo=0, hi=d.length-1; while(hi-lo>1){ const m=(lo+hi)>>1; if(d[m].x<x) lo=m; else hi=m; }
  return Math.abs(d[lo].x-x)<=Math.abs(d[hi].x-x)? d[lo] : d[hi]; }
let chartN=0;
function chart(el,o){
  const W=el.clientWidth||o.w||560, Hh=el.clientHeight||o.h||180, cmp=!!o.compact;
  const pts=o.series.map(s=>Object.assign({},s,{data:thin(s.data,W)})).filter(s=>s.data.length);
  if(!pts.length){ el.replaceChildren(); el._c=null; return; }
  let x0=Infinity, x1=-Infinity, y0=Infinity, y1=-Infinity;
  pts.forEach(s=>s.data.forEach(p=>{ if(p.x<x0)x0=p.x; if(p.x>x1)x1=p.x; if(p.y<y0)y0=p.y; if(p.y>y1)y1=p.y; }));
  if(o.rule!=null){ y0=Math.min(y0,o.rule); y1=Math.max(y1,o.rule); }
  if(o.vrule){ x0=Math.min(x0,o.vrule.x); x1=Math.max(x1,o.vrule.x); }
  if(o.zero) y0=Math.min(0,y0);
  const ny=nice(y0,y1,cmp?3:4), ticks=[];
  for(let i=0; ny.lo+i*ny.st<=ny.hi+ny.st*1e-6; i++) ticks.push(ny.lo+i*ny.st);
  const fy=o.fy||(v=>fmt(v, ny.st<1?(ny.st<0.1?2:1):0)), tl=ticks.map(fy);
  const T=(o.rule!=null||o.vrule)?16:8, B=20, L=Math.max(...tl.map(t=>String(t).length))*6.3+12;
  const sy=v=>T+(1-(v-ny.lo)/(ny.hi-ny.lo||1))*(Hh-T-B);
  const ends=pts.map(s=>({s, y:sy(s.data[s.data.length-1].y)})).sort((a,b)=>a.y-b.y);
  const direct=!cmp && pts.length>1 && pts.length<=4 && ends.every((e,i)=>!i||e.y-ends[i-1].y>=15);
  const R=direct? Math.max(...pts.map(s=>s.name.length))*6.6+18 : 8;
  const sx=v=>L+(x1===x0?0:(v-x0)/(x1-x0))*(W-L-R);
  const gid='g'+(++chartN);
  let g=`<defs>${pts.map((s,i)=>`<linearGradient id="${gid}${i}" x1="0" y1="0" x2="0" y2="1"><stop offset="0" style="stop-color:${s.color}" stop-opacity="1"/><stop offset="1" style="stop-color:${s.color}" stop-opacity="0"/></linearGradient>`).join('')}</defs>`;
  ticks.forEach((v,i)=>{ const y=Math.round(sy(v))+.5;
    g+=`<line class="gl" x1="${L}" y1="${y}" x2="${W-R}" y2="${y}"/><text class="ax" x="${L-8}" y="${y+3.5}" text-anchor="end">${tl[i]}</text>`; });
  const xt=o.xt? o.xt.filter(v=>v>=x0-1e-9&&v<=x1+1e-9) : (cmp||x1===x0? [x0,x1] : [0,1,2,3].map(i=>x0+(x1-x0)*i/3));
  xt.forEach((v,i)=>{ if(i&&v===xt[i-1]) return;
    const px=sx(v), an=o.xt? (px<L+14?'start':px>W-R-14?'end':'middle') : (i?(i===xt.length-1?'end':'middle'):'start');
    g+=`<text class="ax" x="${px.toFixed(1)}" y="${Hh-4}" text-anchor="${an}">${o.fx? o.fx(v) : (i?'':'Шаг ')+Math.round(v)}</text>`; });
  if(o.rule!=null){ const y=Math.round(sy(o.rule))+.5;
    g+=`<line class="rule" x1="${L}" y1="${y}" x2="${W-R}" y2="${y}"/><text class="ax" x="${W-R}" y="${y-5}" text-anchor="end">${o.ruleLabel||''}</text>`; }
  if(o.vrule){ const x=Math.round(sx(o.vrule.x))+.5, an=x>W-R-90?'end':'start';
    g+=`<line class="vr" x1="${x}" y1="${T-4}" x2="${x}" y2="${Hh-B}"/><text class="vrl" x="${an==='end'?x-6:x+6}" y="${T+6}" text-anchor="${an}">${esc(o.vrule.label||'')}</text>`; }
  const yb=sy(Math.max(ny.lo, Math.min(ny.hi, o.zero? 0 : ny.lo)));
  pts.forEach((s,i)=>{
    const Pp=s.data.map(p=>[sx(p.x).toFixed(1), sy(p.y).toFixed(1)]), d=Pp.map((p,j)=>(j?'L':'M')+p[0]+' '+p[1]).join('');
    if(i===0 && !s.dots && o.area!==false && Pp.length>1) g+=`<path class="area" d="${d}L${Pp[Pp.length-1][0]} ${yb.toFixed(1)}L${Pp[0][0]} ${yb.toFixed(1)}Z" fill="url(#${gid}${i})"/>`;
    g+=`<path class="ln" d="${d}" style="stroke:${s.color}"/>`;
    const sparse=s.dots && (Pp.length<2 || (W-L-R)/(Pp.length-1)>=12);
    (sparse? Pp : [Pp[Pp.length-1]]).forEach(p=>{ g+=`<circle class="ring${s.dots?' ev':''}" cx="${p[0]}" cy="${p[1]}" r="3.6" style="fill:${s.color}"/>`; });
  });
  if(direct) ends.forEach(e=>{ g+=`<text class="lbl" x="${W-R+10}" y="${(e.y+4).toFixed(1)}">${esc(e.s.name)}</text>`; });
  g+=`<g class="hov" style="display:none"><line class="cross" y1="${T}" y2="${Hh-B}"/>${pts.map(s=>`<circle class="ring" r="4" style="fill:${s.color}"/>`).join('')}</g>`;
  el.innerHTML=`<svg role="img" aria-label="${esc(o.title||'')}" width="${W}" height="${Hh}">${g}</svg>`;
  el._c={pts,sx,sy,x0,x1,L,R,W,ft:o.ft,tx:o.tx};
  if(!el._on){ el._on=1;
    el.addEventListener('pointermove', e=>{ el._px=e.clientX; el._py=e.clientY; hover(el); });
    el.addEventListener('pointerleave', ()=>{ el._px=null; hover(el); }); }
  if(el._px!=null) hover(el);
}
// Перекрестие ищет шаг, а не линию: целиться в кривую толщиной два пикселя не приходится.
function hover(el){
  const c=el._c, hov=el.querySelector('.hov');
  if(!c||!hov) return;
  const r=el.getBoundingClientRect(), px=el._px==null || (layers.length && !el.closest('#layer'))? -1 : el._px-r.left;
  if(px<c.L-6 || px>c.W-c.R+6){ hov.style.display='none'; tip.style.opacity=0; return; }
  const xv=c.x0+Math.max(0,Math.min(1,(px-c.L)/(c.W-c.L-c.R||1)))*(c.x1-c.x0);
  const hit=c.pts.map(s=>near(s.data,xv)), main=c.pts.reduce((a,s,i)=>s.data.length>c.pts[a].data.length?i:a, 0);
  const cx=c.sx(hit[main].x).toFixed(1), line=hov.querySelector('.cross'), dots=hov.querySelectorAll('circle');
  line.setAttribute('x1',cx); line.setAttribute('x2',cx);
  hit.forEach((p,i)=>{ dots[i].setAttribute('cx',c.sx(p.x).toFixed(1)); dots[i].setAttribute('cy',c.sy(p.y).toFixed(1)); });
  hov.style.display='';
  tip.innerHTML=`<div class="th">${c.tx? c.tx(hit[main].x) : 'Шаг '+Math.round(hit[main].x)}</div>`+c.pts.map((s,i)=>
    `<div class="tr"><i style="background:${s.color}"></i><b>${c.ft?c.ft(hit[i].y):fmt(hit[i].y)}</b><span>${esc(s.name)}${c.pts.length>1&&Math.abs(hit[i].x-hit[main].x)>0.5? ' · Шаг '+Math.round(hit[i].x) : ''}</span></div>`).join('');
  const tw=tip.offsetWidth, left=el._px+14+tw>innerWidth-8? el._px-14-tw : el._px+14;
  tip.style.transform=`translate(${Math.max(8,left)}px,${Math.max(8,el._py-12)}px)`; tip.style.opacity=1;
}

// Подсказки: title любого элемента показывается своей плашкой — в тон оформлению и с задержкой.
// Пояснения в интерфейсе живут только здесь: на виду — факты, за наведением — почему.
const tt=document.createElement('div'); tt.className='tip tt'; tt.setAttribute('role','tooltip'); document.body.append(tt);
let ttEl=null, ttT=0;
function ttHide(){ clearTimeout(ttT); ttEl=null; tt.style.opacity=0; }
function ttShow(el){
  const t=el.dataset.tip; if(!t||!el.isConnected){ ttHide(); return; }
  tt.textContent=t; tt.style.opacity=0; tt.style.transform='translate(-9999px,0)';
  const r=el.getBoundingClientRect(), w=tt.offsetWidth, h=tt.offsetHeight;
  let x=Math.round(r.left+r.width/2-w/2), y=Math.round(r.bottom+8);
  if(y+h>innerHeight-8) y=Math.round(r.top-h-8);
  x=Math.max(8,Math.min(innerWidth-w-8,x));
  tt.style.transform=`translate(${x}px,${y}px)`; tt.style.opacity=1;
}
document.addEventListener('pointerover', e=>{
  if(e.pointerType==='touch') return;
  const el=e.target.closest&&e.target.closest('[title],[data-tip]');
  if(el===ttEl) return;
  if(el && el.hasAttribute('title')){ const t=el.getAttribute('title'); if(t) el.dataset.tip=t; else delete el.dataset.tip; el.removeAttribute('title'); }
  ttHide(); if(!el||!el.dataset.tip||el.closest('.plot .pl,.wplot .pl')) return;
  ttEl=el; ttT=setTimeout(()=>ttShow(el), 420);
}, true);
['pointerdown','scroll','keydown','wheel'].forEach(k=>document.addEventListener(k, ttHide, true));
document.addEventListener('pointerleave', ttHide);
// ============================================================================
// Графики запуска. Каждый знает своё: главное число, кривую и подробности.
// Снаружи — самое нужное, раскрытое окно — до последнего знака.
// ============================================================================
const S1='var(--s1)', S2='var(--s2)', S3='var(--s3)', S4='var(--s4)', S5='var(--s5)', S6='var(--s6)';
const PALETTE=[S1,S2,S3,S4,S5,S6];
const ser=(a,k,name,color,dots)=>({name,color,dots, data:(a||[]).filter(p=>p[k]!=null&&p['шаг']!=null).map(p=>({x:p['шаг'],y:p[k]}))});
const kilo=v=>v>=1000? dec(+(v/1000).toFixed(1))+'k' : Math.round(v);
// Развёртка: потеря против шага обучения. Шаг растёт в разы, поэтому ось — по степеням десяти.
function sweepChart(s){
  const all=(s.развёртка||[]).filter(r=>r[1]>0);
  if(!all.length) return null;
  // За дном потеря улетает в сотни — без отсечки вся полезная часть легла бы в одну линию у оси.
  const capv=1.5*Math.max(...all.slice(0,3).map(r=>r[2])), d=all.filter(r=>r[2]<=capv), cut=all.length-d.length;
  const lo=Math.floor(Math.log10(all[0][1])), hi=Math.ceil(Math.log10(all[all.length-1][1])), xt=[];
  for(let v=lo; v<=hi; v++) xt.push(v);
  const lr=parseFloat((s.план||{}).lr);
  return {title:'Подбор LR', note:'Подбор LR: 40 шагов с растущим LR; выбирается одна десятая значения LR в минимуме сглаженной потери (loss).'+(cut? ` Не показано выше ${fmt(capv,1)}: ${cut}.` : ''),
    series:[{name:'Потеря (loss)', color:S1, dots:true, data:d.map(r=>({x:Math.log10(r[1]), y:r[2]}))}],
    opt:{xt, fx:v=>'1e'+v, tx:v=>'LR '+Math.pow(10,v).toExponential(3), ft:v=>fmt(v,4), area:false,
         vrule: isFinite(lr)&&lr>0? {x:Math.log10(lr), label:'Выбран '+s.план.lr} : null}};
}
const WIN={
  loss:{ title:'Потеря (loss)', note:'Ниже — лучше. Обучение сглажено по 50 шагам, валидация — отдельные замеры.',
    now(s){ return {v:fmt(lastOf(s.шаги,'ср'),4)}; },
    charts(s){ return [{series:[ser(s.шаги,'ср','Обучение',S1), ser(s.оценки,'loss','Валидация',S2,true)], opt:{ft:v=>fmt(v,4)}}]; },
    big(s){ return [
      {title:'Потеря (loss)', series:[ser(s.шаги,'ср','Обучение',S1), ser(s.оценки,'loss','Валидация',S2,true)], opt:{ft:v=>fmt(v,4)}},
      {title:'Потеря (loss) по шагам без сглаживания', note:'Без сглаживания: виден разброс между примерами.', series:[ser(s.шаги,'loss','Шаг',S1)], opt:{ft:v=>fmt(v,4)}}]; },
    stats(s){ const ev=s.оценки, st=s.шаги, L=vals(ev,'loss'), best=min(L), bi=ev.find(e=>e.loss===best);
      const tail=vals(st.slice(-50),'ср'), head=vals(st.slice(-100,-50),'ср');
      return [['Лучший на валидации', best==null?'—':fmt(best,4), bi?('Шаг '+Math.round(bi['шаг'])):''],
              ['Последний на валидации', L.length?fmt(L[L.length-1],4):'—', 'Валидаций: '+ev.length],
              ['Среднее на обучении', fmt(lastOf(st,'ср'),4), 'Последние 50 шагов'],
              ['Перплексия', num(lastOf(st,'ppl'),2), 'exp(loss)'],
              ['Падение за 50 шагов', (avg(head)!=null&&avg(tail)!=null)?fmt(avg(head)-avg(tail),4):'—', 'Разность средних'],
              ['Токенов в расчёте потери', st.length?int(vals(st,'в_loss').reduce((a,b)=>a+b,0)):'—', 'За весь запуск']]; },
    table(s){ return {cols:['Шаг','На валидации','Перплексия','Лучший'], rows:s.оценки.slice().reverse().map(e=>[Math.round(e['шаг']), fmt(e.loss,4),
           e.loss!=null?num(Math.exp(e.loss),2):'—', fmt(e['лучший'],4)])}; } },
  acc:{ title:'Точность на ответе', note:'Доля верно предсказанных токенов ответа.',
    now(s){ return {v:pct(lastOf(s.шаги,'acc_ср'),2)}; },
    charts(s){ return [{series:[ser(s.шаги,'acc_ср','Обучение',S1), ser(s.оценки,'acc','Валидация',S2,true)], opt:{zero:true, fy:v=>Math.round(v)+'%', ft:v=>pct(v,2)}}]; },
    big(s){ return [
      {title:'Точность', series:[ser(s.шаги,'acc_ср','Обучение',S1), ser(s.оценки,'acc','Валидация',S2,true)], opt:{zero:true,fy:v=>Math.round(v)+'%',ft:v=>pct(v,2)}},
      {title:'Длина примеров', note:'Число токенов на шаг и число токенов, учитываемых в расчёте потери (loss).', series:[ser(s.шаги,'токенов','Всего',S1), ser(s.шаги,'в_loss','В расчёте потери',S3)], opt:{zero:true,ft:v=>int(v)}}]; },
    stats(s){ const st=s.шаги, ev=s.оценки, A=vals(ev,'acc');
      return [['На валидации', A.length?pct(A[A.length-1],2):'—', 'Лучшая: '+(A.length?pct(max(A),2):'—')],
              ['На обучении', pct(lastOf(st,'acc_ср'),2), 'Среднее за 50 шагов'],
              ['Токенов в расчёте потери', int(med(vals(st,'в_loss'))), 'Медиана на шаг'],
              ['Токенов всего', int(med(vals(st,'токенов'))), 'Медиана на шаг'],
              ['Примеров в шаге', int(med(vals(st,'примеров'))), 'Медиана'],
              ['Доля в расчёте потери', (()=>{const a=avg(vals(st,'в_loss')), b=avg(vals(st,'токенов')); return (a&&b)? pct(100*a/b,2):'—';})(), 'Остальное — текст вопроса']]; },
    table(s){ return {cols:['Шаг','На валидации','На обучении'], rows:s.оценки.slice().reverse().map(e=>{ const n=s.шаги.filter(r=>r['шаг']<=e['шаг']).pop();
        return [Math.round(e['шаг']), pct(e.acc,2), n?pct(n.acc_ср,2):'—']; })}; } },
  vram:{ title:'Видеопамять', note:'Пик за шаг; пунктир — потолок карты.',
    now(s){ const v=lastOf(s.шаги,'vram'), t=lastOf(s.шаги,'vram_всего');
      return {v:int(v), unit:v==null?'':'МиБ', sub:t?'Из '+int(t)+' МиБ':'', flag:v&&t&&v/t>0.96?'Почти потолок':''}; },
    charts(s){ return [{series:[ser(s.шаги,'vram','Пик',S1)], opt:{zero:true, rule:lastOf(s.шаги,'vram_всего'), ruleLabel:'Потолок', fy:int, ft:v=>int(v)+' МиБ'}}]; },
    big(s){ return [
      {title:'Пик за шаг', series:[ser(s.шаги,'vram','Пик',S1)], opt:{zero:true,rule:lastOf(s.шаги,'vram_всего'),ruleLabel:'Потолок', fy:int, ft:v=>int(v)+' МиБ'}},
      {title:'Состав памяти', note:'Живая память и область видеопамяти для промежуточных данных слоя; рост значения «Вне области» означает фрагментацию памяти.',
       series:[ser(s.шаги,'живых','Живых',S1), ser(s.шаги,'область','Область',S3)], opt:{zero:true,fy:int,ft:v=>int(v)+' МиБ'}}]; },
    stats(s){ const st=s.шаги, V=vals(st,'vram'), last=st[st.length-1]||{}, tot=last.vram_всего;
      return [['Пик за запуск', V.length?int(max(V))+' МиБ':'—', tot?('Из '+int(tot)+' МиБ'):''],
              ['Запас до потолка', (V.length&&tot)?int(tot-max(V))+' МиБ':'—', 'Менее 200 МиБ — риск аварийной остановки.'],
              ['Обучаемая часть в видеопамяти', last.tier1!=null?int(last.tier1):'—', 'Тензоров, постоянно находящихся в памяти'],
              ['Живых', last['живых']!=null?int(last['живых'])+' МиБ':'—', 'Блоков: '+(last['блоков']!=null?int(last['блоков']):'—')],
              ['Область видеопамяти', last['область']!=null?int(last['область'])+' МиБ':'—', 'Вне области: '+(last['мимо']!=null?int(last['мимо']):'—')],
              ['Кэш в ОЗУ', last['озу']!=null?num(last['озу'],2)+' ГиБ':'—', 'Замороженные веса, BF16']]; },
    table(s){ return {cols:['Шаг','Пик','Живых','Блоков','Область','Вне области'], rows:s.шаги.slice(-200).reverse().map(r=>[Math.round(r['шаг']), int(r.vram), int(r['живых']),
           int(r['блоков']), int(r['область']), int(r['мимо'])])}; } },
  temp:{ title:'Температура', note:'Самая горячая точка кристалла, память и край.',
    now(s){ const h=lastOf(s.шаги,'горячая'); return {v:deg(h), flag:h>tWarn('горячая')?'Троттлинг':''}; },
    charts(s){ return [{series:[ser(s.шаги,'горячая','Горячая',S1), ser(s.шаги,'память','Память',S2), ser(s.шаги,'край','Край',S3)], opt:{fy:deg, fk:deg, ft:v=>Math.round(v)+' °C'}}]; },
    big(s){ return [
      {title:'Температуры', series:[ser(s.шаги,'горячая','Горячая',S1), ser(s.шаги,'память','Память',S2), ser(s.шаги,'край','Край',S3)], opt:{fy:deg,ft:v=>Math.round(v)+' °C'}},
      {title:'Вентилятор', note:'Обороты в минуту; ноль на холодной карте — норма.', series:[ser(s.шаги,'вент','об/мин',S1)], opt:{zero:true,ft:v=>int(v)+' об/мин'}}]; },
    stats(s){ const st=s.шаги, hot=vals(st,'горячая'), edge=vals(st,'край'), mem=vals(st,'память'), fan=vals(st,'вент'), over=hot.filter(v=>v>=tWarn('горячая')).length;
      return [['Самая горячая точка', deg(max(hot)), tLim('горячая')? `Предел кристалла ${tLim('горячая')}°` : 'Предел кристалла ≈ 110°'],
              ['Край', deg(max(edge)), 'Сейчас: '+deg(edge[edge.length-1])],
              ['Память', deg(max(mem)), `Выше ${tWarn('память')}° — троттлинг`],
              ['Вентилятор', fan.length?int(max(fan)):'—', 'Максимум, об/мин'],
              [`Шагов выше ${tWarn('горячая')}°`, over, over?'Троттлинг зафиксирован':'Не зафиксировано'],
              ['Рост за запуск', (hot.length>10)? '+'+Math.round(hot[hot.length-1]-hot[0])+'°':'—', 'От начала запуска до текущего момента.']]; },
    table(s){ return {cols:['Шаг','Горячая','Память','Край','Вентилятор'], rows:s.шаги.slice(-200).reverse().map(r=>[Math.round(r['шаг']), fmt(r['горячая'],0), fmt(r['память'],0),
           fmt(r['край'],0), int(r['вент'])])}; } },
  time:{ title:'Длительность шага', note:'Всплески соответствуют валидации и записи снимка весов.',
    now(s){ const v=lastOf(s.шаги,'сек'), t=lastOf(s.шаги,'ток'); return {v:fmt(v,3), unit:v==null?'':'с', sub:t!=null?int(t)+' ток/с':''}; },
    charts(s){ return [{series:[ser(s.шаги,'сек','Секунд',S1)], opt:{zero:true, ft:v=>fmt(v,3)+' с'}}]; },
    big(s){ return [
      {title:'Секунд на шаг', series:[ser(s.шаги,'сек','Секунд',S1)], opt:{zero:true,ft:v=>fmt(v,3)+' с'}},
      {title:'Распределение времени шага', note:'Прямой проход, LM head, обратный проход с шагом оптимизатора.',
       series:[ser(s.шаги,'обратный','Обратный',S1), ser(s.шаги,'прямой','Прямой',S2), ser(s.шаги,'головка','LM head',S3)], opt:{zero:true,ft:v=>fmt(v,3)+' с'}}]; },
    stats(s){ const st=s.шаги, T=vals(st,'сек'), tk=vals(st,'ток'), slow=[...st].sort((a,b)=>(b['сек']||0)-(a['сек']||0))[0], norm=T.filter(v=>v<2*(med(T)||1));
      return [['Обычный шаг', T.length?fmt(med(T),3)+' с':'—', 'Медиана'],
              ['Средний', T.length?fmt(avg(T),3)+' с':'—', 'Со всплесками'],
              ['Самый долгий', T.length?fmt(max(T),2)+' с':'—', slow?('Шаг '+Math.round(slow['шаг'])):''],
              ['Токенов в секунду', int(med(tk)), 'Медиана'],
              ['Всего времени', T.length? hms(T.reduce((a,b)=>a+b,0)) : '—', 'Сумма по шагам'],
              ['Доля всплесков', T.length? pct(100*(1-norm.length/T.length),2):'—', 'Шагов длиннее двух медиан']]; },
    table(s){ return {cols:['Шаг','Секунд','Прямой','LM head','Обратный','ток/с'], rows:[...s.шаги].sort((a,b)=>(b['сек']||0)-(a['сек']||0)).slice(0,60)
            .map(r=>[Math.round(r['шаг']), fmt(r['сек'],3), fmt(r['прямой'],3), fmt(r['головка'],3), fmt(r['обратный'],3), int(r['ток'])])}; } },
  grad:{ title:'Норма градиента', note:s=>'Мера того, насколько сильно требуется изменить веса.'+((s.развёртка||[]).length? ' Ниже — подбор LR.' : ''),
    now(s){ const lr=(s.последний||{}).lr; return {v:num(lastOf(s.шаги,'гн'),1), sub:lr?'LR '+lr:''}; },
    charts(s){ return [{series:[ser(s.шаги,'гн','Норма',S1)], opt:{zero:true, fy:kilo, ft:v=>num(v,1)}}]; },
    big(s){ const out=[
      {title:'Норма по шагам', series:[ser(s.шаги,'гн','Норма',S1), ser(s.шаги,'гн_макс','Максимум',S2)], opt:{zero:true,fy:kilo,ft:v=>num(v,1)}},
      {title:'LR', note:'Разгон в начале, затем плавное снижение.', series:[ser(s.шаги,'lr','LR',S1)], opt:{zero:true,fy:v=>v?v.toExponential(1):'0',ft:v=>v.toExponential(3)}}];
      const sw=sweepChart(s); if(sw) out.push(sw); return out; },
    stats(s){ const G=vals(s.шаги,'гн'), first=G.slice(0,10), last=G.slice(-10);
      return [['Сейчас', G.length?num(G[G.length-1],1):'—', ''],
              ['Максимум за запуск', G.length?num(max(G),1):'—', ''],
              ['В начале', first.length?num(avg(first),1):'—', 'Среднее первых 10 шагов'],
              ['Снижение', (avg(first)&&avg(last))?fmt(avg(first)/avg(last),2)+'×':'—', 'Начало к концу'],
              ['Выбросов', G.filter((v,i)=>i>5&&v>3*avg(G.slice(Math.max(0,i-50),i))).length, 'Втрое выше среднего за 50 шагов.'],
              ['LR', (s.план||{}).lr||'—', s.развёртка&&s.развёртка.length? 'Подобран' : 'Задан заранее']]; },
    table(s){ return {cols:['Шаг','Норма','Максимум','Потеря (loss)'], rows:s.шаги.slice(-200).reverse().map(r=>[Math.round(r['шаг']), num(r['гн'],1), num(r['гн_макс'],1), fmt(r.loss,4)])}; } }
};
const ALL_CHARTS=['loss','acc','vram','temp','time','grad'];
const chartOrder=()=>{ const a=(UI.charts||ALL_CHARTS).filter(k=>WIN[k]); return a.length? a : ALL_CHARTS; };

function keys(c, values){
  if(c.series.length<2) return '';
  const ft=(c.opt||{}).fk||(c.opt||{}).ft||fmt;
  return c.series.map((x,i)=>{ const v=values&&i&&x.data.length? ` <b>${ft(x.data[x.data.length-1].y)}</b>` : '';
    return `<span><i${x.dots?' class="d"':''} style="background:${x.color}"></i>${esc(x.name)}${v}</span>`; }).join('');
}
function plotCols(g){ const n=+UI.ccols||0; if(n) return n; const w=(g&&g.clientWidth)||$('#views').clientWidth||1200; return Math.max(1,Math.min(4,Math.floor((w+12)/430))); }
// Разметку плиток строим один раз на набор: перестройка уводила бы фокус и перекрестие из-под курсора.
function plots(s){
  const g=$('#plots'), order=chartOrder(), sig=order.join(',');
  g.style.setProperty('--cols', plotCols(g)); g.style.setProperty('--chh', (+UI.ch||150)+'px');
  if(g._sig!==sig){ g._sig=sig; g.replaceChildren();
    order.forEach((id,n)=>{ const w=WIN[id], d=document.createElement('div');
      d.className='plot glass'; d.dataset.win=id; d.tabIndex=0; d.setAttribute('role','button'); d.setAttribute('aria-label', 'Подробности: '+w.title+', клавиша '+(n+1));
      d.innerHTML=`<div class="pt"><span>${w.title}</span><span class="kbd" title="Клавиша">${n+1}</span><span class="flag" hidden></span><span class="exp">${ic('expand')}</span></div>
        <div class="pv"><b></b><small></small></div><div class="keys"></div><div class="pl" data-empty="Данные появятся после первых шагов."></div>`;
      d.onclick=()=>openWin(id,d); d.onkeydown=e=>{ if(e.key==='Enter'||e.key===' '){ e.preventDefault(); openWin(id,d); } };
      g.append(d); }); }
  order.forEach(id=>{ const w=WIN[id], d=g.querySelector(`[data-win="${id}"]`), c=w.charts(s)[0], n=w.now(s), fl=d.querySelector('.flag');
    const has=c.series.some(x=>x.data.length);
    d.querySelector('.pv b').textContent=has? n.v : ''; d.querySelector('.pv small').textContent=has? n.unit||'' : '';
    fl.hidden=!n.flag; d.querySelector('.exp').hidden=!!n.flag; if(n.flag) fl.innerHTML=ic('warn')+`<span>${n.flag}</span>`;
    d.querySelector('.keys').innerHTML= has? keys(c,true) || esc(n.sub||'') : '';
    chart(d.querySelector('.pl'), Object.assign({series:c.series,title:w.title,compact:true}, c.opt||{})); });
}

// --- окно подробностей ------------------------------------------------------
function planTable(rows){
  return '<table class="tbl"><thead><tr><th>Параметр</th><th>Значение</th><th>Источник</th></tr></thead><tbody>'
    + rows.map(r=>`<tr><td>${esc(cap(r.что))}</td><td><b>${esc(cap(r.знач))}</b></td><td class="faint">${esc(cap(r.как||''))}</td></tr>`).join('') + '</tbody></table>';
}
const envLine=env=>!env? '' : Object.entries(env).map(([k,v])=>`${k}=${/\s/.test(v)||v===''? '"'+v+'"' : v}`).join(' ');
function fillWin(){
  const s=state; if(!s||!winOpen) return;
  const L=winOpen.el, body=L.querySelector('.winbody'), top=body.scrollTop, id=winOpen.id;
  const T=L.querySelector('h2'), N=L.querySelector('.winhead p'), St=L.querySelector('.w-stats'), Pl=L.querySelector('.wplots'), Tb=L.querySelector('.w-table');
  if(id==='log'){
    T.textContent='Журнал'; N.textContent=`«${s.имя}»`;
    St.innerHTML=''; Pl.innerHTML=''; Pl.dataset.win='';
    const m=s.сообщения.slice().reverse();
    Tb.innerHTML=(m.length? '<div class="tscroll"><table class="tbl"><thead><tr><th>Время</th><th>Событие</th><th>Описание</th></tr></thead><tbody>'
        + m.map(x=>`<tr><td class="faint nw" title="${esc(x.время)}">${esc(when(x.время,1))}</td><td class="nw">${esc(cap(x.событие))}</td><td>${esc(cap(x.текст))}</td></tr>`).join('') + '</tbody></table></div>'
        : '<div class="empty">Событий пока нет</div>')
      + `<h3 class="h3" style="margin-top:18px">Вывод программы <button class="btn sm quiet" data-copy="out">${ic('copy')}Копировать</button></h3>`
      + ((s.вывод||[]).length? `<pre class="out" style="max-height:none">${esc(s.вывод.join('\n'))}</pre>` : '<div class="empty">Вывода пока нет</div>');
    const cb=Tb.querySelector('[data-copy]'); if(cb) cb.onclick=()=>copyText((s.вывод||[]).join('\n'));
  } else if(id==='plan'){
    const v=s.выбор||{}, mo=v.модель||{}, pl=s.план||{};
    T.textContent='Заданные параметры'; N.textContent=v.начало? `«${s.имя}» · Запуск ${when(v.начало)} · Режим «${v.режим_имя||v.режим||''}»` : `«${s.имя}»`;
    St.innerHTML=''; Pl.innerHTML=''; Pl.dataset.win='';
    const facts=[{что:'Модель', знач: mo.модель? `${mo.модель} · ${mo.вид||''} · ${mo.основной||''}` : '—', как: mo.параметров? `${int(mo.параметров)} параметров · ${mo.папка||''}` : ''},
                 {что:'Данные', знач: base(v.датасет)||'—', как:(s.сводка? `${int(s.сводка.examples)} примеров: ${int(s.сводка.обучение)} на обучение, ${int(s.сводка.отложено)} на валидацию · ` : '')+(v.датасет||'')},
                 {что:'LR', знач: pl.lr||'—', как: v.env&&v.env.LR? 'Задан заранее' : 'Подобран'}];
    Tb.innerHTML=planTable(v.строки? facts.concat(v.строки.filter(r=>r.к!=='LR'||!pl.lr)) : facts)
      + (v.строки? '' : '<div class="note" style="margin-top:12px">'+ic('info')+'<span>Запуск из терминала: запись о выбранных параметрах отсутствует.</span></div>')
      + (v.env? `<h3 class="h3" style="margin-top:18px">Переменные окружения движка</h3><pre class="out">${esc(envLine(v.env))}</pre>` : '');
  } else {
    const w=WIN[id], tb=w.table(s);
    T.innerHTML=esc(w.title)+hint(typeof w.note==='function'? w.note(s) : w.note); N.textContent=`«${s.имя}»`;
    St.innerHTML='<div class="tiles">'+w.stats(s).map(x=>`<div class="tile"><span class="tl" title="${esc(x[0])}">${x[0]}</span><span class="tv">${x[1]}</span><span class="tn">${x[2]||''}</span></div>`).join('')+'</div>';
    const big=w.big(s);
    if(Pl.children.length!==big.length || Pl.dataset.win!==id){ Pl.dataset.win=id;
      Pl.innerHTML=big.map((c,i)=>`<div class="wplot${i?' sm':''}"><h3>${c.title}${hint(c.note)}</h3>${c.series.length>1?`<div class="keys">${keys(c)}</div>`:''}<div class="pl"></div></div>`).join(''); }
    big.forEach((c,i)=>chart(Pl.children[i].querySelector('.pl'), Object.assign({series:c.series,title:c.title}, c.opt||{})));
    Tb.innerHTML= tb.rows.length? `<div class="tscroll" style="max-height:420px"><table class="tbl"><thead><tr>${tb.cols.map(c=>`<th>${c}</th>`).join('')}</tr></thead><tbody>`
        + tb.rows.map(r=>`<tr>${r.map(v=>`<td>${v}</td>`).join('')}</tr>`).join('') + '</tbody></table></div>' : '<div class="empty">Данные появятся после первых шагов.</div>';
  }
  body.scrollTop=top;
}
// Окно вырастает из своей плитки: видно, что именно раскрылось.
function openWin(id, src){
  if(!state) return;
  if(winOpen) winOpen.L.close();
  const s=document.createElement('div'); s.className='scrim';
  s.innerHTML=`<div class="pop winbox" role="dialog" aria-modal="true" tabindex="-1"><div class="winhead"><h2></h2><p></p><div class="ca">
    <button class="btn icon quiet" data-x aria-label="Закрыть" title="Закрыть (Esc)">${ic('x')}</button></div></div>
    <div class="winbody"><div class="w-stats"></div><div class="wplots"></div><div class="w-table"></div></div></div>`;
  const L=pushLayer(s, ()=>{ winOpen=null; tip.style.opacity=0; });
  winOpen={id, el:s, L};
  s.querySelector('[data-x]').onclick=()=>L.close();
  s.addEventListener('pointerdown',e=>{ if(e.target===s) L.close(); });
  fillWin();
  const box=s.querySelector('.winbox');
  if(src && box.animate && !calm()){ const a=src.getBoundingClientRect(), b=box.getBoundingClientRect();
    box.animate([{transform:`translate(${a.left-b.left}px,${a.top-b.top}px) scale(${a.width/b.width},${a.height/b.height})`, opacity:.4},{transform:'none', opacity:1}],
      {duration:230, easing:'cubic-bezier(.2,.75,.25,1)'}); }
  box.focus();
}
function journal(s){
  const m=s.сообщения.slice(-3).reverse(), el=$('#log'), key=s.имя+':'+m.map(x=>x.время+x.текст).join('|');
  if(el._key===key) return; el._key=key;
  el.innerHTML=`<span class="jt">Журнал</span><div class="jl">`+(m.length? m.map(x=>
      `<div class="jr"><i>${esc((x.время||'').slice(11))}</i><i>${esc(cap(x.событие))}</i><span title="${esc(cap(x.текст))}">${esc(cap(x.текст))}</span></div>`).join('')
      : '<div class="faint">Событий пока нет</div>')+`</div><span class="exp">${ic('expand')}</span>`;
}

// ============================================================================
// Ход обучения: одна фраза о том, что происходит, этапы, одно главное действие.
// ============================================================================
const STAGES=[['Токены','Токены'],['Подмешивание','Подмешивание'],['Защита','Защита общих знаний'],['Развёртка','Подбор LR'],['Раскладка','Раскладка среза'],['Обучение','Обучение'],['Завершено','Завершено']];
const STATE_WORD={'идёт':'Выполняется','завершено':'Завершено','остановлено':'Остановлено','ошибка':'Ошибка','нет':'Не запускалось'};
const STATE_PILL={'идёт':'live','завершено':'ok','остановлено':'warn','ошибка':'bad','нет':''};
let actErr='', stopping=false;
const liveRun=()=>H&&H.идёт? H.идёт : '';
function bestOf(s){ const L=vals(s.оценки,'loss'), b=min(L), e=s.оценки.find(x=>x.loss===b); return b==null? null : {v:b, step:e&&e['шаг']}; }
function stageText(s){
  return {'Токены':'Подготовка токенов','Подмешивание':'Подмешивание: разметка ответов','Защита':'Защита общих знаний','Развёртка':`Подбор LR · ${(s.развёртка||[]).length} из 40`,
          'Раскладка':'Раскладка среза','Обучение':'Обучение','Завершено':'Завершено'}[s.этап]||'';
}
function loadInfo(s){ const ld=(s.загрузка||'').match(/([\d.]+)\/([\d.]+) ГиБ, ([\d.]+) МБ\/с/); return ld? {a:+ld[1], b:+ld[2], v:+ld[3]} : null; }
function nowSlab(s){
  const st=s.состояние, L=s.последний||{}, p=s.план||{}, cs=L['шаг']||0, tot=p.всего||0, live=s.идёт, best=bestOf(s);
  const bd=$('#n-badge'); bd.className='pill '+(STATE_PILL[st]||''); bd.querySelector('span').textContent=STATE_WORD[st]||cap(st);
  const ep= p.на_эпоху&&cs? Math.min(p.эпох||1, Math.floor((cs-1)/p.на_эпоху)+1) : (L['эпоха']||0);
  let t, sub='';
  if(live){
    const ld=loadInfo(s), noSteps=!cs && !(s.этап==='Развёртка' && (s.развёртка||[]).length);
    if(s.этап==='Развёртка') t=`Подбор LR · ${(s.развёртка||[]).length} из 40`;
    else if(s.этап==='Токены') t='Подготовка токенов';
    else if(s.этап==='Подмешивание') t='Подмешивание: разметка ответов';
    else if(s.этап==='Защита') t='Защита общих знаний';
    else if(s.этап==='Раскладка') t='Раскладка среза по видеопамяти';
    else if(cs) t=`Шаг ${int(cs)}${tot?' из '+int(tot):''}${p.эпох? ` · Эпоха ${ep} из ${p.эпох}` : ''}`;
    else if(s.этап==='Обучение') t='Загрузка весов в ОЗУ';
    else t='Запуск';
    const parts=[];
    if(cs){ if(L.прошло) parts.push(`Прошло <b>${esc(L.прошло)}</b>`); if(L.осталось) parts.push(`Осталось <b>${esc(L.осталось)}</b>`);
      if(best) parts.push(`Лучшее значение <b>${fmt(best.v,4)}</b> на шаге ${int(best.step)}`); }
    else if(s.этап==='Развёртка'){ const d=s.развёртка||[], l=d[d.length-1]; if(l) parts.push(`LR <b>${Number(l[1]).toExponential(3)}</b> · Потеря (loss) <b>${fmt(l[2],4)}</b>`); }
    if(ld && noSteps){ const left=ld.v>0? (ld.b-ld.a)*1073.741824/ld.v : 0;
      t= s.этап==='Развёртка'? 'Подбор LR: загрузка весов' : 'Загрузка весов в ОЗУ';
      parts.length=0; parts.push(`<b>${fmt(ld.a,1)}</b> из ${fmt(ld.b,1)} ГиБ`, `${Math.round(ld.v)} МБ/с`, left>0? `Осталось ≈ <b>${hms(left)}</b>` : 'Завершение'); }
    sub=parts.join(' · ');
  } else if(st==='завершено'){
    t= best? `Лучшая потеря (loss) ${fmt(best.v,4)} на шаге ${int(best.step)}` : 'Завершено';
    sub=[/^остановка:/i.test(s.почему||'')? 'Ранняя остановка' : s.почему? esc(cap(s.почему)) : 'Завершено', cs? `<b>${int(cs)}</b>${tot?' из '+int(tot):''} ${steps_(tot||cs)}` : '', L.прошло? `<b>${esc(L.прошло)}</b>` : ''].filter(Boolean).join(' · ');
  } else if(st==='ошибка'){
    t= cs? `Ошибка на шаге ${int(cs)}` : 'Ошибка запуска';
    sub= s.продолжение? `Снимок весов на шаге <b>${int(s.продолжение-1)}</b>` : 'Снимок весов отсутствует';
  } else if(st==='остановлено'){
    t= cs? `Остановлено на шаге ${int(cs)}${tot?' из '+int(tot):''}` : 'Остановлено';
    sub= s.продолжение? `Снимок весов на шаге <b>${int(s.продолжение-1)}</b> · Продолжение с шага ${int(s.продолжение)}` : cs? 'Снимок весов отсутствует' : '';
    if(best) sub+=(sub?' · ':'')+`Лучшее значение <b>${fmt(best.v,4)}</b> на шаге ${int(best.step)}`;
  } else t='Обучение ещё не начиналось';
  $('#n-title').textContent=t; $('#n-sub').innerHTML=sub;
  const er=$('#n-err'), why= st==='ошибка'? s.почему : actErr;
  er.hidden=!why; er.innerHTML= why? ic('bad')+`<span>${esc(cap(why))}</span>` : '';
  const other=liveRun() && liveRun()!==s.имя;
  let a='';
  if(live) a=`<button class="btn danger" id="n-stop">${ic('stop')}<span>${stopping? 'Остановка' : 'Остановить'}</span></button>`;
  else if(st==='завершено') a=`<button class="btn primary" data-n="chat">${ic('chat')}Разговор</button><button class="btn" data-n="bench">${ic('bench')}Проверка</button><button class="btn quiet" data-go="setup">Новое обучение</button>`;
  else if(st==='нет' && !s.шаги.length && !s.этап) a=`<button class="btn primary" data-go="setup">Новое обучение</button>`;
  else a=`<button class="btn primary" data-n="resume"${other?` disabled title="Занято: выполняется «${esc(liveRun())}»"`:''}>${ic('play')}<span>${s.продолжение? 'Продолжить с шага '+int(s.продолжение) : 'Начать заново'}</span></button>`
        +(s.продолжение? `<button class="btn" data-n="chat">${ic('chat')}Разговор</button>` : '')+`<button class="btn quiet" data-go="setup">Новое обучение</button>`;
  const ae=$('#n-act'); if(ae._h!==a){ ae._h=a; ae.innerHTML=a; }
  stagesView(s); sweepView(s); chipsView(s);
}
function stageFrac(s,k){   // доля готовности этапа: 0..1 или null, если мерить нечем
  const p=s.план||{}, L=s.последний||{}, ld=(s.загрузка||'').match(/([\d.]+)\/([\d.]+) ГиБ/);
  if(k==='Развёртка') return Math.min(1,(s.развёртка||[]).length/40) || (ld? 0.02 : null);
  if(k==='Обучение') return L.шаг&&p.всего? Math.min(1,L.шаг/p.всего) : (ld? 0 : null);
  if(k===s.этап && s.ход_этапа){ const f=parseFloat(s.ход_этапа); if(f>=0) return Math.min(1,f); }   // повтор, защита общих знаний: долю пишет сам этап
  return null;
}
const stageCap=s=>cap((s.ход_этапа||'').split('\t')[1]||'');
function mainBar(s){
  const el=$('#n-bar'), ld=(s.загрузка||'').match(/([\d.]+)\/([\d.]+) ГиБ/), cs=(s.последний||{}).шаг||0;
  let f=null, capt='';
  if(s.идёт){
    if(ld && !cs && !(s.этап==='Развёртка' && (s.развёртка||[]).length)){ f=+ld[1]/+ld[2]; capt='Загрузка весов'; }
    else if(s.этап==='Развёртка'){ f=stageFrac(s,'Развёртка'); capt=`Подбор LR · ${(s.развёртка||[]).length} из 40`; }
    else if(cs){ const p=s.план||{}, L=s.последний||{}; f=stageFrac(s,'Обучение'); capt=`Шаг ${int(cs)}${p.всего?' из '+int(p.всего):''}${L.осталось?' · Осталось '+L.осталось:''}`; }
    else { const g=stageFrac(s,s.этап), nm=(STAGES.find(x=>x[0]===s.этап)||['',''])[1]; f= g==null? -1 : g; capt= stageCap(s)? nm+' · '+stageCap(s) : nm; } }
  else if(s.состояние==='завершено'){ f=1; capt='Завершено'; }
  el.hidden= f==null; if(f==null) return;
  const bar=el.querySelector('.bar'); bar.classList.toggle('ind', f<0); bar.classList.toggle('ok', s.состояние==='завершено');
  el.querySelector('.bar i').style.width= f<0? '' : (100*f).toFixed(1)+'%';
  $('#n-cap').textContent=capt; $('#n-pct').textContent= f<0? '' : Math.round(100*f)+' %';
}
function stagesView(s){
  mainBar(s);
  let ci=STAGES.findIndex(x=>x[0]===s.этап);
  if(ci<0) ci= s.шаги.length? STAGES.findIndex(x=>x[0]==='Обучение') : -1;
  const v=s.выбор||{}, fixed=v.env&&v.env.LR, p=s.план||{}, L=s.последний||{}, best=bestOf(s);
  const slice=(s.сообщения||[]).map(m=>(m.текст.match(/(Целые матрицы: [^,]+, слои [\d,]+)|([\d.]+% строк)/)||[])[0]).filter(Boolean).pop();
  const lr=loraOfRun(s);                          // LoRA: ни защиты общих знаний, ни раскладки среза — адаптеры во всех слоях
  const h=STAGES.map(([k,name],i)=>{
    let c= i<ci? 'done' : i===ci? (s.идёт? 'cur' : s.состояние==='ошибка'? 'bad' : (k==='Завершено'||s.состояние==='завершено')? 'done' : 'stop') : 'todo';
    if(k==='Развёртка' && c!=='todo' && c!=='cur' && (fixed || !(s.развёртка||[]).length)) c='skip';
    const rp=s.повтор || (v.env&&v.env.REPLAY) || '';
    if(k==='Подмешивание' && c!=='cur' && (rp==='0' || (!rp && c!=='todo'))) c='skip';
    if(lr && (k==='Защита'||k==='Раскладка') && c!=='cur') c='skip';
    let capt='';
    if(k==='Токены') capt= s.сводка? `${int(s.сводка.examples)} ${plural(s.сводка.examples,'пример','примера','примеров')}` : '';
    if(k==='Развёртка') capt= c==='cur'? `${(s.развёртка||[]).length} из 40` : c==='skip'? (fixed? 'Задан '+fixed : 'Не выполнялся') : c==='todo'? '40 шагов' : (p.lr? 'Выбран '+p.lr : `${(s.развёртка||[]).length} из 40`);
    if(k==='Подмешивание') capt= c==='skip'? 'Отключено' : c==='cur'? (stageCap(s)||'Выполняется') : (rp? '×'+rp+' к данным' : '');
    if(k==='Защита'){ const mt=(s.способ&&s.способ.MATS)||(v.env&&v.env.MATS)||'';
      if(c!=='cur' && c!=='todo' && !mt && s.этап!=='Защита') c='skip';
      capt= c==='skip'? 'Не требуется' : c==='cur'? stageCap(s) : (mt? mt.replace(':',' ') : ''); }
    if(k==='Раскладка') capt= lr? 'Не требуется' : slice? cap(slice) : (c==='todo'? 'Срез строк по видеопамяти' : '');
    if(k==='Обучение') capt= L.шаг? `Шаг ${int(L.шаг)}${p.всего?' из '+int(p.всего):''}` : (p.всего? int(p.всего)+' '+steps_(p.всего) : '');
    if(k==='Завершено') capt= best? 'Лучшее '+fmt(best.v,4) : '';
    const icn={done:'ok', cur:'dot', stop:'pause', bad:'bad', skip:'skip', todo:'todo'}[c];
    const fr= c==='done'||c==='skip'? 1 : c==='cur'? stageFrac(s,k) : c==='todo'? 0 : stageFrac(s,k);
    const bar=`<span class="bar${c==='cur'&&fr==null?' ind':''}${c==='skip'?' off':''}" style="display:block"><i style="width:${fr==null? (c==='cur'?'':'0') : (100*fr).toFixed(1)}%"></i></span>`;
    return `<div class="stg ${c}"${c==='cur'?' aria-current="step"':''}><b>${ic(icn)}${name}</b><small title="${esc(capt)}">${esc(capt)||'&nbsp;'}</small>${bar}</div>`;
  }).join('');
  const el=$('#n-stages'); if(el._h!==h){ el._h=h; el.innerHTML=h; }
}
function sweepView(s){
  const box=$('#n-sweep'), show=s.идёт && s.этап==='Развёртка'; box.hidden=!show; if(!show) return;
  const c=sweepChart(s), d=s.развёртка||[], l=d[d.length-1];
  $('#n-sweep-v').innerHTML= l? `Точка ${d.length} из 40 · LR <b>${Number(l[1]).toExponential(3)}</b> · Потеря (loss) <b>${fmt(l[2],4)}</b>` : '';
  const el=box.querySelector('.pl'); if(c) chart(el, Object.assign({series:c.series, title:c.title}, c.opt)); else el.replaceChildren();
}
function chipsView(s){
  const v=s.выбор||{}, mo=v.модель||{}, p=s.план||{}, env=v.env||{}, c=[];
  if(mo.модель) c.push(['Модель', `${mo.модель} · ${mo.вид||''} · ${mo.основной||''}`]);
  if(v.режим_имя) c.push(['Режим', v.режим_имя]);
  const g=goalLine(s); if(g) c.push(['Способ', g]);
  if(v.датасет) c.push(['Данные', base(v.датасет)]);
  if(p.lr) c.push(['LR', p.lr]);
  if(p.упаковка!=null) c.push(['Упаковка', p.упаковка==='0'? 'По одному примеру' : p.упаковка+' токенов']);
  if(p.эпох) c.push(['Эпох', p.эпох]);
  if(p.оценка) c.push(['Валидация каждые', p.оценка]);
  if(p.сброс) c.push(['Снимок весов каждые', p.сброс]);
  if(env.STOP_AFTER!=null&&env.STOP_AFTER!=='') c.push(['Ранняя остановка', env.STOP_AFTER==='0'? 'Отключена' : `${env.STOP_AFTER} ${plural(+env.STOP_AFTER,'валидация','валидации','валидаций')} без улучшения`]);
  if(env.X) c.push(['Ключи движка', env.X]);
  const h=c.map(([k,x])=>`<span class="chip">${esc(k)} <b>${esc(x)}</b></span>`).join('') + `<button class="btn sm tint" data-n="plan">Все заданные параметры</button>`;
  const el=$('#n-chips'); if(el._h!==h){ el._h=h; el.innerHTML=h; }
}
const GOAL_NAME={'срез':'Срез строк по всей сети','точно':'Точность: факты','характер':'Характер: голос и манера','lora':'LoRA'};
const GOAL_SHORT={'срез':'Срез строк','точно':'Точность','характер':'Характер','lora':'LoRA'};
const LORA_MATS=[['q','q_proj'],['k','k_proj'],['v','v_proj'],['o','o_proj'],['g','gate_proj'],['u','up_proj'],['d','down_proj']];
// параметры LoRA запуска — из mode.txt (его пишет автопилот), иначе из записи о выборе
const loraOfRun=s=>(s.способ&&s.способ.LORA)||((s.выбор||{}).env||{}).LORA||'';
function loraDesc(a){ const [r,al,tg]=String(a).split(','); return `LoRA r ${r}, α ${al} · `+LORA_MATS.filter(([c])=>(tg||'').includes(c)).map(([,n])=>n.replace('_proj','')).join(' '); }
function goalOfRun(s){ if(loraOfRun(s)) return 'lora'; const mt=(s.способ&&s.способ.MATS)||((s.выбор||{}).env||{}).MATS||''; return mt.startsWith('down_proj')? 'точно' : mt.startsWith('attn')? 'характер' : 'срез'; }
const goalLine=s=>loraOfRun(s)? loraDesc(loraOfRun(s)) : GOAL_NAME[goalOfRun(s)];
function runSelect(s){
  const sel=$('#run'), names=[...new Set([...(H&&H.запуски||[]).map(r=>r.имя), ...(s.запуски||[]), cur].filter(Boolean))];
  if(sel.options.length!==names.length || names.some((n,i)=>sel.options[i].value!==n)) sel.replaceChildren(...names.map(n=>new Option(n,n)));
  if(sel.value!==cur) sel.value=cur;
}
async function resume(name){
  actErr='';
  const r=await post('/api/start',{имя:name||state.имя, продолжить:true}).catch(()=>({ок:false,почему:'Нет связи с программой.'}));
  if(!r.ок){ actErr=r.почему||'Не удалось запустить.'; toast('Не удалось запустить', cap(actErr), 'bad'); } else toast('Обучение продолжено', `«${name||state.имя}»`, 'ok');
  await refreshH(); if(state) nowSlab(state);
}
async function stopRun(){
  if(stopping||!state) return;
  if(+UI.confirmStop){
    const sv=state.продолжение;
    const yes=await dialog({title:`Остановить «${state.имя}»?`, text: sv? `Снимок весов на шаге ${int(sv-1)} · Продолжение с шага ${int(sv)}` : 'Снимок весов отсутствует, продолжение невозможно.',
      ok:'Остановить', danger:true});
    if(!yes) return;
  }
  stopping=true; nowSlab(state);
  await post('/api/stop').catch(()=>{});
  stopping=false; await refreshH(); toast('Обучение остановлено', `«${state&&state.имя||''}»`, 'warn');
}

// ============================================================================
// Запуски: вся библиотека, сравнение, удаление
// ============================================================================
const RS={q:'', f:'all', sort:'когда', dir:-1, sel:new Set(), cmp:null};
const R_FILTERS=[['all','Все'],['идёт','Выполняются'],['завершено','Завершены'],['остановлено','Остановлены'],['ошибка','Ошибки']];
function runsList(){
  const q=RS.q.trim().toLowerCase();
  let L=(H&&H.запуски||[]).filter(r=>(RS.f==='all'||r.состояние===RS.f) && (!q || r.имя.toLowerCase().includes(q) || (r.датасет||'').toLowerCase().includes(q)));
  const k=RS.sort, d=RS.dir, val=r=>k==='лучшая'? (r.лучшая==null? Infinity*-d : r.лучшая) : k==='размер'? (r.размер||0) : k==='шаг'? (r.шаг||0) : String(r[k]||'');
  return L.sort((a,b)=>{ const x=val(a), y=val(b); return (x<y?-1:x>y?1:0)*d; });
}
function renderRuns(){
  if(!H) return;
  const f=$('#r-f'); if(!f.children.length){ f.innerHTML=R_FILTERS.map(([k,t])=>`<button role="radio" data-f="${k}">${t}</button>`).join('');
    f.onclick=e=>{ const b=e.target.closest('[data-f]'); if(b){ RS.f=b.dataset.f; renderRuns(); } }; }
  f.querySelectorAll('[data-f]').forEach(b=>b.setAttribute('aria-checked', b.dataset.f===RS.f));
  const all=H.запуски||[], list=runsList(), tot=all.reduce((s,r)=>s+(r.размер||0),0), dk=H.диск||{};
  $('#r-sum').innerHTML=`<span>Запусков <b>${all.length}</b></span><span>Занято <b>${bytes(tot)}</b></span>`+(dk.свободно? `<span>Свободно <b>${bytes(dk.свободно)}</b></span>` : '');
  for(const n of [...RS.sel]) if(!all.some(r=>r.имя===n)) RS.sel.delete(n);
  const th=(k,t,cls='')=>`<th class="sortable ${cls}" data-sort="${k}"${RS.sort===k?` aria-sort="${RS.dir>0?'ascending':'descending'}"`:''}>${t}</th>`;
  const h=`<thead><tr><th class="cb"><input type="checkbox" id="r-all" aria-label="Выбрать все"${list.length&&list.every(r=>RS.sel.has(r.имя))?' checked':''}></th>${th('имя','Имя')}${th('цель','Способ')}${th('режим','Режим','c-mode')}${th('состояние','Состояние')}${th('шаг','Шаг','r')}${th('лучшая','Лучшая потеря (loss)','r')}${th('прошло','Время','r c-time')}${th('размер','Размер','r c-size')}${th('когда','Дата')}<th></th></tr></thead><tbody>`
    + (list.length? list.map(r=>`<tr data-run="${esc(r.имя)}"${RS.sel.has(r.имя)?' class="sel"':''}>
      <td class="cb"><input type="checkbox" data-cb="${esc(r.имя)}"${RS.sel.has(r.имя)?' checked':''} aria-label="Выбрать"></td>
      <td class="nm" title="${esc(r.датасет||'')}">${esc(r.имя)}</td><td>${esc(GOAL_SHORT[r.цель]||'—')}</td><td class="c-mode">${esc(modeName(r.режим)||'—')}</td>
      <td><span class="pill ${STATE_PILL[r.состояние]||''}"><i></i>${esc(STATE_WORD[r.состояние]||r.состояние)}</span></td>
      <td class="r">${r.шаг? int(r.шаг)+(r.всего?' / '+int(r.всего):'') : '—'}</td><td class="r"><b>${r.лучшая!=null? fmt(r.лучшая,4) : '—'}</b></td>
      <td class="r faint c-time">${esc(r.прошло||'—')}</td><td class="r faint c-size">${r.размер!=null? bytes(r.размер) : '—'}</td><td class="faint nw" title="${esc((r.когда||'').slice(0,19))}">${esc(when(r.когда))}</td>
      <td><div class="rowact">
        <button class="btn icon quiet sm" data-ra="open" title="Открыть">${ic('open')}</button>
        <button class="btn icon quiet sm" data-ra="chat" title="Разговор"${r.веса?'':' disabled'}>${ic('chat')}</button>
        <button class="btn icon quiet sm danger" data-ra="del" title="Удалить"${r.состояние==='идёт'?' disabled':''}>${ic('trash')}</button></div></td></tr>`).join('')
      : `<tr><td colspan="11"><div class="empty">${all.length? 'Ничего не найдено' : 'Запусков пока нет'}</div></td></tr>`) + '</tbody>';
  const t=$('#r-t'); if(t._h!==h){ t._h=h; t.innerHTML=h; }
  topActions();
}
function runsBind(){
  const t=$('#r-t');
  t.addEventListener('click', e=>{
    const s=e.target.closest('[data-sort]'); if(s){ const k=s.dataset.sort; if(RS.sort===k) RS.dir=-RS.dir; else { RS.sort=k; RS.dir= k==='имя'||k==='цель'||k==='режим'? 1 : -1; } renderRuns(); return; }
    if(e.target.id==='r-all'){ const L=runsList(); if(e.target.checked) L.forEach(r=>RS.sel.add(r.имя)); else L.forEach(r=>RS.sel.delete(r.имя)); renderRuns(); return; }
    const cb=e.target.closest('[data-cb]'); if(cb){ cb.checked? RS.sel.add(cb.dataset.cb) : RS.sel.delete(cb.dataset.cb); renderRuns(); return; }
    const tr=e.target.closest('tr[data-run]'); if(!tr) return; const n=tr.dataset.run, a=e.target.closest('[data-ra]');
    if(a){ if(a.dataset.ra==='del') return delRuns([n]); if(a.dataset.ra==='chat') return chatWith(n); }
    openRun(n);
  });
  $('#r-q').addEventListener('input', e=>{ RS.q=e.target.value; renderRuns(); });
  $('#r-cmp-x').onclick=()=>{ RS.cmp=null; $('#r-cmp').hidden=true; };
}
function openRun(n){ cur=n; picked=true; actErr=''; state=null; listen(); go('train'); }
async function delRuns(names){
  names=names.filter(n=>n!==liveRun()); if(!names.length) return;
  const rows=(H.запуски||[]).filter(r=>names.includes(r.имя)), sz=rows.reduce((s,r)=>s+(r.размер||0),0);
  const yes=await dialog({title: names.length===1? `Удалить «${names[0]}»?` : `Удалить запуски: ${names.length}?`,
    text:`Журнал, токены и обученные веса · ${bytes(sz)}. Восстановление невозможно. Исходная модель не затрагивается.`, ok:'Удалить', danger:true});
  if(!yes) return;
  let freed=0, bad=[];
  for(const n of names){ const r=await post('/api/runs/delete',{имя:n}).catch(()=>({ок:false})); if(r.ок) freed+=r.освобождено||0; else bad.push(n); RS.sel.delete(n);
    if(cur===n){ cur=''; state=null; if(es){ es.close(); es=null; } } }
  toast(bad.length? 'Удалено частично' : 'Удалено', `Освобождено ${bytes(freed)}`+(bad.length? ` · Не удалось удалить: ${bad.join(', ')}` : ''), bad.length? 'warn':'ok');
  await refreshH(); if(!cur && H.запуски.length){ cur=H.запуски[0].имя; listen(); }
}
async function compareRuns(names){
  names=names.slice(0,6); if(names.length<2) return;
  $('#r-cmp').hidden=false; $('#r-cmp-s').textContent='Загрузка';
  const S=await Promise.all(names.map(n=>getj('/api/state?run='+encodeURIComponent(n)).catch(()=>null)));
  RS.cmp=S.filter(Boolean); drawCompare();
  $('#r-cmp').scrollIntoView({block:'start', behavior: calm()? 'auto' : 'smooth'});
}
function drawCompare(){
  const S=RS.cmp; if(!S) return;
  $('#r-cmp-s').textContent=S.map(s=>s.имя).join(' · ');
  $('#r-leg').innerHTML=S.map((s,i)=>`<span><i style="background:${PALETTE[i]}"></i>${esc(s.имя)}</span>`).join('');
  const C=[['Потеря (loss) на валидации', s=>ser(s.оценки,'loss',s.имя,null,true), v=>fmt(v,4)], ['Потеря (loss) на обучении', s=>ser(s.шаги,'ср',s.имя), v=>fmt(v,4)],
           ['Точность на валидации', s=>ser(s.оценки,'acc',s.имя,null,true), v=>pct(v,2)], ['Секунд на шаг', s=>ser(s.шаги,'сек',s.имя), v=>fmt(v,3)+' с']];
  const g=$('#r-plots'); g.style.setProperty('--chh','200px');
  if(g.children.length!==C.length) g.innerHTML=C.map(c=>`<div class="plot glass" style="cursor:default"><div class="pt"><span>${c[0]}</span></div><div class="pl"></div></div>`).join('');
  C.forEach((c,i)=>chart(g.children[i].querySelector('.pl'), {series:S.map((s,j)=>Object.assign(c[1](s),{color:PALETTE[j]})), title:c[0], ft:c[2], compact:false, area:false}));
  const row=(t,f)=>`<tr><td class="faint">${t}</td>${S.map(s=>`<td>${f(s)}</td>`).join('')}</tr>`;
  const bestS=s=>{ const b=bestOf(s); return b? `<b>${fmt(b.v,4)}</b> <span class="faint">Шаг ${int(b.step)}</span>` : '—'; };
  const rs=n=>(H.запуски||[]).find(r=>r.имя===n)||{};
  $('#r-ct').innerHTML=`<thead><tr><th></th>${S.map((s,i)=>`<th><span style="display:inline-flex;align-items:center;gap:6px"><i style="width:10px;height:3px;border-radius:2px;background:${PALETTE[i]}"></i>${esc(s.имя)}</span></th>`).join('')}</tr></thead><tbody>`
    + row('Способ', s=>esc(goalLine(s)))+row('Режим', s=>esc((s.выбор||{}).режим_имя||'—'))+row('Данные', s=>esc(base((s.выбор||{}).датасет)||'—'))
    + row('Лучшая потеря (loss) на валидации', bestS)+row('Последняя потеря (loss) на валидации', s=>fmt(lastOf(s.оценки,'loss'),4))+row('Точность на валидации', s=>pct(lastOf(s.оценки,'acc'),2))
    + row('Шагов', s=>`${int((s.последний||{}).шаг)} / ${int((s.план||{}).всего)}`)+row('LR', s=>esc((s.план||{}).lr||'—'))
    + row('Секунд на шаг', s=>fmt(med(vals(s.шаги,'сек')),3))+row('Токенов в секунду', s=>int(med(vals(s.шаги,'ток'))))
    + row('Пик видеопамяти', s=>{ const v=max(vals(s.шаги,'vram')); return v==null?'—':int(v)+' МиБ'; })+row('Время', s=>esc((s.последний||{}).прошло||'—'))
    + row('На диске', s=>bytes(rs(s.имя).размер))+'</tbody>';
}
// ============================================================================
// Новое обучение: модель, режим, способ, данные — и справа ровно то, что пойдёт.
// ============================================================================
const PCIE={1:.25, 2:.5, 3:.985, 4:1.969, 5:3.938, 6:7.563};      // ГБ/с на линию после кодирования
const OLDM={'быстрый':'Быстрый старт','баланс':'Сбалансированный','качество':'Максимум качества'};
const modeName=k=>((H&&H.режимы||[]).find(r=>r.к===k)||{}).имя||OLDM[k]||k;
let dataErr='', launchErr='';
// Сколько займёт замороженная часть в ОЗУ (BF16, как в файле) и сколько её поедет по шине за шаг.
function frozen(){
  const m=H.модель||{}, n=m.параметров||0, emb=m.эмбеддинги||0, lay=n-emb, q=qsel(), qn= q&&q.годна? q.имя.slice(0,-3) : '';
  const cache= qn? Math.max(0, lay*2-(H.база_bf16||0)) + q.байт : lay*2;   // с квантом матрицы слоёв — из .tq, прочее — BF16
  return {cache, bus:2*cache, need:cache + emb*2 + 6*2**30, lay, qn};
}
// Источник базы — BF16 из файла; внутри замороженные матрицы и связанные frozen-тензоры хранятся в FP16 или выбранном кванте
const qsel=()=>{ const nm=H.база||''; return nm? (H.базы||[]).find(b=>b.имя===nm)||null : null; };
const quantBusy=()=>{ const w=H.настройка? H.настройка_что||'' : ''; return /^(квант|импорт):/.test(w)? w : ''; };
let baseRec=false;
function renderBase(){
  const el=$('#m-base'), m=H.модель||{};
  el.hidden=!m.есть; if(!m.есть) return;
  const cur=H.база||'', busy=quantBusy(), can=!!(H.движок && m.движок) && !H.настройка;
  const why= can? '' : H.настройка? 'Выполняется другое задание.' : !H.движок? 'Требуется собранный движок.' : 'Требуется поддерживаемая модель.';
  const rows=[{k:'', name:'BF16', sub:`Как в файле · ${gib(H.база_bf16,1)}`, tip:'Замороженные матрицы слоёв берутся из исходного файла без потерь.', bad:'', del:false}]
    .concat((H.базы||[]).map(b=>({k:b.имя, name:b.имя.slice(0,-3), sub:[b.откуда, gib(b.байт,1), b.ошибка!=null? 'Ошибка '+pct(100*b.ошибка) : ''].filter(Boolean).join(' · '),
      tip:(b.типы||[]).map(([t,n])=>`${t} ×${n}`).join(', ') + (((H.рецепты||[]).find(r=>r.к===b.рецепт)||{}).суть? '. '+(H.рецепты.find(r=>r.к===b.рецепт).суть) : '')
        + (b.ошибка!=null? '. Ошибка весов указана как среднеквадратичная относительно BF16.' : ''),
      bad: b.годна? '' : 'Непригодна', badTip: cap(b.почему||''), del:true})));
  const h=`<div class="bh"><span title="Замороженные матрицы хранятся в FP16 или в выбранном формате квантования. Обучаемые веса читаются из исходного BF16 и хранятся в FP32; замороженные нормировки и эмбеддинги преобразуются в FP16.">База</span>
      <div class="ca"><button class="btn sm quiet" id="b-q" ${can?'':'disabled'} title="${esc(can? 'Собственный квант из BF16, создаваемый на процессоре по правилам llama.cpp. Исходный файл используется только для чтения.' : why)}" aria-expanded="${baseRec&&can}">${ic('build')}Квантовать</button>
      <button class="btn sm quiet" id="b-f" ${can?'':'disabled'} title="${esc(can? 'Готовый квант этой же модели: GGUF любого типа, в том числе динамический Unsloth, или bitsandbytes 4 бита. Блоки переносятся без пересчёта и сверяются с исходной моделью.' : why)}">${ic('file')}Из файла</button></div></div>
    <div class="blist" role="radiogroup" aria-label="База">${rows.map(r=>`<div class="brow" role="radio" aria-checked="${r.k===cur}" data-base="${esc(r.k)}" title="${esc(r.tip)}"${r.bad?' aria-disabled="true"':''} tabindex="${r.k===cur?0:-1}"><span class="tick"></span>
      <span class="bn"><b>${esc(r.name)}</b><span>${esc(r.sub)}</span></span>${r.bad?`<span class="pill bad" title="${esc(r.badTip)}"><i></i>${esc(r.bad)}</span>`:''}${r.del?`<button class="btn sm quiet icon" data-bdel="${esc(r.k)}" title="Удалить файл базы" aria-label="Удалить файл базы">${ic('trash')}</button>`:''}</div>`).join('')}</div>
    ${baseRec&&can? `<div class="togs" id="b-recs" role="group" aria-label="Рецепт">${(H.рецепты||[]).filter(r=>r.байт).map(r=>`<button class="tog" data-rec="${esc(r.к)}" title="${esc(r.суть)}">${esc(r.к)} · ${gib(r.байт,1)}</button>`).join('')}</div>` : ''}
    ${busy? `<div class="bjob"><div class="bar ind" id="b-bar"><i></i></div><p class="hint" id="b-line">${esc((busy.startsWith('квант')? 'Квантование ' : 'Перенос ')+base(busy.slice(busy.indexOf(':')+1)))}</p></div>` : ''}`;
  if(el._h!==h){ el._h=h; el.innerHTML=h;
    el.querySelector('#b-q').onclick=()=>{ baseRec=!baseRec; el._h=''; renderBase(); };
    el.querySelector('#b-f').onclick=importBase;
    el.querySelectorAll('[data-rec]').forEach(b=>b.onclick=()=>{ baseRec=false; setupRun('квант:'+b.dataset.rec); });
    el.querySelectorAll('.brow').forEach(r=>r.onclick=e=>{ if(e.target.closest('[data-bdel]') || r.getAttribute('aria-disabled')==='true' || r.dataset.base===cur) return; setBase(r.dataset.base); });
    el.querySelectorAll('[data-bdel]').forEach(b=>b.onclick=async()=>{ const k=b.dataset.bdel;
      if(!await dialog({title:`Удалить «${k.slice(0,-3)}»?`, text:'Файл базы удаляется с диска. Исходная модель не затрагивается.', ok:'Удалить', danger:true})) return;
      const r=await post('/api/base',{имя:k, удалить:true}).catch(()=>({ок:false,почему:'Нет связи.'}));
      toast(r.ок? 'Удалено' : 'Не удалось удалить', r.ок? k.slice(0,-3) : cap(r.почему), r.ок?'ok':'bad'); if(r.готовность){ H=r.готовность; renderChoose(); } replan(); }); }
  baseJob();
}
// ход задания: квантование — по матрицам; перенос — сначала копия (до 70 %), потом сверка
function baseJob(){
  const bar=$('#b-bar'), ln=$('#b-line'); if(!bar||!ln) return;
  const t=($('#setup-log')||{}).textContent||'', all=[...t.matchAll(/^(Квантование|Перенос|Сверка) (\d+)\/(\d+) (\S+)/gmi)], m=all[all.length-1];
  if(!m) return;
  const kind=m[1].toLowerCase(), i=+m[2], n=Math.max(1,+m[3]), f= kind==='квантование'? i/n : kind==='перенос'? .7*i/n : .7+.3*i/n;
  bar.classList.remove('ind'); bar.firstChild.style.width=(100*f).toFixed(1)+'%'; ln.textContent=`${cap(kind)} ${i}/${n} · ${m[4].replace(/\.weight$/,'')}`;
}
async function setBase(k){
  const r=await post('/api/base',{имя:k}).catch(()=>({ок:false,почему:'Нет связи.'}));
  if(!r.ок) toast('Не удалось выбрать базу', cap(r.почему), 'bad');
  if(r.готовность){ H=r.готовность; renderChoose(); } replan();
}
async function importBase(){
  const p=await dialog({title:'Квантованная база из файла', text:'GGUF (первый файл, если их несколько), model.safetensors в формате bitsandbytes 4 бита либо папка с такой моделью.', input:{ph:'/путь/к/model.gguf'}, ok:'Перенести'});
  if(p) setupRun('импорт:'+p.trim());
}
function busTime(b){ const g=(H.гпу||{}).шина||{}, r=PCIE[g.поколение], w=g.линий; return r&&w? b/(r*w*1e9) : null; }
const busName=()=>{ const b=(H.гпу||{}).шина||{}; return b.поколение&&b.линий? `PCIe ${b.поколение}.0 ×${b.линий}` : 'Шина не определена'; };
const ST_PILL={ok:['ok','Готово'], need:['warn','Требуется действие'], bad:['bad','Ошибка'], wait:['live','Выполняется']};
function setPill(el, st, text){ const p=ST_PILL[st]||['','']; el.className='pill '+p[0]; el.innerHTML=`<i></i>${esc(text||p[1])}`; el.hidden=!st; }

function renderModel(){
  const m=H.модель||{}, el=$('#m-card'), sig=JSON.stringify([m.есть,m.папка,m.байт,m.почему,H.настройка_что]);
  if(el._sig!==sig){ el._sig=sig;
    if(!m.есть) el.innerHTML=`<div class="empty" style="padding:18px 0">${ic('folder')}<h3>Модель не найдена</h3><span>В папке <bdi>${esc(m.папка||'')}</bdi> отсутствует model.safetensors.</span>
        <div class="frow" style="justify-content:center"><button class="btn primary" data-act="папка">${ic('folder')}Указать папку</button><button class="btn" data-act="веса">${ic('save')}Скачать ≈ 22 ГБ</button></div></div>`;
    else el.innerHTML=`<div class="mname">${esc(m.модель||'?')}</div><p class="mline">${esc(m.вид)} · ${esc(m.основной)} · ${gib(m.байт)}</p>
        <div class="facts"><span class="chip"><b>${int(m.параметров)}</b> параметров</span><span class="chip"><b>${int(m.слоёв)}</b> слоёв</span>
          <span class="chip">Скрытая размерность <b>${int(m.скрытый)}</b></span><span class="chip">Словарь <b>${int(m.словарь)}</b></span>${m.архитектура?`<span class="chip">${esc(m.архитектура)}</span>`:''}</div>
        <div class="mpath" title="${esc(m.папка)}">${ic('lock')}<bdi>${esc(m.папка)}</bdi><span class="ro" title="Файл используется только для чтения.">Только чтение</span></div>
        ${m.движок? '' : `<div class="note bad" style="margin-top:12px">${ic('bad')}<span>${esc(cap(m.почему))}</span></div>`}`; }
  const ms=!m.есть? 'need' : !m.движок? 'bad' : 'ok';
  setPill($('#m-st'), ms, ms==='ok'? 'Готова' : ms==='need'? 'Веса отсутствуют' : 'Не поддерживается');
}
// где что лежит при обучении: срез учится в видеопамяти, остальное ждёт в ОЗУ и едет по шине
function renderMem(){
  const m=H.модель||{}, el=$('#m-kdet');
  $('#m-kmore').hidden=!m.параметров;
  if(!m.параметров) return;
  const t1=+((H.железо||{}).tier1_params)||0, bf=frozen(), tb=busTime(bf.bus), g=goalMats();
  const tile=(l,v,n)=>`<div class="tile"><span class="tl">${l}</span><span class="tv">${v}</span><span class="tn">${n}</span></div>`;
  const learn= ch.tab==='lora'? (n=>tile('Обучается', int(n), pct(100*n/m.параметров,2)+' · Адаптеры FP32')+tile('С AdamW в видеопамяти', gib(n*12), 'A, B и два момента'))(loraSize(ch.lora)[1])
    : g? tile('Обучается', int(g), pct(100*g/m.параметров,2)+' · FP32')+tile('С NSGD в видеопамяти', gib(g*6), 'Вес и инерция')
    : tile('Обучается', t1? int(t1) : '—', t1? pct(100*t1/m.параметров,2)+' · FP32' : 'Срез строк FP32')+tile('С Adam в видеопамяти', t1? gib(t1*12) : '—', 'Вес и два момента');
  const h=`<div class="tiles">${learn}
    <div class="tile"><span class="tl">Заморожено в ОЗУ</span><span class="tv">${gib(bf.cache)}</span><span class="tn">${bf.qn? esc(bf.qn) : 'BF16, как в файле'}</span></div>
    <div class="tile"><span class="tl">По шине за шаг</span><span class="tv">${gib(bf.bus)}</span><span class="tn">${tb? `≥ ${fmt(tb,2)} с на ${busName()}` : busName()}</span></div></div>`;
  if(el._h!==h){ el._h=h; el.innerHTML=h; }
}
function modeFacts(v){
  if(!v) return ['Все параметры плана', 'Ключи движка'];
  const f=[v.lr? `LR ${ch.tab==='lora'? LORA_LR : v.lr}` : 'LR подбирается автоматически', `Валидаций ≥ ${v.оценок}`];
  if(v.эпохи!==1) f.push(`Эпох × ${dec(v.эпохи)}`);
  f.push({bf3:'×3 (FP16/BF16)', fp32:'FP32', железо:'По возможностям видеокарты'}[v.умножение]||'');
  f.push(`Ранняя остановка: ${v.терпение}`);
  return f.filter(Boolean);
}
function renderModes(){
  const el=$('#m-modes'), list=H.режимы||[];
  if(el._n!==list.length+ch.tab){ el._n=list.length+ch.tab;           // у вкладки LoRA свои факты режима
    el.innerHTML=list.map(r=>`<button class="opt" role="radio" data-mode="${esc(r.к)}" title="${esc(cap(r.суть))}"><span class="tick"></span><b>${esc(r.имя)}</b>
      <span class="chips">${modeFacts(r.знач).map(x=>`<span class="chip">${esc(x)}</span>`).join('')}</span></button>`).join(''); }
  el.querySelectorAll('.opt').forEach(b=>{ const on=b.dataset.mode===ch.mode; b.setAttribute('aria-checked',on); b.tabIndex=on?0:-1; });
}
const GOAL_FACTS={'срез':['4,4 % строк всех матриц','Нормировки','Adam'], 'точно':['Три целые матрицы down_proj','NSGD + защита общих знаний','Градиент по ответу'],
  'характер':['Внимание ранних слоёв','NSGD + защита общих знаний','Подмешивание сохраняет знания']};
function renderGoals(){
  const el=$('#m-goals'), list=H.цели||[];
  if(el._n!==list.length){ el._n=list.length;
    el.innerHTML=list.map(r=>`<button class="opt" role="radio" data-goal="${esc(r.к)}" title="${esc(cap(r.суть))}"><span class="tick"></span><b>${esc(cap(r.имя))}</b>
      <span class="chips">${(GOAL_FACTS[r.к]||[]).map(x=>`<span class="chip">${esc(x)}</span>`).join('')}</span></button>`).join(''); }
  el.querySelectorAll('.opt').forEach(b=>{ const on=b.dataset.goal===ch.goal; b.setAttribute('aria-checked',on); b.tabIndex=on?0:-1; });
}
// --- LoRA: та же раскладка, что в движке (по описанию выбранной модели): каждый шестой слой — полное внимание, своей v_proj там нет ---
const LORA_LR='2e-4', LORA_SETS=[['qv','q v'],['qkvo','Внимание'],['qkvogud','Все']];
const loraArg=lp=>`${lp.r},${+lp.alpha},${lp.tg},${+lp.drop}`;
function loraParse(a){ const p=String(a||'').split(','), r=parseInt(p[0],10), al=parseFloat(p[1]), d=parseFloat(p[3]);
  const tg=LORA_MATS.map(([c])=>c).filter(c=>(p[2]||'').includes(c)).join('');
  return {r: r>=1&&r<=1024? r : LORA0.r, alpha: al>0? al : LORA0.alpha, tg: tg||LORA0.tg, drop: isFinite(d)? Math.max(0,Math.min(.9,d)) : LORA0.drop}; }
function geom(){ const g=H.геометрия; return g&&g.length===4? g : [0,0,0,[]]; }
function loraSize(lp){ const [H_, I, NH, LL]=geom(); let n=0, p=0;
  for(const [hd, kv, hv] of LL){ const io={q:H_+NH*hd, k:H_+kv*hd, v:H_+kv*hd, o:NH*hd+H_, g:H_+I, u:H_+I, d:I+H_};
    for(const [c] of LORA_MATS) if(lp.tg.includes(c) && !(c==='v'&&!hv)){ n++; p+=lp.r*io[c]; } }
  return [n, p]; }
// целые матрицы «Точно» и «Характер» — сколько чисел учится: как mats_params на сервере, формы слоёв — из описания модели
function goalMats(){ if(ch.goal!=='точно' && ch.goal!=='характер') return 0; const [H_, I, NH, LL]=geom();
  const o=ch.mode==='эксперт'? ch.over['слои правки']||'' : '', d=((H.цели||[]).find(r=>r.к===ch.goal)||{}).слои||'';
  return (/^\d+(,\d+)*$/.test(o)? o : d).split(',').filter(Boolean).map(Number).filter(l=>l<LL.length)
    .reduce((a,l)=>{ const [hd,kv,hv]=LL[l]; return a+(ch.goal==='точно'? H_*I : 2*H_*NH*hd+(hv?2:1)*H_*kv*hd); },0); }
function renderTab(){
  const t=$('#tab'); t.hidden= view!=='setup';
  t.querySelectorAll('[data-tab]').forEach(b=>b.setAttribute('aria-checked', b.dataset.tab===ch.tab));
  $('#c-goal').hidden= ch.tab==='lora'; $('#c-lora').hidden= ch.tab!=='lora';
}
function renderLora(){
  if(ch.tab!=='lora') return;
  const lp=ch.lora, m=H.модель||{}, [n, p]=loraSize(lp), t1=+((H.железо||{}).tier1_params)||0, bud=t1*12, over=!!(bud && p*12>bud);
  const seg=(id,list,v)=>{ const el=$(id), h=list.map(([k,t])=>`<button role="radio" data-v="${esc(k)}" aria-checked="${String(k)===String(v)}">${esc(t)}</button>`).join('');
    if(el._h!==h){ el._h=h; el.innerHTML=h; } };
  seg('#lo-r', [...new Set([4,8,16,32,64,128,lp.r])].sort((a,b)=>a-b).map(x=>[x,x]), lp.r);
  seg('#lo-d', [...new Set([0,.05,.1,+lp.drop])].sort((a,b)=>a-b).map(x=>[x,dec(x)]), +lp.drop);
  seg('#lo-set', LORA_SETS, lp.tg);
  if(document.activeElement!==$('#lo-a')) $('#lo-a').value=String(+lp.alpha);
  const mh=LORA_MATS.map(([c,nm])=>`<button class="tog" data-m="${c}" aria-pressed="${lp.tg.includes(c)}"${lp.tg===c?' title="Должна остаться хотя бы одна матрица."':''}>${nm}</button>`).join('');
  const me=$('#lo-m'); if(me._h!==mh){ me._h=mh; me.innerHTML=mh; }
  const th=[['Адаптеров', int(n), 'Пар A и B'], ['Обучаемых чисел', int(p), m.параметров? pct(100*p/m.параметров,2)+' модели' : ''],
            ['Видеопамять', gib(p*12), 'A, B и два момента'], ['Множитель s', dec(+(lp.alpha/lp.r).toPrecision(4)), 'α / r']]
    .map(([l,v,t])=>`<div class="tile"><span class="tl">${l}</span><span class="tv">${v}</span><span class="tn">${t}</span></div>`).join('');
  const te=$('#lo-tiles'); if(te._h!==th){ te._h=th; te.innerHTML=th; }
  const ov=$('#lo-over'); ov.hidden=!over; if(over){ ov.innerHTML='<i></i>Превышен бюджет'; ov.title=`Превышен бюджет ${gib(bud)}: слишком большой ранг.`; }
  $('#lo-def').hidden= loraArg(lp)===loraArg(LORA0);
}
function renderData(){
  const pr=P&&P.подготовка, nm=P&&P.имя, el=$('#d-state');
  if(document.activeElement!==$('#d-file') && $('#d-file').value!==ch.file) $('#d-file').value=ch.file;
  if(document.activeElement!==$('#d-name') && $('#d-name').value!==ch.name) $('#d-name').value=ch.name;
  const taken=nm&&!nm.ок&&ch.name; $('#d-nst').hidden=!taken; $('#d-nst').title= taken? cap(nm.почему) : '';
  $('#d-nst').innerHTML= taken? `<i></i>${/уже есть/i.test(nm.почему)? 'Имя занято' : 'Недопустимое имя'}` : '';
  let h='', st='need';
  if(dataErr){ h=`<div class="note bad">${ic('bad')}<span>${esc(cap(dataErr))}</span></div>`; st='bad'; }
  else if(pr && pr.идёт){ h=`<div class="loadline"><span>Токенизация «${esc(base(pr.источник&&pr.источник.путь))}»</span><span class="bar ind"><i></i></span></div>`
      + (pr.вывод.length? `<pre class="out" style="margin-top:10px">${esc(pr.вывод.slice(-6).join('\n'))}</pre>` : ''); st='wait'; }
  else if(pr && pr.код){ h=`<div class="note bad">${ic('bad')}<span>Файл не токенизирован, причина указана ниже.</span></div><pre class="out" style="margin-top:10px">${esc(pr.вывод.join('\n'))}</pre>`; st='bad'; }
  else if(pr && pr.сводка){ const s=pr.сводка, src=pr.источник||{};
    h=(pr.свежий? '' : `<div class="note warn" style="margin-bottom:10px">${ic('warn')}<span>Файл изменён после токенизации.</span><button class="btn sm tint" data-act="prep" style="margin-left:auto">Токенизировать заново</button></div>`)
      + `<div class="tiles">
        <div class="tile"><span class="tl">Примеров</span><span class="tv">${int(s.examples)}</span></div>
        <div class="tile"><span class="tl">На обучение</span><span class="tv">${int(s.обучение)}</span></div>
        <div class="tile"><span class="tl">На валидацию</span><span class="tv">${int(s.отложено)}</span></div>
        <div class="tile"><span class="tl">Длина, медиана</span><span class="tv">${int(s.median)}<small>ток.</small></span></div>
        <div class="tile"><span class="tl">Длина, p99</span><span class="tv">${int(s.p99)}<small>ток.</small></span></div>
        <div class="tile"><span class="tl">Самый длинный</span><span class="tv">${int(s.max)}<small>ток.</small></span></div>
        <div class="tile"><span class="tl">Токенов в расчёте потери</span><span class="tv">${int(s.loss_tokens)}</span></div>
        <div class="tile"><span class="tl">Размер файла</span><span class="tv">${bytes(src.размер)}</span></div></div>`;
    st=pr.свежий? (taken? 'bad':'ok') : 'bad'; }
  if(el._h!==h){ el._h=h; el.innerHTML=h; }
  const s0=pr&&pr.сводка, path0=(pr&&pr.источник&&pr.источник.путь)||'';
  if(s0 && s0.instr_rows) sceneSeen={path:path0, n:s0.instr_rows};
  const ir=(s0&&s0.instr_rows) || (pr&&pr.идёт&&sceneSeen.path===path0? sceneSeen.n : 0);
  $('#d-scene').hidden=!ir;
  if(document.activeElement!==$('#d-share')) $('#d-share').value=String(Math.round(ch.scene.доля*100));
  if(document.activeElement!==$('#d-always')) $('#d-always').value=ch.scene.всегда;
  const sn=$('#d-snote'), nt=s0&&s0.instr_rows&&!(pr&&pr.идёт)? `С системной инструкцией: ${int(s0.with_instr)} из ${int(s0.examples)}` : '';
  sn.hidden=!nt; if(sn.textContent!==nt) sn.textContent=nt;
  setPill($('#d-st'), (pr&&pr.источник)||dataErr? st : 'need', st==='ok'? 'Готово' : st==='wait'? 'Токенизация' : st==='bad'? 'Ошибка' : 'Данные отсутствуют');
  const pv=$('#d-prev'), path=pr&&pr.источник&&pr.источник.путь; pv.hidden=!path;
  if(path && pv._path!==path){ pv._path=path; pv._at=0; if(pv.open) loadExamples(); }
}
// Примеры из файла — по одному, с листанием: видно, что именно пойдёт в обучение.
async function loadExamples(){
  const pv=$('#d-prev'), box=$('#d-ex'), path=pv._path; if(!path) return;
  const r=await post('/api/peek',{путь:path, с:pv._at||0, n:1}).catch(()=>null);
  if(!r||!r.ок){ box.innerHTML=`<div class="note bad">${ic('bad')}<span>${esc(cap((r&&r.почему)||'Не удалось прочитать файл.'))}</span></div>`; return; }
  const x=r.примеры[0]||{}, at=r.с, n=r.всего, F=[['instruction','Инструкция'],['input','Ввод'],['output','Ответ']];
  const known=F.filter(([k])=>x[k]!=null&&x[k]!==''), rest=Object.keys(x).filter(k=>!F.some(f=>f[0]===k));
  box.innerHTML=`<div class="exh"><span class="faint">Пример ${int(at+1)} из ${int(n)}</span><span class="pg">
      <button class="btn icon sm quiet" data-pg="-1"${at<=0?' disabled':''} aria-label="Предыдущий пример">${ic('left')}</button>
      <button class="btn icon sm quiet" data-pg="1"${at>=n-1?' disabled':''} aria-label="Следующий пример">${ic('right')}</button></span></div>
    <div class="exb">${known.map(([k,t])=>`<div class="exf"><span>${t}</span><div class="raw">${esc(x[k])}</div></div>`).join('')}
      ${rest.map(k=>`<div class="exf"><span>${esc(k)}</span><div class="raw">${esc(x[k])}</div></div>`).join('')}</div>`;
  box.querySelectorAll('[data-pg]').forEach(b=>b.onclick=()=>{ pv._at=Math.max(0,Math.min(n-1,(pv._at||0)+(+b.dataset.pg))); loadExamples(); });
}
// Проверки — одна строка на условие, в постоянном порядке: ничего не прыгает, видно всё сразу.
function checks(){
  const L=[], m=H.модель||{}, g=H.гпу||{}, pl=P&&P.план, pr=P&&P.подготовка, nm=P&&P.имя;
  const add=(grp,s,t,v,sub,acts)=>L.push({grp,s,t,v,sub:sub||'',acts:acts||[]});
  const busy=H.настройка? H.настройка_что||'?' : '';
  if(!m.есть && busy==='веса') add('выбор','wait','Модель','Загрузка весов', m.папка);
  else if(!m.есть) add('выбор','need','Модель','Отсутствует model.safetensors', m.папка, [['Указать папку','папка',1],['Скачать','веса',0]]);
  else if(m.склеить && busy==='склеить') add('выбор','wait','Модель','Объединение частей весов в один файл', m.папка);
  else if(m.склеить) add('выбор','need','Модель', cap(m.почему), 'Исходные части удаляются только после проверки объединённого файла · '+m.папка, [['Объединить','склеить',1],['Другая папка','папка',0]]);
  else if(!m.движок) add('выбор','bad','Модель', cap(m.почему), m.модель, [['Другая папка','папка',0]]);
  else add('выбор','ok','Модель', `${m.модель} · ${m.основной}`);
  const q=qsel(), qb=quantBusy();
  if(qb) add('выбор','info','База', (qb.startsWith('квант')? 'Квантование ' : 'Перенос ')+base(qb.slice(qb.indexOf(':')+1)), 'Обучению не препятствует: задание выполняется с низким приоритетом.');
  else if(H.база && !q) add('выбор','bad','База', `База ${H.база.slice(0,-3)} отсутствует`, '', [['BF16','base:',1]]);
  else if(q && !q.годна) add('выбор','bad','База', `${q.имя.slice(0,-3)}: ${q.почему}`, '', [['BF16','base:',1]]);
  else if(q) add('выбор','ok','База', `${q.имя.slice(0,-3)} · ${gib(q.байт,1)}`);
  if(dataErr) add('выбор','bad','Данные', cap(dataErr), 'Путь к файлу указывается в карточке «Данные».');
  else if(!pr || (!pr.идёт && !pr.источник)) add('выбор','need','Данные','Не выбраны', 'Путь к файлу указывается в карточке «Данные».');
  else if(pr.идёт) add('выбор','wait','Данные', `Токенизация «${base(pr.источник.путь)}»`);
  else if(pr.код || !pr.сводка) add('выбор','bad','Данные', 'Ошибка токенизации', (pr.вывод||[]).slice(-1)[0]||'', [['Повторить','prep',0]]);
  else if(!pr.свежий) add('выбор','bad','Данные','Файл изменён после токенизации', '', [['Токенизировать заново','prep',1]]);
  else add('выбор','ok','Данные', `${base(pr.источник.путь)} · ${int(pr.сводка.examples)} ${plural(pr.сводка.examples,'пример','примера','примеров')}`);
  if(!ch.name) add('выбор','need','Имя','Не задано');
  else if(nm && !nm.ок) add('выбор','bad','Имя', cap(nm.почему), '', /уже есть/i.test(nm.почему)? [[`Открыть «${ch.name}»`,'open:'+ch.name,0]] : []);
  else if(nm) add('выбор','ok','Имя', ch.name);
  if(pl && pl.ок===false && pr && pr.сводка) add('выбор','bad','План', cap(pl.почему), '', ch.mode==='эксперт' && /неверно/i.test(pl.почему||'')? [['Вернуть подобранные значения','reset',0]] : []);
  if(busy==='движок') add('запуск','wait','Движок','Сборка');
  else if(H.движок) add('запуск','ok','Движок', 'Собран'+(g.архитектура? ' · '+g.архитектура : ''));
  else if(H.компилятор) add('запуск','need','Движок', H.движок_есть? 'Собран из прежней версии' : 'Не собран', 'Сборка под установленную карту занимает несколько минут · '+H.компилятор, [['Собрать','движок',1]]);
  else add('запуск','bad','Движок', H.карта? 'Набор разработки отсутствует' : 'Видеокарта не найдена', H.карта==='nvidia'? 'CUDA Toolkit' : H.карта==='amd'? 'ROCm: rocm-hip-sdk, hipblas, hipblaslt' : 'ROCm (AMD) · CUDA Toolkit (NVIDIA)');
  if(busy==='пакеты') add('запуск','wait','Токенизатор','Установка пакетов');
  else if(!H.tokenizers) add('запуск','need','Токенизатор','Пакет tokenizers отсутствует', '', [['Установить','пакеты',1]]);
  else if(!H.словарь) add('запуск', m.есть? 'need' : 'info', 'Токенизатор', m.есть? 'Файл tokenizer.json отсутствует' : 'Поставляется вместе с весами', m.есть? m.папка : '', m.есть? [['Скачать','веса',0]] : []);
  else add('запуск','ok','Токенизатор','Готов');
  if(ch.tab!=='lora' && ch.goal!=='срез') add('запуск', H.numpy? 'ok' : 'need', 'Защита общих знаний', H.numpy? 'Готово' : 'Пакет numpy отсутствует', '', H.numpy? [] : [['Установить','пакеты',1]]);
  const live=liveRun();
  if(live) add('запуск','bad','Карта', `Занята: «${live}»`, '', [['Перейти','train',1]]);
  const tot=(H.озу||{}).всего||0, f=frozen();
  if(tot && m.параметров){ const rs=H.резерв_озу!=null? H.резерв_озу : Math.min(8*2**30, Math.max(5*2**30, tot/4)), lay=f.cache, cache=Math.max(0, Math.min(lay, tot-rs));   // как движок: занятое сейчас + запас потолка памяти + 1 ГиБ (замер из health)
    if(cache>=lay) add('ресурсы','ok','ОЗУ', `${gib(tot)} · Кэш весов целиком`, `Замороженные веса ${f.qn||'BF16'}: ${gib(lay)}, системе остаётся ${gib(tot-lay)}.`);
    else add('ресурсы','warn','ОЗУ', `Кэш весов ${gib(cache)} из ${gib(lay)}`, `Остальные ${gib(lay-cache)} читаются с диска дважды за шаг, поэтому обучение замедляется. Для разговора требуется ${gib(lay)} ОЗУ.`); }
  const v=H.видеопамять||{};
  if(!live){ const who=(v.кем||[]).map(x=>`${x.кто} ${gib(x.байт)}`).join(', ');
    if(!v.всего) add('ресурсы','info','Видеопамять', 'Проверка при запуске');
    else if(v.занято > 0.15*v.всего) add('ресурсы','warn','Видеопамять', `Занято ${gib(v.занято)} из ${gib(v.всего)}`, who);
    else add('ресурсы','ok','Видеопамять', `Свободно ${gib(v.всего-v.занято)} из ${gib(v.всего)}`, who); }
  const dk=H.диск||{};
  const wr=pl&&pl.запись? pl.запись*1.1 : 20*2**30;   // снимок весов и лучший снимок этого плана с запасом 10 %; плана нет — прежние 20 ГиБ
  if(dk.свободно!=null) add('ресурсы', dk.свободно<wr? 'warn':'ok', 'Диск', `Свободно ${bytes(dk.свободно)}`, 'Рабочая папка: снимки весов и состояние оптимизатора.');
  return L;
}
const blocking=L=>L.filter(c=>['bad','need','wait'].includes(c.s));
function renderLaunch(L){
  const pl=P&&P.план, rows=(pl&&pl.строки)||[], row=k=>rows.find(r=>r.к===k)||{}, B=blocking(L);
  let s='';
  if(pl && pl.шагов) s=[[int(pl.шагов),cap(steps_(pl.шагов))],[row('EPOCHS').знач||'',cap(plural(+row('EPOCHS').знач||0,'эпоха','эпохи','эпох'))],[int(pl.оценок),cap(plural(pl.оценок,'валидация','валидации','валидаций'))],
      [(pl.длительность||'').split(' — ')[0]||'—','Время',cap((pl.длительность||'').split(' — ').slice(1).join(' — '))]].filter(x=>x[0]!=='')
      .map(([v,k,t])=>`<div${t?` title="${esc(t)}"`:''}><b>${esc(String(v))}</b><span>${k}</span></div>`).join('');
  const se=$('#l-sum'); if(se._h!==s){ se._h=s; se.innerHTML=s; }
  const go=$('#l-go'), why=$('#l-why'); if(!go._busy) go.disabled=!!B.length;
  const tg=$('#ta-go'); if(tg) tg.disabled=go.disabled;
  let w;
  if(launchErr){ w=`Не удалось запустить: ${esc(cap(launchErr))}`; why.className='why bad'; }
  else if(B.length){ const b=B[0]; w=`${esc(b.t)}: ${esc(b.v)}${B.length>1? ` · Ещё ${B.length-1}` : ''}`; why.className='why bad'; }
  else { why.className='why'; w=''; }
  if(why._h!==w){ why._h=w; why.innerHTML=w; }
  const st=B.length? (B.some(c=>c.s==='bad')?'bad':B.some(c=>c.s==='wait')?'wait':'need') : 'ok';
  setPill($('#l-pill'), st, st==='ok'? 'Готово' : st==='wait'? 'Подготовка' : st==='bad'? 'Ошибка' : 'Требуется действие');
  const h=L.map(c=>`<div class="kv ${c.s}"><span class="lamp">${ic(c.s==='ok'?'ok':c.s==='need'?'need':c.s==='bad'?'bad':c.s==='warn'?'warn':c.s==='wait'?'wait':'info')}</span>
    <span class="k">${esc(c.t)}</span><span class="v" title="${esc(c.sub)}">${esc(c.v)}</span>${c.acts.length? `<div class="acts">${c.acts.map(a=>`<button class="btn sm ${a[2]?'tint':''}" data-act="${esc(a[1])}">${esc(a[0])}</button>`).join('')}</div>` : ''}</div>`).join('');
  const ce=$('#l-checks'); if(ce._h!==h){ ce._h=h; ce.innerHTML=h; }
  renderPlan();
}
// План — построчно: что задано и каким числом. В «Эксперте» каждое правится; почему так — в подсказке строки.
let planT=0;
function renderPlan(){
  const el=$('#l-plan'), pl=P&&P.план, ex=ch.mode==='эксперт', pr=P&&P.подготовка, own=Object.keys(ch.over).length, eb=$('#l-expert');
  eb.hidden=!(pl&&pl.строки) || (ex && !own); eb.textContent= ex? 'Сбросить собственные значения' : 'Изменить';
  eb.title= ex? 'Вернуть значения, подобранные автоподбором.' : 'Режим эксперта: ручная настройка каждого параметра.';
  $('#l-hw').hidden=!ex;
  if(!pl||!pl.строки){ const t=pr&&pr.идёт? 'Токенизация данных' : 'Данные отсутствуют';
    if(el._sig!=='e'+t){ el._sig='e'+t; el.innerHTML=`<p class="hint">${t}</p>`; } el.title=''; return; }
  const sig=(ex?'x':'r')+pl.строки.map(r=>r.к+'/'+r.что).join('|');
  if(el._sig!==sig){ el._sig=sig;
    el.innerHTML=pl.строки.map((r,i)=>`<div class="prow" data-i="${i}"><span class="pk">${esc(cap(r.что))}</span>${ex&&r.к? inputFor(r) : '<span class="pv"></span>'}</div>`).join('');
    el.querySelectorAll('input[data-k]').forEach(i=>i.addEventListener('input', ()=>{ const v=i.value.trim(); if(v) ch.over[i.dataset.k]=v; else delete ch.over[i.dataset.k];
      clearTimeout(planT); planT=setTimeout(replan, 280); }));
    el.querySelectorAll('select[data-k]').forEach(s=>s.addEventListener('change', ()=>{ ch.over[s.dataset.k]=s.value; replan(); })); }
  const bad=new Set(((pl.почему||'').match(/^неверно: ([^;]*)/i)||['',''])[1].split(', ').filter(Boolean));
  pl.строки.forEach((r,i)=>{ const d=el.querySelector(`.prow[data-i="${i}"]`); if(!d) return;
    const pv=d.querySelector('.pv'), inp=d.querySelector('input,select'); if(pv) pv.textContent=cap(r.знач); d.title=cap(r.как||'');
    if(inp){ if(inp.tagName==='INPUT'){ const raw=r.сырое!=null&&r.сырое!==''? String(r.сырое) : '', zn=cap(r.знач||'');
        inp.placeholder= !raw? zn : /[а-яёa-z]/i.test(zn)&&zn!==raw? raw+' · '+zn : raw; }
      inp.setAttribute('aria-invalid', bad.has(r.к)); } });
  el.title='Переменные окружения движка: '+envLine(pl.env);
}
function inputFor(r){
  if(r.варианты) return `<select class="inp" data-k="${esc(r.к)}" aria-label="${esc(r.что)}">${r.варианты.map(([k,t])=>`<option value="${esc(k)}"${String(ch.over[r.к]??r.сырое)===k?' selected':''}>${esc(cap(t))}</option>`).join('')}</select>`;
  return `<input class="inp" data-k="${esc(r.к)}" aria-label="${esc(r.что)}" value="${esc(ch.over[r.к]??'')}" placeholder="${esc(r.сырое!=null&&r.сырое!==''? r.сырое : r.знач)}" autocomplete="off" spellcheck="false">`;
}
function renderChoose(){
  if(!H) return;
  renderTab(); renderModel(); renderBase(); renderMem(); renderModes(); renderGoals(); renderLora(); renderData(); renderLaunch(checks());
  if(view==='setup') $('#vs').textContent=[ (H.модель||{}).модель, modeName(ch.mode), base(ch.file) ].filter(Boolean).join(' · ');
}

// --- что делают кнопки выбора ------------------------------------------------
let planBusy=false, planAgain=false, planPoll=0;
async function replan(){
  if(planBusy){ planAgain=true; return; }
  planBusy=true;
  try{ P=await post('/api/plan', {имя:ch.name, режим:ch.mode, что:goalNow(), лора:ch.lora, свои: ch.mode==='эксперт'? ch.over : {}}); } catch(e){ P=null; }
  planBusy=false;
  const pr=P&&P.подготовка;
  if(pr && pr.источник && !ch.file){ ch.file=pr.источник.путь; ch.scene={...SCENE0, ...(pr.сцена||{})}; if(ch.nameAuto){ ch.name=freeName(base(ch.file)); planAgain=true; }
    if(pr.сводка && pr.сводка.instr_rows==null && pr.свежий) return setFile(ch.file); }   // разбор прежней версии: без сводки по сценам
  renderChoose(); sideUpdate();
  if(planAgain){ planAgain=false; replan(); return; }
  clearTimeout(planPoll); if(pr && pr.идёт) planPoll=setTimeout(replan, 700);
}
function freeName(b){
  const taken=new Set((H&&H.запуски||[]).map(r=>r.имя));
  const s=(b||'run').replace(/\.[^.]*$/,'').replace(/[^\p{L}\p{N}_.-]+/gu,'-').replace(/^[.-]+|-+$/g,'').slice(0,34)||'run';
  let n=s, i=2; while(taken.has(n)) n=s+'-'+(i++); return n;
}
async function setFile(path){
  path=(path||'').trim(); if(!path) return;
  ch.file=path; $('#d-file').value=path; dataErr='';
  if(ch.nameAuto){ ch.name=freeName(base(path)); $('#d-name').value=ch.name; }
  const r=await post('/api/prepare',{датасет:path, сцена:ch.scene}).catch(()=>({ок:false,почему:'Нет связи с программой.'}));
  if(!r.ок) dataErr=r.почему||'Не удалось подготовить данные.';
  replan();
}
function setScene(){
  const v=$('#d-share').value.replace(',', '.').replace('%','').trim(), d=v===''? 1 : Number(v)/100;
  const words=$('#d-always').value.split(',').map(x=>x.trim()).filter(Boolean).join(', ');
  const sc={доля: isFinite(d)? Math.min(1, Math.max(0, d)) : ch.scene.доля, всегда: words};
  $('#d-share').value=String(Math.round(sc.доля*100)); $('#d-always').value=words;
  if(sc.доля===ch.scene.доля && sc.всегда===ch.scene.всегда) return;
  ch.scene=sc; if(ch.file) setFile(ch.file);
}
async function pickModel(){
  const p=await dialog({title:'Папка модели', text:'Расположение файла model.safetensors', input:{value:(H.модель||{}).папка||'', ph:'/путь/к/папке'}, ok:'Выбрать'});
  if(!p) return;
  await post('/api/settings',{свои:{MODEL:p}}); await refreshH(); replan();
}
function setMode(k){ if(ch.mode===k) return; ch.mode=k; renderModes(); renderPlan(); post('/api/settings',{свои:{MODE:k}}); replan(); }
function setGoal(k){ if(ch.goal===k) return; ch.goal=k; renderGoals(); renderMem(); replan(); }
const goalNow=()=>ch.tab==='lora'? 'lora' : ch.goal;
function setTab(k){ k= k==='lora'? 'lora' : 'tier'; if(ch.tab===k) return; ch.tab=k; post('/api/settings',{свои:{TAB:k}}); renderChoose(); replan(); }
function setLora(k, v){
  let lp={...ch.lora};
  if(k==='def') lp={...LORA0};
  else if(k==='mat') lp.tg=LORA_MATS.map(([c])=>c).filter(c=>c===v? !lp.tg.includes(c) : lp.tg.includes(c)).join('');
  else if(k==='tg') lp.tg=v;
  else if(k==='r'){ const r=parseInt(v,10); if(r>=1 && r<=1024) lp.r=r; }
  else if(k==='alpha'){ const a=parseFloat(String(v).replace(',','.')); if(a>0 && a<=65536) lp.alpha=a; }
  else if(k==='drop'){ const d=parseFloat(v); if(isFinite(d)) lp.drop=Math.max(0, Math.min(.9, d)); }
  if(!lp.tg || loraArg(lp)===loraArg(ch.lora)){ renderLora(); return; }   // без единой матрицы учить нечего: последняя не выключается
  ch.lora=lp; post('/api/settings',{свои:{LORA:loraArg(lp)}}); renderChoose(); replan();
}
async function launch(){
  if(!H || blocking(checks()).length) return;
  const b=$('#l-go'); b._busy=true; b.disabled=true; b.querySelector('span').textContent='Запуск';
  const r=await post('/api/start',{имя:ch.name, режим:ch.mode, что:goalNow(), лора:ch.lora, свои: ch.mode==='эксперт'? ch.over : {}}).catch(()=>({ок:false,почему:'Нет связи с программой.'}));
  b._busy=false; b.querySelector('span').textContent='Начать обучение';
  if(!r.ок){ launchErr=r.почему||'Не удалось запустить.'; renderChoose(); toast('Не удалось запустить', cap(launchErr), 'bad'); return; }
  launchErr=''; cur=ch.name; picked=true; toast('Обучение запущено', `«${ch.name}»`, 'ok');
  // данные переехали под имя запуска: следующий выбор начинается с чистого листа
  ch.file=''; ch.name=''; ch.nameAuto=true; ch.scene={...SCENE0}; $('#d-file').value=''; $('#d-name').value=''; $('#d-prev')._path='';
  await refreshH(); listen(); go('train'); replan();
}
async function act(a){
  if(a==='папка') return pickModel();
  if(a==='prep') return setFile(ch.file);
  if(a==='train'){ cur=liveRun()||cur; picked=true; listen(); return go('train'); }
  if(a==='reset'){ ch.over={}; $('#l-plan')._sig=''; return replan(); }
  if(a.startsWith('open:')) return openRun(a.slice(5));
  if(['движок','веса','пакеты','склеить'].includes(a)) return setupRun(a);
  if(a.startsWith('base:')) return setBase(a.slice(5));
}
// --- сборка движка, загрузка весов, пакеты: вывод виден под проверками -------
let setupES=null;
async function setupRun(what){
  const log=$('#setup-log'); log.hidden=false; log.textContent='';
  if(view!=='setup') go('setup');
  const r=await post('/api/setup',{что:what}).catch(()=>({ок:false,почему:'Нет связи.'}));
  if(!r.ок){ log.textContent=cap(r.почему); return; }
  const kd=what.split(':')[0];
  toast({движок:'Сборка движка',веса:'Скачивание весов',пакеты:'Установка пакетов',квант:'Квантование',импорт:'Перенос квантованной базы'}[kd]||'Подготовка', what.includes(':')? base(what.slice(what.indexOf(':')+1)) : '', 'info');
  setupListen(); refreshH();
}
function setupListen(){
  if(setupES) setupES.close();
  const log=$('#setup-log'); log.hidden=false;
  setupES=new EventSource('/api/setup/stream');
  setupES.onmessage=e=>{ const ev=JSON.parse(e.data);
    if(ev.текст){ log.textContent+=ev.текст; log.scrollTop=1e9; baseJob(); }
    if(ev.конец){ setupES.close(); setupES=null; log.textContent+= ev.код? `\nОшибка (код ${ev.код}), причина указана выше.\n` : '\nВыполнено\n'; log.scrollTop=1e9;
      toast(ev.код? 'Не выполнено' : 'Выполнено', ev.код? `Код ${ev.код}` : '', ev.код? 'bad':'ok'); if(ev.готовность){ H=ev.готовность; renderChoose(); } refreshH(); } };
}
// ============================================================================
// Канал разговора: одна загруженная модель на всю программу, запросы — по очереди.
// Каждый запрос получает ровно один ответ «end»: так разговор и проверка не мешают друг другу.
// ============================================================================
const Chat={es:null, q:[], busy:null, info:null,
  open(){ if(this.es) return Promise.resolve();
    this.es=new EventSource('/api/chat/stream');
    this.es.onmessage=e=>this.ev(JSON.parse(e.data));
    return new Promise(r=>{ if(this.es.readyState===1) r(); else this.es.addEventListener('open', ()=>r(), {once:true}); setTimeout(r, 1500); }); },
  ev(m){
    if(m.load) chatLoad(m.load);
    if(m.ready) chatLoad(null);
    if(m.tok!=null && this.busy && this.busy.tok) this.busy.tok(m.tok);
    if(m.end){ const b=this.busy; this.busy=null;
      if(m.закрыт){ chatLoad(null); const rest=this.q.splice(0), r=m.остановлено? {end:true, остановлено:true} : {end:true, ошибка:m.ошибка||'Движок закрыт.'};
        if(b) b.done(r); rest.forEach(x=>x.done(m.остановлено? r : {end:true, ошибка:'Движок закрыт.'})); refreshH(); return; }
      if(b) b.done(m); this.next(); } },
  send(body, tok){ return new Promise(res=>{ this.q.push({body, tok, done:res}); this.next(); }); },
  async next(){
    if(this.busy || !this.q.length) return;
    const it=this.q.shift(); this.busy=it; await this.open();
    const r=await post('/api/chat', it.body).catch(()=>({ок:false, почему:'Нет связи с программой.'}));
    if(!r.ок){ if(this.busy===it) this.busy=null; it.done({end:true, ошибка:r.почему}); this.next(); }
    else if(r.имя!==undefined) chatRefresh(); },
  cancel(){ const rest=this.q.splice(0); rest.forEach(x=>x.done({end:true, ошибка:'Отменено.'})); }
};
function chatLoad(t){ const el=$('#c-load'); if(!el) return; el.hidden=!t; if(t) $('#c-load-t').textContent=cap(t); chatRefresh(); }

// --- вид «Разговор» -----------------------------------------------------------
const CV={sel:null, convs:{}, sending:false};
const chatModels=()=>[{k:'', n:'Исходная модель', s:(H&&H.модель&&H.модель.модель)||'Без обучения'}]
  .concat((H&&H.запуски||[]).filter(r=>r.веса).map(r=>({k:r.имя, n:r.имя, s:[GOAL_SHORT[r.цель], r.лучшая!=null? 'Потеря (loss) '+fmt(r.лучшая,4) : ''].filter(Boolean).join(' · ')})));
function chatDefault(){ const L=chatModels(); const c=L.find(m=>m.k===cur&&m.k); return (c||L[1]||L[0]).k; }
function chatWith(name){ CV.sel=name; go('chat'); }
function chatRefresh(){
  if(!H) return;
  if(CV.sel==null) CV.sel=chatDefault();
  const L=chatModels(); if(!L.some(m=>m.k===CV.sel)) CV.sel=chatDefault();
  const h=L.map(m=>`<button class="mi" data-m="${esc(m.k)}" aria-current="${m.k===CV.sel}"><b>${esc(m.n)}</b><small>${esc(m.s)}</small></button>`).join('');
  const el=$('#c-models'); if(el._h!==h){ el._h=h; el.innerHTML=h; }
  const ms=$('#c-msel'), oh=L.map(m=>`<option value="${esc(m.k)}">${esc(m.n)}</option>`).join(''); if(ms._h!==oh){ ms._h=oh; ms.innerHTML=oh; } if(ms.value!==CV.sel) ms.value=CV.sel;
  const R=H.разговор||{}, loaded=R.имя, isBase=!CV.sel, ready=R.готов && loaded!=null && (isBase || loaded===CV.sel);
  $('#c-title').textContent= isBase? 'Исходная модель' : CV.sel;
  const st=$('#c-state'); const loading=loaded!=null && !R.готов;
  st.className='pill '+(ready? 'ok' : loading? 'live' : ''); st.innerHTML=`<i></i>${ready? 'Загружена' : loading? 'Загружается' : 'Не загружена'}`;
  st.title= ready? '' : 'Веса загружаются при первом вопросе.';
  $('#c-ab').disabled=isBase; if(isBase) $('#c-ab').checked=false;
  $('#c-unload').hidden=loaded==null;
  if(view==='chat') $('#vs').textContent= ready? 'Модель в видеопамяти' : loading? 'Загрузка весов' : '';
  $('#st-chat').className='st'+(loaded!=null? (R.готов? ' ok':' live') : '');
}
function convOf(k){ return CV.convs[k]||(CV.convs[k]=[]); }
function msgEl(m){
  const d=document.createElement('div');
  if(m.who==='ab'){ d.className='ab'; d.append(msgEl(m.a), msgEl(m.b)); return d; }
  d.className='msg '+(m.who==='me'?'me':m.who==='sys'?'sys':m.who==='b'?'b':m.who==='a'?'a':'');
  const who= m.who==='me'? '' : m.who==='sys'? '' : `<span class="who">${esc(m.name||'')}</span>`;
  d.innerHTML=`${who}<div class="bub${m.typing?' typing':''}"><span class="msg-text"></span></div>${m.meta?`<span class="meta">${esc(m.meta)}</span>`:''}`;
  d.querySelector('.msg-text').textContent=m.text; d.querySelector('.bub').hidden=!m.text&&!m.typing&&m.who!=='me'; m.el=d; return d;
}
function convRender(){
  const box=$('#c-log'), C=convOf(CV.sel);
  box.replaceChildren();
  if(!C.length){ const r=(H&&H.запуски||[]).find(x=>x.имя===CV.sel)||{}, m=(H&&H.модель)||{};
    const f= CV.sel? [GOAL_NAME[r.цель], r.лучшая!=null? 'Потеря (loss) '+fmt(r.лучшая,4) : '', r.шаг? 'Шаг '+int(r.шаг) : '', r.когда? when(r.когда) : ''] : [m.модель, 'Без обучения'];
    box.innerHTML=`<div class="empty">${ic('chat')}<h3>${CV.sel? esc(CV.sel) : 'Исходная модель'}</h3><span>${esc(f.filter(Boolean).join(' · '))}</span></div>`; return; }
  C.forEach(m=>box.append(msgEl(m))); box.scrollTop=1e9;
}
function patchMsg(m){ if(!m.el) return; const t=m.el.querySelector('.msg-text'), b=m.el.querySelector('.bub'); t.textContent=m.text; b.classList.toggle('typing',!!m.typing); b.hidden=!m.text&&!m.typing;
  let me=m.el.querySelector('.meta'); if(m.meta){ if(!me){ me=document.createElement('span'); me.className='meta'; m.el.append(me); } me.textContent=m.meta; } }
const chatOpts=()=>({temp:+UI.chatTemp, max:+UI.chatMax, сцена:UI.chatScene||'', сила:+UI.chatScale});
async function ask1(m, body){
  const box=$('#c-log');
  const r=await Chat.send(Object.assign(chatOpts(), body), piece=>{ m.text+=piece; patchMsg(m); if(box.scrollHeight-box.scrollTop-box.clientHeight<120) box.scrollTop=1e9; });
  m.typing=false;
  if(r.ошибка){ m.text=m.text||cap(r.ошибка); m.meta='Ошибка'; }
  else if(r.остановлено) m.meta='Остановлено';
  else m.meta=[r.токенов? r.токенов+' ток.' : '', r.сек? dec(r.сек)+' с' : '', r.скорость||''].filter(Boolean).join(' · ');
  patchMsg(m); return r;
}
async function chatSend(){
  const inp=$('#c-in'), q=inp.value.trim(); if(!q || CV.sending) return;
  inp.value=''; autoGrow(inp); CV.sending=true; $('#c-send').disabled=true;
  const k=CV.sel, C=convOf(k), box=$('#c-log');
  if(!C.length) box.replaceChildren();
  const me={who:'me', text:q}; C.push(me); box.append(msgEl(me));
  if($('#c-ab').checked && k){
    const a={who:'a', name:k, text:'', typing:true}, b={who:'b', name:'Исходная модель', text:'', typing:true}, pair={who:'ab', a, b};
    C.push(pair); box.append(msgEl(pair)); box.scrollTop=1e9;
    await ask1(a, {имя:k, base:false, q, new:true});
    await ask1(b, {имя:k, base:true, q, new:true});
  } else {
    const m={who:'m', name: k||'Исходная модель', text:'', typing:true}; C.push(m); box.append(msgEl(m)); box.scrollTop=1e9;
    await ask1(m, {имя:k, base:!k, q});
  }
  CV.sending=false; $('#c-send').disabled=false; inp.focus(); refreshH();
}
async function chatNew(){ const k=CV.sel; CV.convs[k]=[]; convRender(); await Chat.send({имя:k, base:!k, new:true}); }
function autoGrow(t){ t.style.height='auto'; t.style.height=Math.min(200, t.scrollHeight+2)+'px'; }
function chatBind(){
  $('#c-models').addEventListener('click', e=>{ const b=e.target.closest('[data-m]'); if(!b) return; CV.sel=b.dataset.m; chatRefresh(); convRender(); });
  $('#c-msel').onchange=e=>{ CV.sel=e.target.value; chatRefresh(); convRender(); };
  $('#c-form').onsubmit=e=>{ e.preventDefault(); chatSend(); };
  $('#c-in').addEventListener('keydown', e=>{ if(e.key==='Enter' && !e.shiftKey){ e.preventDefault(); chatSend(); } });
  $('#c-in').addEventListener('input', e=>autoGrow(e.target));
  $('#c-new').onclick=chatNew;
  $('#c-unload').onclick=async()=>{ BV.stop=true; await post('/api/chat/stop', {}).catch(()=>null); chatLoad(null); refreshH(); };
  $('#c-opt').onclick=()=>{ const o=$('#c-opts'); o.hidden=!o.hidden; };
  [['#c-temp','chatTemp'],['#c-max','chatMax'],['#c-scale','chatScale'],['#c-scene','chatScene']].forEach(([id,k])=>{ const el=$(id); el.value=UI[k];
    el.addEventListener('change', ()=>uiSet({[k]: el.type==='number'? +el.value : el.value})); });
}

// ============================================================================
// Проверка вопросами: обученная против исходной, каждый вопрос с чистого листа.
// Совпадение с ожидаемым ищется по смыслу строки: регистр, ё, знаки и числа словами не мешают.
// ============================================================================
const BV={rows:[], run:false, stop:false, t0:0};
const NUMW={ноль:0,нуль:0,один:1,одна:1,одно:1,два:2,две:2,три:3,четыре:4,пять:5,шесть:6,семь:7,восемь:8,девять:9,десять:10,одиннадцать:11,двенадцать:12,
  тринадцать:13,четырнадцать:14,пятнадцать:15,шестнадцать:16,семнадцать:17,восемнадцать:18,девятнадцать:19,двадцать:20,тридцать:30,сорок:40,пятьдесят:50,сто:100,тысяча:1000,
  zero:0,one:1,two:2,three:3,four:4,five:5,six:6,seven:7,eight:8,nine:9,ten:10,twelve:12,twenty:20,hundred:100};
const normA=s=>String(s||'').toLowerCase().replace(/ё/g,'е').replace(/[«»"'“”„.,!?;:()\[\]{}—–\-*_]/g,' ').split(/\s+/).filter(Boolean).map(w=>NUMW[w]!=null? String(NUMW[w]) : w);
function match(ans, exp){
  const a=normA(ans), e=normA(exp); if(!e.length) return null;
  const A=' '+a.join(' ')+' ', E=' '+e.join(' ')+' ';
  if(A.includes(E)) return true;
  if(e.length<=3) return e.every(w=>a.includes(w)) || (a.length && e[0]===a[0]);
  const hit=e.filter(w=>a.includes(w)).length; return hit/e.length>=0.7;
}
const same=(x,y)=>normA(x).join(' ')===normA(y).join(' ');
function benchModels(){
  const sel=$('#b-run'), L=chatModels(), first=sel._h==null, keep=sel.value;
  const h=L.map(m=>`<option value="${esc(m.k)}">${esc(m.n)}</option>`).join('');
  if(sel._h!==h){ sel._h=h; sel.innerHTML=h; sel.value= !first && L.some(m=>m.k===keep)? keep : chatDefault(); }
  const runOpt=$('#b-src').querySelector('[value="run"]'); runOpt.disabled=!sel.value; if(!sel.value && $('#b-src').value==='run') $('#b-src').value='list';
  const src=$('#b-src').value; $('#b-file-f').hidden=src!=='file'; $('#b-list-f').hidden=src!=='list';
  $('#b-base').disabled=!sel.value; if(!sel.value) $('#b-base').checked=false; else if(sel._was===''&&sel.value) $('#b-base').checked=true;
  sel._was=sel.value; benchCount();
}
async function benchQuestions(){
  const src=$('#b-src').value, run=$('#b-run').value;
  if(src==='list') return $('#b-list').value.split('\n').map(x=>x.trim()).filter(Boolean).map(x=>{ const [q,...e]=x.split('||'); return {q:q.trim(), ждём:e.join('||').trim()}; });
  let path=$('#b-file').value.trim();
  if(src==='run'){ const s=await getj('/api/state?run='+encodeURIComponent(run)).catch(()=>null); path=s&&s.выбор&&s.выбор.датасет||'';
    if(!path) throw new Error('У запуска нет записи о файле данных, требуется выбрать файл.'); }
  if(!path) throw new Error('Файл с вопросами не выбран.');
  const r=await post('/api/questions',{путь:path}); if(!r.ок) throw new Error(r.почему||'Не удалось прочитать файл.');
  return r.вопросы;
}
// Сколько вопросов пойдёт — видно до запуска: из списка считаем сразу, из файла — спрашиваем программу.
let bcT=0, bcKey='';
function benchCount(){
  clearTimeout(bcT); const src=$('#b-src').value, run=$('#b-run').value, el=$('#b-count'), inf=$('#b-src-info');
  const show=(n,w)=>{ el.textContent= n==null? '' : `${n} ${plural(n,'вопрос','вопроса','вопросов')}`; inf.textContent=w||''; };
  if(src==='list'){ const n=$('#b-list').value.split('\n').filter(x=>x.trim()).length; show(n||null, n? 'Собственный список' : 'Собственный список пуст'); return; }
  const key=src+'|'+run+'|'+$('#b-file').value.trim(); if(key===bcKey) return; bcKey=key;
  bcT=setTimeout(async()=>{ try{ const q=await benchQuestions(); if(bcKey===key) show(q.length, src==='run'? `Из данных «${run}»` : base($('#b-file').value)); }
    catch(e){ if(bcKey===key) show(null, cap(e.message)); } }, 250);
}
function benchDraw(){
  const R=BV.rows, n=R.length, cmp=$('#b-base').checked && $('#b-run').value, hasE=R.some(r=>r.ждём);
  const doneA=R.filter(r=>r.a!=null).length, doneB=R.filter(r=>r.b!=null).length, total=n*(cmp?2:1);
  const hitA=R.filter(r=>r.ok===true).length, hitB=R.filter(r=>r.okb===true).length, chg=R.filter(r=>r.a!=null&&r.b!=null&&!same(r.a,r.b)).length;
  $('#b-res').hidden=!n; $('#b-empty').hidden=!!n;
  $('#b-prog').textContent= BV.run? `Выполняется · ${doneA+doneB} из ${total}` : n? `${n} ${plural(n,'вопрос','вопроса','вопросов')} · ${hms((BV.t1||Date.now())/1000-BV.t0/1000)}` : '';
  $('#b-bar').hidden=!BV.run; $('#b-bar').querySelector('i').style.width=(total? 100*(doneA+doneB)/total : 0)+'%';
  $('#b-stop').hidden=!BV.run; $('#b-go').disabled=BV.run;
  $('#b-tiles').innerHTML=(hasE? `<div class="tile"><span class="tl">Совпало с ожидаемым</span><span class="tv">${hitA}<small>из ${n}</small></span><span class="tn">${$('#b-run').value? 'Обученная' : 'Исходная'}</span></div>` : '')
    + (cmp&&hasE? `<div class="tile"><span class="tl">У исходной модели</span><span class="tv">${hitB}<small>из ${n}</small></span><span class="tn">Совпало с ожидаемым</span></div>` : '')
    + (cmp? `<div class="tile"><span class="tl">Ответ изменился</span><span class="tv">${chg}<small>из ${doneB}</small></span><span class="tn">Обученная модель против исходной</span></div>` : '');
  const res=r=>{ if(r.a==null) return BV.run? 'Выполняется' : '—'; const t=[];
    if(r.ждём) t.push(r.ok? '✓ Совпало' : '✗ Не совпало'); if(cmp&&r.b!=null) t.push(same(r.a,r.b)? '= Как у исходной модели' : '≠ Изменился'); return t.join(' · ')||'—'; };
  const cls=r=>r.a==null? '' : r.ждём? (r.ok? 'hit':'miss') : (cmp&&r.b!=null&&same(r.a,r.b)? 'same':'');
  $('#b-t').innerHTML=`<thead><tr><th>#</th><th>Вопрос</th>${hasE?'<th>Ожидаемый ответ</th>':''}<th>${$('#b-run').value? 'Обученная' : 'Исходная'}</th>${cmp?'<th>Исходная</th>':''}<th>Итог</th></tr></thead><tbody>`
    + R.map((r,i)=>`<tr class="${cls(r)}"><td class="faint">${i+1}</td><td class="q"${r.сцена?` title="${esc('Системная инструкция: '+r.сцена)}"`:''}>${esc(r.q)}</td>${hasE?`<td class="ans faint">${esc(r.ждём)}</td>`:''}<td class="ans">${r.a==null? '<span class="faint">—</span>' : esc(r.a)}</td>
      ${cmp?`<td class="ans">${r.b==null? '<span class="faint">—</span>' : esc(r.b)}</td>`:''}<td class="res${r.ждём&&r.a!=null?' mark':''}" data-i="${i}" title="${r.ждём?'Щелчок мышью отмечает результат вручную.':''}">${res(r)}</td></tr>`).join('') + '</tbody>';
}
async function benchRun(){
  if(BV.run) return;
  let qs; try{ qs=await benchQuestions(); }catch(e){ toast('Проверка не запущена', cap(e.message), 'bad'); return; }
  if(!qs.length){ toast('Вопросов нет','', 'warn'); return; }
  const run=$('#b-run').value, cmp=$('#b-base').checked&&run, o={temp:+$('#b-temp').value||0, max:+$('#b-max').value||48, сцена:$('#b-scene').value};
  BV.rows=qs.map(x=>({q:x.q, сцена:x.сцена||'', ждём:x.ждём, a:null, b:null, ok:null, okb:null})); BV.run=true; BV.stop=false; BV.t0=Date.now(); BV.t1=0; benchDraw();
  $('#b-res').scrollIntoView({block:'start', behavior: calm()? 'auto':'smooth'});
  const pass=async(isB)=>{ for(const r of BV.rows){ if(BV.stop) return false; let t='';
      const m=await Chat.send(Object.assign({}, o, {имя:run, base:isB||!run, q:r.q, new:true}, r.сцена? {сцена:r.сцена} : {}), p=>{ t+=p; });
      if(m.остановлено){ BV.stop=true; return false; }
      if(m.ошибка){ toast('Проверка прервана', cap(m.ошибка), 'bad'); BV.stop=true; return false; }
      if(isB){ r.b=t.trim(); r.okb=r.ждём? match(r.b, r.ждём) : null; } else { r.a=t.trim(); r.ok=r.ждём? match(r.a, r.ждём) : null; }
      benchDraw(); } return true; };
  if(await pass(false) && cmp) await pass(true);
  if(run && H && H.разговор && H.разговор.имя!=null) await Chat.send({имя:run, base:false, new:true});
  BV.run=false; BV.t1=Date.now(); benchDraw(); refreshH();
  if(!BV.stop) toast('Проверка завершена', `${BV.rows.length} ${plural(BV.rows.length,'вопрос','вопроса','вопросов')}`, 'ok');
}
function benchTable(){
  const cmp=BV.rows.some(r=>r.b!=null), head=['№','Вопрос','Ожидаемый ответ','Обученная'].concat(cmp?['Исходная']:[]).concat(['Совпало']);
  return {head, rows:BV.rows.map((r,i)=>[i+1, r.q, r.ждём, r.a??''].concat(cmp?[r.b??'']:[]).concat([r.ok==null?'':r.ok?'да':'нет']))};
}
function benchBind(){
  ['#b-run','#b-src'].forEach(id=>$(id).addEventListener('change', ()=>{ bcKey=''; benchModels(); }));
  $('#b-list').addEventListener('input', benchCount); $('#b-file').addEventListener('change', ()=>{ bcKey=''; benchCount(); });
  $('#b-go').onclick=benchRun;
  $('#b-stop').onclick=()=>{ BV.stop=true; Chat.cancel(); };
  $('#b-t').addEventListener('click', e=>{ const c=e.target.closest('.res.mark'); if(!c) return; const r=BV.rows[+c.dataset.i]; r.ok=!r.ok; benchDraw(); });
  $('#b-copy').onclick=()=>{ const t=benchTable(); copyText([t.head].concat(t.rows).map(r=>r.map(v=>String(v).replace(/\s+/g,' ')).join('\t')).join('\n'), 'Таблица скопирована'); };
  $('#b-save').onclick=async()=>{ const t=benchTable(), r=await post('/api/bench/save',{имя:$('#b-run').value, заголовок:t.head, строки:t.rows}).catch(()=>({ок:false}));
    r.ок? toast('Таблица сохранена', r.путь, 'ok', 6000) : toast('Не удалось сохранить', cap(r.почему||''), 'bad'); };
}
// ============================================================================
// Виды, боковая панель, поток состояния
// ============================================================================
const VIEWS={setup:'Новое обучение', train:'Ход обучения', runs:'Запуски', chat:'Разговор', bench:'Проверка', settings:'Настройки'};
const VKEYS=['setup','train','runs','chat','bench','settings'];
let es=null, tRender=0, tWin=0, navN=0;
function go(v){
  navN++;
  if(v==='обучение') v='train'; if(v==='выбор'||v==='запуск') v='setup'; if(!VIEWS[v]) v='setup';
  view=v;
  const blank= v==='train' && !cur;
  VKEYS.forEach(k=>{ const el=$('#v-'+k); if(el) el.hidden= k!==v || (k==='train'&&blank); });
  $('#v-blank').hidden=!blank;
  $$('.ni[data-go]').forEach(b=>b.setAttribute('aria-current', b.dataset.go===v? 'page' : 'false'));
  $('#vt').textContent=VIEWS[v]; $('#vs').textContent=''; $('#tab').hidden= v!=='setup';
  topActions();
  if(v==='setup') renderChoose();
  if(v==='train' && state){ tRender=0; render(); }
  if(v==='runs') renderRuns();
  if(v==='chat'){ chatRefresh(); convRender(); Chat.open(); setTimeout(()=>$('#c-in').focus(), 30); }
  if(v==='bench') benchModels();
  if(v==='settings') settingsRender();
  if(v!=='chat') $('#views').scrollTop=0;
  try{ localStorage.setItem('view', v); }catch(e){}
}
// Действия в полосе заголовка — только те, что относятся к виду.
// В узком окне «Запуск» уходит под выбор — кнопка старта остаётся в полосе заголовка.
const NARROW=matchMedia('(max-width:1100px)'); NARROW.addEventListener('change', ()=>topActions());
function topActions(){
  const t=$('#tact'); let h='';
  if(view==='runs'){ const n=RS.sel.size;
    h=`<button class="btn sm" id="ta-cmp"${n<2?' disabled':''} title="Требуется выбрать от 2 до 6 запусков.">${ic('compare')}Сравнить${n>=2?' · '+n:''}</button>
       <button class="btn sm danger" id="ta-del"${!n?' disabled':''}>${ic('trash')}Удалить${n?' · '+n:''}</button>`; }
  if(view==='train' && state) h=`<button class="btn sm quiet" id="ta-log" title="Журнал (J)">${ic('file')}Журнал</button><button class="btn sm quiet" id="ta-plan">${ic('settings')}Параметры</button>`;
  if(view==='setup' && NARROW.matches) h=`<button class="btn sm primary" id="ta-go" title="Ctrl+Enter"${$('#l-go').disabled?' disabled':''}>${ic('play')}Начать обучение</button>`;
  if(t._h===h) return; t._h=h; t.innerHTML=h;
  const b=id=>t.querySelector(id);
  if(b('#ta-cmp')) b('#ta-cmp').onclick=()=>compareRuns([...RS.sel]);
  if(b('#ta-del')) b('#ta-del').onclick=()=>delRuns([...RS.sel]);
  if(b('#ta-log')) b('#ta-log').onclick=()=>openWin('log',$('#log'));
  if(b('#ta-plan')) b('#ta-plan').onclick=()=>openWin('plan',$('#n-chips'));
  if(b('#ta-go')) b('#ta-go').onclick=launch;
}
function listen(){
  if(es){ es.close(); es=null; }
  tRender=0;
  if(!cur){ state=null; return; }
  es=new EventSource('/api/stream?run='+encodeURIComponent(cur));
  es.onmessage=e=>{ const s=JSON.parse(e.data); if(s.имя!==cur) return; state=s; render(); };
  es.onerror=()=>{ const b=$('#n-badge span'); if(b&&view==='train') b.textContent='Связь прервана'; };
}
function render(){
  if(!state) return;
  edgeUpdate(state); sideUpdate();
  const L=state.последний||{}, t=(state.план||{}).всего;
  document.title= state.идёт? `TierLM · «${state.имя}» ${L.шаг? int(L.шаг)+(t?' / '+int(t):'') : stageText(state)}` : 'TierLM';
  if(view!=='train') return;
  runSelect(state); nowSlab(state); journal(state); topActions();
  $('#vs').textContent=`«${state.имя}»`;
  const now=Date.now();
  if(now-tRender>900){ tRender=now; plots(state); }
  if(winOpen && now-tWin>2000){ tWin=now; fillWin(); }
}
// Кромка полосы заголовка — ход идущего обучения: видно с любого вида.
function edgeUpdate(s){
  const edge=$('#edge'), fill=$('#fill');
  const live=s&&s.идёт, L=(s&&s.последний)||{}, P_=(s&&s.план)||{}, c=L['шаг']||0, total=P_.всего||0, frac=live&&total? Math.min(1,c/total) : 0;
  edge.hidden=!live;
  if(edge._cur!==c || edge._live!==live){
    const sec=lastOf(s&&s.шаги||[],'сек')||0, goOn=live && total && c && sec && !calm();
    if(edge._cur==null || !goOn){ fill.style.transition= edge._cur==null? 'none' : 'width .4s ease'; fill.style.width=(100*frac)+'%'; fill.offsetWidth; }
    if(goOn){ fill.style.transition=`width ${Math.min(90,Math.max(.3,sec))}s linear`; fill.style.width=(100*Math.min(1,(c+1)/total))+'%'; }
    edge._cur=c; edge._live=live; }
  edge.setAttribute('aria-valuenow', Math.round(100*frac));
}
// Боковая панель: идущее обучение, готовность нового, число запусков, место на диске.
function sideUpdate(){
  if(!H) return;
  const live=liveRun(), s= state&&state.имя===live? state : null, r=(H.запуски||[]).find(x=>x.имя===live);
  const lv=$('#live'); lv.hidden=!live;
  if(live){ const L=(s&&s.последний)||{}, p=(s&&s.план)||{}, st=L.шаг||(r&&r.шаг)||0, tot=p.всего||(r&&r.всего)||0, f=tot? st/tot : null;
    $('#lv-name').textContent=live; $('#lv-pct').textContent= f!=null&&st? Math.round(100*f)+' %' : '';
    const bar=$('#lv-bar'); bar.style.width= f!=null&&st? (100*f).toFixed(1)+'%' : '30%'; bar.parentElement.classList.toggle('ind', !(f!=null&&st));
    $('#lv-sub').textContent= s? (st? `Шаг ${int(st)}${tot?' из '+int(tot):''}${L.осталось? ' · Осталось '+L.осталось : ''}` : stageText(s)||'Запуск') : (st? `Шаг ${int(st)}${tot?' из '+int(tot):''}` : 'Выполняется'); }
  const B=blocking(checks()), so=$('#st-setup');
  so.className= live? 'st' : 'st '+(B.some(c=>c.s==='bad')? 'bad' : B.length? 'need' : 'ok'); so.title= live? '' : B.length? B[0].t+': '+B[0].v : 'Готово к запуску';
  const t=$('#st-train'); t.className='st'+(live? ' live' : ''); t.title= live? `Выполняется «${live}»` : '';
  const n=(H.запуски||[]).length; $('#nb-runs').textContent= n? n : '';
  const dk=H.диск||{}, d=$('#disk');
  if(dk.всего){ const used=1-dk.свободно/dk.всего; d.hidden=false;
    d.innerHTML=`Свободно ${bytes(dk.свободно)} из ${bytes(dk.всего)}<span class="bar${used>.92?' bad':''}" style="display:block"><i style="width:${(100*used).toFixed(1)}%"></i></span>`; }
  $('#st-bench').className='st'+(BV.run? ' live' : '');
}
let lastLive=null;
async function refreshH(){
  try{ H=await getj('/api/health'); }catch(e){ return; }
  const live=H.идёт;
  if(lastLive && lastLive!==live){            // обучение только что кончилось: тост с итогом
    const r=(H.запуски||[]).find(x=>x.имя===lastLive);
    if(r) toast(r.состояние==='завершено'? 'Обучение завершено' : r.состояние==='ошибка'? 'Обучение остановлено из-за ошибки' : 'Обучение остановлено',
      `«${r.имя}»`+(r.лучшая!=null? ` · Потеря (loss) ${fmt(r.лучшая,4)}` : ''), r.состояние==='завершено'? 'ok' : r.состояние==='ошибка'? 'bad' : 'warn', 8000); }
  lastLive=live;
  if(live && live!==refreshH._live && cur!==live && !picked){ cur=live; listen(); }
  refreshH._live=live;
  if(!cur && (H.запуски||[]).length){ cur=H.запуски[0].имя; listen(); }
  $('#ver').textContent=H.версия||'';
  renderChoose(); sideUpdate();
  if(view==='train' && state) nowSlab(state);
  if(view==='runs') renderRuns();
  if(view==='chat') chatRefresh();
  if(view==='bench') benchModels();
}

// ============================================================================
// Настройки: оформление, графики, поведение, разговор, оборудование, пути, клавиши, о программе
// ============================================================================
const SECS=[['look','Оформление','palette'],['charts','Графики','chart'],['behave','Поведение','settings'],['chatd','Разговор','chat'],
  ['hw','Оборудование','chip'],['paths','Пути','folder'],['keys','Клавиши','keys'],['about','О программе','info']];
let sec='look', apT=0, uiT=0, cfg=null;
function apSave(){ clearTimeout(apT); apT=setTimeout(()=>post('/api/settings',{свои:{APPEAR:JSON.stringify(AP),THEME:AP.theme==='light'?'light':''}}),300); }
function apSet(p, keep){ apApply(Object.assign({},AP,p)); apSave(); if(!keep && view==='settings' && sec==='look') settingsRender(true); if(state&&view==='train') plots(state); }
function uiSave(){ clearTimeout(uiT); uiT=setTimeout(()=>post('/api/settings',{свои:{UI:JSON.stringify(UI)}}),300); }
function uiSet(p, rerender){ Object.assign(UI,p); uiSave(); if(rerender && view==='settings') settingsRender(true); if(state){ $('#plots')._sig=''; if(view==='train') plots(state); } }
const segH=(id,list,v)=>`<span class="seg" role="radiogroup" id="${id}">${list.map(([k,t])=>`<button role="radio" data-v="${esc(k)}" aria-checked="${String(k)===String(v)}">${t}</button>`).join('')}</span>`;
const rngH=(id,lo,hi,stp,v,out)=>`<input type="range" class="rng" id="${id}" min="${lo}" max="${hi}" step="${stp}" value="${v}"><output id="${id}-o">${out}</output>`;
const swH=(id,on)=>`<input type="checkbox" class="sw" id="${id}"${on?' checked':''}>`;
const row=(label,ctl,small='')=>`<div class="srow"><span class="sl">${label}${hint(small)}</span><div class="sc">${ctl}</div></div>`;
const card=(title,body,acts='')=>`<section class="card glass">${title||acts? `<div class="ch"><h2>${title}</h2>${acts?`<div class="ca">${acts}</div>`:''}</div>` : ''}${body}</section>`;
function presetCard(n,p,user,i){
  const a=Object.assign({},AP_DEF,p), B=AP_BG[a.bg], light=a.theme==='light', w=a.bg==='solid'?a.solid:a.bg==='image'?'#555':(light?B.l:B.d);
  const on=['theme','bg','accent','radius','line','glass','ff','fs','weight','density'].every(k=>String(a[k])===String(AP[k]));
  return `<button class="pcard" data-pre="${user?'u'+i:'b'+i}" aria-checked="${on}"><i style="background:${w}">
    <s style="position:absolute;left:10px;top:10px;bottom:10px;width:30%;border-radius:${Math.round(a.radius/3)}px;background:${light?'rgba(246,246,249,.85)':'rgba(26,26,31,.8)'}"></s>
    <s style="position:absolute;left:calc(30% + 16px);right:10px;top:10px;height:42%;border-radius:${Math.round(a.radius/3)}px;background:${light?'rgba(255,255,255,.85)':'rgba(40,40,46,.85)'};box-shadow:inset 0 0 0 ${a.line}px ${light?'rgba(0,0,0,.1)':'rgba(255,255,255,.14)'}"></s>
    <s style="position:absolute;right:16px;bottom:12px;width:26px;height:8px;border-radius:4px;background:${a.accent}"></s></i>
    <span>${esc(n)}${user?' <span class="faint">· Собственный</span>':''}</span>${user?`<span class="del" data-del="${i}" title="Удалить набор">${ic('x')}</span>`:''}</button>`;
}
function settingsRender(keepScroll){
  const nav=$('#s-nav');
  nav.innerHTML=SECS.map(([k,t,i])=>`<button class="ni" data-s="${k}" aria-current="${k===sec?'page':'false'}">${ic(i)}<span>${t}</span></button>`).join('');
  nav.onclick=e=>{ const b=e.target.closest('[data-s]'); if(b){ sec=b.dataset.s; settingsRender(); } };
  const body=$('#s-body'), top=$('#views').scrollTop;
  $('#vs').textContent=(SECS.find(x=>x[0]===sec)||[])[1]||'';
  body.innerHTML=({look:secLook, charts:secCharts, behave:secBehave, chatd:secChat, hw:secHw, paths:secPaths, keys:secKeys, about:secAbout}[sec]||secLook)();
  body.querySelectorAll('.rng').forEach(rangeFill);
  ({look:bindLook, charts:bindCharts, behave:bindBehave, chatd:bindChat, hw:bindHw, paths:bindPaths}[sec]||(()=>{}))(body);
  if(keepScroll) $('#views').scrollTop=top;
}
function secLook(){
  const acc=['#0a84ff','#5e5ce6','#bf5af2','#ff375f','#c06c84','#ff9f0a','#ffd60a','#30d158','#64d2ff','#8e8e93','#e5e5ea','#1d1d1f'];
  const bgs=Object.entries(AP_BG).map(([k,B])=>`<button class="pcard" data-bg="${k}" aria-checked="${AP.bg===k}"><i style="background:${AP.theme==='light'?B.l:B.d};height:46px"></i><span>${B.n}</span></button>`).join('')
    + `<button class="pcard" data-bg="solid" aria-checked="${AP.bg==='solid'}"><i style="background:${AP.solid};height:46px"></i><span>Собственный цвет</span></button>`
    + (AP.img? `<button class="pcard" data-bg="image" aria-checked="${AP.bg==='image'}"><i style="height:46px;background:url(/bg?t=${AP.img}) center/cover"></i><span>Изображение</span></button>` : '');
  const fonts=Object.entries(AP_FONTS).map(([k,f])=>`<option value="${k}"${AP.ff===k?' selected':''} style="font-family:${f[1].replace(/"/g,"'")}">${f[0]}</option>`).join('');
  return card('Наборы', `<div class="presets">${AP_PRE.map(([n,p],i)=>presetCard(n,p,false,i)).join('')}${(UI.presets||[]).map((x,i)=>presetCard(x.n,x.a,true,i)).join('')}</div>`,
      `<button class="btn sm" id="ap-save">${ic('plus')}Сохранить текущее оформление</button><button class="btn sm quiet" id="ap-exp">${ic('copy')}Экспорт</button><button class="btn sm quiet" id="ap-imp">Импорт</button>`)
    + card('Цвет', row('Тема', segH('ap-theme',[['dark','Тёмная'],['light','Светлая']],AP.theme))
      + row('Акцент', `<div class="swatches" id="ap-acc">${acc.map(c=>`<button data-c="${c}" style="background:${c};color:${c}" aria-checked="${c.toLowerCase()===String(AP.accent).toLowerCase()}" title="${c}"></button>`).join('')}<input type="color" class="sw-c" id="ap-accc" value="${/^#[0-9a-f]{6}$/i.test(AP.accent)?AP.accent:'#0a84ff'}" title="Собственный цвет"></div>`))
    + card('Фон', `<div class="presets" style="grid-template-columns:repeat(auto-fill,minmax(110px,1fr))" id="ap-bgs">${bgs}</div><div style="margin-top:12px">`
      + (AP.bg==='solid'? row('Собственный цвет', `<input type="color" class="sw-c" id="ap-solid" value="${AP.solid}">`) : '')
      + (AP.bg==='image'&&AP.img? row('Изображение', `<button class="btn sm quiet" id="ap-imgx">Убрать</button>`, 'Выбирается в окне программы.')
      + row('Затемнение изображения', rngH('ap-dim',0,.85,.05,AP.dim,Math.round(AP.dim*100)+' %'))
      + row('Размытие изображения', rngH('ap-iblur',0,40,1,AP.iblur,AP.iblur+' px')) : '') + '</div>')
    + card('Поверхности', row('Непрозрачность', rngH('ap-glass',0,1,.02,AP.glass,Math.round(AP.glass*100)+' %'), 'Относится к карточкам и панелям.')
      + row('Размытие под стеклом', rngH('ap-blur',0,60,1,AP.blur,AP.blur+' px'))
      + row('Тени', segH('ap-shadow',[[0,'Нет'],[1,'Мягкие'],[2,'Глубокие']],AP.shadow))
      + row('Линии', segH('ap-line',[[0,'Нет'],[.5,'Тонкие'],[1,'Обычные'],[1.5,'Чёткие']],AP.line))
      + row('Скругление', rngH('ap-rad',0,28,1,AP.radius,AP.radius+' px')))
    + card('Текст', row('Шрифт', `<select class="inp" id="ap-ff" style="max-width:260px">${fonts}</select>`)
      + row('Размер', rngH('ap-fs',12,17,.5,AP.fs,dec(AP.fs)+' px'))
      + row('Начертание', segH('ap-wt',[[300,'Лёгкое'],[400,'Обычное'],[500,'Плотное']],AP.weight)))
    + card('Компоновка', row('Плотность', segH('ap-den',[[.85,'Плотно'],[1,'Обычно'],[1.15,'Просторно']],AP.density))
      + row('Панель', segH('ap-side',[['left','Слева'],['right','Справа']],AP.side))
      + row('Ширина панели', rngH('ap-sidew',200,340,4,AP.sidew,AP.sidew+' px'))
      + row('Ширина содержимого', segH('ap-wide',[[0,'Ограничена'],[1,'Во всю ширину']],AP.wide))
      + row('Анимации', swH('ap-motion',+AP.motion)))
    + card('Собственный CSS', `<textarea class="inp code" id="ap-css" spellcheck="false" placeholder=".card { border-radius: 4px }">${esc(AP.css||'')}</textarea>
        <p class="hint" style="margin-top:8px">Доступные переменные: --accent, --card, --mat, --r, --fs, --ff, --wall.</p>`,
        `<button class="btn sm quiet" id="ap-cssx">Очистить</button>`)
    + card('Сброс оформления', row('Вернуть оформление по умолчанию', `<button class="btn danger" id="ap-reset">Сбросить</button>`));
}
function bindLook(b){
  const q=s=>b.querySelector(s);
  b.querySelectorAll('.pcard[data-pre]').forEach(c=>c.onclick=e=>{
    const d=e.target.closest('[data-del]'); if(d){ e.stopPropagation(); const i=+d.dataset.del; UI.presets.splice(i,1); uiSet({presets:UI.presets},true); return; }
    const k=c.dataset.pre, p= k[0]==='u'? UI.presets[+k.slice(1)].a : AP_PRE[+k.slice(1)][1];
    apSet(Object.assign({},AP_DEF,{img:AP.img,css:AP.css,sidew:AP.sidew,side:AP.side},p)); });
  q('#ap-save').onclick=async()=>{ const n=await dialog({title:'Сохранить набор', input:{ph:'Название'}, ok:'Сохранить'}); if(!n) return;
    const a={}; Object.keys(AP_DEF).forEach(k=>{ if(k!=='img'&&k!=='css') a[k]=AP[k]; }); UI.presets=(UI.presets||[]).concat([{n:n.trim().slice(0,40), a}]); uiSet({presets:UI.presets},true); toast('Набор сохранён', n,'ok'); };
  q('#ap-exp').onclick=()=>copyText(JSON.stringify(AP), 'Оформление скопировано');
  q('#ap-imp').onclick=async()=>{ const t=await dialog({title:'Импорт оформления', text:'Требуется вставить JSON, скопированный кнопкой «Экспорт».', input:{area:true, mono:true, ph:'{"theme":"dark", …}'}, ok:'Применить'});
    if(!t) return; try{ const a=JSON.parse(t); if(typeof a!=='object'||!a) throw 0; apSet(Object.assign({},AP_DEF,a)); toast('Оформление применено','', 'ok'); }catch(e){ toast('Некорректный JSON','', 'bad'); } };
  const seg=(id,k,num)=>{ const s=q('#'+id); if(s) s.onclick=e=>{ const x=e.target.closest('[data-v]'); if(x) apSet({[k]: num? +x.dataset.v : x.dataset.v}); }; };
  seg('ap-theme','theme'); seg('ap-shadow','shadow',1); seg('ap-line','line',1); seg('ap-wt','weight',1); seg('ap-den','density',1); seg('ap-side','side'); seg('ap-wide','wide',1);
  q('#ap-acc').onclick=e=>{ const x=e.target.closest('[data-c]'); if(x) apSet({accent:x.dataset.c}); };
  q('#ap-accc').oninput=e=>apSet({accent:e.target.value}, true);
  q('#ap-accc').onchange=()=>settingsRender(true);
  q('#ap-bgs').onclick=e=>{ const x=e.target.closest('[data-bg]'); if(x) apSet({bg:x.dataset.bg}); };
  if(q('#ap-solid')) q('#ap-solid').oninput=e=>apSet({bg:'solid',solid:e.target.value}, true);
  if(q('#ap-imgx')) q('#ap-imgx').onclick=()=>{ post('/api/settings',{свои:{BG:''}}); apSet({img:0,bg:AP.bg==='image'?AP_DEF.bg:AP.bg}); };
  const rng=(id,k,f)=>{ const r=q('#'+id); if(!r) return; r.oninput=()=>{ rangeFill(r); q('#'+id+'-o').textContent=f(+r.value); apSet({[k]:+r.value}, true); }; };
  rng('ap-dim','dim',v=>Math.round(v*100)+' %'); rng('ap-iblur','iblur',v=>v+' px'); rng('ap-glass','glass',v=>Math.round(v*100)+' %'); rng('ap-blur','blur',v=>v+' px');
  rng('ap-rad','radius',v=>v+' px'); rng('ap-fs','fs',v=>dec(v)+' px'); rng('ap-sidew','sidew',v=>v+' px');
  q('#ap-ff').onchange=e=>apSet({ff:e.target.value});
  q('#ap-motion').onchange=e=>apSet({motion:e.target.checked?1:0}, true);
  let cssT=0; q('#ap-css').oninput=e=>{ clearTimeout(cssT); cssT=setTimeout(()=>apSet({css:e.target.value}, true), 250); };
  q('#ap-cssx').onclick=()=>{ q('#ap-css').value=''; apSet({css:''}, true); };
  q('#ap-reset').onclick=async()=>{ if(await dialog({title:'Сбросить оформление?', text:'Собственные наборы сохраняются.', ok:'Сбросить', danger:true})) apSet(Object.assign({},AP_DEF)); };
}
function secCharts(){
  const on=new Set(chartOrder()), all=chartOrder().concat(ALL_CHARTS.filter(k=>!on.has(k)));
  return card('Плитки на экране «Ход обучения»', `<div class="clist" id="ch-list">${all.map((k,i)=>`<div class="citem" data-k="${k}">${swH('chk-'+k,on.has(k))}<span>${WIN[k].title}</span>
      <button class="btn icon sm quiet" data-mv="-1" title="Выше"${i?'':' disabled'}>${ic('up')}</button><button class="btn icon sm quiet" data-mv="1" title="Ниже"${i<all.length-1?'':' disabled'}>${ic('down')}</button></div>`).join('')}</div>`)
    + card('Вид графиков', row('Колонок', segH('ch-cols',[[0,'Авто'],[1,'1'],[2,'2'],[3,'3'],[4,'4']],UI.ccols||0))
      + row('Высота графика', rngH('ch-h',100,280,10,UI.ch||150,(UI.ch||150)+' px'))
      + row('Толщина линии', rngH('ch-lw',1,3.5,.25,AP.lw,dec(AP.lw)+' px'))
      + row('Заливка под кривой', swH('ch-fill',+AP.cfill)) + row('Сетка', swH('ch-grid',+AP.cgrid)) + row('Точки валидации', swH('ch-dots',+AP.cdots)));
}
function bindCharts(b){
  const list=b.querySelector('#ch-list'), cur_=()=>[...list.querySelectorAll('.citem')].filter(x=>x.querySelector('.sw').checked).map(x=>x.dataset.k);
  list.addEventListener('change', ()=>{ const a=cur_(); uiSet({charts: a.length? a : ['loss']}, true); });
  list.addEventListener('click', e=>{ const m=e.target.closest('[data-mv]'); if(!m) return; const it=m.closest('.citem'), d=+m.dataset.mv, sib= d<0? it.previousElementSibling : it.nextElementSibling;
    if(!sib) return; d<0? list.insertBefore(it,sib) : list.insertBefore(sib,it); uiSet({charts:cur_()}, true); });
  b.querySelector('#ch-cols').onclick=e=>{ const x=e.target.closest('[data-v]'); if(x) uiSet({ccols:+x.dataset.v}, true); };
  const r=b.querySelector('#ch-h'); r.oninput=()=>{ rangeFill(r); b.querySelector('#ch-h-o').textContent=r.value+' px'; uiSet({ch:+r.value}); };
  const l=b.querySelector('#ch-lw'); l.oninput=()=>{ rangeFill(l); b.querySelector('#ch-lw-o').textContent=dec(l.value)+' px'; apSet({lw:+l.value}, true); };
  [['ch-fill','cfill'],['ch-grid','cgrid'],['ch-dots','cdots']].forEach(([id,k])=>b.querySelector('#'+id).onchange=e=>apSet({[k]:e.target.checked?1:0}, true));
}
function secBehave(){
  return card('Поведение', row('Стартовый экран', segH('ui-start',[['auto','Авто'],['setup','Новое обучение'],['train','Ход обучения'],['runs','Запуски']],UI.start), 'Режим «Авто» открывает ход обучения, если обучение выполняется.')
    + row('Уведомление по окончании', swH('ui-notify',+UI.notify), 'Уведомление на рабочем столе, в том числе при закрытом окне.')
    + row('В трей при закрытии', swH('ui-tray',+UI.tray), 'Действует, пока выполняется обучение.')
    + row('Подтверждать остановку', swH('ui-confirm',+UI.confirmStop))
    + row('Десятичный знак', segH('ui-dec',[['dot','0.5137'],['comma','0,5137']],UI.dec)));
}
function bindBehave(b){
  b.querySelector('#ui-start').onclick=e=>{ const x=e.target.closest('[data-v]'); if(x) uiSet({start:x.dataset.v}, true); };
  b.querySelector('#ui-dec').onclick=e=>{ const x=e.target.closest('[data-v]'); if(x){ uiSet({dec:x.dataset.v}, true); refreshH(); } };
  [['ui-notify','notify'],['ui-tray','tray'],['ui-confirm','confirmStop']].forEach(([id,k])=>b.querySelector('#'+id).onchange=e=>uiSet({[k]:e.target.checked?1:0}));
}
function secChat(){
  return card('Разговор по умолчанию', row('Температура', `<input class="inp" type="number" id="cd-temp" min="0" max="2" step="0.05" value="${UI.chatTemp}" style="max-width:120px">`, 'При значении 0 всегда выбирается самый вероятный ответ.')
    + row('Длина ответа', `<input class="inp" type="number" id="cd-max" min="1" max="2048" value="${UI.chatMax}" style="max-width:120px">`, 'В токенах.')
    + row('Сила обучения', `<input class="inp" type="number" id="cd-scale" min="0" max="3" step="0.1" value="${UI.chatScale}" style="max-width:120px">`, '1 — результат обучения без изменений, 0 — исходная модель.')
    + row('Системная инструкция', `<input class="inp" id="cd-scene" value="${esc(UI.chatScene)}" placeholder="Текст системной инструкции">`));
}
function bindChat(b){
  [['cd-temp','chatTemp',1],['cd-max','chatMax',1],['cd-scale','chatScale',1],['cd-scene','chatScene',0]].forEach(([id,k,n])=>b.querySelector('#'+id).onchange=e=>{
    uiSet({[k]: n? +e.target.value : e.target.value}); const m={chatTemp:'#c-temp',chatMax:'#c-max',chatScale:'#c-scale',chatScene:'#c-scene'}[k]; if($(m)) $(m).value=UI[k]; });
}
const CFG_HW=[['mem_cap_pct','Потолок памяти, %','Доля памяти всей машины, выше которой движок не занимает ОЗУ и видеопамять. Допустимы значения от 50 до 99, по умолчанию 98. При нехватке ОЗУ обучение сохраняет снимок весов и останавливается.'],['headroom_mib','Запас видеопамяти, МиБ','Видеопамять, резервируемая под промежуточные данные шага.'],['tier1_params','Бюджет среза строк','Количество обучаемых чисел в видеопамяти: вес и два момента Adam, 12 байт на число.'],
  ['bf16x3','FP16×3 / BF16×3','Сначала применяется FP16×3; если проба не пройдена, используется BF16×3 при успешной пробе, иначе FP32.',1],['prefetch','Опережающая подкачка следующего слоя','Следующий слой передаётся по шине, пока считается текущий.',1]];
function secHw(){
  const g=(H&&H.гпу)||{}, o=(H&&H.озу)||{}, v=(H&&H.видеопамять)||{}, dk=(H&&H.диск)||{};
  const facts=`<div class="tiles">
    <div class="tile"><span class="tl">Карта</span><span class="tv" style="font-size:calc(var(--fs) + 1px)">${esc(g.название||g.архитектура||'—')}</span><span class="tn">${esc(g.архитектура||'')}</span></div>
    <div class="tile"><span class="tl">Видеопамять</span><span class="tv">${v.всего? gib(v.всего,1) : g.видеопамять? gib(g.видеопамять*2**20,1) : '—'}</span><span class="tn">${v.всего? 'Занято '+gib(v.занято,1) : ''}</span></div>
    <div class="tile"><span class="tl">ОЗУ</span><span class="tv">${gib(o.всего,1)}</span><span class="tn">Доступно ${gib(o.доступно,1)}</span></div>
    <div class="tile"><span class="tl">Шина</span><span class="tv">${(g.шина||{}).поколение? `PCIe ${g.шина.поколение}.0` : '—'}</span><span class="tn">${(g.шина||{}).линий? '×'+g.шина.линий : ''}</span></div>
    <div class="tile"><span class="tl">Диск</span><span class="tv">${bytes(dk.свободно)}</span><span class="tn">Свободно из ${bytes(dk.всего)}</span></div></div>`;
  return card('Машина', facts)
    + card('Настройка оборудования', `<div id="hw-rows"><p class="hint">Загрузка</p></div>`, `<button class="btn sm primary" id="hw-save">Сохранить</button>`)
    + card('Движок', row('Состояние', `<span class="pill ${H&&H.движок?'ok':'warn'}"><i></i>${H&&H.движок? 'Собран'+(g.архитектура?' · '+esc(g.архитектура):'') : H&&H.движок_есть? 'Собран из прежней версии' : 'Не собран'}</span>`)
      + row('Сборка', `<button class="btn" id="hw-build"${H&&H.компилятор?'':' disabled'}>${ic('build')}Пересобрать</button><span class="faint">${esc((H&&H.компилятор)||'Компилятор отсутствует')}</span>`, 'Занимает несколько минут.'));
}
async function bindHw(b){
  cfg=await getj('/api/settings').catch(()=>null); if(!cfg) return;
  b.querySelector('#hw-rows').innerHTML=CFG_HW.map(([k,l,s,sw])=>row(l, sw? `<input type="checkbox" class="sw" id="h_${k}" data-h="${k}"${String(cfg.железо[k]??'0')!=='0'?' checked':''}>`
    : `<input class="inp" id="h_${k}" data-h="${k}" value="${esc(cfg.железо[k]??'')}" style="max-width:220px">`, s)).join('');
  b.querySelector('#hw-save').onclick=async()=>{ const железо={}; b.querySelectorAll('[data-h]').forEach(i=>{ if(i.type==='checkbox') железо[i.dataset.h]=i.checked?'1':'0'; else if(i.value.trim()) железо[i.dataset.h]=i.value.trim(); });
    const r=await post('/api/settings',{свои:{}, железо}); r.ок? toast('Сохранено','Действует со следующего запуска движка.','ok') : toast('Не удалось сохранить', cfg.пути['настройка оборудования'], 'bad'); refreshH(); };
  b.querySelector('#hw-build').onclick=()=>setupRun('движок');
}
function secPaths(){
  return card('Расположение файлов', `<div id="p-rows"><p class="hint">Загрузка</p></div>`, `<button class="btn sm primary" id="p-save">Сохранить</button>`)
    + card('Программа', `<div id="p-prog"></div>`);
}
async function bindPaths(b){
  cfg=await getj('/api/settings').catch(()=>null); if(!cfg) return;
  const P_=[['MODEL','Папка модели','Расположение файла model.safetensors. Используется только для чтения.',1],['WORK','Рабочая папка','Журналы, токены, обученные веса.',1],['HF_REPO','Источник загрузки весов','Hugging Face',0]];
  b.querySelector('#p-rows').innerHTML=P_.map(([k,l,s,pick])=>row(l, `<input class="inp" id="p_${k}" data-p="${k}" value="${esc(cfg.свои[k]??'')}" spellcheck="false">`, s)).join('');
  b.querySelector('#p-prog').innerHTML=Object.entries(cfg.пути).map(([k,v])=>row(cap(k), `<code class="faint" style="overflow-wrap:anywhere">${esc(v)}</code>`)).join('');
  b.querySelector('#p-save').onclick=async()=>{ const свои={}; b.querySelectorAll('[data-p]').forEach(i=>свои[i.dataset.p]=i.value.trim());
    await post('/api/settings',{свои}); toast('Сохранено','','ok'); await refreshH(); listen(); replan(); };
}
const KEYS=[['Ctrl+K','Команды и поиск'],['Ctrl+1 … Ctrl+6','Разделы по порядку'],['Ctrl+,','Настройки'],['Ctrl+Enter','Начать обучение'],['Ctrl+Shift+L','Сменить тему'],
  ['1 … 6','Плитки графиков на экране «Ход обучения»'],['J','Журнал запуска'],['/','Поиск по запускам'],['Esc','Закрыть окно или диалог'],['F11','Во весь экран'],['Ctrl+Q','Завершить работу программы']];
function secKeys(){ return card('Клавиши', KEYS.map(([k,t])=>row(t, k.split(' … ').map(x=>x.split('+').map(y=>`<kbd>${esc(y)}</kbd>`).join('+')).join(' … '))).join('')); }
function secAbout(){
  return card('', `<div class="about"><svg class="mark" viewBox="0 0 64 64" aria-hidden="true"><g stroke-linejoin="round" stroke-width="2.4"><polygon class="l1" points="7,18 32,5.5 57,18 32,30.5"/><polygon class="l2" points="15.5,29.75 32,38 57,25.5 57,31 32,43.5 15.5,35.25"/><polygon class="l3" points="7,38 32,50.5 49,42 55,45 32,56.5 7,45"/></g></svg>
      <div><div class="mname">TierLM ${esc((H&&H.версия)||'')}</div><p class="mtag">Дообучение моделей, которые не помещаются в видеопамять</p><p class="mline">Веса хранятся в ОЗУ и подаются на видеокарту послойно с опережением, поэтому модель 12B в BF16 обучается на 8 ГБ. Изменяются собственные веса модели, а не только адаптер. Поддерживаются AMD и NVIDIA.</p></div></div>`)
    + card('Связь с автором', CONTACTS.map(([n,t,u])=>row(n, `<button class="btn sm quiet" data-copy="${esc(t)}" title="Скопировать">${ic('copy')}${esc(t)}</button>`
        + (u? `<a class="btn sm quiet" href="${u}" target="_blank" rel="noopener" title="Открыть в браузере">${ic('open')}</a>` : ''))).join(''))
    + card('Шрифты', '<p class="hint">Inter, Golos Text, Onest, Manrope, Jost, Geologica — SIL Open Font License 1.1.</p>');
}
const CONTACTS=[['Telegram','@So_bad_catty','https://t.me/So_bad_catty'],['ВКонтакте','Eberhard Gottman',''],['Discord','so_bad_cat','']];
document.addEventListener('click', e=>{ const b=e.target.closest('[data-copy]'); if(b) copyText(b.dataset.copy, 'Скопировано · '+b.dataset.copy); });

// ============================================================================
// Палитра команд: всё, что умеет программа, — по названию
// ============================================================================
function commands(){
  const C=[];
  VKEYS.forEach((k,i)=>C.push({g:'Разделы', t:VIEWS[k], i:{setup:'setup',train:'train',runs:'runs',chat:'chat',bench:'bench',settings:'settings'}[k], k:'Ctrl+'+(i+1), f:()=>go(k)}));
  SECS.forEach(([k,t,i])=>C.push({g:'Настройки', t:'Настройки: '+t, i, f:()=>{ sec=k; go('settings'); }}));
  if(H && !blocking(checks()).length) C.push({g:'Действия', t:'Начать обучение', i:'play', k:'Ctrl+Enter', f:launch});
  if(liveRun()) C.push({g:'Действия', t:`Остановить «${liveRun()}»`, i:'stop', f:()=>{ cur=liveRun(); listen(); go('train'); setTimeout(stopRun,300); }});
  C.push({g:'Действия', t: AP.theme==='light'? 'Тёмная тема' : 'Светлая тема', i:'theme', k:'Ctrl+Shift+L', f:()=>apSet({theme:AP.theme==='light'?'dark':'light'})});
  C.push({g:'Действия', t: LANG==='en'? 'Русский язык' : 'English', i:'globe', f:()=>setLang(LANG==='en'?'ru':'en')});
  C.push({g:'Действия', t:'Собрать движок', i:'build', f:()=>setupRun('движок')});
  C.push({g:'Действия', t:'Завершить работу программы', i:'power', k:'Ctrl+Q', f:killAll});
  AP_PRE.forEach(([n,p])=>C.push({g:'Оформление', t:'Набор «'+n+'»', i:'palette', f:()=>apSet(Object.assign({},AP_DEF,{img:AP.img,css:AP.css},p))}));
  (UI.presets||[]).forEach(x=>C.push({g:'Оформление', t:'Собственный набор «'+x.n+'»', i:'palette', f:()=>apSet(Object.assign({},AP_DEF,x.a))}));
  (H&&H.запуски||[]).forEach(r=>{ C.push({g:'Запуски', t:'Открыть «'+r.имя+'»', i:'open', s:STATE_WORD[r.состояние]||'', f:()=>openRun(r.имя)});
    if(r.веса) C.push({g:'Запуски', t:'Разговор с «'+r.имя+'»', i:'chat', f:()=>chatWith(r.имя)}); });
  return C;
}
function palette(){
  if(layers.some(l=>l.el.classList.contains('topal'))) return;
  const s=document.createElement('div'); s.className='scrim topal';
  s.innerHTML=`<div class="pop palette" role="dialog" aria-modal="true" aria-label="Команды"><div class="pin">${ic('search')}<input placeholder="Команда, раздел или запуск" spellcheck="false" autocomplete="off"><kbd>Esc</kbd></div><div class="plist" role="listbox"></div></div>`;
  const L=pushLayer(s, ()=>{}), inp=s.querySelector('input'), list=s.querySelector('.plist'), all=commands(); let sel=0, shown=[];
  const score=(t,q)=>{ t=t.toLowerCase(); if(!q) return 1; if(t.includes(q)) return 3-(t.indexOf(q)>0?1:0); let i=0; for(const c of t){ if(c===q[i]) i++; if(i===q.length) return 1; } return 0; };
  const draw=()=>{ const q=inp.value.trim().toLowerCase(); shown=all.map(c=>({c, s:score(T(c.t),q)||score(c.t,q)})).filter(x=>x.s>0).sort((a,b)=>b.s-a.s).map(x=>x.c);
    if(q==='' ) shown=all; sel=Math.min(sel, Math.max(0,shown.length-1)); let g='', h='';
    shown.slice(0,60).forEach((c,i)=>{ if(c.g!==g){ g=c.g; h+=`<div class="pg">${c.g}</div>`; } h+=`<button class="pi" role="option" data-i="${i}" aria-selected="${i===sel}">${ic(c.i||'arrow')}<span>${esc(c.t)}</span>${c.s?`<em>${esc(c.s)}</em>`:''}${c.k?`<em>${esc(c.k)}</em>`:''}</button>`; });
    list.innerHTML=h||'<div class="pnone">Ничего не найдено</div>'; const a=list.querySelector('[aria-selected="true"]'); if(a) a.scrollIntoView({block:'nearest'}); };
  const run=i=>{ const c=shown[i]; if(!c) return; L.close(); setTimeout(c.f, 10); };
  inp.addEventListener('input', ()=>{ sel=0; draw(); });
  inp.addEventListener('keydown', e=>{ if(e.key==='ArrowDown'){ e.preventDefault(); sel=Math.min(shown.length-1,sel+1); draw(); } else if(e.key==='ArrowUp'){ e.preventDefault(); sel=Math.max(0,sel-1); draw(); }
    else if(e.key==='Enter'){ e.preventDefault(); run(sel); } });
  list.addEventListener('click', e=>{ const b=e.target.closest('[data-i]'); if(b) run(+b.dataset.i); });
  s.addEventListener('pointerdown', e=>{ if(e.target===s) L.close(); });
  draw(); setTimeout(()=>inp.focus(), 10);
}

// ============================================================================
// Окно, язык, тема, завершение, клавиши, запуск
// ============================================================================
async function setLang(l){ if(l===window.LANG) return; await post('/api/settings',{свои:{LANG:l}}); location.reload(); }
async function killAll(){
  const live=liveRun();
  const yes=await dialog({title:'Завершить работу программы?', text: live? `Обучение «${live}» будет остановлено, программа будет закрыта полностью.` : 'Программа будет закрыта полностью.', ok:'Завершить', danger:true});
  if(!yes) return;
  try{ await post('/api/kill'); }catch(e){}
  document.body.innerHTML='<p style="margin:44vh auto;text-align:center;color:#888;font-family:sans-serif">Работа TierLM завершена</p>';
}
$$('#lang button').forEach(b=>{ b.setAttribute('aria-checked',b.dataset.l===window.LANG); b.onclick=()=>setLang(b.dataset.l); });
$('#th').onclick=()=>apSet({theme:AP.theme==='light'?'dark':'light'});
$('#kill').onclick=killAll;
$('#cmdk').onclick=palette;
document.addEventListener('click', e=>{
  const t=e.target.closest('[data-go],[data-act],[data-n]'); if(!t || t.disabled) return;
  if(t.dataset.go){ if(t.dataset.go==='train'){ const l=liveRun(); if(l && cur!==l && (t.id==='live'||!picked)){ cur=l; listen(); } } return go(t.dataset.go); }
  if(t.dataset.act) return act(t.dataset.act);
  const n=t.dataset.n;
  if(n==='resume') resume(); else if(n==='chat') chatWith(state&&state.имя||cur); else if(n==='bench'){ go('bench'); $('#b-run').value=state&&state.имя||''; benchModels(); }
  else if(n==='plan') openWin('plan', $('#n-chips'));
});
$('#n-act').addEventListener('click', e=>{ if(e.target.closest('#n-stop')) stopRun(); });
$('#m-folder').onclick=pickModel;
$('#m-modes').addEventListener('click', e=>{ const b=e.target.closest('[data-mode]'); if(b) setMode(b.dataset.mode); });
$('#m-goals').addEventListener('click', e=>{ const b=e.target.closest('[data-goal]'); if(b) setGoal(b.dataset.goal); });
$('#tab').addEventListener('click', e=>{ const b=e.target.closest('[data-tab]'); if(b) setTab(b.dataset.tab); });
for(const [id,k] of [['#lo-r','r'],['#lo-d','drop'],['#lo-set','tg']]) $(id).addEventListener('click', e=>{ const b=e.target.closest('[data-v]'); if(b) setLora(k, b.dataset.v); });
$('#lo-m').addEventListener('click', e=>{ const b=e.target.closest('[data-m]'); if(b) setLora('mat', b.dataset.m); });
$('#lo-def').onclick=()=>setLora('def');
$('#lo-a').addEventListener('input', ()=>{ const t=$('#lo-a'), v=t.value.replace(/[^0-9]/g,'').slice(0,4); if(v!==t.value) t.value=v; });
$('#lo-a').addEventListener('change', ()=>{ setLora('alpha', $('#lo-a').value); $('#lo-a').value=String(+ch.lora.alpha); });
$('#lo-a').addEventListener('keydown', e=>{ if(e.key==='Enter') $('#lo-a').blur(); });
function arrows(group, attr, set){
  $(group).addEventListener('keydown', e=>{ const d={ArrowRight:1,ArrowDown:1,ArrowLeft:-1,ArrowUp:-1}[e.key]; if(!d) return;
    const bs=[...$(group).querySelectorAll(`[${attr}]`)], i=bs.indexOf(document.activeElement); if(i<0) return;
    e.preventDefault(); const n=bs[(i+d+bs.length)%bs.length]; set(n.getAttribute(attr)); n.focus(); });
}
arrows('#m-modes','data-mode', setMode); arrows('#m-goals','data-goal', setGoal);
$('#d-file').addEventListener('keydown', e=>{ if(e.key==='Enter') setFile($('#d-file').value); });
$('#d-file').addEventListener('change', ()=>{ if($('#d-file').value.trim()!==ch.file) setFile($('#d-file').value); });
$('#d-prev').addEventListener('toggle', ()=>{ if($('#d-prev').open) loadExamples(); });
for(const id of ['#d-share','#d-always']){ $(id).addEventListener('change', setScene); $(id).addEventListener('keydown', e=>{ if(e.key==='Enter') setScene(); }); }
$('#d-share').addEventListener('input', ()=>{ const t=$('#d-share'), v=t.value.replace(/[^0-9]/g,'').slice(0,3); if(v!==t.value) t.value=v; });
let nameT=0;
$('#d-name').addEventListener('input', ()=>{ ch.name=$('#d-name').value.trim(); ch.nameAuto=!ch.name; launchErr='';
  if(!ch.name && ch.file) ch.name=freeName(base(ch.file)); clearTimeout(nameT); nameT=setTimeout(replan, 250); });
$('#l-go').onclick=launch;
$('#l-expert').onclick=()=>{ if(ch.mode==='эксперт') act('reset'); else setMode('эксперт'); };
$('#l-hwbtn').onclick=e=>{ e.preventDefault(); sec='hw'; go('settings'); };
$('#run').onchange=()=>{ cur=$('#run').value; picked=true; actErr=''; state=null; listen(); };
$('#log').onclick=()=>openWin('log',$('#log'));
$('#log').onkeydown=e=>{ if(e.key==='Enter'||e.key===' '){ e.preventDefault(); openWin('log',$('#log')); } };
runsBind(); chatBind(); benchBind();
addEventListener('keydown', e=>{
  const k=e.key, mod=e.ctrlKey||e.metaKey;
  if(k==='Escape'){ const L=topLayer(); if(L){ e.preventDefault(); L.close(); } return; }
  if(mod && (k==='k'||k==='K'||k==='л'||k==='Л'||e.code==='KeyK'||e.code==='KeyP')){ e.preventDefault(); palette(); return; }
  if(mod && (k==='q'||k==='й'||e.code==='KeyQ')){ e.preventDefault(); killAll(); return; }
  if(mod && e.shiftKey && e.code==='KeyL'){ e.preventDefault(); apSet({theme:AP.theme==='light'?'dark':'light'}); return; }
  if(mod && (k===','||e.code==='Comma')){ e.preventDefault(); go('settings'); return; }
  if(mod && k==='Enter' && view==='setup'){ e.preventDefault(); launch(); return; }
  if(mod && /^[1-6]$/.test(k)){ e.preventDefault(); go(VKEYS[+k-1]); return; }
  if(layers.length || e.target.closest('input,select,textarea') || mod || e.altKey) return;
  if(view==='train' && state){ const n=parseInt(k,10), o=chartOrder(); if(n>=1 && n<=o.length){ openWin(o[n-1], $(`.plot[data-win="${o[n-1]}"]`)); return; }
    if(k==='j'||k==='о'){ openWin('log',$('#log')); return; } }
  if(view==='runs' && k==='/'){ e.preventDefault(); $('#r-q').focus(); }
});
// Окно меняет размер — кривые перечерчиваются под новый.
let raf=0;
new ResizeObserver(()=>{ cancelAnimationFrame(raf); raf=requestAnimationFrame(()=>{
  if(view==='train' && state){ plots(state); sweepView(state); } if(winOpen) fillWin(); if(view==='runs' && RS.cmp) drawCompare(); }); }).observe($('#views'));
setInterval(()=>{ if(!document.hidden) refreshH(); }, 4000);
(async function init(){
  const n0=navN;
  try{ const c=await getj('/api/settings'), o=c.свои||{}; if(o.MODE) ch.mode=o.MODE; ch.tab= o.TAB==='lora'? 'lora' : 'tier'; if(o.LORA) ch.lora=loraParse(o.LORA); }catch(e){}
  await refreshH();
  if(H && H.идёт && !picked) cur=H.идёт;
  listen();
  let v=UI.start||'auto'; if(v==='auto') v= H&&H.идёт? 'train' : 'setup';
  if(navN===n0) go(v); else go(view);     // пока шёл первый опрос, раздел уже выбрали — его и оставляем
  if(H && H.настройка) setupListen();
  replan();
})();
</script></body></html>
"""


if __name__ == '__main__':
    main()
