#!/usr/bin/env python3
# а1
import concurrent.futures, hashlib, http.client, json, os, re, shutil, struct, sys, tempfile, threading, time, urllib.error, urllib.parse, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import arch, tmpl
try:
    import hwprobe                                                  # а2
except Exception:
    hwprobe = None

# а3
KNOWN_FILE = os.path.join(HERE, 'hub_known.json')
# а4
SEED = [
    ('google/gemma-4-12B-it', 'карта', 'Основная модель TierLM'),
    ('Qwen/Qwen3-8B', 'формулы', 'Рассуждения отключаются шаблоном'),
    ('Qwen/Qwen3-4B', 'формулы', 'Быстрая проверка идей'),
    ('Qwen/Qwen3-1.7B', 'формулы', 'Целиком в видеопамяти'),
]
# а5
SORTS = {'downloads': 'downloads', 'likes': 'likes', 'trending': 'trendingScore', 'modified': 'lastModified', 'created': 'createdAt'}
TTL = {'downloads': 86400, 'likes': 86400, 'created': 86400, 'trending': 3600, 'modified': 3600}     # а6
# а7
TASKS = ('text-generation', 'image-text-to-text', 'any-to-any')
EXPAND = ('gated', 'disabled', 'safetensors', 'downloads', 'likes', 'lastModified', 'createdAt', 'trendingScore', 'config', 'pipeline_tag')
KEEP = ('config.json', 'generation_config.json', 'tokenizer.json', 'tokenizer_config.json', 'special_tokens_map.json',
        'chat_template.jinja', 'chat_template.json', 'model.safetensors.index.json')
HEAD1 = 1 << 17                                  # а8
HF = (os.environ.get('HF_ENDPOINT') or 'https://huggingface.co').rstrip('/')
NETERR = (urllib.error.URLError, OSError, ValueError, http.client.HTTPException)   # а9


def repo_of(s):
    # б1
    s = (s or '').strip()
    m = re.match(r'^(?:https?://)?(?:www\.)?(?:huggingface\.co|hf\.co)/(?:models/)?([^/\s?#]+/[^/\s?#]+)', s)
    r = m.group(1) if m else s
    r = r.rstrip('/')
    return r if re.fullmatch(r'[\w.\-]+/[\w.\-]+', r) else ''


def wanted(f):
    # в1
    return '/' not in f and (f in KEEP or (f.endswith('.safetensors') and not f.startswith('consolidated')))


_DIRECT = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def _req(url, token, rng=None, timeout=20):
    h = {'User-Agent': 'TierLM'}
    if token:
        h['Authorization'] = 'Bearer ' + token
    if rng:
        h['Range'] = f'bytes={rng[0]}-{rng[1]}'
    rq = urllib.request.Request(url, headers=h)
    try:
        return urllib.request.urlopen(rq, timeout=timeout)
    except urllib.error.HTTPError:
        raise
    except (urllib.error.URLError, OSError):
        # г1
        if not any((os.environ.get(v) or '').startswith('https://') for v in ('HTTPS_PROXY', 'https_proxy', 'HTTP_PROXY', 'http_proxy')): raise
        return _DIRECT.open(urllib.request.Request(url, headers=h), timeout=timeout)   # г2


def _json(url, token):
    with _req(url, token) as r:
        return json.loads(r.read())


def _why_http(e, repo):
    if isinstance(e, urllib.error.HTTPError):
        if e.code in (401, 403):
            return f'{repo}: Закрытая модель. Требуется согласие на странице модели и ключ Hugging Face в настройках.'
        if e.code == 404:
            return f'{repo}: Модель не найдена на Hugging Face.'
        return f'{repo}: Hugging Face вернул код {e.code}.'
    return f'{repo}: Нет связи с Hugging Face ({getattr(e, "reason", e)}).'


def st_header_remote(repo, fn, token, rev='main'):
    # ж1
    url = f'{HF}/{repo}/resolve/{rev}/{urllib.parse.quote(fn)}'
    with _req(url, token, (0, HEAD1 - 1)) as r:                       # ж2
        buf = r.read(HEAD1)                                            # ж3
    n = struct.unpack('<Q', buf[:8])[0]
    if not 2 <= n <= 200 << 20:
        raise ValueError(f'{fn}: недопустимая длина заголовка {n}')
    if len(buf) >= 8 + n:
        h = json.loads(buf[8:8 + n])
    else:
        with _req(url, token, (8, 7 + n), timeout=60) as r:
            h = json.loads(r.read(n))
    h.pop('__metadata__', None)
    return {k: (v['dtype'], v['shape']) for k, v in h.items()}


def cache_dir():
    d = os.path.join(os.environ.get('TIER_DATA') or os.path.join(os.environ.get('XDG_DATA_HOME') or os.path.join(os.path.expanduser('~'), '.local', 'share'), 'tierlm'), 'hub')
    os.makedirs(d, exist_ok=True)
    return d


def info(repo_or_url, token=None, fresh=False, max_age=86400, max_shards=0):
    # и1
    repo = repo_of(repo_or_url)
    if not repo:
        return {'ошибка': f'«{repo_or_url}» не является именем модели. Требуется «организация/модель» или ссылка на huggingface.co.'}
    cp = os.path.join(cache_dir(), repo.replace('/', '__') + '.json')
    if not fresh:
        try:
            c = json.load(open(cp, encoding='utf-8'))
            if time.time() - c.get('когда', 0) < max_age and not c.get('ошибка'):
                return c
        except (OSError, ValueError):
            pass
    out = {'repo': repo, 'когда': time.time(), 'ссылка': f'{HF}/{repo}'}
    try:
        api = _json(f'{HF}/api/models/{repo}?blobs=true', token)
    except NETERR as e:
        out['ошибка'] = _why_http(e, repo)
        return out
    sib = [(s.get('rfilename', ''), s.get('size') or (s.get('lfs') or {}).get('size') or 0) for s in api.get('siblings') or []]
    want = [(f, n) for f, n in sib if wanted(f)]
    st = [(f, n) for f, n in want if f.endswith('.safetensors')]
    cd = api.get('cardData') or {}
    out.update(закрыта=bool(api.get('gated')), лицензия=cd.get('license') or '', загрузок=api.get('downloads') or 0,
               нравится=api.get('likes') or 0, изменена=(api.get('lastModified') or '')[:10], файлов=len(want),
               байт_весов=sum(n for _, n in st), байт_всего=sum(n for _, n in want), кусков=len(st),
               шаблон=any(f in ('chat_template.jinja', 'chat_template.json') for f, _ in want) or
               bool(((api.get('config') or {}).get('tokenizer_config') or {}).get('chat_template')),
               токенизатор=any(f == 'tokenizer.json' for f, _ in want), задача=api.get('pipeline_tag') or '',
               всего_параметров=(api.get('safetensors') or {}).get('total') or 0,
               типы=(api.get('safetensors') or {}).get('parameters') or {})
    if not st:
        out['почему'] = 'В корне нет весов safetensors, движку нечего читать.' + (' Есть только GGUF, его можно использовать как квантованную базу.' if any(f.endswith('.gguf') for f, _ in sib) else '')
        out['поддержка'] = False
        _save(cp, out)
        return out
    if max_shards and len(st) > max_shards:
        out['ошибка'] = f'{repo}: Веса разделены на {len(st)} частей. Проверка поддержки выполняется при открытии модели.'
        return out
    td = tempfile.mkdtemp(prefix='tierhub')
    try:
        for f in ('config.json', 'generation_config.json', 'tokenizer_config.json'):
            if any(x == f for x, _ in want):
                try:
                    with _req(f'{HF}/{repo}/resolve/main/{f}', token) as r:
                        open(os.path.join(td, f), 'wb').write(r.read())
                except NETERR as e:
                    if f == 'config.json':
                        raise
        T = {}
        with concurrent.futures.ThreadPoolExecutor(min(8, len(st))) as ex:        # и2
            for h in ex.map(lambda f: st_header_remote(repo, f[0], token), st):
                T.update(h)
        A, why = arch.describe(td, (T, [f for f, _ in st]))
        if arch.load_json(os.path.join(td, 'tokenizer_config.json'), {}).get('chat_template'):
            out['шаблон'] = True
        cfg = arch.text_cfg(arch.load_json(os.path.join(td, 'config.json'), {}))
    except NETERR + (KeyError, IndexError, TypeError, AttributeError, ArithmeticError, struct.error, SystemExit) as e:   # и3
        out['ошибка'] = _why_http(e, repo) if isinstance(e, urllib.error.URLError) else f'{repo}: Не удалось прочитать заголовки ({e}).'
        return out
    finally:
        shutil.rmtree(td, ignore_errors=True)
    merge = bool(A) and bool(why) and all('merge' in w for w in why)
    hard = [w for w in why if 'merge' not in w]
    if not out['шаблон'] and not tmpl.manual((A or {}).get('family', '')):   # и4
        hard.append('Нет шаблона диалога (chat_template): модель базовая и не предназначена для разговора.')
    if not out['токенизатор']:
        hard.append('Нет tokenizer.json.')
    if A:
        pre, lp = A['prefix'], A['layer_prefix']
        text = sum(_prod(s) for k, (_, s) in T.items() if k.startswith(pre) or k == A.get('head'))
        lay = sum(_prod(s) for k, (_, s) in T.items() if k.startswith(lp))
        out.update(семья=A['family'], слоёв=A['layers'], ширина=A['hidden'], mlp=A['inter'], голов=A['heads'], kv=A['L'][0]['nkv'] if A['L'] else 0,
                   словарь=A['vocab'], полных=sum(L['full'] for L in A['L']), параметров=text, параметров_слоёв=lay,
                   связана_головка=bool(A['tied']), сводка=arch.summary(A, hard),
                   kv_на_позицию=sum(2 * L['hd'] * L['nkv'] * 4 for L in A['L']))   # и5
    out.update(контекст=int(cfg.get('max_position_embeddings') or 0), склейка=merge, поддержка=bool(A) and not hard,
               почему=' '.join(hard), основной=max(out['типы'], key=out['типы'].get) if out['типы'] else '')
    _save(cp, out)
    return out


def _prod(s):
    n = 1
    for d in s:
        n *= d
    return n


def _save(p, d):
    t = f'{p}.{os.getpid()}.{threading.get_ident()}.part'      # л1
    try:
        with open(t, 'w', encoding='utf-8') as f:
            json.dump(d, f, ensure_ascii=False)
        os.replace(t, p)
    except OSError:
        try:
            os.remove(t)
        except OSError:
            pass


# л2
_down = [0.0]        # л3
_bad = {}            # л4
_kn = {}


def known_all():
    # м1
    try:
        mt = os.stat(KNOWN_FILE).st_mtime
    except OSError:
        mt = 0
    if _kn.get('mt') != mt:
        try:
            d = json.load(open(KNOWN_FILE, encoding='utf-8')).get('статусы') or {}
        except (OSError, ValueError, AttributeError):
            d = {r: {'проверка': v, 'заметка': n} for r, v, n in SEED}
        _kn.update(mt=mt, v={r.lower(): dict(v, repo=r) for r, v in d.items() if isinstance(v, dict)})
    return _kn['v']


def known(repo):
    return known_all().get((repo or '').lower()) or {'проверка': 'нет', 'заметка': ''}


def cached_info(repo, max_age=86400):
    # о1
    try:
        c = json.load(open(os.path.join(cache_dir(), repo.replace('/', '__') + '.json'), encoding='utf-8'))
    except (OSError, ValueError):
        return None
    return c if isinstance(c, dict) and time.time() - c.get('когда', 0) < max_age and not c.get('ошибка') else None


SHOW = ('поддержка', 'почему', 'склейка', 'байт_весов', 'байт_всего', 'кусков', 'параметров', 'параметров_слоёв', 'семья', 'слоёв', 'контекст', 'шаблон')


def mark(rows, max_age=30 * 86400):
    # п1
    for r in rows:
        k, c = known(r['repo']), cached_info(r['repo'], max_age)
        note = ' · '.join(x for x in (k.get('заметка'), 'Требуется согласие на сайте' if r.get('закрыта') else '') if x)
        new = {'проверка': k.get('проверка') or 'нет', 'заметка': note, 'поддержка': None}
        if c:
            new.update({f: c[f] for f in SHOW if f in c})
            new['закрыта'] = bool(r.get('закрыта') or c.get('закрыта'))
        b = _bad.get(r['repo'])
        if new['поддержка'] is None:
            if b and time.time() - b[0] < 600:
                new['проверка_ошибка'] = b[1]
            elif time.time() < _down[0]:
                new['проверка_ошибка'] = 'Нет связи с Hugging Face. Проверка будет выполнена после восстановления связи.'
        for f in SHOW + ('проверка_ошибка',):                        # п2
            if f not in new:
                r.pop(f, None)
        r.update(new)
    return rows


def check(repos, token=None, workers=6, cb=None, cancel=None, fresh=False):
    # р1
    todo = [] if time.time() < _down[0] else [r for r in dict.fromkeys(repos) if fresh or not cached_info(r)]     # р2
    done = [0]

    def one(r):
        if cancel and cancel():
            return r, None
        try:
            m = info(r, token, fresh, max_shards=16)
        except Exception as e:                                       # р3
            m = {'ошибка': f'{r}: Не удалось разобрать ({type(e).__name__}: {e}).'}
        if m.get('ошибка'):
            _bad[r] = (time.time(), m['ошибка'])
        else:
            _bad.pop(r, None)
        return r, m

    if todo:
        with concurrent.futures.ThreadPoolExecutor(max(1, min(workers, len(todo)))) as ex:
            for f in concurrent.futures.as_completed([ex.submit(one, r) for r in todo]):
                r, m = f.result()
                if m is not None:
                    done[0] += 1
                    if cb:
                        cb(r, m)
    return done[0]


def _row(m):
    # с1
    cfg, st = m.get('config') or {}, m.get('safetensors') or {}
    return {'repo': m.get('id') or m.get('modelId') or '', 'загрузок': m.get('downloads') or 0, 'нравится': m.get('likes') or 0,
            'тренд': m.get('trendingScore') or 0, 'изменена': (m.get('lastModified') or '')[:10], 'создана': (m.get('createdAt') or '')[:10],
            'закрыта': bool(m.get('gated')), 'задача': m.get('pipeline_tag') or '', 'всего_параметров': st.get('total') or 0,
            'типы': st.get('parameters') or {}, 'тип_модели': cfg.get('model_type') or '', 'архитектуры': cfg.get('architectures') or []}


def _page(url, token):
    # т1
    if not url.startswith(HF + '/api/'):
        raise ValueError('Ссылка на продолжение ведёт не на Hugging Face.')
    with _req(url, token, timeout=15) as r:
        data = json.loads(r.read())
        link = r.headers.get('Link') or ''
    if not isinstance(data, list):
        raise ValueError('Ответ не является списком.')
    m = re.search(r'<([^>]+)>\s*;\s*rel="next"', link)
    return data, (m.group(1) if m and m.group(1).startswith(HF + '/api/') else '')


_CUR = {}   # т2


def _fetch(task, search, sort, limit, org, min_b, max_b, token, state=None):
    # у1
    size = 100   # у2
    q = [('pipeline_tag', task), ('filter', 'safetensors'), ('sort', SORTS[sort]), ('direction', '-1'), ('limit', size)]
    if search:
        q.append(('search', search))
    if org:
        q.append(('author', org))
    q += [('expand[]', x) for x in EXPAND]
    rows, url = (list(state[0]), state[1]) if state else ([], f'{HF}/api/models?' + urllib.parse.urlencode(q))
    seen, pages = {r['repo'] for r in rows}, 0
    while url and len(rows) < limit and pages < 50:   # у3
        data, url = _page(url, token)
        pages += 1
        for m in data:
            r = _row(m)
            if not r['repo'] or r['repo'] in seen or m.get('disabled'):
                continue
            n = r['всего_параметров']
            if (min_b or max_b) and not (n and (not min_b or n >= min_b * 1e9) and (not max_b or n <= max_b * 1e9)):
                continue
            seen.add(r['repo'])
            rows.append(r)
    _CUR.setdefault((search, sort, org, min_b, max_b), {})[task] = (rows, url)
    return rows[:limit], bool(url) or len(rows) > limit


SORTKEY = {'downloads': 'загрузок', 'likes': 'нравится', 'trending': 'тренд', 'modified': 'изменена', 'created': 'создана'}


def _by(rows, sort):
    k = SORTKEY[sort]
    return sorted(rows, key=lambda r: r.get(k) or ('' if k in ('изменена', 'создана') else 0), reverse=True)


def _offline(search, org, min_b, max_b, sort):
    # х1
    pool = {}
    try:
        names = sorted(os.listdir(cache_dir()))
    except OSError:
        names = []
    for fn in names:
        p = os.path.join(cache_dir(), fn)
        try:
            c = json.load(open(p, encoding='utf-8'))
        except (OSError, ValueError):
            continue
        if fn.startswith('catalog_') and isinstance(c, dict):
            for r in c.get('rows') or []:
                if r.get('repo') and c.get('когда', 0) >= pool.get(r['repo'], (0,))[0]:
                    pool[r['repo']] = (c.get('когда', 0), r)
        elif fn.endswith('.json') and isinstance(c, dict) and c.get('repo'):
            r = {k: c.get(k, d) for k, d in (('repo', ''), ('загрузок', 0), ('нравится', 0), ('изменена', ''), ('закрыта', False), ('задача', ''), ('типы', {}))}
            r['всего_параметров'] = c.get('всего_параметров') or 0
            if c.get('когда', 0) >= pool.get(r['repo'], (0,))[0]:
                pool[r['repo']] = (c.get('когда', 0), dict(pool.get(r['repo'], (0, {}))[1], **r))
    rows = [r for _, r in pool.values()]
    src = 'сохранённые ответы'
    if not rows:
        rows, src = [{'repo': r, 'загрузок': 0, 'нравится': 0, 'изменена': '', 'закрыта': False, 'всего_параметров': 0} for r, _, _ in SEED], 'запасной список'
    words = search.lower().split()
    rows = [r for r in rows if all(w in r['repo'].lower() for w in words) and (not org or r['repo'].lower().startswith(org.lower() + '/'))
            and (not (min_b or max_b) or (r.get('всего_параметров') and (not min_b or r['всего_параметров'] >= min_b * 1e9) and (not max_b or r['всего_параметров'] <= max_b * 1e9)))]
    return _by(rows, sort), src


def catalog(search='', sort='downloads', limit=30, org='', min_b=0, max_b=0, tasks=TASKS, token=None, offline=False, fresh=False):
    # ц1
    sort = sort if sort in SORTS else 'downloads'
    limit = max(1, int(limit or 30))   # ц2
    search, org = (search or '').strip(), (org or '').strip().strip('/')
    if repo_of(search):
        search = repo_of(search)
    if '/' in search and not org:                                      # ц3
        org, _, search = search.partition('/')
    search, tasks = search.strip(), tuple(tasks or TASKS)
    cp = os.path.join(cache_dir(), 'catalog_' + hashlib.sha1(json.dumps([search, sort, limit, org, min_b, max_b, tasks, HF]).encode()).hexdigest()[:20] + '.json')
    try:
        c = json.load(open(cp, encoding='utf-8'))
        c = c if isinstance(c, dict) and isinstance(c.get('rows'), list) else None
    except (OSError, ValueError):
        c = None
    out = {'rows': [], 'источник': '', 'ошибка': '', 'ещё': False, 'когда': 0}
    if c and not fresh and (offline or time.time() - c.get('когда', 0) < TTL[sort]):
        out.update(rows=c['rows'], ещё=bool(c.get('ещё')), когда=c.get('когда', 0), источник='кэш')
    else:
        why = ''
        if offline:
            why = 'Режим без сети.'
        elif time.time() < _down[0]:
            why = 'Нет связи, предыдущая попытка не удалась. Повторная попытка будет выполнена позже.'
        else:
            try:
                st = {} if fresh else _CUR.get((search, sort, org, min_b, max_b), {})
                with concurrent.futures.ThreadPoolExecutor(len(tasks)) as ex:
                    got = list(ex.map(lambda t: _fetch(t, search, sort, limit, org, min_b, max_b, token, st.get(t)), tasks))
                rows = _by([r for g, _ in got for r in g], sort)
                more = any(m for _, m in got) or len(rows) > limit
                _down[0] = 0
                out.update(rows=rows[:limit], ещё=more, когда=time.time(), источник='сеть')
                _save(cp, {'когда': out['когда'], 'rows': out['rows'], 'ещё': more})
            except NETERR as e:
                _down[0] = time.time() + 60
                why = _why_http(e, 'Hugging Face').split(': ', 1)[-1]
        if not out['источник']:
            if c:
                out.update(rows=c['rows'], ещё=bool(c.get('ещё')), когда=c.get('когда', 0), источник='кэш (устаревший)')
            else:
                rows, src = _offline(search, org, min_b, max_b, sort)
                out.update(rows=rows[:limit], ещё=len(rows) > limit, источник=src)
            out['ошибка'] = f'{why} Показано: {out["источник"]}.'
    out['rows'] = mark([dict(r) for r in out['rows']])
    return out


def main_list(a):
    # ч1
    import argparse
    p = argparse.ArgumentParser(prog='hub.py list', description='Каталог Hugging Face: текстовые модели с safetensors и заключение о поддержке движком.')
    p.add_argument('--search', default='', help='Часть названия; «автор/часть» задаёт также автора.')
    p.add_argument('--sort', default='downloads', choices=list(SORTS), help='Порядок: downloads (по умолчанию), likes, trending, modified, created.')
    p.add_argument('--limit', type=int, default=30)
    p.add_argument('--org', default='', help='Только этого автора (организации).')
    p.add_argument('--min-b', type=float, default=0, help='Не менее указанного числа млрд параметров.')
    p.add_argument('--max-b', type=float, default=0, help='Не более указанного числа млрд параметров.')
    p.add_argument('--task', action='append', help=f'Значение pipeline_tag на сайте; допускается несколько раз (по умолчанию {", ".join(TASKS)}).')
    p.add_argument('--supported', action='store_true', help='Только модели, поддерживаемые движком (список расширяется до --limit).')
    p.add_argument('--no-check', action='store_true', help='Не проверять поддержку по заголовкам, использовать только кэш.')
    p.add_argument('--fresh', action='store_true', help='Игнорировать кэш списка.')
    p.add_argument('--offline', action='store_true', help='Без сети: кэш и сохранённые сведения.')
    n = p.parse_args(a)
    tok, lim, rows, res = os.environ.get('HF_TOKEN'), n.limit, [], {}
    for _ in range(6):
        res = catalog(n.search, n.sort, lim, n.org, n.min_b, n.max_b, n.task or TASKS, tok, n.offline, n.fresh)
        if not n.no_check and not n.offline and time.time() >= _down[0]:
            check([r['repo'] for r in res['rows']], tok)
        mark(res['rows'])
        rows = [r for r in res['rows'] if not n.supported or r.get('поддержка')]
        if not n.supported or len(rows) >= n.limit or not res['ещё'] or res['источник'] not in ('сеть', 'кэш'):
            break
        lim *= 2                                                       # ч2
    rows = rows[:n.limit]
    sys.stderr.write(f'Источник: {res["источник"]}' + (f' ({res["ошибка"]})' if res['ошибка'] else '') + f'; строк: {len(rows)}; проверено: {sum(r.get("поддержка") is not None for r in rows)}'
                     + ('; есть продолжение (требуется увеличить --limit)' if res['ещё'] else '') + '\n')
    keep = ('repo', 'проверка', 'заметка', 'поддержка', 'почему', 'склейка', 'загрузок', 'нравится', 'изменена', 'закрыта', 'всего_параметров', 'задача', 'тип_модели', 'байт_весов', 'проверка_ошибка')
    print(json.dumps([{k: r[k] for k in keep if k in r} for r in rows], ensure_ascii=False, indent=1))

# ч3
LANE = {1: .25, 2: .5, 3: .985, 4: 1.969, 5: 3.938, 6: 7.563}        # ч4
QBYTES = {'BF16': 2.0, 'Q8_0': 34 / 32, 'Q6_K': 210 / 256, 'Q5_K_M': 0.71, 'Q4_K_M': 0.6}


def pcie_gbs(gen, width):
    g, w = int(gen or 0), int(width or 0)
    if not (g and w) and hwprobe:                                    # ш1
        try:
            g2, w2 = hwprobe.pcie(); g, w = g or g2, w or w2
        except Exception:
            pass
    return LANE.get(g, 1.969) * (w or 16) * 0.8 * 1e9, bool(g and w)


def reserve(ram):
    # э1
    if hwprobe:
        try:
            return hwprobe.ram_reserve_now()
        except Exception:
            pass
    return min(8 << 30, max(5 << 30, ram // 4))


def dims_of(m):
    # ю1
    try:
        d = {'hidden': int(m['ширина']), 'inter': int(m['mlp']), 'heads': int(m['голов']), 'layers': int(m['слоёв'])}
    except (KeyError, TypeError, ValueError):
        return {}
    d['kv_per_pos'] = int(m.get('kv_на_позицию') or d['layers'] * 2 * int(m.get('kv') or d['heads']) * (d['hidden'] // max(1, d['heads'])) * 4)
    return d


def train_margin(vram, d):
    # я1
    if hwprobe and d and vram:
        try:
            P = hwprobe.pack_fit(vram // hwprobe.TRAIN_ACT_SHARE, d) or hwprobe.PACK_LO
            return hwprobe.act_bytes(P, d['hidden'], d['inter'], d['heads'], d['layers'])
        except Exception:
            pass
    return 1536 << 20


def chat_margin(vram, d, ctx_max):
    # аа1
    if d.get('kv_per_pos') and vram:
        kv = d['kv_per_pos']; n = max(256, min(vram // 4 // kv, ctx_max or (vram // 4 // kv)) // 256 * 256)
        return n * kv
    return 1 << 30


def forecast(m, hw):
    # аб1
    rows = []
    P, PL = m.get('параметров') or 0, m.get('параметров_слоёв') or 0
    if not P:
        return rows
    lay = PL * 2                                                     # аб2
    ram, vram, free = hw.get('ram') or 0, hw.get('vram') or 0, hw.get('disk_free')
    bw, known = pcie_gbs(hw.get('pcie_gen'), hw.get('pcie_width'))
    have = hw.get('есть_байт') or 0                                   # аб3
    need = max(0, (m.get('байт_всего') or 0) - have) + ((m.get('байт_весов') or 0) if m.get('кусков', 0) > 1 or m.get('склейка') else 0)
    if free is not None:
        rows.append(('Диск', f'{gib(need)} из {gib(free)} свободных', 'Загрузка' + (' и объединение частей (на время объединения требуется вторая копия)' if m.get('склейка') else ''),
                     'ok' if free > need + (2 << 30) else 'bad'))
    cache = max(0, ram - reserve(ram)) if ram else 0
    if ram:
        if cache >= lay:
            rows.append(('ОЗУ при обучении', f'Кэш весов целиком: {gib(lay)}', f'Остаётся {gib(ram - lay)}', 'ok'))
        else:
            rows.append(('ОЗУ при обучении', f'В кэше {gib(cache)} из {gib(lay)}', f'Остальные {gib(lay - cache)} читаются с диска на каждом шаге, что замедляет работу', 'warn'))
        fit = [q for q in ('BF16', 'Q8_0', 'Q6_K', 'Q5_K_M', 'Q4_K_M') if PL * QBYTES[q] <= cache]
        if fit and fit[0] == 'BF16':
            rows.append(('ОЗУ в разговоре', f'BF16 {gib(lay)}: помещается', '', 'ok'))
        elif fit:
            rows.append(('ОЗУ в разговоре', f'Требуется квантованная база {fit[0]}: {gib(PL * QBYTES[fit[0]])}', f'BF16 {gib(lay)} не помещается в {gib(cache)}', 'warn'))
        else:
            rows.append(('ОЗУ в разговоре', f'Даже Q4_K_M ({gib(PL * QBYTES["Q4_K_M"])}) превышает {gib(cache)}', 'Веса будут читаться с диска во время разговора', 'bad'))
    t1 = hw.get('tier1') or 0
    if t1:
        sh = min(1.0, t1 / PL)
        rows.append(('Обучаемая доля (срез строк)', f'{100 * sh:.1f} % строк каждой матрицы', f'Бюджет видеопамяти: {t1 / 1e6:.0f} млн чисел', 'ok' if sh >= 0.01 else 'warn'))
    dm = dims_of(m)
    res = max(0, vram - train_margin(vram, dm) - t1 * 12) if vram else 0       # аб4
    a = hw.get('anchor_step')
    if a and a[1]:
        st = a[0] * max(0.05, (lay - min(lay, res)) / max(1, a[1] - min(a[1], res)))
        how = f'По замеру запуска «{a[2]}»: {a[0]:.1f} с на {gib(a[1])}' if len(a) > 2 else f'По замеру запуска: {a[0]:.1f} с'
    else:
        st = 2 * max(0, lay - res) / bw + 0.3
        how = f'По шине PCIe {bw / 1e9:.0f} ГБ/с' + ('' if known else ' (поколение неизвестно, принято 4.0 x16)')
    if ram and cache < lay:
        dbps = 2.5e9                                                     # аб5
        if hwprobe:
            try:
                dbps = hwprobe.disk_read_bps(hw.get('disk_path')) or dbps   # аб6
            except Exception:
                pass
        st += 2 * (lay - cache) / dbps                                   # аб7
        how += f'; дополнительно чтение с диска ({dbps / 1e9:.1f} ГБ/с)'
    rows.append(('Время шага обучения', f'≈ {st:.1f} с', how, 'ok' if st < 15 else 'warn'))
    cres = max(0, vram - chat_margin(vram, dm, m.get('контекст'))) if vram else 0   # аб8
    stream = max(0, lay - cres)
    at = hw.get('anchor_tok')                                         # аб9
    if at and at[0] > 0 and at[1]:
        spt = at[0] * max(0.1, stream / max(1, at[1] - min(at[1], cres)))
        rows.append(('Разговор, BF16', f'≈ {1 / spt:.1f} ток/с', f'По замеру разговора с {at[2]}: {1 / at[0]:.1f} ток/с', 'ok' if spt < 0.25 else 'warn'))
    elif stream:
        rows.append(('Разговор, BF16', f'Не более {bw / stream:.1f} ток/с', f'На каждый токен по шине передаётся {gib(stream)}; замер ещё не выполнен', 'warn'))
    else:
        rows.append(('Разговор, BF16', 'Все веса в видеопамяти', 'Шина не ограничивает скорость; замер ещё не выполнен', 'ok'))
    q = PL * QBYTES['Q4_K_M']
    if stream and vram:
        rows.append(('Квантованная база Q4_K_M', f'{gib(q)}: ' + ('целиком в видеопамяти' if q <= cres else f'в видеопамяти {gib(cres)}, остальное по шине'),
                     'На каждый токен передаётся меньше весов, ответ быстрее', 'ok'))
    return rows


def gib(n):
    return f'{(n or 0) / 2**30:.1f} ГиБ'


if __name__ == '__main__':
    a = sys.argv[1:]
    if len(a) >= 2 and a[0] == 'info':
        print(json.dumps(info(a[1], os.environ.get('HF_TOKEN'), '--fresh' in a), ensure_ascii=False, indent=1))
    elif a and a[0] == 'list':
        main_list(a[1:])
    else:
        print(open(__file__, encoding='utf-8').read().split('\nimport', 1)[0])
        sys.exit(1)
