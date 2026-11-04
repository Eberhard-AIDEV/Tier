# а1
import json, math, os, queue, re, shutil, subprocess, sys, threading, time

from PySide6.QtCore import (Property, QObject, QPointF, QRectF, QSize, Qt, QTimer, QUrl, Signal, Slot)
from PySide6.QtGui import (QBrush, QClipboard, QColor, QDesktopServices, QFont, QFontDatabase, QFontMetricsF, QGuiApplication, QIcon,
                           QImage, QLinearGradient, QPainter, QPainterPath, QPen, QPolygonF, QRadialGradient, QTransform)
from PySide6.QtQml import QJSValue, QQmlApplicationEngine, qmlRegisterType
from PySide6.QtQuick import QQuickPaintedItem, QQuickWindow
from PySide6.QtSvg import QSvgRenderer
from PySide6.QtWidgets import QApplication, QColorDialog, QFileDialog, QMenu, QSystemTrayIcon

T = None            # а2
BK = None           # а3
LANG = 'ru'
DEC = 'dot'
NBSP = ' '     # а4


# б1
def okn(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def dec(s):
    return re.sub(r'(\d)\.(\d)', r'\1,\2', s) if DEC == 'comma' else s


def group(t):
    m = re.match(r'^(-?)(\d+)(.*)$', t)
    if not m or len(m.group(2)) < 5:
        return t
    return m.group(1) + f'{int(m.group(2)):,}'.replace(',', NBSP) + m.group(3)


def fmt(v, d=2):
    return dec(f'{v:.{d}f}') if okn(v) else '—'


def int_(v):
    return group(str(int(round(v)))) if okn(v) else '—'


def num(v, d=2):
    return dec(group(f'{v:.{d}f}')) if okn(v) else '—'


def gib(b, d=2):
    return num(b / 2 ** 30, d) + NBSP + tr('ГиБ') if okn(b) else '—'


def bytes_(b):
    if not okn(b):
        return '—'
    if b >= 2 ** 30:
        return num(b / 2 ** 30, 0 if b >= 100 * 2 ** 30 else 1) + NBSP + tr('ГиБ')
    if b >= 2 ** 20:
        return num(b / 2 ** 20, 0) + NBSP + tr('МиБ')
    if b >= 1024:
        return num(b / 1024, 0) + NBSP + tr('КиБ')
    return int_(b) + NBSP + tr('Б')


def pct(v, d=1):
    return dec(f'{v:.{d}f}') + NBSP + '%' if okn(v) else '—'


def deg(v):
    return f'{round(v)}°' if okn(v) else '—'


def hms(s):
    s = max(0, round(s or 0))
    if s >= 3600:
        return f'{s // 3600} ч {s % 3600 // 60:02d} мин'
    return f'{s // 60} мин {s % 60:02d} с' if s >= 60 else f'{s} с'


def plural(n, a, b, c):
    n = abs(round(n or 0))
    t, h = n % 10, n % 100
    return a if t == 1 and h != 11 else b if 2 <= t <= 4 and not 10 <= h < 20 else c


def steps_(n):
    return plural(n, 'шаг', 'шага', 'шагов')


def cap(s):
    s = str(s or '')
    return s[0].upper() + s[1:] if s and 'а' <= s[0].lower() <= 'я' or s[:1] == 'ё' else s


def base(p):
    return os.path.basename(str(p or '').rstrip('/'))


def when(s, sec=False):
    m = re.match(r'^(\d{4})-(\d\d)-(\d\d)[ T](\d\d):(\d\d)(?::(\d\d))?', str(s or ''))
    if not m:
        return s or '—'
    y, mo, d, hh, mm, ss = m.groups()
    t = time.localtime()
    today = time.mktime((t.tm_year, t.tm_mon, t.tm_mday, 0, 0, 0, 0, 0, -1))
    day = time.mktime((int(y), int(mo), int(d), 0, 0, 0, 0, 0, -1))
    k = round((today - day) / 86400)
    hm = f'{hh}:{mm}' + (f':{ss}' if sec and ss else '')
    return f'Сегодня {hm}' if k == 0 else f'Вчера {hm}' if k == 1 else f'{d}.{mo}.{y} {hm}'


def vals(a, k):
    return [r[k] for r in a or [] if okn(r.get(k))]


def last_of(a, k):
    for r in reversed(a or []):
        if r.get(k) is not None:
            return r[k]
    return None


def med(a):
    return sorted(a)[len(a) >> 1] if a else None


def avg(a):
    return sum(a) / len(a) if a else None


# х1
EN, _rx, _trc = {}, None, {}
CYR = re.compile('[а-яёА-ЯЁ]')


_GL = []   # х2


def load_gloss(path):
    # ц1
    _GL.clear()
    try:
        lines = open(path, encoding='utf-8').read().splitlines()
    except OSError:
        return
    for line in lines:
        if not line.strip() or line.startswith('#'):
            continue
        c = line.split('\t')
        if len(c) < 6:
            continue
        try:
            _GL.append((c[1], c[2], re.compile(c[3], re.I), c[4], c[5]))
        except re.error:
            continue


def gloss_for(text):
    # ч1
    t = (text or '').strip()
    if len(t) < 2 or len(t) > 160:
        return ''
    hits = []
    for ru, en, rx, dru, den in _GL:
        m = rx.search(t)
        if m:
            hits.append((m.start(), en if LANG == 'en' else ru, den if LANG == 'en' else dru))
    hits.sort()
    return '\n'.join(f'{n}: {d}' for _, n, d in hits[:3])


def load_en(path):
    global _rx
    EN.clear(); _trc.clear()
    try:
        lines = open(path, encoding='utf-8').read().splitlines()
    except OSError:
        lines = []
    for line in lines:
        if '|' not in line:
            continue
        parts = re.split(r'(?<!\|)\|(?!\|)', line, 1)
        if len(parts) != 2:
            continue
        k, v = parts[0].strip(), parts[1].strip()
        if k and k not in EN:
            EN[k] = v
        if k and k[0].islower() and v:
            K = k[0].upper() + k[1:]
            if K not in EN:
                EN[K] = v[0].upper() + v[1:]
    keys = sorted(EN, key=len, reverse=True)
    # ш1
    L, lead = 'а-яёА-ЯЁ', [k for k in keys if not CYR.match(k[0])]
    _rx = re.compile('(' + ''.join(re.escape(k) + (f'(?![{L}])' if CYR.match(k[-1]) else '') + '|' for k in lead) +
                     f'(?<![{L}])(?:' + '|'.join(re.escape(k) for k in keys if CYR.match(k[0])) + f')(?![{L}]))') if keys else None


def tr(s):
    if LANG != 'en' or not s or not isinstance(s, str) or not CYR.search(s) or _rx is None:
        return s
    r = _trc.get(s)
    if r is None:
        r = _rx.sub(lambda m: EN[m.group(1)], s)
        if len(_trc) < 20000:
            _trc[s] = r
    return r


# ю1
def ser(a, k, name, color, dots=False):
    return {'name': tr(name), 'color': color, 'dots': dots,
            'data': [[p['шаг'], p[k]] for p in a or [] if okn(p.get(k)) and okn(p.get('шаг'))]}


def kilo(v):
    return dec(f'{v / 1000:.1f}'.rstrip('0').rstrip('.')) + 'k' if v >= 1000 else str(round(v))


def sweep_chart(s):
    rows = [r for r in s.get('развёртка') or [] if r[1] > 0]
    if not rows:
        return None
    capv = 1.5 * max(r[2] for r in rows[:3])        # аа1
    d = [r for r in rows if r[2] <= capv]
    cut = len(rows) - len(d)
    lo, hi = math.floor(math.log10(rows[0][1])), math.ceil(math.log10(rows[-1][1]))
    try:
        lr = float((s.get('план') or {}).get('lr') or 'nan')
    except ValueError:
        lr = float('nan')
    note = 'Подбор LR: 40 шагов с растущим LR; выбирается одна десятая значения LR в минимуме сглаженной потери (loss).' + (
        f' Не показано точек выше {fmt(capv, 1)}: {cut}.' if cut else '')
    return {'title': tr('Подбор LR'), 'note': tr(note),
            'series': [{'name': tr('Потеря (loss)'), 'color': 's1', 'dots': True, 'data': [[math.log10(r[1]), r[2]] for r in d]}],
            'opts': {'xt': list(range(lo, hi + 1)), 'fx': 'pow10', 'tx': 'lr10', 'ft': 'f4', 'area': False,
                     'vrule': {'x': math.log10(lr), 'label': tr('Выбрано: ') + s['план']['lr']} if lr == lr and lr > 0 else None}}


def _loss_stats(s):
    ev, st = s['оценки'], s['шаги']
    L = vals(ev, 'loss')
    best = min(L) if L else None
    bi = next((e for e in ev if e['loss'] == best), None)
    tail, head = vals(st[-50:], 'ср'), vals(st[-100:-50], 'ср')
    return [['Лучшая на валидации', fmt(best, 4), f'Шаг {round(bi["шаг"])}' if bi else ''],
            ['Последняя на валидации', fmt(L[-1], 4) if L else '—', f'Валидаций: {len(ev)}'],
            ['Среднее на обучении', fmt(last_of(st, 'ср'), 4), 'Последние 50 шагов'],
            ['Перплексия (ppl)', num(last_of(st, 'ppl'), 2), 'exp(loss)'],
            ['Падение за 50 шагов', fmt(avg(head) - avg(tail), 4) if head and tail else '—', 'Разность средних'],
            ['Токенов в расчёте потери', int_(sum(vals(st, 'в_loss'))) if st else '—', 'За весь запуск']]


def _acc_stats(s):
    st, ev = s['шаги'], s['оценки']
    A = vals(ev, 'acc')
    a, b = avg(vals(st, 'в_loss')), avg(vals(st, 'токенов'))
    return [['На валидации', pct(A[-1], 2) if A else '—', 'Лучшая: ' + (pct(max(A), 2) if A else '—')],
            ['На обучении', pct(last_of(st, 'acc_ср'), 2), 'Среднее за 50 шагов'],
            ['Токенов в расчёте потери', int_(med(vals(st, 'в_loss'))), 'Медиана на шаг'],
            ['Токенов всего', int_(med(vals(st, 'токенов'))), 'Медиана на шаг'],
            ['Примеров в шаге', int_(med(vals(st, 'примеров'))), 'Медиана'],
            ['Доля в расчёте потери', pct(100 * a / b, 2) if a and b else '—', 'Остальное — текст вопроса']]


def _vram_stats(s):
    st = s['шаги']
    V, last = vals(st, 'vram'), (st[-1] if st else {})
    tot = last.get('vram_всего')
    mib = lambda v: int_(v) + NBSP + 'МиБ' if okn(v) else '—'
    return [['Пик за запуск', mib(max(V)) if V else '—', f'Из {int_(tot)} МиБ' if tot else ''],
            ['Запас до потолка', mib(tot - max(V)) if V and tot else '—', 'Менее 200 МиБ — риск аварийной остановки.'],
            ['Обучаемая часть в видеопамяти', int_(last.get('tier1')), 'Тензоров в памяти постоянно'],
            ['Живая память', mib(last.get('живых')), f'Блоков: {int_(last.get("блоков"))}'],
            ['Область видеопамяти', mib(last.get('область')), f'Вне области: {int_(last.get("мимо"))}'],
            ['Кэш в ОЗУ', num(last.get('озу'), 2) + NBSP + 'ГиБ' if okn(last.get('озу')) else '—', 'Замороженные веса в формате BF16']]


# аг1
_HWC = {}           # аг2
TEMP_KEY = {'junction': 'горячая', 'hotspot': 'горячая', 'gpu': 'горячая', 'edge': 'край', 'mem': 'память', 'memory': 'память'}


def _hwp():
    try:
        import hwprobe
        return hwprobe
    except Exception:
        return None


def gpu_dev():
    # ае1
    if 'pci' not in _HWC:
        import glob
        best, nv = (0, ''), ''
        for d in glob.glob('/sys/class/drm/card[0-9]*/device'):
            v, a = T.read_text(d + '/vendor').strip(), os.path.basename(os.path.realpath(d))
            if v == '0x10de':
                nv = nv or a
            elif v == '0x1002':
                best = max(best, (T.num(T.read_text(d + '/mem_info_vram_total').strip()) or 0, a))
        try:
            nvm = T.nvidia_mem()[0] if nv else 0
        except Exception:
            nvm = 0
        _HWC['pci'] = nv if nv and nvm > best[0] else best[1] or nv
    return _HWC['pci']


def temp_lim():
    # аж1
    if 'темп' not in _HWC:
        lim = {}
        try:
            for x in _hwp().temp_sensors(gpu_dev()) or []:
                k, c = TEMP_KEY.get(str(x.get('label') or '').strip().lower()), x.get('crit_c')
                if k and okn(c) and c > 0 and k not in lim:
                    lim[k] = c
        except Exception:
            pass
        _HWC['темп'] = lim
    return _HWC['темп']


def temp_warn(k):
    # аз1
    c = temp_lim().get(k)
    return c - 10 if c else 95


def kv_per_token(path):
    # аи1
    n = 0
    try:
        for line in open(path, encoding='utf-8'):
            f = line.split()       # аи2
            if len(f) >= 5 and f[0] == 'layer':
                n += int(f[3]) * int(f[4])
    except (OSError, ValueError, TypeError):
        return 0
    return 2 * n * 4


def ctx_fit():
    # ак1
    mx = T.ctx_max()
    try:
        kv = kv_per_token((T.model_info() or {}).get('описание'))
        v, eng = T.vram_info(), os.path.basename(T.engine())[:15]
        free = v['всего'] - v['занято'] + sum(x['байт'] for x in v.get('кем') or [] if x['кто'] == eng)
    except Exception:
        kv = free = 0
    if not kv or free <= 0:
        return mx, min(1024, mx)          # ак2
    r = lambda b: max(256, int(b // kv) // 256 * 256)
    return min(mx, r(free)), min(mx, r(free / 10))


def temp_series(s):
    # ал1
    a = [ser(s['шаги'], 'горячая', 'Горячая точка', 's1'), ser(s['шаги'], 'память', 'Память', 's2'), ser(s['шаги'], 'край', 'Край кристалла', 's3')]
    return [x for x in a if x['data']] or a


def _temp_stats(s):
    st = s['шаги']
    hot, edge, mem, fan = vals(st, 'горячая'), vals(st, 'край'), vals(st, 'память'), vals(st, 'вент')
    L, wh = temp_lim(), temp_warn('горячая')
    over = len([v for v in hot if v >= wh])
    return [['Самая горячая точка', deg(max(hot)) if hot else '—', f'Предел кристалла {deg(L["горячая"])}' if L.get('горячая') else 'Датчик не сообщает предел'],
            ['Край кристалла', deg(max(edge)) if edge else '—', 'Сейчас: ' + (deg(edge[-1]) if edge else '—')],
            ['Память', deg(max(mem)) if mem else '—', f'Предел памяти {deg(L["память"])}' if L.get('память') else 'Датчик не сообщает предел'],
            ['Вентилятор', int_(max(fan)) if fan else '—', 'Максимум, об/мин'],
            [f'Шагов выше {deg(wh)}', str(over), 'Зафиксирован троттлинг' if over else 'Не зафиксировано'],
            ['Рост за запуск', f'+{round(hot[-1] - hot[0])}°' if len(hot) > 10 else '—', 'От начала запуска до текущего момента.']]


def _time_stats(s):
    st = s['шаги']
    Tt, tk = vals(st, 'сек'), vals(st, 'ток')
    slow = max(st, key=lambda r: r.get('сек') or 0) if st else None
    m = med(Tt)
    norm = [v for v in Tt if v < 2 * (m or 1)]
    return [['Обычный шаг', fmt(m, 3) + NBSP + 'с' if Tt else '—', 'Медиана'],
            ['Средний шаг', fmt(avg(Tt), 3) + NBSP + 'с' if Tt else '—', 'Со всплесками'],
            ['Самый долгий шаг', fmt(max(Tt), 2) + NBSP + 'с' if Tt else '—', f'Шаг {round(slow["шаг"])}' if slow and okn(slow.get('шаг')) else ''],
            ['Токенов в секунду', int_(med(tk)), 'Медиана'],
            ['Общее время', hms(sum(Tt)) if Tt else '—', 'Сумма по шагам'],
            ['Доля всплесков', pct(100 * (1 - len(norm) / len(Tt)), 2) if Tt else '—', 'Шагов длиннее двух медиан']]


def _grad_stats(s):
    G = vals(s['шаги'], 'гн')
    first, last = G[:10], G[-10:]
    sw = s.get('развёртка') or []
    return [['Текущее значение', num(G[-1], 1) if G else '—', ''],
            ['Максимум за запуск', num(max(G), 1) if G else '—', ''],
            ['В начале запуска', num(avg(first), 1) if first else '—', 'Среднее первых 10 шагов'],
            ['Снижение', fmt(avg(first) / avg(last), 2) + '×' if first and last and avg(last) else '—', 'Отношение начала к концу'],
            ['Выбросов', str(len([v for i, v in enumerate(G) if i > 5 and v > 3 * avg(G[max(0, i - 50):i])])), 'Втрое выше среднего за 50 шагов.'],
            ['LR', (s.get('план') or {}).get('lr') or '—', 'Подобрано автоматически' if sw else 'Задано заранее']]


# ао1
WIN = {
    'loss': {'title': 'Потеря (loss)', 'note': 'Чем ниже, тем лучше. Обучение сглажено по 50 шагам, валидация — отдельные замеры.',
             'now': lambda s: {'v': fmt(last_of(s['шаги'], 'ср'), 4)},
             'chart': lambda s: {'series': [ser(s['шаги'], 'ср', 'Обучение', 's1'), ser(s['оценки'], 'loss', 'Валидация', 's2', True)], 'opts': {'ft': 'f4'}},
             'big': lambda s: [
                 {'title': 'Потеря (loss)', 'series': [ser(s['шаги'], 'ср', 'Обучение', 's1'), ser(s['оценки'], 'loss', 'Валидация', 's2', True)], 'opts': {'ft': 'f4'}},
                 {'title': 'Исходная потеря (loss) по шагам', 'note': 'Без сглаживания; виден разброс между примерами.',
                  'series': [ser(s['шаги'], 'loss', 'Шаг', 's1')], 'opts': {'ft': 'f4'}}],
             'stats': _loss_stats,
             'table': lambda s: (['Шаг', 'На валидации', 'Перплексия (ppl)', 'Лучшая'],
                                 [[int_(e['шаг']), fmt(e['loss'], 4), num(math.exp(e['loss']), 2) if okn(e['loss']) and e['loss'] < 50 else '—',
                                   fmt(e['лучший'], 4)] for e in reversed(s['оценки'])])},
    'acc': {'title': 'Точность (acc) на ответе', 'note': 'Доля угаданных токенов ответа.',
            'now': lambda s: {'v': pct(last_of(s['шаги'], 'acc_ср'), 2)},
            'chart': lambda s: {'series': [ser(s['шаги'], 'acc_ср', 'Обучение', 's1'), ser(s['оценки'], 'acc', 'Валидация', 's2', True)],
                                'opts': {'zero': True, 'fy': 'pct0', 'ft': 'pct2'}},
            'big': lambda s: [
                {'title': 'Точность (acc)', 'series': [ser(s['шаги'], 'acc_ср', 'Обучение', 's1'), ser(s['оценки'], 'acc', 'Валидация', 's2', True)],
                 'opts': {'zero': True, 'fy': 'pct0', 'ft': 'pct2'}},
                {'title': 'Длина примеров', 'note': 'Число токенов на шаг и число токенов, учитываемых в расчёте потери (loss).',
                 'series': [ser(s['шаги'], 'токенов', 'Всего', 's1'), ser(s['шаги'], 'в_loss', 'В расчёте потери', 's3')], 'opts': {'zero': True, 'ft': 'int'}}],
            'stats': _acc_stats,
            'table': lambda s: (['Шаг', 'На валидации', 'На обучении'],
                                [[int_(e['шаг']), pct(e['acc'], 2),
                                  (lambda n: pct(n['acc_ср'], 2) if n else '—')(next((r for r in reversed(s['шаги']) if okn(r.get('шаг')) and r['шаг'] <= e['шаг']), None))]
                                 for e in reversed(s['оценки'])])},
    'vram': {'title': 'Видеопамять', 'note': 'Пик за шаг; пунктиром показан потолок памяти карты.',
             'now': lambda s: (lambda v, t: {'v': int_(v), 'unit': 'МиБ' if okn(v) else '', 'sub': f'Из {int_(t)} МиБ' if t else '',
                                             'flag': 'Близко к потолку' if okn(v) and t and v / t > 0.96 else ''})(last_of(s['шаги'], 'vram'), last_of(s['шаги'], 'vram_всего')),
             'chart': lambda s: {'series': [ser(s['шаги'], 'vram', 'Пик', 's1')],
                                 'opts': {'zero': True, 'rule': last_of(s['шаги'], 'vram_всего'), 'ruleLabel': tr('Потолок'), 'fy': 'int', 'ft': 'mib'}},
             'big': lambda s: [
                 {'title': 'Пик за шаг', 'series': [ser(s['шаги'], 'vram', 'Пик', 's1')],
                  'opts': {'zero': True, 'rule': last_of(s['шаги'], 'vram_всего'), 'ruleLabel': tr('Потолок'), 'fy': 'int', 'ft': 'mib'}},
                 {'title': 'Состав памяти', 'note': 'Живая память и область видеопамяти для промежуточных данных слоя. Рост значения «Вне области» указывает на фрагментацию памяти.',
                  'series': [ser(s['шаги'], 'живых', 'Живая память', 's1'), ser(s['шаги'], 'область', 'Область видеопамяти', 's3')], 'opts': {'zero': True, 'fy': 'int', 'ft': 'mib'}}],
             'stats': _vram_stats,
             'table': lambda s: (['Шаг', 'Пик', 'Живая память', 'Блоков', 'Область видеопамяти', 'Вне области'],
                                 [[int_(r['шаг']), int_(r['vram']), int_(r['живых']), int_(r['блоков']), int_(r['область']), int_(r['мимо'])]
                                  for r in reversed(s['шаги'][-200:])])},
    'temp': {'title': 'Температура', 'note': 'Самая горячая точка кристалла, память и край кристалла.',
             'now': lambda s: (lambda h: {'v': deg(h), 'flag': 'Троттлинг' if okn(h) and h > temp_warn('горячая') else ''})(last_of(s['шаги'], 'горячая')),
             'chart': lambda s: {'series': temp_series(s), 'opts': {'fy': 'deg', 'fk': 'deg', 'ft': 'degc'}},
             'big': lambda s: [
                 {'title': 'Температуры', 'series': temp_series(s), 'opts': {'fy': 'deg', 'ft': 'degc'}},
                 {'title': 'Вентилятор', 'note': 'Обороты в минуту. Нулевое значение на холодной карте является нормой.',
                  'series': [ser(s['шаги'], 'вент', 'Об/мин', 's1')], 'opts': {'zero': True, 'ft': 'rpm'}}],
             'stats': _temp_stats,
             'table': lambda s: (['Шаг', 'Горячая точка', 'Память', 'Край кристалла', 'Вентилятор'],
                                 [[int_(r['шаг']), fmt(r['горячая'], 0), fmt(r['память'], 0), fmt(r['край'], 0), int_(r['вент'])]
                                  for r in reversed(s['шаги'][-200:])])},
    'time': {'title': 'Длительность шага', 'note': 'Всплески соответствуют валидации и сохранению снимка весов.',
             'now': lambda s: (lambda v, t: {'v': fmt(v, 3), 'unit': 'с' if okn(v) else '', 'sub': f'{int_(t)} ток/с' if okn(t) else ''})(
                 last_of(s['шаги'], 'сек'), last_of(s['шаги'], 'ток')),
             'chart': lambda s: {'series': [ser(s['шаги'], 'сек', 'Секунд', 's1')], 'opts': {'zero': True, 'ft': 'sec'}},
             'big': lambda s: [
                 {'title': 'Секунд на шаг', 'series': [ser(s['шаги'], 'сек', 'Секунд', 's1')], 'opts': {'zero': True, 'ft': 'sec'}},
                 {'title': 'Состав шага', 'note': 'Прямой проход, LM head, обратный проход с шагом оптимизатора.',
                  'series': [ser(s['шаги'], 'обратный', 'Обратный проход', 's1'), ser(s['шаги'], 'прямой', 'Прямой проход', 's2'),
                             ser(s['шаги'], 'головка', 'LM head', 's3')], 'opts': {'zero': True, 'ft': 'sec'}}],
             'stats': _time_stats,
             'table': lambda s: (['Шаг', 'Секунд', 'Прямой проход', 'LM head', 'Обратный проход', 'Ток/с'],
                                 [[int_(r['шаг']), fmt(r['сек'], 3), fmt(r['прямой'], 3), fmt(r['головка'], 3), fmt(r['обратный'], 3), int_(r['ток'])]
                                  for r in sorted(s['шаги'], key=lambda r: -(r.get('сек') or 0))[:60]])},
    'grad': {'title': 'Норма градиента (|g|)', 'note': 'Показатель того, насколько сильно требуется изменить веса.',
             'now': lambda s: (lambda lr: {'v': num(last_of(s['шаги'], 'гн'), 1), 'sub': f'LR {lr}' if lr else ''})((s.get('последний') or {}).get('lr')),
             'chart': lambda s: {'series': [ser(s['шаги'], 'гн', 'Норма', 's1')], 'opts': {'zero': True, 'fy': 'kilo', 'ft': 'num1'}},
             'big': lambda s: [x for x in [
                 {'title': 'Норма по шагам', 'series': [ser(s['шаги'], 'гн', 'Норма', 's1'), ser(s['шаги'], 'гн_макс', 'Максимум', 's2')],
                  'opts': {'zero': True, 'fy': 'kilo', 'ft': 'num1'}},
                 {'title': 'LR', 'note': 'Разгон в начале, затем плавное снижение.',
                  'series': [ser(s['шаги'], 'lr', 'LR', 's1')], 'opts': {'zero': True, 'fy': 'exp1', 'ft': 'exp3'}},
                 sweep_chart(s)] if x],
             'stats': _grad_stats,
             'table': lambda s: (['Шаг', 'Норма', 'Максимум', 'Потеря (loss)'],
                                 [[int_(r['шаг']), num(r['гн'], 1), num(r['гн_макс'], 1), fmt(r['loss'], 4)] for r in reversed(s['шаги'][-200:])])},
}
ALL_CHARTS = ['loss', 'acc', 'vram', 'temp', 'time', 'grad']
STAGES = [('Токены', 'Токены'), ('Подмешивание', 'Подмешивание'), ('Замер', 'Замер слоёв'), ('Защита', 'Защита общих знаний'), ('Развёртка', 'Подбор LR'),
          ('Раскладка', 'Раскладка среза'), ('Обучение', 'Обучение'), ('Завершено', 'Завершено')]
STATE_WORD = {'идёт': 'Выполняется', 'завершено': 'Завершено', 'остановлено': 'Остановлено', 'ошибка': 'Ошибка', 'нет': 'Не начато'}
STATE_PILL = {'идёт': 'live', 'завершено': 'ok', 'остановлено': 'warn', 'ошибка': 'bad', 'нет': ''}
GOAL_NAME = {'срез': 'Срез строк по всей сети', 'точно': 'Точность: факты', 'характер': 'Характер: голос и манера', 'lora': 'LoRA'}
GOAL_SHORT = {'срез': 'Срез строк', 'точно': 'Точность', 'характер': 'Характер', 'lora': 'LoRA'}


def lora_of_run(s):
    # ап1
    a = (s.get('способ') or {}).get('LORA') or (((s.get('выбор') or {}).get('env')) or {}).get('LORA') or ''
    return T.lora_norm(a) if a else None


def goal_of_run(s):
    if lora_of_run(s):
        return 'lora'
    mt = (s.get('способ') or {}).get('MATS') or (((s.get('выбор') or {}).get('env')) or {}).get('MATS') or ''
    return 'точно' if mt.startswith('down_proj') else 'характер' if mt.startswith('attn') else 'срез'


def goal_line(s):
    # ас1
    lp = lora_of_run(s)
    return T.lora_desc(lp) if lp else GOAL_NAME[goal_of_run(s)]


def best_of(s):
    L = vals(s['оценки'], 'loss')
    if not L:
        return None
    b = min(L)
    e = next((x for x in s['оценки'] if x['loss'] == b), {})
    return {'v': b, 'step': e.get('шаг')}


def load_info(s):
    m = re.search(r'([\d.]+)/([\d.]+) ГиБ, ([\d.]+) МБ/с', s.get('загрузка') or '')
    return {'a': float(m.group(1)), 'b': float(m.group(2)), 'v': float(m.group(3))} if m else None


def stage_frac(s, k):
    p, L = s.get('план') or {}, s.get('последний') or {}
    ld = load_info(s)
    if k == 'Развёртка':
        return min(1, len(s.get('развёртка') or []) / 40) or (0.02 if ld else None)
    if k == 'Обучение':
        return min(1, L['шаг'] / p['всего']) if L.get('шаг') and p.get('всего') else (0 if ld else None)
    if k == s.get('этап') and s.get('ход_этапа'):
        try:
            f = float(s['ход_этапа'].split('\t')[0])
            return min(1, f) if f >= 0 else None
        except ValueError:
            return None
    return None


def stage_cap(s):
    return cap((s.get('ход_этапа') or '').split('\t')[1] if '\t' in (s.get('ход_этапа') or '') else '')


def stage_text(s):
    return {'Токены': 'Подготовка токенов', 'Подмешивание': 'Разметка подмешивания', 'Замер': 'Замер слоёв', 'Защита': 'Защита общих знаний: поиск главных направлений',
            'Развёртка': f'Подбор LR · {len(s.get("развёртка") or [])} из 40', 'Раскладка': 'Раскладка среза',
            'Обучение': 'Обучение', 'Завершено': 'Завершено'}.get(s.get('этап') or '', '')


def train_vm(s, live_name, stopping=False, act_err=''):
    # ач1
    st, L, p, live = s['состояние'], s.get('последний') or {}, s.get('план') or {}, s['идёт']
    cs, tot, best = L.get('шаг') or 0, p.get('всего') or 0, best_of(s)
    ep = min(p.get('эпох') or 1, math.floor((cs - 1) / p['на_эпоху']) + 1) if p.get('на_эпоху') and cs else (L.get('эпоха') or 0)
    sub_parts, t = [], ''
    B = lambda x: f'<b>{x}</b>'
    if live:
        ld = load_info(s)
        no_steps = not cs and not (s.get('этап') == 'Развёртка' and s.get('развёртка'))
        e = s.get('этап')
        if e == 'Развёртка':
            t = f'Подбор LR · {len(s.get("развёртка") or [])} из 40'
        elif e == 'Токены':
            t = 'Подготовка токенов'
        elif e == 'Подмешивание':
            t = 'Подмешивание: разметка ответов'
        elif e == 'Защита':
            t = 'Защита общих знаний'
        elif e == 'Раскладка':
            t = 'Раскладка среза по видеопамяти'
        elif cs:
            t = f'Шаг {int_(cs)}' + (f' из {int_(tot)}' if tot else '') + (f' · Эпоха {ep} из {p["эпох"]}' if p.get('эпох') else '')
        elif e == 'Обучение':
            t = 'Загрузка весов в ОЗУ'
        else:
            t = 'Запуск'
        if cs:
            if L.get('прошло'):
                sub_parts.append(f'Прошло {B(L["прошло"])}')
            if L.get('осталось'):
                sub_parts.append(f'Осталось {B(L["осталось"])}')
            if best:
                sub_parts.append(f'Лучшая потеря (loss) {B(fmt(best["v"], 4))} на шаге {int_(best["step"])}')
        elif e == 'Развёртка' and s.get('развёртка'):
            l = s['развёртка'][-1]
            sub_parts.append(f'LR {B(f"{l[1]:.3e}")} · Потеря (loss) {B(fmt(l[2], 4))}')
        if ld and no_steps:
            left = (ld['b'] - ld['a']) * 1073.741824 / ld['v'] if ld['v'] > 0 else 0
            t = 'Подбор LR: загрузка весов' if e == 'Развёртка' else 'Загрузка весов в ОЗУ'
            sub_parts = [f'{B(fmt(ld["a"], 1))} из {fmt(ld["b"], 1)} ГиБ', f'{round(ld["v"])} МБ/с',
                         f'Осталось ≈ {B(hms(left))}' if left > 0 else 'Завершение']
    elif st == 'завершено':
        t = f'Лучшая потеря (loss) {fmt(best["v"], 4)} на шаге {int_(best["step"])}' if best else 'Завершено'
        sub_parts = [x for x in ['Ранняя остановка' if (s.get('почему') or '').startswith('Остановка:') else cap(s.get('почему')) if s.get('почему') else 'Завершено',
                                 f'{B(int_(cs))}' + (f' из {int_(tot)}' if tot else '') + ' ' + steps_(tot or cs) if cs else '',
                                 B(L['прошло']) if L.get('прошло') else ''] if x]
    elif st == 'ошибка':
        t = f'Ошибка на шаге {int_(cs)}' if cs else 'Ошибка запуска'
        sub_parts = [f'Снимок весов на шаге {B(int_(s["продолжение"] - 1))}' if s.get('продолжение') else 'Снимка весов нет']
    elif st == 'остановлено':
        t = f'Остановлено на шаге {int_(cs)}' + (f' из {int_(tot)}' if tot else '') if cs else 'Остановлено'
        if s.get('продолжение'):
            sub_parts = [f'Снимок весов на шаге {B(int_(s["продолжение"] - 1))}', f'Продолжение с шага {int_(s["продолжение"])}']
        elif cs:
            sub_parts = ['Снимка весов нет']
        if best:
            sub_parts.append(f'Лучшая потеря (loss) {B(fmt(best["v"], 4))} на шаге {int_(best["step"])}')
    else:
        t = 'Обучение не начато'
    err = s.get('почему') if st == 'ошибка' else act_err
    other = bool(live_name) and live_name != s['имя']
    acts = []
    if live:
        acts.append({'id': 'stop', 'text': 'Остановка' if stopping else 'Остановить', 'kind': 'danger', 'icon': 'stop', 'on': not stopping})
    elif st == 'завершено':
        acts += [{'id': 'chat', 'text': 'Разговор', 'kind': 'primary', 'icon': 'chat', 'on': True},
                 {'id': 'bench', 'text': 'Проверка', 'kind': '', 'icon': 'bench', 'on': True},
                 {'id': 'setup', 'text': 'Новое обучение', 'kind': 'quiet', 'icon': 'setup', 'on': True}]
    elif st == 'нет' and not s['шаги'] and not s.get('этап'):
        acts.append({'id': 'setup', 'text': 'Новое обучение', 'kind': 'primary', 'icon': 'setup', 'on': True})
    else:
        acts.append({'id': 'resume', 'text': f'Продолжить с шага {int_(s["продолжение"])}' if s.get('продолжение') else 'Начать заново',
                     'kind': 'primary', 'icon': 'play', 'on': not other, 'tip': f'Недоступно: выполняется обучение «{live_name}».' if other else ''})
        if s.get('продолжение'):
            acts.append({'id': 'chat', 'text': 'Разговор', 'kind': '', 'icon': 'chat', 'on': True})
        acts.append({'id': 'setup', 'text': 'Новое обучение', 'kind': 'quiet', 'icon': 'setup', 'on': True})
    for a in acts:
        a['text'] = tr(a['text'])
        a['tip'] = tr(a.get('tip', ''))
    # ач2
    f, capt = None, ''
    ld = load_info(s)
    if live:
        if ld and not cs and not (s.get('этап') == 'Развёртка' and s.get('развёртка')):
            f, capt = ld['a'] / ld['b'] if ld['b'] else 0, 'Загрузка весов'
        elif s.get('этап') == 'Развёртка':
            f, capt = stage_frac(s, 'Развёртка'), f'Подбор LR · {len(s.get("развёртка") or [])} из 40'
        elif cs:
            f = stage_frac(s, 'Обучение')
            capt = f'Шаг {int_(cs)}' + (f' из {int_(tot)}' if tot else '') + (f' · Осталось {L["осталось"]}' if L.get('осталось') else '')
        else:
            g = stage_frac(s, s.get('этап'))
            nm = dict(STAGES).get(s.get('этап') or '', '')
            f = -1 if g is None else g
            capt = f'{nm} · {stage_cap(s)}' if stage_cap(s) else nm
    elif st == 'завершено':
        f, capt = 1, 'Завершено'
    bar = {'show': f is not None, 'frac': -1 if f is None else f, 'cap': tr(capt), 'pct': '' if f is None or f < 0 else f'{round(100 * f)}{NBSP}%',
           'ok': st == 'завершено'}
    # ач3
    ci = next((i for i, x in enumerate(STAGES) if x[0] == s.get('этап')), -1)
    if ci < 0:
        ci = 5 if s['шаги'] else -1
    v = s.get('выбор') or {}
    env = v.get('env') or {}
    fixed = env.get('LR')
    msgs = s.get('сообщения') or []
    slice_ = next((m2.group(0) for m2 in (re.search(r'(Целые матрицы: [^,]+, слои [\d,]+)|([\d.]+% строк)', m['текст']) for m in reversed(msgs)) if m2), '')
    rp = s.get('повтор') or env.get('REPLAY') or ''
    lrun = lora_of_run(s)                           # ач4
    stages = []
    for i, (k, name) in enumerate(STAGES):
        c = 'done' if i < ci else ('cur' if live else 'bad' if st == 'ошибка' else 'done' if (k == 'Завершено' or st == 'завершено') else 'stop') if i == ci else 'todo'
        if k == 'Развёртка' and c not in ('todo', 'cur') and (fixed or not s.get('развёртка')):
            c = 'skip'
        if k == 'Подмешивание' and c != 'cur' and (rp == '0' or (not rp and c != 'todo')):
            c = 'skip'
        mt = (s.get('способ') or {}).get('MATS') or env.get('MATS') or ''
        if k == 'Защита' and c not in ('cur', 'todo') and not mt and s.get('этап') != 'Защита':
            c = 'skip'
        if k == 'Замер' and c not in ('cur', 'todo') and s.get('этап') != 'Замер' and (not mt or env.get('MATS_AUTO') != '1'):
            c = 'skip'
        if lrun and k in ('Замер', 'Защита', 'Раскладка') and c != 'cur':
            c = 'skip'
        cp = ''
        if k == 'Токены':
            sv = s.get('сводка')
            cp = f'{int_(sv["examples"])} {plural(sv["examples"], "пример", "примера", "примеров")}' if sv else ''
        elif k == 'Развёртка':
            n = len(s.get('развёртка') or [])
            cp = f'{n} из 40' if c == 'cur' else ('Задано: ' + fixed if fixed else 'Не выполнялся') if c == 'skip' else '40 шагов' if c == 'todo' else (
                'Выбрано: ' + p['lr'] if p.get('lr') else f'{n} из 40')
        elif k == 'Подмешивание':
            cp = 'Отключено' if c == 'skip' else (stage_cap(s) or 'Выполняется') if c == 'cur' else (f'×{rp} к данным' if rp else '')
        elif k == 'Замер':
            cp = 'По выбору' if c == 'skip' and mt else 'Не требуется' if c == 'skip' else 'Шесть шагов' if c in ('cur', 'todo') else 'Слои ' + mt.split(':')[-1]
        elif k == 'Защита':
            cp = 'Не требуется' if c == 'skip' else stage_cap(s) if c == 'cur' else mt.replace(':', ' ')
        elif k == 'Раскладка':
            cp = 'Не требуется' if lrun else cap(slice_) if slice_ else ('Срез строк по видеопамяти' if c == 'todo' else '')
        elif k == 'Обучение':
            cp = f'Шаг {int_(L["шаг"])}' + (f' из {int_(tot)}' if tot else '') if L.get('шаг') else (f'{int_(tot)} {steps_(tot)}' if tot else '')
        elif k == 'Завершено':
            cp = f'Лучшая потеря {fmt(best["v"], 4)}' if best and c != 'todo' else ''
        fr = 1 if c in ('done', 'skip') else stage_frac(s, k) if c in ('cur', 'stop', 'bad') else 0
        stages.append({'k': k, 'name': tr(name), 'cls': c, 'cap': tr(cp), 'frac': -1 if fr is None else fr,
                       'icon': {'done': 'ok', 'cur': 'dot', 'stop': 'pause', 'bad': 'bad', 'skip': 'skip', 'todo': 'todo'}[c]})
    # ач5
    sw = None
    if live and s.get('этап') == 'Развёртка':
        sw = sweep_chart(s)
        d = s.get('развёртка') or []
        sw_cap = f'Точка {len(d)} из 40 · LR <b>{d[-1][1]:.3e}</b> · Потеря (loss) <b>{fmt(d[-1][2], 4)}</b>' if d else ''
    # ач6
    mo = v.get('модель') or {}
    chips = []
    if mo.get('модель'):
        chips.append(['Модель', f'{mo["модель"]} · {mo.get("вид") or ""} · {mo.get("основной") or ""}'])
    if v.get('режим_имя'):
        chips.append(['Режим', v['режим_имя']])
    chips.append(['Способ обучения', goal_line(s)])
    if v.get('датасет'):
        chips.append(['Данные', base(v['датасет'])])
    if p.get('lr'):
        chips.append(['LR', p['lr']])
    if p.get('упаковка') is not None:
        chips.append(['Упаковка', 'По одному примеру' if p['упаковка'] == '0' else p['упаковка'] + ' токенов'])
    if p.get('эпох'):
        chips.append(['Число эпох', str(p['эпох'])])
    if p.get('оценка'):
        chips.append(['Валидация каждые', p['оценка']])
    if p.get('сброс'):
        chips.append(['Снимок весов каждые', p['сброс']])
    if env.get('STOP_AFTER') not in (None, ''):
        sa = env['STOP_AFTER']
        chips.append(['Ранняя остановка', 'Отключена' if sa == '0' else f'{sa} {plural(int(sa), "валидация", "валидации", "валидаций")} без улучшения'])
    if env.get('X'):
        chips.append(['Дополнительные параметры', env['X']])
    chips = [[tr(a), tr(b)] for a, b in chips]
    journal = [{'time': (m.get('время') or '')[11:], 'event': tr(cap(m.get('событие'))), 'text': tr(cap(m.get('текст')))} for m in reversed(msgs[-3:])]
    return {'name': s['имя'], 'live': live, 'state': st, 'word': tr(STATE_WORD.get(st, cap(st))), 'pill': STATE_PILL.get(st, ''),
            'title': tr(t), 'sub': tr(' · '.join(sub_parts)), 'err': tr(cap(err or '')), 'acts': acts, 'bar': bar, 'stages': stages,
            'sweep': {'show': bool(sw), 'cap': tr(sw_cap) if sw else '', 'title': sw['title'] if sw else '', 'note': sw['note'] if sw else ''},
            'chips': chips, 'journal': journal, 'step': cs, 'total': tot, 'frac': min(1, cs / tot) if live and tot else 0,
            'sec': last_of(s['шаги'], 'сек') or 0, 'hasSteps': bool(s['шаги']), 'blank': False,
            'doc': f'«{s["имя"]}» ' + (f'{int_(cs)}' + (f' / {int_(tot)}' if tot else '') if cs else tr(stage_text(s))) if live else '',
            'stopNote': tr(f'Снимок весов на шаге {int_(s["продолжение"] - 1)} · Продолжение с шага {int_(s["продолжение"])}' if s.get('продолжение')
                           else 'Снимка весов пока нет, продолжение невозможно.') if live else ''}, sw


# ач7
BLANK_TV = {'name': '', 'live': False, 'state': '', 'word': '', 'pill': '', 'title': '', 'sub': '', 'err': '', 'acts': [],
            'bar': {'show': False, 'frac': 0, 'cap': '', 'pct': '', 'ok': False}, 'stages': [], 'sweep': {'show': False, 'cap': '', 'title': '', 'note': ''},
            'chips': [], 'journal': [], 'step': 0, 'total': 0, 'frac': 0, 'sec': 0, 'hasSteps': False, 'blank': True, 'doc': '', 'tiles': {}, 'stopNote': ''}


def tile_vm(s, k):
    w = WIN[k]
    n = w['now'](s)
    c = w['chart'](s)
    has = any(x['data'] for x in c['series'])
    ft = (c['opts'].get('fk') or c['opts'].get('ft') or 'f2')
    keys = []
    if len(c['series']) > 1 and has:
        for i, x in enumerate(c['series']):
            keys.append({'name': x['name'], 'color': x['color'], 'dots': x['dots'], 'value': fmtv(ft, x['data'][-1][1]) if i and x['data'] else ''})
    return {'title': tr(w['title']), 'value': n.get('v', '—') if has else '', 'unit': tr(n.get('unit', '')) if has else '',
            'sub': tr(n.get('sub', '')) if has else '', 'flag': tr(n.get('flag', '')), 'keys': keys, 'has': has}, c


def fmtv(spec, v, axis_step=None):
    # аэ1
    if not okn(v):
        return '—'
    u = lambda x: NBSP + tr(x)
    if spec == 'f4':
        return fmt(v, 4)
    if spec == 'f3':
        return fmt(v, 3)
    if spec == 'f2':
        return fmt(v, 2)
    if spec == 'f1':
        return fmt(v, 1)
    if spec == 'int':
        return int_(v)
    if spec == 'pct0':
        return f'{round(v)}%'
    if spec == 'pct2':
        return pct(v, 2)
    if spec == 'deg':
        return deg(v)
    if spec == 'degc':
        return f'{round(v)}{NBSP}°C'
    if spec == 'mib':
        return int_(v) + u('МиБ')
    if spec == 'sec':
        return fmt(v, 3) + u('с')
    if spec == 'kilo':
        return kilo(v)
    if spec == 'num1':
        return num(v, 1)
    if spec == 'exp1':
        return f'{v:.1e}' if v else '0'
    if spec == 'exp3':
        return f'{v:.3e}'
    if spec == 'rpm':
        return int_(v) + u('об/мин')
    if spec == 'pow10':
        return f'1e{round(v)}'
    if spec == 'lr10':
        return 'LR ' + f'{10 ** v:.3e}'
    if spec == 'step':
        return tr('Шаг') + ' ' + str(round(v))
    if axis_step is not None:
        return fmt(v, 0 if axis_step >= 1 else 1 if axis_step >= 0.1 else 2)
    return fmt(v, 2)


def win_vm(s, k):
    # аю1
    w = WIN[k]
    big = w['big'](s)
    cols, rows = w['table'](s)
    return {'title': tr(w['title']), 'note': tr(w['note']), 'sub': f'«{s["имя"]}»',
            'stats': [[tr(a), b, tr(c)] for a, b, c in w['stats'](s)],
            'charts': [{'title': tr(c['title']), 'note': tr(c.get('note', '')),
                        'keys': [{'name': x['name'], 'color': x['color'], 'dots': x['dots']} for x in c['series']] if len(c['series']) > 1 else []}
                       for c in big],
            'cols': [tr(c) for c in cols], 'rows': rows}, big


def log_vm(s):
    m = list(reversed(s.get('сообщения') or []))
    return {'title': tr('Журнал'), 'sub': f'«{s["имя"]}»',
            'events': [{'when': tr(when(x.get('время'), True)), 'full': x.get('время') or '', 'event': tr(cap(x.get('событие'))), 'text': tr(cap(x.get('текст')))}
                       for x in m],
            'out': '\n'.join(s.get('вывод') or [])}


def plan_vm(s):
    v, pl = s.get('выбор') or {}, s.get('план') or {}
    mo = v.get('модель') or {}
    sv = s.get('сводка')
    facts = [{'что': 'Модель', 'знач': f'{mo["модель"]} · {mo.get("вид") or ""} · {mo.get("основной") or ""}' if mo.get('модель') else '—',
              'как': f'{int_(mo["параметров"])} параметров · {mo.get("папка") or ""}' if mo.get('параметров') else ''},
             {'что': 'Данные', 'знач': base(v.get('датасет')) or '—',
              'как': (f'{int_(sv["examples"])} {plural(sv["examples"], "пример", "примера", "примеров")}: {int_(sv.get("обучение"))} на обучение, {int_(sv.get("отложено"))} на валидацию · ' if sv else '') + (v.get('датасет') or '')},
             {'что': 'LR', 'знач': pl.get('lr') or '—', 'как': 'Задано заранее' if (v.get('env') or {}).get('LR') else 'Подобрано автоматически'}]
    rows = facts + [r for r in (v.get('строки') or []) if r.get('к') != 'LR' or not pl.get('lr')] if v.get('строки') else facts
    sub = f'«{s["имя"]}» · Запуск {when(v["начало"])} · Режим «{v.get("режим_имя") or v.get("режим") or ""}»' if v.get('начало') else f'«{s["имя"]}»'
    env = v.get('env')
    return {'title': tr('Заданные параметры'), 'sub': tr(sub),
            'rows': [[tr(cap(r.get('что'))), tr(cap(str(r.get('знач', '')))), tr(cap(r.get('как') or ''))] for r in rows],
            'note': '' if v.get('строки') else tr('Запуск выполнен из терминала, записи о выбранных параметрах нет.'),
            'env': ' '.join(f'{k}="{x}"' if (re.search(r'\s', str(x)) or x == '') else f'{k}={x}' for k, x in env.items()) if env else ''}


# ба1
PCIE = {1: .25, 2: .5, 3: .985, 4: 1.969, 5: 3.938, 6: 7.563}      # ба2
GOAL_FACTS = {'срез': ['4,4 % строк всех матриц', 'Нормировки', 'Adam'], 'точно': ['Три матрицы down_proj целиком', 'NSGD + защита общих знаний', 'Градиент по ответу'],
              'характер': ['Внимание ранних слоёв', 'NSGD + защита общих знаний', 'Подмешивание сохраняет знания']}


def mode_facts(v, lora=False):
    if not v:
        return ['Все поля плана', 'Параметры движка']
    f = [f'LR {T.LORA_LR if lora else v["lr"]}' if v['lr'] else 'Подбор LR', f'Валидаций ≥ {v["оценок"]}']
    if v['эпохи'] != 1:
        f.append('Эпох × ' + dec('%g' % v['эпохи']))
    g = {'bf3': 'FP16×3 / BF16×3 с резервным режимом', 'fp32': 'FP32', 'железо': 'Умножение по оборудованию'}.get(v['умножение'], '')
    if g:
        f.append(g)
    f.append(f'Остановка: {v["терпение"]}')
    return f


def bus(H):
    # бв1
    g = (H.get('гпу') or {}).get('шина') or {}
    gen, w = int(g.get('поколение') or 0), int(g.get('линий') or 0)
    if not (gen and w):
        if 'pcie' not in _HWC:
            try:
                _HWC['pcie'] = tuple(int(x or 0) for x in _hwp().pcie(gpu_dev()))[:2]
            except Exception:
                _HWC['pcie'] = (0, 0)
        gen, w = _HWC['pcie'] if len(_HWC['pcie']) == 2 else (0, 0)
    return gen, w


def bus_time(H, b):
    gen, w = bus(H)
    r = PCIE.get(gen)
    return b / (r * w * 1e9) if r and w else None


def bus_name(H):
    gen, w = bus(H)
    return f'PCIe {gen}.0 ×{w}' if gen and w else 'шине'


def qsel(H):
    # бе1
    nm = H.get('база') or ''
    return next((b for b in H.get('базы') or [] if b['имя'] == nm), None) if nm else None


def frozen(H, lay):
    # бж1
    q = qsel(H)
    if q and q.get('годна'):
        return max(0, lay * 2 - (H.get('база_bf16') or 0)) + q['байт'], q['имя'][:-3]
    return lay * 2, ''


def quant_busy(H):
    w = H.get('настройка_что') or '' if H.get('настройка') else ''
    return w if w.startswith(('квант', 'импорт')) else ''


def base_vm(H):
    # би1
    cur, busy = H.get('база') or '', quant_busy(H)
    rows = [{'k': '', 'name': 'BF16', 'sub': tr('Как в исходном файле') + ' · ' + gib(H.get('база_bf16'), 1), 'on': not cur, 'bad': '', 'badTip': '', 'del': False,
             'tip': tr('Замороженные матрицы слоёв берутся из исходного файла без потерь.')}]
    for b in H.get('базы') or []:
        tys = ', '.join(f'{t} ×{n}' for t, n in b.get('типы') or [])
        err = b.get('ошибка')
        sub = [tr(b.get('откуда') or ''), gib(b['байт'], 1)] + ([tr('Ошибка') + ' ' + pct(100 * err)] if err is not None else [])
        tip = tys + ('. ' + tr(T.RECIPE_TIP[b['рецепт']]) if b.get('рецепт') in T.RECIPE_TIP else '') + \
            ('. ' + tr('Ошибка весов среднеквадратичная, относительно BF16.') if err is not None else '')
        rows.append({'k': b['имя'], 'name': b['имя'][:-3], 'sub': ' · '.join(x for x in sub if x), 'on': b['имя'] == cur, 'tip': tip,
                     'bad': tr('Непригодна') if not b.get('годна') else '', 'badTip': tr(cap(b.get('почему') or '')), 'del': True})
    m = H.get('модель') or {}
    can = bool(H.get('движок') and m.get('движок')) and not H.get('настройка')
    why = '' if can else tr('Выполняется другое задание') if H.get('настройка') else tr('Требуется собранный движок') if not H.get('движок') else tr('Требуется поддерживаемая модель')
    return {'rows': rows, 'cur': cur, 'busy': tr({'квант': 'Квантование', 'импорт': 'Перенос'}[busy.split(':')[0]]) + ' ' + base(busy.split(':', 1)[1]) if busy else '',
            'recipes': [[r['к'], f'{r["к"]} · {gib(r["байт"], 1)}'] for r in H.get('рецепты') or [] if r.get('байт')],
            'can': can, 'why': why, 'dir': H.get('база_папка') or ''}


def checks(H, P, ch, live):
    L, m, g = [], H.get('модель') or {}, H.get('гпу') or {}
    pl, pr, nm = (P or {}).get('план'), (P or {}).get('подготовка'), (P or {}).get('имя')

    def add(s, t, v, sub='', acts=()):
        L.append({'s': s, 't': t, 'v': v, 'sub': sub, 'acts': [list(a) for a in acts]})
    busy = (H.get('настройка_что') or '?') if H.get('настройка') else ''
    if not m.get('есть') and busy == 'веса':
        add('wait', 'Модель', 'Загрузка весов', m.get('папка', ''))
    elif not m.get('есть'):
        add('need', 'Модель', 'Отсутствует model.safetensors', m.get('папка', ''), [('Указать папку', 'папка', 1), ('Выбрать модель', 'веса', 0)])
    elif m.get('склеить') and busy == 'склеить':
        add('wait', 'Модель', 'Объединение частей весов в один файл', m.get('папка', ''))
    elif m.get('склеить'):
        add('need', 'Модель', cap(m.get('почему')), 'Исходные части удаляются только после проверки объединённого файла · ' + m.get('папка', ''),
            [('Объединить', 'склеить', 1), ('Другая папка', 'папка', 0)])
    elif not m.get('движок'):
        add('bad', 'Модель', cap(m.get('почему')), m.get('модель', ''), [('Другая папка', 'папка', 0)])
    else:
        add('ok', 'Модель', f'{m.get("модель")} · {m.get("вид")} · {m.get("основной")}')
    q, qb = qsel(H), quant_busy(H)
    if qb:
        add('info', 'База', ('Квантование ' if qb.startswith('квант') else 'Перенос ') + base(qb.split(':', 1)[1]),
            'Обучению не мешает: задание выполняется с низким приоритетом.')
    elif H.get('база') and not q:
        add('bad', 'База', f'База {H["база"][:-3]} отсутствует', '', [('BF16', 'base:', 1)])
    elif q and not q.get('годна'):
        add('bad', 'База', f'{q["имя"][:-3]}: {cap(q.get("почему"))}', '', [('BF16', 'base:', 1)])
    elif q:
        add('ok', 'База', q['имя'][:-3] + ' · ' + gib(q['байт'], 1))
    if ch['dataErr']:
        add('bad', 'Данные', cap(ch['dataErr']), '', [('Другой файл', 'pick', 0)])
    elif not pr or (not pr.get('идёт') and not pr.get('источник')):
        add('need', 'Данные', 'Не выбраны', '', [('Выбрать файл', 'pick', 1)])
    elif pr.get('идёт'):
        add('wait', 'Данные', f'Токенизация «{base(pr["источник"]["путь"])}»')
    elif pr.get('код') or not pr.get('сводка'):
        add('bad', 'Данные', 'Ошибка токенизации', (pr.get('вывод') or [''])[-1] if pr.get('вывод') else '', [('Повторить', 'prep', 0)])
    elif not pr.get('свежий'):
        add('bad', 'Данные', 'Файл изменён после токенизации', '', [('Токенизировать заново', 'prep', 1)])
    else:
        n = pr['сводка']['examples']
        add('ok', 'Данные', f'{base(pr["источник"]["путь"])} · {int_(n)} {plural(n, "пример", "примера", "примеров")}')
    if not ch['name']:
        add('need', 'Имя', 'Не задано')
    elif nm and not nm.get('ок'):
        add('bad', 'Имя', cap(nm.get('почему')), '', [(f'Открыть «{ch["name"]}»', 'open:' + ch['name'], 0)] if 'уже есть' in (nm.get('почему') or '') else [])
    elif nm:
        add('ok', 'Имя', ch['name'])
    if pl and pl.get('ок') is False and pr and pr.get('сводка'):
        add('bad', 'План', cap(pl.get('почему')), '', [('Вернуть подобранные значения', 'reset', 0)] if ch['mode'] == 'эксперт' and 'неверно' in (pl.get('почему') or '') else [])
    if busy == 'движок':
        add('wait', 'Движок', 'Сборка')
    elif H.get('движок'):
        add('ok', 'Движок', 'Собран' + (f' · {g["архитектура"]}' if g.get('архитектура') else ''))
    elif H.get('компилятор'):
        add('need', 'Движок', 'Собран для прежней версии' if H.get('движок_есть') else 'Не собран', 'Сборка под установленную карту занимает несколько минут · ' + H['компилятор'],
            [('Собрать', 'движок', 1)])
    else:
        add('bad', 'Движок', 'Набор разработки не установлен' if H.get('карта') else 'Видеокарта не найдена',
            'CUDA Toolkit' if H.get('карта') == 'nvidia' else 'ROCm: rocm-hip-sdk, hipblas, hipblaslt' if H.get('карта') == 'amd' else 'ROCm (AMD) · CUDA Toolkit (NVIDIA)')
    if busy == 'пакеты':
        add('wait', 'Токенизатор', 'Установка пакетов')
    elif not H.get('tokenizers'):
        add('need', 'Токенизатор', 'Пакет tokenizers не установлен', '', [('Установить', 'пакеты', 1)])
    elif not H.get('словарь'):
        add('need' if m.get('есть') else 'info', 'Токенизатор', 'Отсутствует tokenizer.json' if m.get('есть') else 'Поставляется вместе с весами',
            m.get('папка', '') if m.get('есть') else '', [('Скачать', 'веса', 0)] if m.get('есть') else [])
    else:
        add('ok', 'Токенизатор', 'Готов')
    if ch.get('tab') != 'lora' and ch['goal'] != 'срез':
        add('ok' if H.get('numpy') else 'need', 'Защита общих знаний', 'Готово' if H.get('numpy') else 'Пакет numpy не установлен', '', [] if H.get('numpy') else [('Установить', 'пакеты', 1)])
    if live:
        add('bad', 'Карта', f'Занята обучением «{live}»', '', [('Перейти', 'train', 1)])
    tot = (H.get('озу') or {}).get('всего') or 0
    if tot and m.get('параметров'):
        # бк1
        lay, qn = frozen(H, m['параметров'] - (m.get('эмбеддинги') or 0))
        cp = T.num((H.get('железо') or {}).get('mem_cap_pct'))
        try:
            rsv = _hwp().ram_reserve(tot, (H.get('озу') or {}).get('доступно') or tot, **({'cap_pct': int(cp)} if cp else {}))
        except Exception:
            rsv = min(8 << 30, max(5 << 30, tot // 4))      # бк2
        cache = max(0, min(lay, tot - rsv))
        if cache >= lay:
            add('ok', 'ОЗУ', f'{gib(tot)} · ' + tr('Кэш весов целиком'), f'Замороженные веса ({qn or "BF16"}): {gib(lay)}; системе остаётся {gib(tot - lay)}')
        else:
            add('warn', 'ОЗУ', f'Кэш весов {gib(cache)} из {gib(lay)}',
                f'Остальные {gib(lay - cache)} читаются с диска дважды за шаг, что замедляет обучение. Для разговора требуется {gib(lay)} ОЗУ.')
    v = H.get('видеопамять') or {}
    if not live:
        who = ', '.join(f'{x["кто"]} {gib(x["байт"])}' for x in v.get('кем') or [])
        if not v.get('всего'):
            add('info', 'Видеопамять', 'Проверка выполняется при запуске')
        elif v.get('занято', 0) > 0.15 * v['всего']:
            add('warn', 'Видеопамять', f'Занято {gib(v["занято"])} из {gib(v["всего"])}', who)
        else:
            add('ok', 'Видеопамять', f'Свободно {gib(v["всего"] - v.get("занято", 0))} из {gib(v["всего"])}', who)
    dk = H.get('диск') or {}
    if dk.get('свободно') is not None:
        # бк3
        sn = snap_bytes(H, ch)
        need = sn + (1 << 30) if sn else 20 * 2 ** 30
        add('warn' if dk['свободно'] < need else 'ok', 'Диск', f'Свободно {bytes_(dk["свободно"])}' + (f', нужно {bytes_(need)}' if sn and dk['свободно'] < need else ''),
            'Рабочая папка: снимки весов и состояние оптимизатора.')
    for c in L:
        c['t'], c['v'], c['sub'] = tr(c['t']), tr(c['v']), tr(c['sub'])
        for a in c['acts']:
            a[0] = tr(a[0])
    return L


def snap_bytes(H, ch):
    # бл1
    m = H.get('модель') or {}
    if not m.get('параметров'):
        return 0
    try:
        if ch.get('tab') == 'lora':
            return T.lora_size(ch['lora'])[1] * 12
        if ch['goal'] != 'срез':
            ly = ch['over'].get('слои правки', '') if ch['mode'] == 'эксперт' else ''
            return T.mats_params(ch['goal'], ly if re.fullmatch(r'\d+(,\d+)*', ly) else T.goal_layers(ch['goal'])) * 6
        return int(T.num((H.get('железо') or {}).get('tier1_params')) or 0) * 12
    except Exception:
        return 0


def setup_vm(H, P, ch, live, launching=False):
    m = H.get('модель') or {}
    pl, pr, nm = (P or {}).get('план'), (P or {}).get('подготовка'), (P or {}).get('имя')
    L = checks(H, P, ch, live)
    B = [c for c in L if c['s'] in ('bad', 'need', 'wait')]
    # бм1
    ms = 'need' if not m.get('есть') or m.get('склеить') else 'bad' if not m.get('движок') else 'ok'
    model = {'есть': bool(m.get('есть')), 'name': m.get('модель') or '?', 'path': m.get('папка') or '',
             'line': f'{m.get("вид", "")} · {m.get("основной", "")} · {gib(m.get("байт"))}' if m.get('есть') else '',
             'arch': m.get('архитектура') or '', 'why': tr(cap(m.get('почему') or '')) if m.get('есть') and not m.get('движок') else '',
             'chips': [f'<b>{int_(m.get("параметров"))}</b> ' + tr('параметров'), f'<b>{int_(m.get("слоёв"))}</b> ' + tr('слоёв'),
                       tr('Скрытая размерность') + f' <b>{int_(m.get("скрытый"))}</b>', tr('Словарь') + f' <b>{int_(m.get("словарь"))}</b>'] if m.get('параметров') else [],
             'pill': ms, 'pillText': tr('Требуется объединение частей' if m.get('склеить') else {'ok': 'Готова', 'need': 'Нет весов', 'bad': 'Не поддерживается'}[ms]),
             'base': base_vm(H) if m.get('есть') else None}
    # бм2
    mem = []
    t1 = T.num((H.get('железо') or {}).get('tier1_params')) or 0
    lo = ch.get('tab') == 'lora'
    if m.get('параметров'):
        lay = m['параметров'] - (m.get('эмбеддинги') or 0)
        if lo:                                       # бм3
            n_ = T.lora_size(ch['lora'])[1]
            learn = [[tr('Обучается'), int_(n_), pct(100 * n_ / m['параметров'], 2) + ' · ' + tr('Адаптеры FP32')],
                     [tr('С состоянием AdamW в видеопамяти'), gib(n_ * 12), tr('A, B и два момента')]]
        elif ch['goal'] != 'срез':                   # бм4
            ly = ch['over'].get('слои правки', '') if ch['mode'] == 'эксперт' else ''
            n_ = T.mats_params(ch['goal'], ly if re.fullmatch(r'\d+(,\d+)*', ly) else T.goal_layers(ch['goal']))
            learn = [[tr('Обучается'), int_(n_), pct(100 * n_ / m['параметров'], 2) + ' · FP32'],
                     [tr('С NSGD в видеопамяти'), gib(n_ * 6), tr('Вес и инерция')]]
        else:
            learn = [[tr('Обучается'), int_(t1) if t1 else '—', (pct(100 * t1 / m['параметров'], 2) + ' · FP32') if t1 else tr('Срез строк, FP32')],
                     [tr('С Adam в видеопамяти'), gib(t1 * 12) if t1 else '—', tr('Вес и два момента')]]
        fz, qn = frozen(H, lay)
        tb = bus_time(H, 2 * fz)
        mem = learn + [[tr('Заморожено в ОЗУ'), gib(fz), qn or tr('BF16, как в исходном файле')],
                       [tr('По шине за шаг'), gib(2 * fz), tr(f'≥ {fmt(tb, 2)} с на {bus_name(H)}' if tb else bus_name(H))]]
    # бм5
    lora = None
    if lo:
        lp = ch['lora']
        n_ad, n_par = T.lora_size(lp)
        bud = t1 * 12 if t1 else None
        over = bool(bud and n_par * 12 > bud)
        lora = {'r': str(lp['r']), 'alpha': f'{lp["alpha"]:g}', 'drop': f'{lp["drop"]:g}', 'tg': lp['tg'],
                'ranks': [[str(x), str(x)] for x in sorted({4, 8, 16, 32, 64, 128, lp['r']})],
                'drops': [[f'{x:g}', dec(f'{x:g}')] for x in sorted({0.0, 0.05, 0.1, lp['drop']})],
                'mats': [{'k': c, 'name': nm, 'on': c in lp['tg'], 'last': lp['tg'] == c} for c, nm in T.LORA_MATS],
                'sets': [['qv', 'q v'], ['qkvo', tr('Внимание')], ['qkvogud', tr('Все')]],
                'tiles': [[tr('Адаптеров'), int_(n_ad), tr('Пар A и B')],
                          [tr('Обучаемых параметров'), int_(n_par), pct(100 * n_par / m['параметров'], 2) + ' ' + tr('модели') if m.get('параметров') else ''],
                          [tr('Видеопамять'), gib(n_par * 12), tr('A, B и два момента')],
                          [tr('Множитель s'), dec(f'{lp["alpha"] / lp["r"]:.4g}'), 'α / r']],
                'over': over, 'overText': tr(f'Превышен бюджет {gib(bud)}: ранг слишком велик.') if over else '',
                'isDef': lp == T.LORA_DEF}
    modes = [{'k': r['к'], 'name': tr(r['имя']), 'tip': tr(cap(r['суть'])), 'chips': [tr(x) for x in mode_facts(r['знач'], lo)], 'on': r['к'] == ch['mode']}
             for r in H.get('режимы') or []]
    goals = [{'k': r['к'], 'name': tr(cap(r['имя'])), 'tip': tr(cap(r['суть'])), 'chips': [tr(x) for x in GOAL_FACTS.get(r['к'], [])], 'on': r['к'] == ch['goal']}
             for r in H.get('цели') or []]
    # бм6
    taken = bool(nm and not nm.get('ок') and ch['name'])
    data = {'state': 'none', 'tiles': [], 'out': '', 'err': '', 'instr': 0, 'sceneNote': '',
            'taken': tr(('Имя занято' if 'уже есть' in (nm.get('почему') or '') else 'Недопустимое имя') if taken else ''),
            'takenTip': tr(cap(nm.get('почему'))) if taken else '', 'path': (pr.get('источник') or {}).get('путь', '') if pr else '', 'pill': 'need', 'pillText': tr('Нет данных')}
    if ch['dataErr']:
        data.update(state='error', err=tr(cap(ch['dataErr'])), pill='bad', pillText=tr('Ошибка'))
    elif pr and pr.get('идёт'):
        data.update(state='parsing', cap=tr(f'Токенизация «{base((pr.get("источник") or {}).get("путь"))}»'), out='\n'.join(pr.get('вывод', [])[-6:]), pill='wait', pillText=tr('Токенизация'))
    elif pr and pr.get('код'):
        data.update(state='error', err=tr('Файл не токенизирован, причина указана ниже.'), out='\n'.join(pr.get('вывод', [])), pill='bad', pillText=tr('Ошибка'))
    elif pr and pr.get('сводка'):
        sv, src = pr['сводка'], pr.get('источник') or {}
        tok = tr('ток.')
        data.update(state='ready' if pr.get('свежий') else 'stale',
                    tiles=[[tr('Примеров'), int_(sv['examples']), ''], [tr('На обучение'), int_(sv.get('обучение')), ''], [tr('На валидацию'), int_(sv.get('отложено')), ''],
                           [tr('Длина, медиана'), int_(sv.get('median')), tok], [tr('Длина, p99'), int_(sv.get('p99')), tok], [tr('Самый длинный'), int_(sv.get('max')), tok],
                           [tr('Токенов в расчёте потери'), int_(sv.get('loss_tokens')), ''], [tr('Размер файла'), bytes_(src.get('размер')), '']],
                    pill='bad' if taken or not pr.get('свежий') else 'ok', pillText=tr('Ошибка' if taken or not pr.get('свежий') else 'Готово'))
        if sv.get('instr_rows'):
            data.update(instr=sv['instr_rows'], sceneNote=tr(f'С системной инструкцией: {int_(sv.get("with_instr"))} из {int_(sv["examples"])}'))
            ch['instrSeen'] = (data['path'], sv['instr_rows'])
    if data['state'] == 'parsing' and (ch.get('instrSeen') or ('',))[0] == data['path']:
        data['instr'] = ch['instrSeen'][1]              # бм7
    # бм8
    rows = (pl or {}).get('строки') or []
    row = lambda k: next((r for r in rows if r.get('к') == k), {})
    bigs = []
    if pl and pl.get('шагов'):
        ep = row('EPOCHS').get('знач') or ''
        dur = (pl.get('длительность') or '').split(' — ')
        bigs = [[int_(pl['шагов']), tr(cap(steps_(pl['шагов']))), ''],
                [ep, tr(cap(plural(int(ep) if str(ep).isdigit() else 0, 'эпоха', 'эпохи', 'эпох'))), ''],
                [int_(pl['оценок']), tr(cap(plural(pl['оценок'], 'валидация', 'валидации', 'валидаций'))), ''],
                [tr(dur[0]) if dur[0] else '—', tr('Время'), tr(cap(' — '.join(dur[1:])))]]
    why = ''
    if ch['launchErr']:
        why = tr('Не удалось запустить: ') + tr(cap(ch['launchErr']))
    elif B:
        more = len(B) - 1
        why = f'{B[0]["t"]}: {B[0]["v"]}' + ((f' · +{more}' if LANG == 'en' else f' · Ещё {more}') if more else '')
    st = ('bad' if any(c['s'] == 'bad' for c in B) else 'wait' if any(c['s'] == 'wait' for c in B) else 'need') if B else 'ok'
    ex = ch['mode'] == 'эксперт'
    bad = set(((re.match(r'^неверно: ([^;]*)', (pl or {}).get('почему') or '', re.I) or [None, ''])[1] or '').split(', ')) - {''}
    prow = []
    for r in rows:
        raw = '' if r.get('сырое') is None else str(r.get('сырое'))
        zn = cap(str(r.get('знач', '')))
        ph = zn if raw == '' else (f'{raw} · {zn}' if re.search('[а-яёa-z]', zn, re.I) and zn != raw else raw)
        prow.append({'k': r.get('к') or '', 'label': tr(cap(r.get('что'))), 'value': dec(tr(zn)), 'tip': dec(tr(cap(r.get('как') or ''))),
                     'edit': bool(ex and r.get('к')), 'ph': tr(ph), 'own': ch['over'].get(r.get('к') or '', ''), 'bad': (r.get('к') or '') in bad,
                     'opts': [[k2, tr(cap(t2))] for k2, t2 in r.get('варианты') or []], 'cur': str(ch['over'].get(r.get('к'), r.get('сырое', '')))})
    launch = {'pill': st, 'pillText': tr({'ok': 'Готово', 'wait': 'Подготовка', 'bad': 'Ошибка', 'need': 'Требуется действие'}[st]),
              'bigs': bigs, 'can': not B and not launching, 'why': why, 'checks': L, 'rows': prow, 'expert': ex,
              'planEmpty': tr('Токенизация данных' if pr and pr.get('идёт') else 'Нет данных') if not rows else '',
              'ownCount': len(ch['over']), 'env': (pl or {}).get('env') or {}}
    sub = ' · '.join(x for x in [m.get('модель'), next((tr(r['имя']) for r in H.get('режимы') or [] if r['к'] == ch['mode']), ''), base(ch['file'])] if x)
    return {'model': model, 'mem': mem, 'modes': modes, 'goals': goals, 'data': data, 'launch': launch, 'sub': sub,
            'blocked': [c['t'] + ': ' + c['v'] for c in B], 'tab': 'lora' if lo else 'tier', 'lora': lora}


def load_line(t):
    # бн1
    m = re.search(r'([\d.]+)/([\d.]+) ГиБ(?:, ([\d.]+) МБ/с)?', t or '')
    if not m:
        return tr(cap(t)), -1.0
    a, b = float(m.group(1)), float(m.group(2))
    return tr('Загрузка весов') + f' · {fmt(a, 1)} ' + tr('из') + f' {fmt(b, 1)} ' + tr('ГиБ') + (f' · {round(float(m.group(3)))} ' + tr('МБ/с') if m.group(3) else ''), (a / b if b else -1.0)


def free_name(H, b):
    taken = {r['имя'] for r in H.get('запуски') or []}
    s = re.sub(r'\.[^.]*$', '', b or 'run')
    s = re.sub(r'[^\w.-]+', '-', s).strip('.-')[:34] or 'run'
    n, i = s, 2
    while n in taken:
        n, i = f'{s}-{i}', i + 1
    return n


# бо1
R_FILTERS = [['all', 'Все'], ['идёт', 'Выполняются'], ['завершено', 'Завершены'], ['остановлено', 'Остановлены'], ['ошибка', 'Ошибки']]


def runs_vm(H, q, f, sort, d):
    q = (q or '').strip().lower()
    L = [r for r in H.get('запуски') or [] if (f == 'all' or r['состояние'] == f) and (not q or q in r['имя'].lower() or q in (r.get('датасет') or '').lower())]

    def key(r):
        if sort == 'лучшая':
            return (r['лучшая'] is None, r['лучшая'] or 0) if d > 0 else (r['лучшая'] is not None, -(r['лучшая'] or 0))
        if sort in ('размер', 'шаг'):
            return r.get(sort) or 0
        return str(r.get(sort) or '')
    L.sort(key=key, reverse=(d < 0 and sort != 'лучшая'))
    if sort == 'лучшая' and d < 0:
        L.sort(key=lambda r: (r['лучшая'] is None, -(r['лучшая'] or 0)))
    mode_name = {r['к']: r['имя'] for r in H.get('режимы') or []}
    rows = [{'name': r['имя'], 'goal': tr(GOAL_SHORT.get(r.get('цель'), '—')), 'mode': tr(mode_name.get(r.get('режим'), '') or '—'),
             'state': r['состояние'], 'word': tr(STATE_WORD.get(r['состояние'], r['состояние'])), 'pill': STATE_PILL.get(r['состояние'], ''),
             'step': (int_(r['шаг']) + (f' / {int_(r["всего"])}' if r.get('всего') else '')) if r.get('шаг') else '—',
             'best': fmt(r['лучшая'], 4) if r.get('лучшая') is not None else '—', 'time': r.get('прошло') or '—',
             'size': bytes_(r['размер']) if r.get('размер') is not None else '—', 'when': tr(when(r.get('когда'))), 'whenFull': (r.get('когда') or '')[:19],
             'data': r.get('датасет') or '', 'weights': bool(r.get('веса')), 'live': r['состояние'] == 'идёт'} for r in L]
    all_ = H.get('запуски') or []
    tot = sum(r.get('размер') or 0 for r in all_)
    dk = H.get('диск') or {}
    return {'rows': rows, 'count': len(all_), 'size': bytes_(tot), 'free': bytes_(dk['свободно']) if dk.get('свободно') else '',
            'empty': tr('Ничего не найдено' if all_ else 'Запусков пока нет')}


def compare_vm(states):
    pal = ['s1', 's2', 's3', 's4', 's5', 's6']
    C = [('Потеря (loss) на валидации', lambda s: ser(s['оценки'], 'loss', s['имя'], None, True), 'f4'),
         ('Потеря (loss) на обучении', lambda s: ser(s['шаги'], 'ср', s['имя'], None), 'f4'),
         ('Точность (acc) на валидации', lambda s: ser(s['оценки'], 'acc', s['имя'], None, True), 'pct2'),
         ('Секунд на шаг', lambda s: ser(s['шаги'], 'сек', s['имя'], None), 'sec')]
    charts = []
    for title, f, ft in C:
        series = []
        for j, s in enumerate(states):
            x = f(s)
            x['name'] = s['имя']
            x['color'] = pal[j]
            series.append(x)
        charts.append({'title': tr(title), 'series': series, 'opts': {'ft': ft, 'area': False}})
    rows = []
    def row(t, fn):
        rows.append([tr(t)] + [fn(s) for s in states])
    row('Способ', lambda s: tr(goal_line(s)))
    row('Режим', lambda s: tr((s.get('выбор') or {}).get('режим_имя') or '—'))
    row('Данные', lambda s: base((s.get('выбор') or {}).get('датасет')) or '—')
    row('Лучшая потеря (loss) на валидации', lambda s: (lambda b: f'{fmt(b["v"], 4)} · {tr("Шаг")} {int_(b["step"])}' if b else '—')(best_of(s)))
    row('Последняя потеря (loss) на валидации', lambda s: fmt(vals(s['оценки'], 'loss')[-1], 4) if vals(s['оценки'], 'loss') else '—')
    row('Точность (acc) на валидации', lambda s: pct(vals(s['оценки'], 'acc')[-1], 2) if vals(s['оценки'], 'acc') else '—')
    row('Шагов', lambda s: (int_((s.get('последний') or {}).get('шаг')) + (f' / {int_(s["план"]["всего"])}' if (s.get('план') or {}).get('всего') else '')))
    row('LR', lambda s: (s.get('план') or {}).get('lr') or '—')
    row('Секунд на шаг', lambda s: (fmt(med(vals(s['шаги'], 'сек')), 3) + NBSP + tr('с')) if vals(s['шаги'], 'сек') else '—')
    row('Время', lambda s: (s.get('последний') or {}).get('прошло') or '—')
    row('Состояние', lambda s: tr(STATE_WORD.get(s['состояние'], s['состояние'])))
    return {'names': [s['имя'] for s in states], 'colors': pal[:len(states)], 'charts': [{'title': c['title']} for c in charts], 'rows': rows}, charts


# бр1
NUMW = {'ноль': 0, 'нуль': 0, 'один': 1, 'одна': 1, 'одно': 1, 'два': 2, 'две': 2, 'три': 3, 'четыре': 4, 'пять': 5, 'шесть': 6, 'семь': 7, 'восемь': 8,
        'девять': 9, 'десять': 10, 'одиннадцать': 11, 'двенадцать': 12, 'тринадцать': 13, 'четырнадцать': 14, 'пятнадцать': 15, 'шестнадцать': 16,
        'семнадцать': 17, 'восемнадцать': 18, 'девятнадцать': 19, 'двадцать': 20, 'тридцать': 30, 'сорок': 40, 'пятьдесят': 50, 'сто': 100, 'тысяча': 1000,
        'zero': 0, 'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5, 'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10, 'twelve': 12, 'twenty': 20, 'hundred': 100}


def norm_a(s):
    s = re.sub(r'[«»"\'“”„.,!?;:()\[\]{}—–\-*_]', ' ', str(s or '').lower().replace('ё', 'е'))
    return [str(NUMW[w]) if w in NUMW else w for w in s.split()]


def match(ans, exp):
    a, e = norm_a(ans), norm_a(exp)
    if not e:
        return None
    A, E = ' ' + ' '.join(a) + ' ', ' ' + ' '.join(e) + ' '
    if E in A:
        return True
    if len(e) <= 3:
        return all(w in a for w in e) or bool(a and e[0] == a[0])
    return len([w for w in e if w in a]) / len(e) >= 0.7


def same(x, y):
    return ' '.join(norm_a(x)) == ' '.join(norm_a(y))


# бф1
def _prop(attr, sig):
    def g(self):
        return getattr(self, attr)

    def s(self, v):
        if isinstance(v, QJSValue):
            v = v.toVariant()
        if getattr(self, attr) != v:
            setattr(self, attr, v)
            getattr(self, sig).emit()
            self.update()
    return g, s


def _nice(lo, hi, n):
    if not (math.isfinite(lo) and math.isfinite(hi)):
        lo, hi = 0, 1
    if lo == hi:
        pad = abs(lo) * 0.05 or 1
        lo, hi = lo - pad, hi + pad
    raw = (hi - lo) / n
    p = 10 ** math.floor(math.log10(raw))
    f = raw / p
    stp = (1 if f < 1.5 else 2 if f < 3 else 5 if f < 7 else 10) * p
    return math.floor(lo / stp + 1e-9) * stp, math.ceil(hi / stp - 1e-9) * stp, stp


def _thin(a, W):
    n, capn = len(a), max(120, round(W * 0.75))
    if n <= capn * 2:
        return a
    out = []
    for b in range(capn):
        i0, i1 = b * n // capn, (b + 1) * n // capn
        if i1 <= i0:
            continue
        lo = hi = i0
        for i in range(i0 + 1, i1):
            if a[i][1] < a[lo][1]:
                lo = i
            if a[i][1] > a[hi][1]:
                hi = i
        out.append(a[min(lo, hi)])
        if lo != hi:
            out.append(a[max(lo, hi)])
    if out[-1] is not a[-1]:
        out.append(a[-1])
    return out


def _near(d, x):
    lo, hi = 0, len(d) - 1
    while hi - lo > 1:
        m = (lo + hi) >> 1
        if d[m][0] < x:
            lo = m
        else:
            hi = m
    return d[lo] if abs(d[lo][0] - x) <= abs(d[hi][0] - x) else d[hi]


THEME = {
    'dark': {'ink': '#ededf0', 'ink2': '#a3a3ad', 'ink3': '#76767f', 'ink4': '#45454c', 'sep': '#1effffff', 'grid': '#14ffffff',
             'pop': '#f02e2e34', 'popEdge': '#24ffffff', 'solid': '#2b2b31', 's2': '#ff9f0a', 's3': '#ff6482', 's4': '#64d2ff', 's5': '#bf5af2', 's6': '#30d158'},
    'light': {'ink': '#1d1d1f', 'ink2': '#5a5a63', 'ink3': '#8a8a93', 'ink4': '#c4c4ca', 'sep': '#14000000', 'grid': '#0f000000',
              'pop': '#f5fcfcfd', 'popEdge': '#1a000000', 'solid': '#ffffff', 's2': '#d67a00', 's3': '#d63864', 's4': '#0a8fc8', 's5': '#9a3cd0', 's6': '#1f9d3a'}}


class Chart(QQuickPaintedItem):
    # бш1
    keyChanged, revChanged, lightChanged, accentChanged, lwChanged, fillChanged = Signal(), Signal(), Signal(), Signal(), Signal(), Signal()
    gridChanged, dotsChanged, familyChanged, pxChanged, compactChanged, emptyChanged = Signal(), Signal(), Signal(), Signal(), Signal(), Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._key, self._rev, self._light, self._accent, self._lw, self._fill = '', 0, False, QColor('#0a84ff'), 1.75, True
        self._grid, self._dots, self._family, self._px, self._compact, self._empty = True, True, 'TL Inter', 11.0, False, ''
        self._hx = None
        self.setAcceptHoverEvents(True)
        self.setAntialiasing(True)

    key = Property(str, *_prop('_key', 'keyChanged'), notify=keyChanged)
    rev = Property(int, *_prop('_rev', 'revChanged'), notify=revChanged)
    light = Property(bool, *_prop('_light', 'lightChanged'), notify=lightChanged)
    accent = Property(QColor, *_prop('_accent', 'accentChanged'), notify=accentChanged)
    lw = Property(float, *_prop('_lw', 'lwChanged'), notify=lwChanged)
    fill = Property(bool, *_prop('_fill', 'fillChanged'), notify=fillChanged)
    grid = Property(bool, *_prop('_grid', 'gridChanged'), notify=gridChanged)
    dots = Property(bool, *_prop('_dots', 'dotsChanged'), notify=dotsChanged)
    family = Property(str, *_prop('_family', 'familyChanged'), notify=familyChanged)
    px = Property(float, *_prop('_px', 'pxChanged'), notify=pxChanged)
    compact = Property(bool, *_prop('_compact', 'compactChanged'), notify=compactChanged)
    empty = Property(str, *_prop('_empty', 'emptyChanged'), notify=emptyChanged)

    def hoverMoveEvent(self, e):
        self._hx = e.position().x(), e.position().y()
        self.update()

    def hoverLeaveEvent(self, e):
        self._hx = None
        self.update()

    def color(self, c):
        th = THEME['light' if self._light else 'dark']
        if not c or c == 's1':
            return QColor(self._accent)
        return QColor(th.get(c, c))

    def paint(self, p):
        c = BK.charts.get(self._key) if BK else None
        th = THEME['light' if self._light else 'dark']
        W, H = self.width(), self.height()
        if W < 20 or H < 20:
            return
        p.setRenderHint(QPainter.Antialiasing)
        p.setRenderHint(QPainter.TextAntialiasing)
        font = QFont(self._family)
        font.setPixelSize(max(9, round(self._px - 2.5)))
        p.setFont(font)
        fm = QFontMetricsF(font)
        series = [dict(x, data=_thin(x['data'], W)) for x in (c or {}).get('series') or [] if x.get('data')]
        if not series:
            if self._empty:
                p.setPen(QColor(th['ink3']))
                p.drawText(QRectF(0, 0, W, H), Qt.AlignCenter, self._empty)
            return
        o = c.get('opts') or {}
        cmp_ = self._compact
        x0 = min(d[0][0] for d in (s['data'] for s in series))
        x1 = max(d[-1][0] for d in (s['data'] for s in series))
        ys = [y for s in series for _x, y in s['data']]
        y0, y1 = min(ys), max(ys)
        if o.get('rule') is not None and okn(o.get('rule')):
            y0, y1 = min(y0, o['rule']), max(y1, o['rule'])
        vr = o.get('vrule')
        if vr and okn(vr.get('x')):
            x0, x1 = min(x0, vr['x']), max(x1, vr['x'])
        if o.get('zero'):
            y0 = min(0, y0)
        lo, hi, st = _nice(y0, y1, 3 if cmp_ else 4)
        ticks, i = [], 0
        while lo + i * st <= hi + st * 1e-6:
            ticks.append(lo + i * st)
            i += 1
        fy = o.get('fy')
        tl = [fmtv(fy, v) if fy else fmtv(None, v, st) for v in ticks]
        Tm = 16 if (o.get('rule') is not None or vr) else 8
        Bm = 20
        Lm = max(fm.horizontalAdvance(t) for t in tl) + 12
        sy = lambda v: Tm + (1 - (v - lo) / ((hi - lo) or 1)) * (H - Tm - Bm)
        ends = sorted(({'s': s, 'y': sy(s['data'][-1][1])} for s in series), key=lambda e: e['y'])
        direct = not cmp_ and 1 < len(series) <= 4 and all(i == 0 or ends[i]['y'] - ends[i - 1]['y'] >= 15 for i in range(len(ends)))
        Rm = max(fm.horizontalAdvance(s['name']) for s in series) + 18 if direct else 8
        sx = lambda v: Lm + (0 if x1 == x0 else (v - x0) / (x1 - x0)) * (W - Lm - Rm)
        ax = QColor(th['ink3'])
        # бш2
        for v, t in zip(ticks, tl):
            y = round(sy(v)) + .5
            if self._grid:
                p.setPen(QPen(QColor(th['grid']), 1))
                p.drawLine(QPointF(Lm, y), QPointF(W - Rm, y))
            p.setPen(ax)
            p.drawText(QRectF(0, y - 8, Lm - 8, 16), Qt.AlignRight | Qt.AlignVCenter, t)
        # бш3
        xt = [v for v in o['xt'] if x0 - 1e-9 <= v <= x1 + 1e-9] if o.get('xt') else ([x0, x1] if cmp_ or x1 == x0 else [x0 + (x1 - x0) * k / 3 for k in range(4)])
        prev = None
        for k, v in enumerate(xt):
            if prev is not None and v == prev:
                continue
            prev = v
            px = sx(v)
            if o.get('fx'):
                lab = fmtv(o['fx'], v)
            else:
                lab = (tr('Шаг') + ' ' if k == 0 else '') + str(round(v))
            w = fm.horizontalAdvance(lab)
            if o.get('xt'):
                left = px if px < Lm + 14 else px - w if px > W - Rm - 14 else px - w / 2
            else:
                left = px if k == 0 else px - w if k == len(xt) - 1 else px - w / 2
            p.setPen(ax)
            p.drawText(QPointF(left, H - 5), lab)
        # бш4
        if o.get('rule') is not None and okn(o.get('rule')):
            y = round(sy(o['rule'])) + .5
            pen = QPen(QColor(th['ink4']), 1)
            pen.setDashPattern([4, 4])
            p.setPen(pen)
            p.drawLine(QPointF(Lm, y), QPointF(W - Rm, y))
            if o.get('ruleLabel'):
                p.setPen(ax)
                p.drawText(QRectF(Lm, y - 18, W - Rm - Lm, 14), Qt.AlignRight | Qt.AlignVCenter, o['ruleLabel'])
        if vr and okn(vr.get('x')):
            x = round(sx(vr['x'])) + .5
            pen = QPen(QColor(self._accent), 1)
            pen.setDashPattern([3, 3])
            p.setPen(pen)
            p.drawLine(QPointF(x, Tm - 4), QPointF(x, H - Bm))
            lab = vr.get('label') or ''
            w = fm.horizontalAdvance(lab)
            p.setPen(QColor(self._accent))
            p.drawText(QPointF(x - 6 - w if x > W - Rm - w - 12 else x + 6, Tm + 6), lab)
        # бш5
        yb = sy(max(lo, min(hi, 0 if o.get('zero') else lo)))
        for idx, s in enumerate(series):
            col = self.color(s.get('color'))
            pts = [QPointF(sx(x), sy(y)) for x, y in s['data']]
            if idx == 0 and not s.get('dots') and o.get('area', True) is not False and self._fill and len(pts) > 1:
                path = QPainterPath(pts[0])
                for q in pts[1:]:
                    path.lineTo(q)
                path.lineTo(pts[-1].x(), yb)
                path.lineTo(pts[0].x(), yb)
                path.closeSubpath()
                g = QLinearGradient(0, Tm, 0, yb)
                c1 = QColor(col)
                c1.setAlphaF(0.20 if not self._light else 0.16)
                c2 = QColor(col)
                c2.setAlphaF(0.0)
                g.setColorAt(0, c1)
                g.setColorAt(1, c2)
                p.setPen(Qt.NoPen)
                p.setBrush(QBrush(g))
                p.drawPath(path)
                p.setBrush(Qt.NoBrush)
            pen = QPen(col, self._lw)
            pen.setCapStyle(Qt.RoundCap)
            pen.setJoinStyle(Qt.RoundJoin)
            p.setPen(pen)
            if len(pts) > 1:
                p.drawPolyline(QPolygonF(pts))
            sparse = s.get('dots') and (len(pts) < 2 or (W - Lm - Rm) / max(1, len(pts) - 1) >= 12)
            ring = QPen(QColor(th['solid']), 1.5)
            for q in (pts if sparse and self._dots else pts[-1:]):
                p.setPen(ring)
                p.setBrush(col)
                p.drawEllipse(q, 3.6, 3.6)
            p.setBrush(Qt.NoBrush)
        if direct:
            for e in ends:
                p.setPen(QColor(th['ink2']))
                p.drawText(QPointF(W - Rm + 10, e['y'] + 4), e['s']['name'])
        # бш6
        if self._hx:
            hx, hy = self._hx
            if Lm - 6 <= hx <= W - Rm + 6:
                xv = x0 + max(0, min(1, (hx - Lm) / ((W - Lm - Rm) or 1))) * (x1 - x0)
                hits = [_near(s['data'], xv) for s in series]
                main = max(range(len(series)), key=lambda i: len(series[i]['data']))
                cx = sx(hits[main][0])
                pen = QPen(QColor(th['ink3']), 1)
                pen.setDashPattern([2, 3])
                p.setPen(pen)
                p.drawLine(QPointF(cx, Tm), QPointF(cx, H - Bm))
                for s, h in zip(series, hits):
                    p.setPen(QPen(QColor(th['solid']), 1.5))
                    p.setBrush(self.color(s.get('color')))
                    p.drawEllipse(QPointF(sx(h[0]), sy(h[1])), 4, 4)
                p.setBrush(Qt.NoBrush)
                ft = o.get('ft')
                head = fmtv(o['tx'], hits[main][0]) if o.get('tx') else tr('Шаг') + ' ' + str(round(hits[main][0]))
                lines = []
                for s, h in zip(series, hits):
                    extra = (' · ' + tr('Шаг') + ' ' + str(round(h[0]))) if len(series) > 1 and abs(h[0] - hits[main][0]) > 0.5 else ''
                    lines.append((self.color(s.get('color')), fmtv(ft, h[1]) if ft else fmt(h[1]), s['name'] + extra))
                bf = QFont(font)
                bf.setWeight(QFont.DemiBold)
                fb = QFontMetricsF(bf)
                lh = fm.height() + 3
                bw = max([fm.horizontalAdvance(head)] + [14 + fb.horizontalAdvance(v) + 7 + fm.horizontalAdvance(n) for _c, v, n in lines]) + 20
                bh = lh * (len(lines) + 1) + 12
                bx = hx + 14 if hx + 14 + bw <= W else hx - 14 - bw
                bx = max(0, bx)
                by = max(0, min(H - bh, hy - bh / 2))
                p.setPen(QPen(QColor(th['popEdge']), 1))
                p.setBrush(QColor(th['pop']))
                p.drawRoundedRect(QRectF(bx, by, bw, bh), 8, 8)
                p.setBrush(Qt.NoBrush)
                p.setPen(QColor(th['ink3']))
                p.drawText(QPointF(bx + 10, by + 6 + fm.ascent()), head)
                for i, (col, v, n) in enumerate(lines):
                    yy = by + 6 + lh * (i + 1)
                    p.setPen(Qt.NoPen)
                    p.setBrush(col)
                    p.drawEllipse(QPointF(bx + 14, yy + fm.ascent() / 2 + 1), 4, 4)
                    p.setBrush(Qt.NoBrush)
                    p.setFont(bf)
                    p.setPen(QColor(th['ink']))
                    p.drawText(QPointF(bx + 24, yy + fm.ascent()), v)
                    p.setFont(font)
                    p.setPen(QColor(th['ink2']))
                    p.drawText(QPointF(bx + 24 + fb.horizontalAdvance(v) + 7, yy + fm.ascent()), n)


BG = {   # бш7
    'aurora': {'d': ('#0a0b10', [(.12, 0, 1200, 700, '#1d2b5a', .6), (.95, .15, 900, 600, '#3a1f4f', .55), (.6, 1.1, 1000, 800, '#0f3b3a', .6)]),
               'l': ('#f5f5f7', [(.12, 0, 1200, 700, '#d8e3ff', .6), (.95, .15, 900, 600, '#f1dcf2', .55), (.6, 1.1, 1000, 800, '#d5f0ea', .6)])},
    'ocean': {'d': ('#070b12', [(0, 1, 1100, 800, '#0b3d5c', .6), (1, 0, 900, 700, '#123a6b', .6)]),
              'l': ('#f3f6fa', [(0, 1, 1100, 800, '#cfe9f7', .6), (1, 0, 900, 700, '#d9e4fb', .6)])},
    'dawn': {'d': ('#120d10', [(.2, 0, 1100, 700, '#4a2a3a', .6), (.9, 1, 900, 700, '#3b2c22', .6)]),
             'l': ('#faf6f4', [(.2, 0, 1100, 700, '#fbe1e6', .6), (.9, 1, 900, 700, '#fdeedd', .6)])},
    'ember': {'d': ('#0d0909', [(.85, 1.1, 1000, 700, '#4d1d0e', .6), (.1, 0, 800, 600, '#2a1430', .6)]),
              'l': ('#faf7f5', [(.85, 1.1, 1000, 700, '#fde3d3', .6), (.1, 0, 800, 600, '#efe0f5', .6)])},
    'lavender': {'d': ('#0c0b14', [(.5, -.1, 1200, 800, '#30295c', .65)]), 'l': ('#f7f6fc', [(.5, -.1, 1200, 800, '#e6e1fb', .65)])},
    'mist': {'d': ('#0e1013', [('lin', '#1a1d22', '#0e1013')]), 'l': ('#e3e8ee', [('lin', '#eef1f5', '#e3e8ee')])},
    'ink': {'d': ('#050506', [(.5, -.1, 900, 600, '#24262b', .7)]), 'l': ('#e9e9eb', [(.5, -.1, 900, 600, '#ffffff', .7)])},
    'plain': {'d': ('#0b0c10', []), 'l': ('#f5f5f7', [])}}


class Wall(QQuickPaintedItem):
    # бэ1
    presetChanged, lightChanged, solidChanged, imgChanged, dimChanged, blurChanged, radiusChanged = (Signal() for _ in range(7))

    def __init__(self, parent=None):
        super().__init__(parent)
        self._preset, self._light, self._solid, self._img, self._dim, self._blur, self._radius = 'aurora', False, '#0e0f13', '', .35, 0.0, 12.0
        self._cache = None
        self.setAntialiasing(True)

    preset = Property(str, *_prop('_preset', 'presetChanged'), notify=presetChanged)
    light = Property(bool, *_prop('_light', 'lightChanged'), notify=lightChanged)
    solid = Property(str, *_prop('_solid', 'solidChanged'), notify=solidChanged)
    img = Property(str, *_prop('_img', 'imgChanged'), notify=imgChanged)
    dim = Property(float, *_prop('_dim', 'dimChanged'), notify=dimChanged)
    blur = Property(float, *_prop('_blur', 'blurChanged'), notify=blurChanged)
    radius = Property(float, *_prop('_radius', 'radiusChanged'), notify=radiusChanged)

    def _image(self, W, H):
        key = (self._img, int(W), int(H), round(self._blur))
        if self._cache and self._cache[0] == key:
            return self._cache[1]
        im = QImage(self._img)
        if im.isNull():
            self._cache = (key, None)
            return None
        im = im.scaled(int(W), int(H), Qt.KeepAspectRatioByExpanding, Qt.SmoothTransformation)
        im = im.copy((im.width() - int(W)) // 2, (im.height() - int(H)) // 2, int(W), int(H))
        if self._blur > 0:          # бэ2
            k = max(1, int(self._blur))
            sm = im.scaled(max(1, int(W) // k), max(1, int(H) // k), Qt.IgnoreAspectRatio, Qt.SmoothTransformation)
            im = sm.scaled(int(W), int(H), Qt.IgnoreAspectRatio, Qt.SmoothTransformation)
        self._cache = (key, im)
        return im

    def paint(self, p):
        W, H = self.width(), self.height()
        p.setRenderHint(QPainter.Antialiasing)
        clip = QPainterPath()
        clip.addRoundedRect(QRectF(0, 0, W, H), self._radius, self._radius)
        p.setClipPath(clip)
        if self._preset == 'image' and self._img:
            im = self._image(W, H)
            if im is not None:
                p.drawImage(QRectF(0, 0, W, H), im)
                c = QColor('#ffffff' if self._light else '#000000')
                c.setAlphaF(max(0, min(.9, self._dim)))
                p.fillRect(QRectF(0, 0, W, H), c)
                return
        if self._preset == 'solid':
            p.fillRect(QRectF(0, 0, W, H), QColor(self._solid))
            return
        base_, blobs = BG.get(self._preset, BG['aurora'])['l' if self._light else 'd']
        p.fillRect(QRectF(0, 0, W, H), QColor(base_))
        for b in blobs:
            if b[0] == 'lin':
                g = QLinearGradient(0, 0, 0, H)
                g.setColorAt(0, QColor(b[1]))
                g.setColorAt(1, QColor(b[2]))
                p.fillRect(QRectF(0, 0, W, H), QBrush(g))
                continue
            cx, cy, rx, ry, col, stop = b
            g = QRadialGradient(QPointF(0, 0), 1.0)
            c0, c1 = QColor(col), QColor(col)
            c1.setAlpha(0)
            g.setColorAt(0, c0)
            g.setColorAt(stop, c1)
            g.setColorAt(1, c1)
            br = QBrush(g)
            br.setTransform(QTransform().translate(cx * W, cy * H).scale(rx, ry))
            p.fillRect(QRectF(0, 0, W, H), br)


class Logo(QQuickPaintedItem):
    inkChanged, accentChanged = Signal(), Signal()
    P = [[(7, 18), (32, 5.5), (57, 18), (32, 30.5)], [(15.5, 29.75), (32, 38), (57, 25.5), (57, 31), (32, 43.5), (15.5, 35.25)],
         [(7, 38), (32, 50.5), (49, 42), (55, 45), (32, 56.5), (7, 45)]]

    def __init__(self, parent=None):
        super().__init__(parent)
        self._ink, self._accent = QColor('#ffffff'), QColor('#0a84ff')
        self.setAntialiasing(True)

    ink = Property(QColor, *_prop('_ink', 'inkChanged'), notify=inkChanged)
    accent = Property(QColor, *_prop('_accent', 'accentChanged'), notify=accentChanged)

    def paint(self, p):
        p.setRenderHint(QPainter.Antialiasing)
        k = min(self.width(), self.height()) / 64
        p.scale(k, k)
        for pts, (col, a) in zip(self.P, ((self._ink, .46), (self._accent, 1), (self._ink, .28))):
            c = QColor(col)
            c.setAlphaF(a * col.alphaF())
            pen = QPen(c, 2.4)
            pen.setJoinStyle(Qt.RoundJoin)
            p.setPen(pen)
            p.setBrush(c)
            p.drawPolygon(QPolygonF([QPointF(x, y) for x, y in pts]))


ICONS = {
    'ok': '<path d="M3 8.5l3.2 3.2L13 4.5"/>', 'need': '<circle cx="8" cy="8" r="5.5"/><path d="M7 5.6 9.4 8 7 10.4"/>',
    'bad': '<circle cx="8" cy="8" r="5.5"/><path d="M6.1 6.1l3.8 3.8M9.9 6.1l-3.8 3.8"/>', 'warn': '<path d="M8 2.6l5.8 10.2H2.2z"/><path d="M8 6.8v2.8M8 11.3v.1"/>',
    'wait': '<path d="M8 2.5a5.5 5.5 0 1 1-5.5 5.5"/>', 'info': '<circle cx="8" cy="8" r="5.5"/><path d="M8 7.4v3.4M8 5.2v.1"/>', 'todo': '<circle cx="8" cy="8" r="5"/>',
    'dot': '<circle cx="8" cy="8" r="3.2" fill="C" stroke="none"/>', 'pause': '<path d="M6 4.5v7M10 4.5v7"/>', 'skip': '<path d="M4 8h8"/>',
    'stop': '<rect x="4.5" y="4.5" width="7" height="7" rx="1.2"/>', 'play': '<path d="M5 3.6v8.8L12.4 8z" fill="C"/>', 'arrow': '<path d="M3 8h10M9.5 4.5 13 8l-3.5 3.5"/>',
    'expand': '<path d="M9.5 3H13v3.5M13 3L9 7M6.5 13H3V9.5M3 13l4-4"/>', 'trash': '<path d="M3 4.5h10M6.5 4.5V3h3v1.5M4.5 4.5l.6 8.5h5.8l.6-8.5"/>',
    'chat': '<path d="M2.8 4.2c0-.9.7-1.6 1.6-1.6h7.2c.9 0 1.6.7 1.6 1.6v5c0 .9-.7 1.6-1.6 1.6H7.2L4.4 13v-2.2c-.9 0-1.6-.7-1.6-1.6z"/>',
    'open': '<path d="M6 3.5H3.5v9h9V10M9 3.5h3.5V7M12.5 3.5 7.5 8.5"/>',
    'copy': '<rect x="5.5" y="5.5" width="8" height="8" rx="1.5"/><path d="M10.5 5.5V3.8c0-.7-.6-1.3-1.3-1.3H3.8c-.7 0-1.3.6-1.3 1.3v5.4c0 .7.6 1.3 1.3 1.3h1.7"/>',
    'save': '<path d="M8 2.5v8M4.8 7.5 8 10.7l3.2-3.2M3 13.5h10"/>', 'refresh': '<path d="M13 8a5 5 0 1 1-1.5-3.6M13 2.8v2.7h-2.7"/>',
    'compare': '<path d="M2.5 12.5l3.5-4 3 2.5 4.5-6"/><path d="M2.5 9l3.5-2 3 1.5 4.5-4" opacity=".45"/>',
    'folder': '<path d="M2.5 5c0-.8.7-1.5 1.5-1.5h2.4l1.4 1.5H12c.8 0 1.5.7 1.5 1.5v5c0 .8-.7 1.5-1.5 1.5H4c-.8 0-1.5-.7-1.5-1.5z"/>',
    'file': '<path d="M4 2.5h5l3 3v8H4z"/><path d="M9 2.5v3h3"/>', 'lock': '<rect x="3.5" y="7" width="9" height="6.5" rx="1.5"/><path d="M5.5 7V5.3a2.5 2.5 0 0 1 5 0V7"/>',
    'left': '<path d="M10 3.5 5.5 8l4.5 4.5"/>', 'right': '<path d="M6 3.5 10.5 8 6 12.5"/>', 'up': '<path d="M3.5 10 8 5.5l4.5 4.5"/>', 'down': '<path d="M3.5 6 8 10.5 12.5 6"/>',
    'plus': '<path d="M8 3.5v9M3.5 8h9"/>', 'x': '<path d="M4 4l8 8M12 4l-8 8"/>', 'search': '<circle cx="7" cy="7" r="4.3"/><path d="M10.2 10.2 13.5 13.5"/>',
    'setup': '<rect x="2.5" y="2.5" width="11" height="11" rx="3"/><path d="M8 5.5v5M5.5 8h5"/>', 'train': '<path d="M1.8 8.5h2.6l1.8-4.5 3 8 1.8-3.5h3.2"/>',
    'runs': '<rect x="2.5" y="3" width="11" height="3" rx="1"/><path d="M3.5 6v6.5h9V6M6.5 8.5h3"/>', 'bench': '<path d="M2.5 4.5l1.2 1.2L6 3.4M2.5 10l1.2 1.2L6 8.9M8.5 4.8h5M8.5 10.3h5"/>',
    'settings': '<path d="M2.5 5h5M11.5 5h2M2.5 11h2M8.5 11h5"/><circle cx="9.5" cy="5" r="1.8"/><circle cx="6.5" cy="11" r="1.8"/>',
    'theme': '<circle cx="8" cy="8" r="5.5"/><path d="M8 2.5a5.5 5.5 0 0 1 0 11z" fill="C" stroke="none"/>', 'power': '<path d="M8 2.5v5"/><path d="M4.6 4.6a5 5 0 1 0 6.8 0"/>',
    'palette': '<circle cx="8" cy="8" r="5.6"/><circle cx="6" cy="6.2" r=".9" fill="C" stroke="none"/><circle cx="9.6" cy="5.8" r=".9" fill="C" stroke="none"/><circle cx="10.6" cy="9" r=".9" fill="C" stroke="none"/><path d="M8 13.6c-1-.6-.8-2.2.6-2.2"/>',
    'chart': '<path d="M2.5 13.5h11M4 11V8M7 11V5M10 11V7M13 11V3.5"/>',
    'chip': '<rect x="4" y="4" width="8" height="8" rx="1.5"/><path d="M6 1.8V4M10 1.8V4M6 12v2.2M10 12v2.2M1.8 6H4M1.8 10H4M12 6h2.2M12 10h2.2"/>',
    'keys': '<rect x="1.5" y="4" width="13" height="8" rx="1.5"/><path d="M4 7h.01M6.5 7h.01M9 7h.01M11.5 7h.01M5 9.5h6"/>',
    'globe': '<circle cx="8" cy="8" r="5.5"/><path d="M2.5 8h11M8 2.5c1.6 1.6 2.4 3.4 2.4 5.5S9.6 11.9 8 13.5C6.4 11.9 5.6 10.1 5.6 8S6.4 4.1 8 2.5z"/>',
    'bolt': '<path d="M9 1.8 3.5 9h4l-.5 5.2L12.5 7h-4z"/>', 'send': '<path d="M8 13V3.5M4 7.5 8 3.5l4 4"/>', 'edit': '<path d="M10.5 2.8l2.7 2.7L6.4 12.3 3.2 13l.7-3.2z"/>',
    'build': '<path d="M9.8 2.8a3 3 0 0 0-3.6 3.9L2.5 10.4l3.1 3.1 3.7-3.7a3 3 0 0 0 3.9-3.6l-1.9 1.9-1.6-.5-.5-1.6z"/>',
    'min': '<path d="M3.5 12h9"/>', 'max': '<rect x="3.5" y="5" width="7.5" height="7.5" rx="1.6"/><path d="M6 3.5h5a1.5 1.5 0 0 1 1.5 1.5v5"/>',
    'check': '<path d="M3.5 8.3l2.8 2.8 6.2-6.4"/>', 'grip': '<path d="M5.5 4h.01M5.5 8h.01M5.5 12h.01M10.5 4h.01M10.5 8h.01M10.5 12h.01"/>',
    'link': '<path d="M6.5 9.5l3-3M7 4.5l1.2-1.2a2.6 2.6 0 0 1 3.6 3.6L10.6 8M9 11.5l-1.2 1.2a2.6 2.6 0 0 1-3.6-3.6L5.4 8"/>',
    'image': '<rect x="2.5" y="3" width="11" height="10" rx="1.6"/><circle cx="6" cy="6.5" r="1.2"/><path d="M3 11.5l3.2-3 2.3 2.1 1.7-1.6 2.8 2.5"/>',
}
_svg = {}


class Icon(QQuickPaintedItem):
    # бя1
    nameChanged, colorChanged, swChanged = Signal(), Signal(), Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._name, self._color, self._sw = '', QColor('#ffffff'), 1.5
        self.setAntialiasing(True)

    name = Property(str, *_prop('_name', 'nameChanged'), notify=nameChanged)
    color = Property(QColor, *_prop('_color', 'colorChanged'), notify=colorChanged)
    sw = Property(float, *_prop('_sw', 'swChanged'), notify=swChanged)

    def paint(self, p):
        body = ICONS.get(self._name)
        if not body:
            return
        key = (self._name, self._color.rgba(), self._sw)
        r = _svg.get(key)
        if r is None:
            c = QColor(self._color)
            col = c.name(QColor.HexRgb)
            op = f'{c.alphaF():.3f}'
            inner = body.replace('fill="C"', 'fill="%s" fill-opacity="%s"' % (col, op))
            svg = ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 16 16" fill="none" stroke="%s" stroke-opacity="%s" '
                   'stroke-width="%s" stroke-linecap="round" stroke-linejoin="round">%s</svg>') % (col, op, self._sw, inner)
            r = QSvgRenderer(svg.encode())
            _svg[key] = r
        p.setRenderHint(QPainter.Antialiasing)
        r.render(p, QRectF(0, 0, self.width(), self.height()))


# ва1
def _js(v):
    return v.toVariant() if isinstance(v, QJSValue) else v


class Bridge(QObject):
    H_changed, S_changed, T_changed, cmp_changed, rev_changed, setup_changed, misc_changed = (Signal() for _ in range(7))
    toast = Signal(str, str, str)                    # вб1
    chatTok = Signal(int, str)
    chatEnd = Signal(int, 'QVariant')
    chatLoad = Signal(str, float)                    # вб2
    benchChanged = Signal()
    _deliver = Signal(object)                        # вб3

    def __init__(self, engine_reload):
        super().__init__()
        self._H, self._S, self._T, self._cmp, self._rev = {}, {}, dict(BLANK_TV), {}, 0
        self.charts = {}
        self._reload = engine_reload
        self._deliver.connect(self._apply, Qt.QueuedConnection)
        s = T.S()
        lp = T.lora_norm(s.get('LORA') or None)
        self.ch = {'mode': T.mode_key(s.get('MODE')), 'goal': 'срез', 'file': '', 'name': '', 'nameAuto': True, 'over': {}, 'dataErr': '', 'launchErr': '',
                   'scene': T.scene_norm(None), 'tab': 'lora' if s.get('TAB') == 'lora' else 'tier',
                   'lora': lp if lp['tg'] else dict(lp, tg=T.LORA_DEF['tg'])}
        self.cur, self.picked, self.stopping, self.act_err, self.launching = '', False, False, '', False
        self.P = None
        self._plan_lock, self._plan_again, self._plan_busy = threading.Lock(), False, False
        self._state, self._win = None, ''
        self._setup = {'log': '', 'end': None, 'what': ''}
        self._bench = {'rows': [], 'run': False, 'stop': False, 't0': 0, 't1': 0, 'cmp': False, 'model': '', 'src': ''}
        self._chat_q, self._chat_busy, self._rid = [], None, 0
        self._chat_lock = threading.Lock()
        self._voice = None
        self.tray = None
        self.relabel = lambda: None     # вб4
        self._sig_H = ''
        threading.Thread(target=self._health_loop, daemon=True).start()
        threading.Thread(target=self._state_loop, daemon=True).start()
        threading.Thread(target=self._chat_loop, daemon=True).start()
        self.replan()                                    # вб5

    # вб6
    def post(self, kind, payload=None):
        self._deliver.emit((kind, payload))

    @Slot(object)
    def _apply(self, item):
        kind, x = item
        if kind == 'H':
            self._H = x
            self.ensure_cur()
            self.H_changed.emit()
            self._remake_setup()
        elif kind == 'P':
            self.P = x
            self._remake_setup()
        elif kind == 'state':
            self._set_state(x)
        elif kind == 'toast':
            self.toast.emit(*x)
        elif kind == 'setup':
            self.setup_changed.emit()
        elif kind == 'tok':
            self.chatTok.emit(*x)
        elif kind == 'end':
            self.chatEnd.emit(*x)
        elif kind == 'load':
            self.chatLoad.emit(*x)
        elif kind == 'bench':
            self.benchChanged.emit()
        elif kind == 'call':
            x()

    # вб7
    def _health_loop(self):
        while True:
            try:
                self.refresh_now()
            except Exception as e:      # вб8
                print('health:', e, file=sys.stderr)
            for _ in range(40):
                time.sleep(0.1)
                if getattr(self, '_poke', False):
                    self._poke = False
                    break

    def refresh_now(self):
        H = T.health()
        H['уведомления'] = bool(self.tray) or bool(shutil.which('notify-send'))
        H['трей'] = bool(self.tray)
        sig = json.dumps(H, ensure_ascii=False, sort_keys=True, default=str)
        if sig != self._sig_H:
            self._sig_H = sig
            self.post('H', H)

    @Slot()
    def refresh(self):
        self._poke = True

    def _remake_setup(self):
        if not self._H:
            return
        P = self.P
        pr = (P or {}).get('подготовка') or {}
        if pr.get('источник') and not self.ch['file']:      # вб9
            self.ch['file'], self.ch['scene'] = pr['источник']['путь'], T.scene_norm(pr.get('сцена'))
            if self.ch['nameAuto']:
                self.ch['name'] = free_name(self._H, base(self.ch['file']))
            self.misc_changed.emit()
            if pr.get('сводка') and 'instr_rows' not in pr['сводка'] and pr.get('свежий'):
                self.setFile(self.ch['file'])            # вб10
            else:
                self.replan()
        self._S = setup_vm(self._H, P, self.ch, self._H.get('идёт') or '', self.launching)
        self.S_changed.emit()

    def _get_H(self):
        return self._H
    health = Property('QVariant', _get_H, notify=H_changed)

    def _get_S(self):
        return self._S
    sv = Property('QVariant', _get_S, notify=S_changed)

    def _get_T(self):
        return self._T
    tv = Property('QVariant', _get_T, notify=T_changed)

    def _get_cmp(self):
        return self._cmp
    cmp = Property('QVariant', _get_cmp, notify=cmp_changed)

    def _get_rev(self):
        return self._rev
    rev = Property(int, _get_rev, notify=rev_changed)

    def _get_setup(self):
        d = dict(self._setup)
        if str(d.get('what') or '').startswith(('квант', 'импорт')) and d.get('end') is None:
            # вб11
            fr, line = -1.0, ''
            for m in re.finditer(r'^(Квантование|Перенос|Сверка) (\d+)/(\d+) (\S+)', d.get('log') or '', re.M):
                i, n = int(m.group(2)), max(1, int(m.group(3)))
                fr = i / n if m.group(1) == 'Квантование' else .7 * i / n if m.group(1) == 'Перенос' else .7 + .3 * i / n
                line = tr(m.group(1)) + f' {i}/{n} · ' + re.sub(r'\.weight$', '', m.group(4))
            d['frac'], d['line'] = fr, line
        return d
    job = Property('QVariant', _get_setup, notify=setup_changed)

    def _get_bench(self):
        b = self._bench
        R, n = b['rows'], len(b['rows'])
        cmp_ = b['cmp']
        hasE = any(r['ждём'] for r in R)
        doneA, doneB = len([r for r in R if r['a'] is not None]), len([r for r in R if r['b'] is not None])
        total = n * (2 if cmp_ else 1)
        hitA, hitB = len([r for r in R if r['ok'] is True]), len([r for r in R if r['okb'] is True])
        chg = len([r for r in R if r['a'] is not None and r['b'] is not None and not same(r['a'], r['b'])])
        tiles = []
        if hasE:
            tiles.append([tr('Совпало с ожидаемым'), str(hitA), tr(f'из {n}'), tr('Обученная модель' if b['model'] else 'Исходная модель')])
        if cmp_ and hasE:
            tiles.append([tr('У исходной модели'), str(hitB), tr(f'из {n}'), tr('Совпало с ожидаемым')])
        if cmp_:
            tiles.append([tr('Ответ изменён'), str(chg), tr(f'из {doneB}'), tr('Обученная модель в сравнении с исходной')])
        el = ((b['t1'] or time.time()) - b['t0']) if b['t0'] else 0
        rows = []
        for i, r in enumerate(R):
            res = []
            if r['a'] is None:
                res = ['Выполняется'] if b['run'] else ['—']
            else:
                if r['ждём']:
                    res.append(tr('Совпало' if r['ok'] else 'Не совпало'))
                if cmp_ and r['b'] is not None:
                    res.append(tr('Как у исходной модели' if same(r['a'], r['b']) else 'Ответ изменён'))
            cls = '' if r['a'] is None else ('hit' if r['ok'] else 'miss') if r['ждём'] else ('same' if cmp_ and r['b'] is not None and same(r['a'], r['b']) else '')
            rows.append({'i': i + 1, 'q': r['q'], 'scene': r.get('сцена') or '', 'exp': r['ждём'], 'a': r['a'], 'b': r['b'], 'res': ' · '.join(res) or '—',
                         'cls': cls, 'mark': bool(r['ждём'] and r['a'] is not None)})
        prog = (tr(f'Выполняется · {doneA + doneB} из {total}') if b['run'] else
                f'{n} {tr(plural(n, "вопрос", "вопроса", "вопросов"))} · {tr(hms(el))}' if n else '')
        return {'run': b['run'], 'n': n, 'hasE': hasE, 'cmp': cmp_, 'model': b['model'], 'tiles': tiles, 'rows': rows, 'prog': prog,
                'frac': (doneA + doneB) / total if total else 0}
    bench = Property('QVariant', _get_bench, notify=benchChanged)

    # вб12
    hubChanged = Signal()

    def _get_hub(self):
        return getattr(self, '_hub', None) or {'list': [], 'local': [], 'sel': None, 'busy': '', 'err': '', 'cat': {'busy': False, 'ошибка': '', 'источник': '', 'ещё': False, 'pend': 0, 'q': '', 'sort': 'downloads', 'limit': 30}}
    hub = Property('QVariant', _get_hub, notify=hubChanged)

    _hub_lock = threading.Lock()      # вб13

    def _hub_set(self, **kw):
        with self._hub_lock:
            h = dict(self._get_hub())
            h.update({k: v(h) if callable(v) else v for k, v in kw.items()})     # вб14
            self._hub = h
        self.post('call', self.hubChanged.emit)

    @Slot()
    def hubRefresh(self):
        def run():
            try:
                self._hub_set(local=T.local_models(), list=lambda h: T.hub_mark(list(h['list'])))     # вб15
            except Exception as e:
                self._hub_set(err=str(e))
        threading.Thread(target=run, daemon=True).start()
        c = self._get_hub()['cat']
        if not self._get_hub()['list'] and not c.get('busy'):
            self.hubCatalog(c.get('q', ''), c.get('sort', 'downloads'), c.get('limit', 30), False)       # вб16

    @Slot(str, str, int, bool)
    def hubCatalog(self, q, sort, limit, fresh):
        # вб17
        self._cat_gen = getattr(self, '_cat_gen', 0) + 1
        gen, q, sort, limit = self._cat_gen, (q or '').strip(), sort or 'downloads', max(1, int(limit or 30))
        self._hub_set(cat=dict(self._get_hub()['cat'], busy=True, q=q, sort=sort, limit=limit))

        def run():
            try:
                res = T.hub_catalog(q, sort, limit, fresh)
            except Exception as e:
                if gen == self._cat_gen:
                    self._hub_set(cat=dict(self._get_hub()['cat'], busy=False, ошибка=str(e)))
                return
            if gen != self._cat_gen:
                return
            rows = res['rows']
            pend = lambda: sum(1 for r in rows if r.get('поддержка') is None and not r.get('проверка_ошибка'))
            meta = dict(busy=False, q=q, sort=sort, limit=limit, источник=res['источник'], ошибка=res['ошибка'], ещё=res['ещё'], когда=res['когда'], pend=pend())
            self._hub_set(list=rows, cat=meta)
            if not meta['pend']:
                return
            by, last = {r['repo']: r for r in rows}, [0.0]

            def one(repo, m):
                if gen != self._cat_gen:
                    return
                T.hub_mark([by[repo]])
                if time.time() - last[0] > 0.4 or not pend():       # вб18
                    last[0] = time.time()
                    self._hub_set(list=list(rows), cat=dict(self._get_hub()['cat'], pend=pend()))
            try:
                T.hub_check([r['repo'] for r in rows if r.get('поддержка') is None], one, lambda: gen != self._cat_gen)
            except Exception:
                pass
            if gen == self._cat_gen:
                self._hub_set(list=list(rows), cat=dict(self._get_hub()['cat'], pend=pend()))
        threading.Thread(target=run, daemon=True).start()

    @Slot(str, bool)
    def hubLook(self, q, fresh):
        # вб19
        q = (q or '').strip()
        if not q:
            return
        self._hub_set(busy=q, err='')

        def run():
            try:
                m = T.hub_info(q, fresh)
            except Exception as e:
                m = {'ошибка': str(e)}
            self._hub_set(sel=m, busy='', err=m.get('ошибка') or '')
            if not m.get('ошибка'):
                self._hub_set(list=lambda h: T.hub_mark(list(h['list'])))     # вб20
        threading.Thread(target=run, daemon=True).start()

    @Slot(str)
    def hubGet(self, q):
        ok, why = T.hub_get(q)
        if not ok:
            self.toast.emit(tr('Не выполнено'), tr(cap(why)), 'bad')
            return
        self._setup = {'log': '', 'end': None, 'what': 'веса'}
        self.setup_changed.emit()
        self.toast.emit(tr('Скачивание весов'), q, 'info')
        threading.Thread(target=self._setup_tail, daemon=True).start()
        self.hubRefresh(); self.refresh()

    @Slot(str)
    def modelUse(self, path):
        ok, why = T.model_use(path)
        self.toast.emit(*((tr('Модель выбрана'), path, 'ok') if ok else (tr('Не выполнено'), tr(cap(why)), 'bad')))
        self.hubRefresh(); self.refresh()

    # вб21
    @Slot(str)
    def setMode(self, k):
        k = T.mode_key(k)
        if self.ch['mode'] != k:
            self.ch['mode'] = k
            T.S_save({'MODE': k})
            self.replan()

    @Slot(str)
    def setGoal(self, k):
        if self.ch['goal'] != k:
            self.ch['goal'] = k
            self.replan()

    @Slot(str)
    def setTab(self, k):
        # вб22
        k = 'lora' if k == 'lora' else 'tier'
        if self.ch['tab'] != k:
            self.ch['tab'] = k
            T.S_save({'TAB': k})
            self._remake_setup()
            self.replan()

    @Slot(str)
    def setBase(self, k):
        # вб23
        ok, why = T.base_pick(k)
        if not ok:
            self.toast.emit(tr('Не выполнено'), tr(cap(why)), 'bad')
        self.refresh_now()
        self.replan()

    @Slot(str)
    def delBase(self, k):
        ok, why = T.base_delete(k)
        self.toast.emit(tr('Удалено') if ok else tr('Не выполнено'), k[:-3] if ok else tr(cap(why)), 'ok' if ok else 'bad')
        self.refresh_now()
        self.replan()

    @Slot(str, str)
    def setLora(self, k, v):
        # вб24
        lp = dict(self.ch['lora'])
        if k == 'def':
            new = dict(T.LORA_DEF)
        elif k == 'mat':
            new = T.lora_norm(dict(lp, tg=lp['tg'].replace(v, '') if v in lp['tg'] else lp['tg'] + v))
        elif k in ('r', 'alpha', 'drop', 'tg'):
            new = T.lora_norm(dict(lp, **{k: v}))
        else:
            return
        if not new['tg'] or new == self.ch['lora']:      # вб25
            return
        self.ch['lora'] = new
        T.S_save({'LORA': T.lora_arg(new)})
        self._remake_setup()
        self.replan()

    @Slot(str)
    def setName(self, n):
        n = n.strip()
        self.ch['name'], self.ch['nameAuto'], self.ch['launchErr'] = n, not n, ''
        if not n and self.ch['file']:
            self.ch['name'] = free_name(self._H, base(self.ch['file']))
        self.replan()

    @Slot(str, str)
    def setOver(self, k, v):
        v = v.strip()
        if v:
            self.ch['over'][k] = v
        else:
            self.ch['over'].pop(k, None)
        self.replan()

    @Slot()
    def resetOver(self):
        self.ch['over'] = {}
        self.replan()

    @Slot(result='QVariant')
    def choice(self):
        sc = self.ch['scene']
        return {'file': self.ch['file'], 'name': self.ch['name'], 'mode': self.ch['mode'], 'goal': self.ch['goal'],
                'share': f'{round(sc["доля"] * 100, 1):g}', 'always': sc['всегда'], 'tab': self.ch['tab'],
                'alpha': f'{self.ch["lora"]["alpha"]:g}'}

    @Slot(str, str)
    def setScene(self, share, words):
        # вб26
        try:
            d = float((share or '').replace(',', '.').replace('%', '').strip() or 100) / 100
        except ValueError:
            d = self.ch['scene']['доля']
        sc = T.scene_norm({'доля': d, 'всегда': words})
        if sc == self.ch['scene']:
            self.misc_changed.emit()                     # вб27
            return
        self.ch['scene'] = sc
        if self.ch['file']:
            self.setFile(self.ch['file'])
        else:
            self.misc_changed.emit()

    @Slot(str)
    def setFile(self, path):
        path = (path or '').strip()
        if not path:
            return
        self.ch['file'], self.ch['dataErr'] = path, ''
        if self.ch['nameAuto']:
            self.ch['name'] = free_name(self._H, base(path))
        ok, why = T.prep_start(T.DRAFT, path, self.ch['scene'])
        if not ok:
            self.ch['dataErr'] = why or 'Не удалось выполнить разбор файла.'
        self.misc_changed.emit()
        self.replan()

    @Slot()
    def replan(self):
        with self._plan_lock:
            if self._plan_busy:
                self._plan_again = True
                return
            self._plan_busy = True
        threading.Thread(target=self._plan_job, daemon=True).start()

    def _plan_job(self):
        while True:
            ch = self.ch
            try:
                P = {'план': T.plan(T.DRAFT, ch['mode'], ch['over'] if ch['mode'] == 'эксперт' else {},
                                    'lora' if ch['tab'] == 'lora' else ch['goal'], ch['lora']),
                     'подготовка': T.prep_state(T.DRAFT), 'имя': T.name_state(ch['name'])}
            except Exception as e:
                print('plan:', e, file=sys.stderr)
                P = None
            self.post('P', P)
            with self._plan_lock:
                if not self._plan_again:
                    self._plan_busy = False
                    break
                self._plan_again = False
        if P and (P.get('подготовка') or {}).get('идёт'):        # вб28
            threading.Timer(0.7, self.replan).start()

    @Slot()
    def launch(self):
        S = self._S
        if not S or not S.get('launch', {}).get('can'):
            return
        self.launching = True
        self._remake_setup()
        ch = self.ch
        threading.Thread(target=self._launch_job, args=(dict(ch, over=dict(ch['over'])),), daemon=True).start()

    def _launch_job(self, ch):
        had_chat = T.chat_alive()                        # вб29
        ok, why = T.start_from_window({'имя': ch['name'], 'режим': ch['mode'], 'что': 'lora' if ch['tab'] == 'lora' else ch['goal'],
                                       'лора': ch['lora'], 'свои': ch['over'] if ch['mode'] == 'эксперт' else {}})

        def done():
            self.launching = False
            if not ok:
                self.ch['launchErr'] = why or 'Не удалось запустить обучение.'
                self.toast.emit(tr('Запуск не выполнен'), tr(cap(self.ch['launchErr'])), 'bad')
                self._remake_setup()
                return
            self.ch.update(launchErr='', file='', name='', nameAuto=True, scene=T.scene_norm(None))
            self.toast.emit(tr('Обучение запущено'), f'«{ch["name"]}»' + (' · ' + tr('Модель разговора выгружена') if had_chat and not T.chat_alive() else ''), 'ok')
            self.cur, self.picked = ch['name'], True
            self._state = None
            self.misc_changed.emit()
            self.go_train.emit()
            self.refresh()
            QTimer.singleShot(1500, self.refresh)        # вб30
            self.replan()
        self.post('call', done)
    go_train = Signal()

    # вб31
    def _state_loop(self):
        seen, beat, name = None, 0, None
        while True:
            time.sleep(0.3)
            n = self.cur
            if not n:
                continue
            try:
                now = (n, T.log_stamp(n), T.train_state(), T.stamp(T.run_file(n, 'stage.txt')), T.stamp(T.run_file(n, 'lrfind.txt')),
                       T.stamp(T.run_file(n, 'load.txt')), T.stamp(T.run_file(n, 'stagep.txt')), self.stopping, self.act_err, DEC, LANG)
                if now != seen or time.time() - beat > 10 or getattr(self, '_kick', False):
                    self._kick = False
                    seen, beat = now, time.time()
                    s = T.read_log(n)
                    self.post('state', s)
            except Exception as e:
                print('state:', e, file=sys.stderr)

    def _set_state(self, s):
        if s['имя'] != self.cur:
            return
        self._state = s
        tv, sw = train_vm(s, self._H.get('идёт') or '', self.stopping, self.act_err)
        tiles = {}
        for k in ALL_CHARTS:
            t, c = tile_vm(s, k)
            tiles[k] = t
            self.charts['tile:' + k] = c
        tv['tiles'] = tiles
        if sw:
            self.charts['sweep'] = sw
        if self._win:
            self._fill_win()
        self._T = tv
        self._rev += 1
        self.T_changed.emit()
        self.rev_changed.emit()

    @Slot(str)
    def openRun(self, n):
        if n == self.cur and self._T.get('name') == n:
            return
        self.cur, self.picked, self.act_err = n, True, ''
        self._kick = True
        self._T = dict(BLANK_TV, name=n)
        self.T_changed.emit()

    @Slot(result=str)
    def current(self):
        return self.cur

    def ensure_cur(self):
        if not self.cur and self._H.get('запуски'):
            self.cur = self._H.get('идёт') or self._H['запуски'][0]['имя']
            self._kick = True

    @Slot()
    def stop(self):
        if self.stopping:
            return
        self.stopping = True
        self._kick = True

        def job():
            T.train_stop()
            self.stopping = False
            self._kick = True
            self.post('toast', (tr('Обучение остановлено'), f'«{self.cur}»', 'warn'))
            self.refresh()
        threading.Thread(target=job, daemon=True).start()

    @Slot()
    def resume(self):
        name = self.cur
        self.act_err = ''

        def job():
            ok, why = T.start_from_window({'имя': name, 'продолжить': True})
            if not ok:
                self.act_err = why or 'Не удалось запустить обучение.'
                self.post('toast', (tr('Запуск не выполнен'), tr(cap(self.act_err)), 'bad'))
            else:
                self.post('toast', (tr('Обучение продолжено'), f'«{name}»', 'ok'))
            self._kick = True
            self.refresh()
        threading.Thread(target=job, daemon=True).start()

    # вб32
    win_changed = Signal()

    @Slot(str)
    def openWin(self, k):
        self._win = k
        self._fill_win()

    @Slot()
    def closeWin(self):
        self._win = ''
        self._wv = {}
        self.win_changed.emit()

    def _fill_win(self):
        s, k = self._state, self._win
        if not s or not k:
            return
        if k == 'log':
            self._wv = dict(log_vm(s), kind='log')
        elif k == 'plan':
            self._wv = dict(plan_vm(s), kind='plan')
        else:
            v, big = win_vm(s, k)
            for i, c in enumerate(big):
                self.charts[f'win:{i}'] = c
            self._wv = dict(v, kind='chart')
        self._rev += 1
        self.win_changed.emit()
        self.rev_changed.emit()

    def _get_wv(self):
        return getattr(self, '_wv', {})
    wv = Property('QVariant', _get_wv, notify=win_changed)

    # вб33
    @Slot(str, str, str, int, result='QVariant')
    def runs(self, q, f, sort, d):
        return runs_vm(self._H, q, f, sort, d)

    @Slot('QVariant')
    def compare(self, names):
        names = [n for n in _js(names) or []][:6]

        def job():
            S = [T.read_log(n) for n in names]
            v, charts = compare_vm(S)

            def done():
                for i, c in enumerate(charts):
                    self.charts[f'cmp:{i}'] = c
                self._cmp = v
                self._rev += 1
                self.cmp_changed.emit()
                self.rev_changed.emit()
            self.post('call', done)
        threading.Thread(target=job, daemon=True).start()

    @Slot()
    def closeCompare(self):
        self._cmp = {}
        self.cmp_changed.emit()

    @Slot('QVariant', result='QVariant')
    def runsSize(self, names):
        names = _js(names) or []
        rows = [r for r in self._H.get('запуски') or [] if r['имя'] in names]
        return bytes_(sum(r.get('размер') or 0 for r in rows))

    @Slot('QVariant')
    def deleteRuns(self, names):
        names = [n for n in _js(names) or [] if n != (self._H.get('идёт') or '')]

        def job():
            freed, bad = 0, []
            for n in names:
                r = T.run_delete(n)
                if r['ок']:
                    freed += r.get('освобождено') or 0
                else:
                    bad.append(n)
            if self.cur in names:
                self.cur = ''
                self._state = None
                self.post('call', lambda: (setattr(self, '_T', dict(BLANK_TV)), self.T_changed.emit()))
            self.post('toast', (tr('Удалено частично' if bad else 'Удалено'), tr(f'Освобождено {bytes_(freed)}') + (tr(' · Не удалено: ') + ', '.join(bad) if bad else ''),
                                'warn' if bad else 'ok'))
            self.refresh()
        threading.Thread(target=job, daemon=True).start()

    # вб34
    @Slot(str, int, result='QVariant')
    def peek(self, path, at):
        r = T.read_examples(path, max(0, at), 1)
        if not r.get('ок'):
            return {'ok': False, 'why': tr(cap(r.get('почему') or 'Не удалось прочитать файл.'))}
        x = (r.get('примеры') or [{}])[0]
        F = [('instruction', 'Инструкция'), ('input', 'Ввод'), ('output', 'Ответ')]
        fields = [[tr(t), str(x[k])] for k, t in F if x.get(k) not in (None, '')] + [[k, str(v)] for k, v in x.items() if k not in dict(F)]
        return {'ok': True, 'at': r['с'], 'n': r['всего'], 'fields': fields}

    # вб35
    @Slot(str)
    def setupRun(self, what):
        ok, why = T.setup_start(what)
        if not ok:
            self.toast.emit(tr('Не выполнено'), tr(cap(why)), 'bad')
            return
        self._setup = {'log': '', 'end': None, 'what': what}
        self.setup_changed.emit()
        self.toast.emit(tr({'движок': 'Сборка движка', 'веса': 'Скачивание весов', 'склеить': 'Объединение частей весов', 'пакеты': 'Установка пакетов', 'квант': 'Квантование',
                            'импорт': 'Перенос квантованной базы'}.get(what.split(':')[0], 'Подготовка')), base(what.split(':', 1)[1]) if ':' in what else '', 'info')
        threading.Thread(target=self._setup_tail, daemon=True).start()
        self.refresh()

    def _setup_tail(self):
        pos = 0
        while True:
            try:
                with open(T.SETUP_LOG, 'rb') as f:
                    f.seek(pos)
                    chunk = f.read()
                    pos = f.tell()
            except OSError:
                chunk = b''
            if chunk:
                txt = chunk.decode('utf-8', 'replace').replace('\r', '\n')
                self._setup['log'] = (self._setup['log'] + txt)[-60000:]
                self.post('setup')
            if not T.setup_alive():
                code = T.setup_proc.returncode if T.setup_proc else 0
                self._setup['end'] = code
                self._setup['log'] += ('\n' + tr(f'Ошибка (код {code}), причина указана выше.') + '\n') if code else ('\n' + tr('Готово') + '\n')
                self.post('setup')
                self.post('toast', (tr('Не выполнено') if code else tr('Готово'), tr(f'Код {code}') if code else '', 'bad' if code else 'ok'))
                self.refresh()
                return
            time.sleep(0.3)

    # вб36
    def _chat_loop(self):
        q = queue.Queue()
        T.chat_subs.append(q)
        while True:
            ev = q.get()
            with self._chat_lock:
                cur = self._chat_busy
            if ev.get('load'):
                self.post('load', load_line(ev['load']))
            if ev.get('ready'):
                self.post('load', ('', -1))
                self.refresh()
            if ev.get('tok') is not None and cur:
                piece = ev['tok'] if cur['text'] else ev['tok'].lstrip()     # вб37
                if piece:
                    cur['text'] += piece
                    if cur.get('emit', True):
                        self.post('tok', (cur['rid'], piece))
            if ev.get('end'):
                if ev.get('закрыт'):
                    self.post('load', ('', -1))
                    with self._chat_lock:
                        rest, self._chat_q, self._chat_busy = self._chat_q, [], None
                    err = ev.get('ошибка') or 'Движок завершил работу.'
                    for it in ([cur] if cur else []) + rest:
                        self._chat_done(it, {'end': True, 'остановлено': True} if ev.get('остановлено') else {'end': True, 'ошибка': err})
                    self.refresh()
                    continue
                with self._chat_lock:
                    self._chat_busy = None
                if cur:
                    self._chat_done(cur, ev)
                self._chat_next()

    def _chat_done(self, it, ev):
        if it.get('fin'):                                # вб38
            return
        it['fin'] = True
        stopped = bool(ev.get('остановлено'))
        info = {'error': '' if stopped else tr(cap(ev.get('ошибка') or '')), 'tokens': ev.get('токенов') or 0, 'sec': ev.get('сек') or 0,
                'speed': tr(ev.get('скорость') or ''), 'text': it['text'].rstrip(), 'stopped': stopped}
        info['meta'] = tr('Остановлено') if stopped else ' · '.join(x for x in [f'{info["tokens"]} {tr("ток.")}' if info['tokens'] else '',
                                                                               (dec(str(info['sec'])) + NBSP + tr('с')) if info['sec'] else '', info['speed']] if x)
        if stopped and not it.get('emit', True):
            info['error'] = tr('Модель выгружена')        # вб39
        if it.get('done'):
            it['done'](info)
        if it.get('emit', True):
            self.post('end', (it['rid'], info))

    def chat_send(self, body, emit=True, done=None):
        self._rid += 1
        it = {'rid': self._rid, 'body': body, 'text': '', 'emit': emit, 'done': done}
        with self._chat_lock:
            self._chat_q.append(it)
        self._chat_next()
        return self._rid

    def _chat_next(self):
        with self._chat_lock:
            if self._chat_busy or not self._chat_q:
                return
            it = self._chat_q.pop(0)
            self._chat_busy = it
        threading.Thread(target=self._chat_issue, args=(it,), daemon=True).start()

    def _chat_issue(self, it):
        body = it['body']
        nm, base_ = body.get('имя') or '', bool(body.get('base'))
        why = ''
        if nm and not os.path.exists(T.run_file(nm, 'step.txt')):
            why = f'У «{nm}» нет снимка весов: обучение не дошло до первого сохранения.'
        elif not nm and not base_:
            why = 'Модель не выбрана.'
        elif not os.path.exists(T.engine()) or not os.path.exists(os.path.join(T.S()['MODEL'], 'model.safetensors')):
            why = 'Отсутствует движок или веса модели.'
        elif it.get('fin'):                              # вб40
            return
        elif T.chat_blocked(nm):
            why = T.chat_blocked(nm)
        elif not T.chat_start(nm, body.get('ctx'), body.get('база')):
            why = 'Не удалось открыть разговор.'
        else:
            try:
                T.chat_proc.stdin.write(json.dumps({k: body[k] for k in ('q', 'new', 'base', 'temp', 'max', 'сцена', 'сила', 'top_k', 'top_p', 'min_p', 'повтор', 'зерно', 'присутствие', 'частота', 'окно', 'думать', 'история') if k in body}, ensure_ascii=False) + '\n')
                T.chat_proc.stdin.flush()
                self.refresh()
                return
            except (BrokenPipeError, AttributeError, OSError):
                why = 'Разговор закрыт.'
        with self._chat_lock:
            if self._chat_busy is it:
                self._chat_busy = None
        self._chat_done(it, {'end': True, 'ошибка': why})
        self._chat_next()

    @Slot('QVariant', result=int)
    def ask(self, body):
        return self.chat_send(dict(_js(body) or {}))

    @Slot()
    def chatStop(self):
        # вб41
        self._bench['stop'] = True
        if not T.chat_stop():                            # вб42
            with self._chat_lock:
                items = ([self._chat_busy] if self._chat_busy else []) + self._chat_q
                self._chat_busy, self._chat_q = None, []
            for it in items:
                self._chat_done(it, {'end': True, 'остановлено': True})
        self.post('load', ('', -1))
        self.refresh()

    @Slot(result='QVariant')
    def chatModels(self):
        L = [{'k': '', 'n': tr('Исходная модель'), 's': (self._H.get('модель') or {}).get('модель') or tr('Без обучения')}]
        for r in self._H.get('запуски') or []:
            if r.get('веса'):
                L.append({'k': r['имя'], 'n': r['имя'], 's': ' · '.join(x for x in [tr(GOAL_SHORT.get(r.get('цель'), '')),
                                                                                     tr('Потеря (loss)') + ' ' + fmt(r['лучшая'], 4) if r.get('лучшая') is not None else ''] if x),
                          'goal': tr(GOAL_NAME.get(r.get('цель'), '')), 'best': fmt(r['лучшая'], 4) if r.get('лучшая') is not None else '',
                          'step': int_(r['шаг']) if r.get('шаг') else '', 'when': tr(when(r.get('когда')))})
        return L

    # вб43
    def _chats_dir(self):
        d = os.path.join(T.DATA, 'chats'); os.makedirs(d, exist_ok=True); return d

    def _chat_path(self, cid):
        if not re.fullmatch(r'[A-Za-z0-9_-]{1,64}', cid or ''): return None
        return os.path.join(self._chats_dir(), cid + '.json')

    @Slot(result='QVariant')
    def chatList(self):
        out = []
        for fn in os.listdir(self._chats_dir()):
            if not fn.endswith('.json'): continue
            try:
                with open(os.path.join(self._chats_dir(), fn), encoding='utf-8') as f: c = json.load(f)
                out.append({'id': c['id'], 'title': c.get('title') or tr('Без названия'), 'model': c.get('model') or '',
                            'base': c.get('base') or '', 'updated': c.get('updated') or 0, 'n': len(c.get('messages') or [])})
            except (OSError, ValueError, KeyError, TypeError): continue
        return sorted(out, key=lambda x: -x['updated'])

    @Slot(str, result='QVariant')
    def chatGet(self, cid):
        p = self._chat_path(cid)
        try:
            with open(p, encoding='utf-8') as f: return json.load(f)
        except (OSError, ValueError, TypeError): return {}

    @Slot(str, str, result=bool)
    def chatSave(self, cid, text):
        p = self._chat_path(cid)
        if not p: return False
        try: c = json.loads(text)
        except ValueError: return False
        tmp = p + '.tmp'
        with open(tmp, 'w', encoding='utf-8') as f: json.dump(c, f, ensure_ascii=False)
        os.replace(tmp, p)
        return True

    @Slot(str)
    def chatDrop(self, cid):
        p = self._chat_path(cid)
        if p and os.path.exists(p): os.remove(p)

    # вб44
    @Slot(str, int, result='QVariant')
    def fileText(self, path, lim):
        LIM = max(200, min(200000, int(lim or 4000)))
        try:
            if os.path.getsize(path) > 8 * 2**20: return {'error': tr('Размер файла превышает 8 МБ.')}
            with open(path, 'rb') as f: raw = f.read(LIM * 4)
            if b'\x00' in raw[:4096]: return {'error': tr('Файл не является текстовым: изображения, документы PDF и архивы не читаются.')}
            t = raw.decode('utf-8', errors='replace')
            cut = len(t) > LIM
            return {'name': os.path.basename(path), 'text': t[:LIM], 'cut': cut}
        except OSError as e: return {'error': str(e)}

    # вб45
    def _v(self):
        if self._voice is None:
            import voice
            self._voice = voice.Voice(on_error=lambda m: self.post('toast', (tr('Озвучка'), tr(cap(m)), 'bad')))
        return self._voice

    @Slot(str)
    def voiceConf(self, js):
        c = _js(js) if not isinstance(js, str) else json.loads(js or '{}')
        self._v().configure(c)

    @Slot(result='QVariant')
    def voiceList(self):
        return [dict(x, why=tr(x['why']) if x['why'] else '') for x in self._v().listing()]

    @Slot(int, str)
    def voiceFeed(self, rid, piece):
        if self._voice is not None: self._voice.feed(rid, piece)

    @Slot(int)
    def voiceEnd(self, rid):
        if self._voice is not None: self._voice.end(rid)

    @Slot(str)
    def voiceSay(self, text):
        if self._voice is not None: self._voice.say(text)

    @Slot()
    def voiceStop(self):
        if self._voice is not None: self._voice.stop()

    @Slot(result=bool)
    def voiceBusy(self):
        return self._voice is not None and self._voice.speaking()

    # вб46
    @Slot(str, str, str, str, result='QVariant')
    def questions(self, src, run, path, text):
        if src == 'list':
            out = []
            for line in (text or '').splitlines():
                line = line.strip()
                if line:
                    q, _, e = line.partition('||')
                    out.append({'q': q.strip(), 'ждём': e.strip(), 'сцена': ''})
            return {'ok': True, 'qs': out, 'info': tr('Собственный список' if out else 'Собственный список пуст')}
        if src == 'run':
            ch = T.run_choice(run) or {}
            path = ch.get('датасет') or ''
            if not path:
                return {'ok': False, 'why': tr('У запуска нет записи о файле данных; требуется выбрать файл.')}
        if not path:
            return {'ok': False, 'why': tr('Файл с вопросами не выбран.')}
        r = T.questions(path)
        if not r.get('ок'):
            return {'ok': False, 'why': tr(cap(r.get('почему') or 'Не удалось прочитать файл.'))}
        return {'ok': True, 'qs': r['вопросы'], 'info': tr(f'Из данных «{run}»') if src == 'run' else base(path)}

    @Slot('QVariant')
    def benchRun(self, o):
        o = dict(_js(o) or {})
        b = self._bench
        if b['run']:
            return
        r = self.questions(o.get('src', 'list'), o.get('run', ''), o.get('file', ''), o.get('text', ''))
        if not r['ok']:
            self.toast.emit(tr('Проверка не запущена'), r['why'], 'bad')
            return
        if not r['qs']:
            self.toast.emit(tr('Нет вопросов'), '', 'warn')
            return
        run = o.get('run') or ''
        b.update(rows=[{'q': x['q'], 'сцена': x.get('сцена') or '', 'ждём': x.get('ждём') or '', 'a': None, 'b': None, 'ok': None, 'okb': None} for x in r['qs']],
                 run=True, stop=False, t0=time.time(), t1=0, cmp=bool(o.get('cmp') and run), model=run)
        self.benchChanged.emit()
        threading.Thread(target=self._bench_job, args=(o,), daemon=True).start()

    def _bench_job(self, o):
        b, run = self._bench, o.get('run') or ''
        base_opts = {'temp': float(o.get('temp') or 0), 'max': int(o.get('max') or 48), 'сцена': o.get('scene') or ''}

        def one(r, isB):
            ev = threading.Event()
            box = {}
            body = dict(base_opts, имя=run, base=isB or not run, q=r['q'], new=True)
            if r.get('сцена'):
                body['сцена'] = r['сцена']
            self.chat_send(body, emit=False, done=lambda info: (box.update(info), ev.set()))
            ev.wait()
            return box
        stopped = False
        for isB in ([False, True] if b['cmp'] else [False]):
            for r in b['rows']:
                if b['stop']:
                    stopped = True
                    break
                info = one(r, isB)
                if info.get('error'):
                    if not b['stop']:
                        self.post('toast', (tr('Проверка прервана'), info['error'], 'bad'))
                    b['stop'] = stopped = True
                    break
                t = (info.get('text') or '').strip()
                if isB:
                    r['b'], r['okb'] = t, match(t, r['ждём']) if r['ждём'] else None
                else:
                    r['a'], r['ok'] = t, match(t, r['ждём']) if r['ждём'] else None
                self.post('bench')
            if stopped:
                break
        if run and T.chat_alive():                      # вб47
            ev = threading.Event()
            self.chat_send({'имя': run, 'base': False, 'new': True}, emit=False, done=lambda _i: ev.set())
            ev.wait(30)
        b['run'], b['t1'] = False, time.time()
        self.post('bench')
        if not stopped:
            n = len(b['rows'])
            self.post('toast', (tr('Проверка завершена'), f'{n} {tr(plural(n, "вопрос", "вопроса", "вопросов"))}', 'ok'))
        self.refresh()

    @Slot()
    def benchStop(self):
        self._bench['stop'] = True
        with self._chat_lock:
            rest, self._chat_q = [x for x in self._chat_q if x.get('emit', True)], [x for x in self._chat_q if not x.get('emit', True)]
            drop, self._chat_q = self._chat_q, rest
        for it in drop:
            self._chat_done(it, {'end': True, 'ошибка': 'Запрос отменён.'})

    @Slot(int)
    def benchMark(self, i):
        r = self._bench['rows'][i]
        r['ok'] = not r['ok']
        self.benchChanged.emit()

    def _bench_table(self):
        b = self._bench
        cmp_ = any(r['b'] is not None for r in b['rows'])
        head = ['№', 'Вопрос', 'Ожидается', 'Обученная модель'] + (['Исходная модель'] if cmp_ else []) + ['Совпало']
        rows = [[i + 1, r['q'], r['ждём'], r['a'] or ''] + ([r['b'] or ''] if cmp_ else []) + ['' if r['ok'] is None else 'Да' if r['ok'] else 'Нет']
                for i, r in enumerate(b['rows'])]
        return [tr(h) for h in head], rows

    @Slot()
    def benchCopy(self):
        head, rows = self._bench_table()
        QGuiApplication.clipboard().setText('\n'.join('\t'.join(re.sub(r'\s+', ' ', str(v)) for v in r) for r in [head] + rows))
        self.toast.emit(tr('Таблица скопирована'), '', 'ok')

    @Slot()
    def benchSave(self):
        head, rows = self._bench_table()
        r = T.bench_save(self._bench['model'], head, rows)
        if r.get('ок'):
            self.toast.emit(tr('Таблица сохранена'), r['путь'], 'ok')
        else:
            self.toast.emit(tr('Не удалось сохранить'), tr(cap(r.get('почему') or '')), 'bad')

    @Slot(result='QVariant')
    def chatBases(self):
        # вб48
        H = self._H or {}
        out = [['', 'BF16' + (' · ' + gib(H.get('база_bf16'), 1) if H.get('база_bf16') else '')]]
        have = set()
        for b in H.get('базы') or []:
            if b.get('годна'):
                out.append([b['имя'], b['имя'][:-3] + ' · ' + gib(b.get('байт'), 1)]); have.add(b.get('рецепт'))
        for r in H.get('рецепты') or []:
            if r.get('байт') and r['к'] not in have:
                out.append(['make:' + r['к'], r['к'] + ' · ' + tr('Создание') + ' ≈ ' + gib(r['байт'], 1)])
        return out

    @Slot(result='QVariant')
    def genDefaults(self):
        return T.gen_defaults()

    @Slot(result=int)
    def ctxMax(self):
        return ctx_fit()[0]

    @Slot(result=int)
    def ctxDefault(self):
        return ctx_fit()[1]

    # вб49
    @Slot(result='QVariant')
    def settings(self):
        return {'свои': T.S(), 'железо': T.conf_read(), 'conf': T.conf_path(),
                'пути': [[tr('Программа'), T.APP], [tr('Данные программы'), T.DATA], [tr('Настройка оборудования'), T.conf_path()]]}

    @Slot('QVariant')
    def saveSettings(self, d):
        d = dict(_js(d) or {})
        T.S_save(d)
        if 'MODEL' in d or 'WORK' in d:
            self.cur, self._state = '', None
            self.refresh()
            self.replan()

    @Slot('QVariant', result=bool)
    def saveConf(self, d):
        d = {k: str(v).strip() for k, v in (dict(_js(d) or {})).items() if k in ('headroom_mib', 'tier1_params', 'bf16x3', 'prefetch', 'mem_cap_pct') and str(v).strip()}
        ok = T.conf_write(d) if d else True
        T._ginfo.clear()
        self.refresh()
        return bool(ok)

    @Slot(str)
    def saveAppear(self, j):
        T.S_save({'APPEAR': j})

    @Slot(str)
    def saveUI(self, j):
        global DEC
        T.S_save({'UI': j})
        try:
            DEC = json.loads(j).get('dec', 'dot')
        except ValueError:
            pass
        self._kick = True
        self.refresh()

    @Slot(str)
    def setLang(self, l):
        global LANG
        if l == LANG:
            return
        T.S_save({'LANG': l})
        LANG = l
        _trc.clear()
        self._sig_H = ''
        self._kick = True               # вб50
        self._remake_setup()
        self.refresh()
        self.relabel()
        QTimer.singleShot(0, self._reload)

    @Slot(result=str)
    def lang(self):
        return LANG

    @Slot(str, result=str)
    def loc(self, s):
        return tr(s)

    @Slot(str, result=str)
    def gloss(self, s):
        return gloss_for(s)

    # вб51
    @Slot(str, result=str)
    def pick(self, kind):
        en = LANG == 'en'
        if kind == 'папка':
            return QFileDialog.getExistingDirectory(None, 'Select folder' if en else 'Выбор папки') or ''
        if kind == 'квант':
            r, _ = QFileDialog.getOpenFileName(None, 'Select GGUF or bitsandbytes file' if en else 'Выбор файла GGUF или bitsandbytes', '',
                                               'GGUF, safetensors (*.gguf *.safetensors);;*')
            return r or ''
        if kind == 'картинка':
            r, _ = QFileDialog.getOpenFileName(None, 'Select image' if en else 'Выбор изображения', '', 'Images (*.png *.jpg *.jpeg *.webp *.avif *.gif *.bmp)')
            return r or ''
        r, _ = QFileDialog.getOpenFileName(None, 'Select file' if en else 'Выбор файла', '', 'JSONL (*.jsonl *.json *.txt);;*')
        return r or ''

    @Slot(str, result=str)
    def pickColor(self, init):
        c = QColorDialog.getColor(QColor(init or '#0a84ff'), None, 'Colour' if LANG == 'en' else 'Цвет')
        return c.name() if c.isValid() else ''

    @Slot(str)
    def copy(self, t):
        cb = QGuiApplication.clipboard()
        cb.setText(t, QClipboard.Clipboard)
        if cb.supportsSelection():                       # вб52
            cb.setText(t, QClipboard.Selection)

    @Slot(str)
    def openUrl(self, u):
        QDesktopServices.openUrl(QUrl(u))

    # вб53
    @Slot()
    def kill(self):
        if T.train_state()[0]:
            T.train_stop()
        T.chat_stop()
        if self._voice is not None: self._voice.stop()
        for pr in [T.setup_proc] + [j['p'] for j in T.prep_jobs.values()]:
            try:
                if pr and pr.poll() is None:
                    pr.kill()
            except OSError:
                pass
        QTimer.singleShot(150, lambda: os._exit(0))

    @Slot(result=bool)
    def hideToTray(self):
        return bool(self.tray) and bool(T.train_state()[0]) and bool(T.ui_pref('tray', 1))

    @Slot()
    def trayHint(self):
        if self.tray and not getattr(self, '_told', False):
            self._told = True
            self.tray.showMessage('TierLM', tr('Обучение продолжается. Окно свёрнуто в системный трей.'), QSystemTrayIcon.Information, 4000)


# вв1
def run(mod):
    # вв2
    global T, BK, LANG, DEC
    T = mod
    LANG = T.S().get('LANG') or 'ru'
    try:
        DEC = json.loads(T.S().get('UI') or '{}').get('dec', 'dot')
    except ValueError:
        DEC = 'dot'
    load_en(os.path.join(T.APP, 'en.txt'))
    load_gloss(os.path.join(T.APP, 'glossary.tsv'))
    app = QApplication.instance() or QApplication(['tierlm'])      # вв3
    app.setApplicationName('TierLM')
    app.setApplicationDisplayName('TierLM')
    app.setDesktopFileName('tierlm')
    app.setQuitOnLastWindowClosed(False)
    icon = QIcon(os.path.join(T.APP, 'logo.svg'))
    app.setWindowIcon(icon)
    fd = os.path.join(T.APP, 'fonts', 'qt')
    for f in sorted(os.listdir(fd)) if os.path.isdir(fd) else []:
        if f.endswith('.ttf'):
            QFontDatabase.addApplicationFont(os.path.join(fd, f))
    for cls, name in ((Chart, 'Chart'), (Wall, 'Wall'), (Logo, 'Logo'), (Icon, 'Icon')):
        qmlRegisterType(cls, 'Tier', 1, 0, name)
    engine = QQmlApplicationEngine()
    holder = {}

    def load():
        for o in engine.rootObjects():
            o.deleteLater()
        engine.clearComponentCache()
        engine.load(QUrl.fromLocalFile(os.path.join(T.APP, 'ui.qml')))
        if engine.rootObjects():
            holder['w'] = engine.rootObjects()[-1]

    BK = Bridge(load)
    engine.rootContext().setContextProperty('B', BK)
    engine.rootContext().setContextProperty('APPVER', T.VERSION)
    tray = None
    if QSystemTrayIcon.isSystemTrayAvailable():
        tray = QSystemTrayIcon(icon, app)
        tray.setToolTip('TierLM')
        m = QMenu()
        a_show, a_stop = m.addAction(''), m.addAction('')
        m.addSeparator()
        a_quit = m.addAction('')

        def relabel():              # вв4
            a_show.setText(tr('Показать окно'))
            a_stop.setText(tr('Остановить обучение'))
            a_quit.setText(tr('Завершить работу'))
        relabel()
        BK.relabel = relabel
        show = lambda: holder.get('w') and (holder['w'].show(), holder['w'].raise_(), holder['w'].requestActivate())
        a_show.triggered.connect(show)
        a_stop.triggered.connect(lambda: threading.Thread(target=T.train_stop, daemon=True).start())
        a_quit.triggered.connect(app.quit)
        m.aboutToShow.connect(lambda: a_stop.setEnabled(bool(T.train_state()[0])))
        tray.setContextMenu(m)
        tray.activated.connect(lambda r: show() if r in (QSystemTrayIcon.Trigger, QSystemTrayIcon.DoubleClick) else None)
        tray.show()
        holder['tray_menu'] = m
    BK.tray = tray

    def note(title, text, urgent=False):
        if not tray:
            raise RuntimeError('Системный трей недоступен')
        BK.post('call', lambda: tray.showMessage(title, tr(text), QSystemTrayIcon.Critical if urgent else QSystemTrayIcon.Information, 8000))
    T.NOTIFY = note
    threading.Thread(target=T.watch_runs, daemon=True).start()
    load()
    if not engine.rootObjects():
        return False
    def teardown():                 # вв5
        import shiboken6
        for o in engine.rootObjects():
            shiboken6.delete(o)
    app.aboutToQuit.connect(teardown)
    app.exec()
    return True
