# а1
import json, random, struct, sys, os
from tokenizers import Tokenizer
E = os.environ.get
# а2
NAME  = E('NAME',  'run')                                        # а3
DATA  = E('DATA',  './dataset.jsonl')  # а4
MODEL = E('MODEL', './model')  # а5
WORK  = E('WORK',  './work')                                 # а6
os.makedirs(WORK, exist_ok=True)
OUT   = E('OUT',   f'{WORK}/{NAME}_tokens.bin')                   # а7
EVAL  = E('EVAL',  f'{WORK}/{NAME}_eval.bin')                     # а8
FRAC  = float(E('INSTR', '1.0'))    # а9
KEEP  = [k.strip() for k in E('INSTR_ALWAYS', '').split(',') if k.strip()]   # а10
HOLD  = int(E('HOLD', '20'))        # а11
HOLD_MAX = int(E('HOLD_MAX', '500'))  # а12
SEED  = int(E('SEED', '42'))
# а13
if len(sys.argv) > 1: DATA = sys.argv[1]
if len(sys.argv) > 2: OUT = sys.argv[2]
# а14
if os.path.abspath(OUT) == os.path.abspath(DATA) or OUT.endswith(('.jsonl', '.json')):
    sys.exit(f'Отказ: токены были бы записаны поверх датасета ({OUT}). Формат вызова: prepare.py датасет.jsonl [токены.bin].')
if not os.path.isfile(DATA): sys.exit(f'Нет датасета: файл «{DATA}» не найден.')
# а15
M = MODEL
tok = Tokenizer.from_file(M + '/tokenizer.json')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tmpl import Fmt
F = Fmt(M, tok)   # а16
enc = lambda s: tok.encode(s, add_special_tokens=False).ids
print('Запуск:', NAME, '| Датасет:', DATA, '| Модель:', M)
rows = []
for n, l in enumerate(open(DATA, encoding='utf-8'), 1):
    if not l.strip(): continue
    try: r = json.loads(l)
    except ValueError as e: sys.exit(f'Строка {n}: не является JSON ({e.msg}).')
    if not isinstance(r, dict) or not str(r.get('output') or '').strip(): sys.exit(f'Строка {n}: нет поля output.')
    rows.append(r)
if not rows: sys.exit('В файле нет примеров.')
ins = lambda r: str(r.get('instruction') or '').strip()
has = [i for i, r in enumerate(rows) if ins(r)]
# а17
order = list(range(len(rows))); random.Random(SEED).shuffle(order)
with_instr = set(has) if FRAC >= 1 else set(random.Random(SEED + 1).sample(has, round(len(has) * max(FRAC, 0))))
if KEEP: with_instr |= {i for i in has if any(k in ins(rows[i]) for k in KEEP)}
recs, lens, outs = [], [], []
for i in order:
    r = rows[i]
    # а18
    prompt, answer = F.example(ins(r) if i in with_instr else '', str(r.get('input') or ''), str(r['output']))
    recs.append((prompt, answer)); lens.append(len(prompt) + len(answer)); outs.append(len(answer))
ne = min(len(recs) // HOLD, HOLD_MAX)
def dump(path, rs):
    with open(path, 'wb') as f:
        f.write(struct.pack('<i', len(rs)))
        for p, a in rs:
            t = p + a; f.write(struct.pack('<ii', len(t), len(p))); f.write(struct.pack('<%di' % len(t), *t))
dump(OUT, recs[ne:]); dump(EVAL, recs[:ne])
print('Обучение:', len(recs) - ne, '→', OUT, '| Валидация:', ne, '→', EVAL, f'(из них с системной инструкцией: {sum(i in with_instr for i in order[:ne])})')
s = sorted(lens); n = len(s)
print('Записано', n, 'примеров в', OUT, '| С системной инструкцией:', len(with_instr), 'из', len(has), '(примеров с полем instruction)')
print('Длина: минимум', s[0], 'медиана', s[n // 2], 'p99', s[int(n * .99)], 'максимум', s[-1])
noi = [l for l, i in zip(lens, order) if i not in with_instr]
# б1
with open(f'{WORK}/{NAME}_stats.txt','w') as st:
    st.write(f'examples {n}\nmedian {s[n//2]}\np99 {s[int(n*.99)]}\nmax {s[-1]}\nloss_tokens {sum(outs)}\ninstr_rows {len(has)}\nwith_instr {len(with_instr)}\n')
# б2
from collections import Counter
sc = Counter(ins(rows[i]) for i in with_instr)
with open(f'{WORK}/{NAME}_scenes.json', 'w', encoding='utf-8') as fs:
    json.dump([{'сцена': k, 'примеров': v} for k, v in sc.most_common()] + [{'сцена': '', 'примеров': len(rows) - len(with_instr)}], fs, ensure_ascii=False)
print('Без системной инструкции: макс', max(noi) if noi else '—', '| Токенов в расчёте потери (loss): медиана', sorted(outs)[n // 2], 'сумма', sum(outs))
print('Пример:', repr(tok.decode(recs[0][0] + recs[0][1], skip_special_tokens=False))[:300])
k = next((j for j, i in enumerate(order) if i in with_instr), None)
if k is not None: print('С системной инструкцией:', repr(tok.decode(recs[k][0] + recs[k][1], skip_special_tokens=False))[:120], '→', repr(tok.decode(recs[k][0][-12:] + recs[k][1], skip_special_tokens=False)))
