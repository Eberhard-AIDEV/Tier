#!/bin/bash
# а1
set -e
DST="${MODEL:-./model}"; REPO="${HF_REPO:-}"
[ -n "$REPO" ] || { echo "Модель не выбрана. Требуется указать имя на Hugging Face или ссылку в разделе «Модели»."; exit 1; }
mkdir -p "$DST"
"${TIER_PY:-python3}" -c 'import huggingface_hub' 2>/dev/null || {
    echo "Установка huggingface_hub"
    "${TIER_PY:-python3}" -m pip install --user huggingface_hub 2>/dev/null || "${TIER_PY:-python3}" -m pip install --user --break-system-packages huggingface_hub; }
"${TIER_PY:-python3}" - "$REPO" "$DST" "${TIER_APP:-$(dirname "$0")}" <<'PY'
import os, re, shutil, sys
from huggingface_hub import HfApi, snapshot_download
repo, dst, app = sys.argv[1], sys.argv[2], sys.argv[3]
m = re.match(r'^(?:https?://)?(?:www\.)?(?:huggingface\.co|hf\.co)/([^/\s]+/[^/\s?#]+)', repo.strip())
repo = (m.group(1) if m else repo.strip()).rstrip('/')
tok = os.environ.get('HF_TOKEN') or None
KEEP = ('config.json', 'generation_config.json', 'tokenizer.json', 'tokenizer_config.json', 'special_tokens_map.json',
        'chat_template.jinja', 'chat_template.json', 'model.safetensors.index.json')
try:
    info = HfApi().model_info(repo, files_metadata=True, token=tok)
except Exception as e:
    sys.exit(f'{repo}: Не удалось открыть на Hugging Face: {e}'.replace('\n', ' '))
files = [(s.rfilename, s.size or 0) for s in info.siblings]
# только веса текстовой модели из корня: consolidated*.safetensors (Mistral) и original/ — те же веса в другом виде
want = [(f, n) for f, n in files if '/' not in f and (f in KEEP or (f.endswith('.safetensors') and not f.startswith('consolidated')))]
if not any(f.endswith('.safetensors') for f, _ in want):
    sys.exit(f'{repo}: В корне нет весов safetensors, движку нечего читать.')
need = sum(n for f, n in want if not os.path.exists(os.path.join(dst, f)) or os.path.getsize(os.path.join(dst, f)) != n)
wb = sum(n for f, n in want if f.endswith('.safetensors'))
shards = sum(1 for f, _ in want if f.endswith('.safetensors')) > 1
free = shutil.disk_usage(dst).free
extra = wb if shards else 0                     # при склейке на время нужна вторая копия
print(f'{repo}: файлов {len(want)}, веса {wb / 2**30:.1f} ГиБ; требуется докачать {need / 2**30:.1f} ГиБ, свободно {free / 2**30:.1f} ГиБ.', flush=True)
if free < need + extra + (1 << 30):
    sys.exit(f'Недостаточно места: требуется {(need + extra) / 2**30:.1f} ГиБ' + (' (с запасом на объединение частей)' if extra else '') + f', свободно {free / 2**30:.1f} ГиБ.')
snapshot_download(repo, local_dir=dst, allow_patterns=[f for f, _ in want], token=tok)
print('Веса загружены:', dst, flush=True)
sys.path.insert(0, app)
import arch
A, why = arch.describe(dst)
if A and why and all('merge' in w for w in why):
    print('Объединение частей в один model.safetensors (BF16)', flush=True)
    arch.merge(dst)
    A, why = arch.describe(dst)
print(arch.summary(A, why))
PY
