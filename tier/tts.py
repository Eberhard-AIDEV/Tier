#!/usr/bin/env python3
# а1
import csv, glob, json, os, random, struct, sys
E = os.environ.get

# б1
def _read(p):
    try:
        with open(p, encoding='utf-8') as f: return f.read()
    except OSError: return ''

def detect(mdir):
    # в1
    cfg = {}
    try: cfg = json.loads(_read(os.path.join(mdir, 'config.json')) or '{}')
    except ValueError: pass
    tok = _read(os.path.join(mdir, 'tokenizer.json'))
    names = os.listdir(mdir) if os.path.isdir(mdir) else []
    mt = str(cfg.get('model_type') or cfg.get('model') or '')
    if 'model.pth' in names and ('xtts' in mt.lower() or 'xtts' in json.dumps(cfg)[:4000].lower() or 'speakers_xtts.pth' in names):
        return {'семейство': 'xtts', 'путь': 'external', 'кодек': 'DVAE (1024 кода мел-токенов)',
                'почему': 'GPT-2 на 440 млн параметров над мел-кодами и HiFi-GAN. Обучение выполняется обычным PyTorch на одной карте, потоковая загрузка не требуется.',
                'нужно': ['torch', 'coqui-tts', 'dvae.pth', 'mel_stats.pth']}
    if '<custom_token_' in tok:
        return {'семейство': 'orpheus', 'путь': 'engine', 'кодек': 'SNAC 24 кГц (3 уровня, 7 токенов на кадр)', 'нужно': ['snac', 'torch', 'torchaudio']}
    if '<|s_' in tok and 'SPEECH_GENERATION_START' in tok:
        return {'семейство': 'llasa', 'путь': 'engine', 'кодек': 'XCodec2', 'нужно': ['xcodec2', 'torch'],
                'замечание': 'Кодировщик подключается так же, как SNAC. Адаптер пока не реализован.'}
    if 'bicodec' in tok.lower():
        return {'семейство': 'spark', 'путь': 'engine', 'кодек': 'BiCodec', 'замечание': 'Адаптер пока не реализован.'}
    if mt in ('f5-tts', 'e2-tts') or 'f5' in mt.lower():
        return {'семейство': 'flow-matching', 'путь': 'external', 'замечание': 'Не языковая модель. Обучение выполняется собственным средством f5-tts.'}
    return {'семейство': '?', 'путь': '?', 'замечание': 'Семейство не определено. Требуются config.json и tokenizer.json.'}

# г1
def read_manifest(path):
    base = os.path.dirname(os.path.abspath(path)); rows = []
    def full(a):
        for c in (a, os.path.join(base, a), os.path.join(base, 'wavs', a), os.path.join(base, 'wavs', a + '.wav'), os.path.join(base, a + '.wav')):
            if os.path.isfile(c): return os.path.abspath(c)
        return ''
    if path.endswith('.jsonl'):
        for n, l in enumerate(open(path, encoding='utf-8'), 1):
            if not l.strip(): continue
            r = json.loads(l); a = full(str(r.get('audio') or r.get('file') or ''))
            if a and str(r.get('text') or '').strip(): rows.append({'audio': a, 'text': str(r['text']).strip(), 'speaker': str(r.get('speaker') or 'spk')})
    else:
        for line in csv.reader(open(path, encoding='utf-8'), delimiter='|', quotechar='\x00'):
            if len(line) < 2: continue
            a = full(line[0].strip())
            if a and line[1].strip(): rows.append({'audio': a, 'text': (line[2] if len(line) > 2 and line[2].strip() else line[1]).strip(), 'speaker': 'spk'})
    return rows

def split(rows, hold=20, hold_max=500):
    ev = rows[hold - 1::hold][:hold_max]; keep = {id(r) for r in ev}
    return [r for r in rows if id(r) not in keep], ev

# д1
SOH, EOT, EOH, SOAI, SOS, EOS, EOAI = 128259, 128009, 128260, 128261, 128257, 128258, 128262
AUDIO0, SLOT = 128266, 4096

def snac_pack(c0, c1, c2):
    # е1
    out = []
    for i in range(len(c0)):
        f = (c0[i], c1[2 * i], c2[4 * i], c2[4 * i + 1], c1[2 * i + 1], c2[4 * i + 2], c2[4 * i + 3])
        out += [AUDIO0 + k * SLOT + v for k, v in enumerate(f)]
    return out

def snac_unpack(ids):
    ids = [t for t in ids if t >= AUDIO0]; ids = ids[:len(ids) // 7 * 7]
    c0, c1, c2 = [], [], []
    for i in range(0, len(ids), 7):
        v = [ids[i + k] - AUDIO0 - k * SLOT for k in range(7)]
        c0.append(v[0]); c1 += [v[1], v[4]]; c2 += [v[2], v[3], v[5], v[6]]
    return c0, c1, c2

def orpheus_example(text_ids, audio_ids):
    # з1
    return [SOH] + text_ids + [EOT, EOH], [SOAI, SOS] + audio_ids + [EOS, EOAI]

def write_bin(path, ex):
    with open(path, 'wb') as f:
        f.write(struct.pack('<i', len(ex)))
        for p, a in ex:
            t = p + a; f.write(struct.pack('<ii', len(t), len(p))); f.write(struct.pack('<%di' % len(t), *t))

def encode_snac(paths, log=print):
    import torch, torchaudio
    from snac import SNAC
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    m = SNAC.from_pretrained('hubertsiuzdak/snac_24khz').to(dev).eval()
    for n, p in enumerate(paths, 1):
        w, sr = torchaudio.load(p)
        w = torchaudio.functional.resample(w.mean(0, keepdim=True), sr, 24000)[None].to(dev)
        with torch.inference_mode(): c = m.encode(w)
        yield [x[0].tolist() for x in c]
        if n % 50 == 0: log(f'Звук → токены: {n}/{len(paths)}')

def prep_orpheus(mdir, manifest, name='run', work='./work', voice=''):
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(os.path.join(mdir, 'tokenizer.json'))
    rows = read_manifest(manifest)
    if not rows: sys.exit('Манифест пуст, либо рядом с ним не найдены файлы звука.')
    os.makedirs(work, exist_ok=True); ex = []
    for r, codes in zip(rows, encode_snac([r['audio'] for r in rows])):
        text = f"{voice}: {r['text']}" if voice else r['text']
        ids = tok.encode(text, add_special_tokens=False).ids
        ex.append(orpheus_example(ids, snac_pack(*codes)))
    tr, ev = split(ex)
    write_bin(os.path.join(work, f'{name}_tokens.bin'), tr); write_bin(os.path.join(work, f'{name}_eval.bin'), ev)
    print(f'Готово: {len(tr)} примеров обучения, {len(ev)} примеров валидации → {work}/{name}_tokens.bin.')

# м1
def xtts_prep(manifest, out):
    rows = read_manifest(manifest)
    if not rows: sys.exit('Манифест пуст, либо рядом с ним не найдены файлы звука.')
    tr, ev = split(rows, hold=10)
    os.makedirs(out, exist_ok=True)
    for fn, rs in (('metadata_train.csv', tr), ('metadata_eval.csv', ev)):
        with open(os.path.join(out, fn), 'w', encoding='utf-8') as f:
            f.write('audio_file|text|speaker_name\n')
            for r in rs: f.write(f"{r['audio']}|{r['text'].replace('|', ' ')}|{r['speaker']}\n")
    print(f'Готово: {len(tr)} примеров обучения, {len(ev)} примеров валидации → {out}.')

XTTS_TPL = '''# Дообучение GPT-части XTTS v2 — по официальному рецепту Coqui (recipes/ljspeech/xtts_v2/train_gpt_xtts.py). Создано tts.py.
# Не запускалось на этой машине: torch и coqui-tts ставятся (pip install coqui-tts), dvae.pth и mel_stats.pth берутся из coqui/XTTS-v2.
import os
from trainer import Trainer, TrainerArgs
from TTS.config.shared_configs import BaseDatasetConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.layers.xtts.trainer.gpt_trainer import GPTArgs, GPTTrainer, GPTTrainerConfig, XttsAudioConfig
from TTS.utils.manage import ModelManager
MODEL = {model!r}; DATA = {data!r}; OUT = {out!r}; LANG = {lang!r}
CK = os.path.join(OUT, 'base'); os.makedirs(CK, exist_ok=True)
def need(name, url):
    p = os.path.join(CK, name)
    if not os.path.isfile(p): ModelManager._download_model_files([url], CK, progress_bar=True)
    return p
DVAE = need('dvae.pth', 'https://huggingface.co/coqui/XTTS-v2/resolve/main/dvae.pth')
MEL = need('mel_stats.pth', 'https://huggingface.co/coqui/XTTS-v2/resolve/main/mel_stats.pth')
TOK = os.path.join(MODEL, 'vocab.json'); XCK = os.path.join(MODEL, 'model.pth')
ds = BaseDatasetConfig(formatter='coqui', dataset_name='tierlm_tts', path=DATA, meta_file_train='metadata_train.csv', meta_file_val='metadata_eval.csv', language=LANG)
args = GPTArgs(max_conditioning_length=132300, min_conditioning_length=66150, debug_loading_failures=False, max_wav_length=255995, max_text_length=200,
               mel_norm_file=MEL, dvae_checkpoint=DVAE, xtts_checkpoint=XCK, tokenizer_file=TOK, gpt_num_audio_tokens=1026, gpt_start_audio_token=1024,
               gpt_stop_audio_token=1025, gpt_use_masking_gt_prompt_approach=True, gpt_use_perceiver_resampler=True)
audio = XttsAudioConfig(sample_rate=22050, dvae_sample_rate=22050, output_sample_rate=24000)
cfg = GPTTrainerConfig(output_path=OUT, model_args=args, run_name='tierlm_xtts', project_name='tierlm', run_description='tierlm', dashboard_logger='tensorboard',
                       audio=audio, batch_size={batch}, batch_group_size=48, eval_batch_size={batch}, num_loader_workers=4, eval_split_max_size=256,
                       print_step=50, plot_step=100, log_model_step=1000, save_step={save}, save_n_checkpoints=1, save_checkpoints=True,
                       optimizer='AdamW', optimizer_wd_only_on_weights=True, optimizer_params={{'betas': [0.9, 0.96], 'eps': 1e-8, 'weight_decay': 1e-2}},
                       lr={lr}, lr_scheduler='MultiStepLR', lr_scheduler_params={{'milestones': [50000 * 18, 150000 * 18, 300000 * 18], 'gamma': 0.5, 'last_epoch': -1}},
                       epochs={epochs})
model = GPTTrainer.init_from_config(cfg)
train, ev = load_tts_samples([ds], eval_split=True, eval_split_max_size=cfg.eval_split_max_size, eval_split_size=cfg.eval_split_size)
Trainer(TrainerArgs(restore_path=None, skip_train_epoch=False, start_with_eval=False, grad_accum_steps={accum}), cfg, output_path=OUT, model=model,
        train_samples=train, eval_samples=ev).fit()
'''

def xtts_script(model, data, out, lang='ru', batch=2, accum=32, lr=5e-6, epochs=10, save=1000):
    os.makedirs(out, exist_ok=True); p = os.path.join(out, 'train_xtts.py')
    open(p, 'w', encoding='utf-8').write(XTTS_TPL.format(model=os.path.abspath(model), data=os.path.abspath(data), out=os.path.abspath(out), lang=lang,
                                                           batch=batch, accum=accum, lr=lr, epochs=epochs, save=save))
    print('Сценарий:', p, '\nЗапуск: python3', p); return p

# о1
def selftest():
    random.seed(1); T = 11
    c0 = [random.randrange(4096) for _ in range(T)]; c1 = [random.randrange(4096) for _ in range(2 * T)]; c2 = [random.randrange(4096) for _ in range(4 * T)]
    ids = snac_pack(c0, c1, c2)
    assert len(ids) == 7 * T and min(ids) >= AUDIO0 and max(ids) < AUDIO0 + 7 * SLOT, 'Выход за диапазон токенов звука'
    assert snac_unpack([SOS] + ids + [EOS]) == (c0, c1, c2), 'Распаковка не вернула исходные коды'
    p, a = orpheus_example([5, 6, 7], ids); assert p[0] == SOH and p[-2:] == [EOT, EOH] and a[:2] == [SOAI, SOS] and a[-2:] == [EOS, EOAI]
    import tempfile
    f = os.path.join(tempfile.mkdtemp(), 't.bin'); write_bin(f, [(p, a)] * 3)
    b = open(f, 'rb').read(); n, ln, pl = struct.unpack('<iii', b[:12])
    assert n == 3 and ln == len(p) + len(a) and pl == len(p), 'Формат .bin расходится с prepare.py'
    print('Самопроверка пройдена: упаковка SNAC туда и обратно, границы токенов и формат .bin корректны.')

if __name__ == '__main__':
    a = sys.argv[1:]
    if not a: sys.exit(__doc__ if __doc__ else open(__file__, encoding='utf-8').read().split('\n\n')[0])
    if a[0] == 'detect': print(json.dumps(detect(a[1]), ensure_ascii=False, indent=1))
    elif a[0] == 'prep': prep_orpheus(a[1], a[2], *(a[3:4] or ['run']), work=E('WORK', './work'), voice=E('VOICE', ''))
    elif a[0] == 'xtts-prep': xtts_prep(a[1], a[2])
    elif a[0] == 'xtts-script': xtts_script(a[1], a[2], a[3], lang=E('LANG_TTS', 'ru'))
    elif a[0] == 'selftest': selftest()
    else: sys.exit('Доступные команды: detect, prep, xtts-prep, xtts-script, selftest.')
