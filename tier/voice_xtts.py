#!/usr/bin/env python3
# а1
import json, os, subprocess, sys

def say(o): sys.stdout.write(json.dumps(o, ensure_ascii=False) + '\n'); sys.stdout.flush()

def need():
    try: import TTS, torch   # noqa: F401
    except ImportError:
        say({'info': 'Установка coqui-tts.'})
        for cmd in ([sys.executable, '-m', 'pip', 'install', '--user', 'coqui-tts'], [sys.executable, '-m', 'pip', 'install', '--user', '--break-system-packages', 'coqui-tts']):
            if subprocess.run(cmd, capture_output=True).returncode == 0: break
        else: say({'error': 'Не удалось установить coqui-tts (torch: на Arch — pacman -S python-pytorch).'}); sys.exit(1)

def main():
    d, spk, lang = sys.argv[1], (sys.argv[2] if len(sys.argv) > 2 else ''), (sys.argv[3] if len(sys.argv) > 3 else 'ru')
    need()
    import torch, torchaudio
    from TTS.tts.configs.xtts_config import XttsConfig
    from TTS.tts.models.xtts import Xtts
    cfg = XttsConfig(); cfg.load_json(os.path.join(d, 'config.json'))
    m = Xtts.init_from_config(cfg); m.load_checkpoint(cfg, checkpoint_dir=d, eval=True, use_deepspeed=False)
    dev = os.environ.get('XTTS_DEVICE', 'auto'); dev = ('cuda' if torch.cuda.is_available() else 'cpu') if dev == 'auto' else dev
    m.to(dev)
    if spk and os.path.isfile(spk): lat, emb = m.get_conditioning_latents(audio_path=[spk])
    else:
        sp = m.speaker_manager.speakers if m.speaker_manager is not None else {}
        if not sp: say({'error': 'В папке нет speakers_xtts.pth. Требуется указать wav с образцом голоса.'}); return
        k = spk if spk in sp else next(iter(sp)); lat, emb = sp[k]['gpt_cond_latent'], sp[k]['speaker_embedding']
    say({'ready': True})
    for line in sys.stdin:
        try:
            r = json.loads(line); o = m.inference(r['text'], lang, lat, emb)
            torchaudio.save(r['out'], torch.tensor(o['wav']).unsqueeze(0), 24000); say({'ok': True})
        except Exception as e: say({'error': f'XTTS: {e}'})

if __name__ == '__main__': main()
