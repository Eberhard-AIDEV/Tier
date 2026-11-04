#!/usr/bin/env python3
# а1
import json, os, queue, re, shutil, subprocess, sys, tempfile, threading

PLAYERS = [['pw-play'], ['paplay'], ['aplay', '-q'], ['ffplay', '-nodisp', '-autoexit', '-loglevel', 'quiet'], ['play', '-q']]

def player():
    for c in PLAYERS:
        if shutil.which(c[0]): return c
    return None

# в1
def clean(t):
    t = re.sub(r'```.*?```', ' ', t, flags=re.S)                 # в2
    t = re.sub(r'`([^`]*)`', r'\1', t)
    t = re.sub(r'\[([^\]]*)\]\([^)]*\)', r'\1', t)               # в3
    t = re.sub(r'https?://\S+', ' ', t)
    t = re.sub(r'[*_#>~|]+', ' ', t)
    return re.sub(r'[ \t\r\f\v]+', ' ', t).strip()

SENT = re.compile(r'(.+?[.!?…]+[»"\')\]]*(?=\s)|.+?\n)', re.S)

OPEN = {'<think>': '</think>', '```': '```'}       # в4

class Splitter:
    # г1
    def __init__(self, min_len=12, max_len=260):
        self.buf, self.text, self.skip, self.min, self.max = '', '', None, min_len, max_len
    @staticmethod
    def _hold(buf, marks):
        # г2
        h = 0
        for o in marks:
            for n in range(min(len(o) - 1, len(buf)), 0, -1):
                if buf.endswith(o[:n]): h = max(h, n); break
        return h
    def _drop(self):
        s = ''
        while True:
            if self.skip:
                i = self.buf.find(self.skip)
                if i < 0:
                    h = self._hold(self.buf, [self.skip]); self.buf = self.buf[len(self.buf) - h:] if h else ''; return s
                self.buf = self.buf[i + len(self.skip):]; self.skip = None
            hits = [(self.buf.find(o), o) for o in OPEN if o in self.buf]
            if hits:
                i, o = min(hits); s += self.buf[:i]; self.buf = self.buf[i + len(o):]; self.skip = OPEN[o]; continue
            h = self._hold(self.buf, OPEN)
            s += self.buf[:len(self.buf) - h]; self.buf = self.buf[len(self.buf) - h:]; return s
    def feed(self, piece):
        self.buf += piece; out = []; self.text += self._drop()
        while True:
            cur, pos = '', 0
            for m in SENT.finditer(self.text):           # г3
                cur += m.group(1); pos = m.end()
                if len(cur) >= self.min: break
            else:
                if len(self.text) >= self.max:           # г4
                    cut = max(self.text.rfind(', ', 0, self.max), self.text.rfind(' ', 0, self.max))
                    cut = cut if cut > self.min else self.max
                    out.append(self.text[:cut + 1]); self.text = self.text[cut + 1:]; continue
                break
            out.append(cur); self.text = self.text[pos:]
        return [x for x in (clean(o) for o in out) if x]
    def flush(self):
        t = clean(self.text + ('' if self.skip else self.buf)); self.buf = self.text = ''; self.skip = None
        return [t] if t else []

# д1
class Espeak:
    id, name = 'espeak', 'Системный (espeak-ng)'
    def __init__(self, o): self.lang = o.get('lang') or 'ru'; self.speed = str(int(o.get('speed') or 165))
    @staticmethod
    def ready(o): return None if shutil.which('espeak-ng') else 'Не установлен espeak-ng (Arch: pacman -S espeak-ng).'
    def synth(self, text, out):
        r = subprocess.run(['espeak-ng', '-v', self.lang, '-s', self.speed, '-w', out, text], capture_output=True, text=True)
        return None if r.returncode == 0 and os.path.getsize(out) > 44 else (r.stderr.strip() or 'Не удалось получить звук от espeak-ng.')
    def close(self): pass

class Xtts:
    # е1
    id, name = 'xtts', 'XTTS'
    def __init__(self, o):
        self.o, self.p, self.lock = o, None, threading.Lock()
    @staticmethod
    def ready(o):
        d = o.get('xtts_dir') or ''
        if not (d and os.path.isfile(os.path.join(d, 'model.pth')) and os.path.isfile(os.path.join(d, 'config.json'))): return 'Требуется указать папку XTTS (model.pth, config.json, vocab.json).'
        return None
    def _start(self):
        env = dict(os.environ); env['XTTS_DEVICE'] = self.o.get('device') or 'auto'
        self.p = subprocess.Popen([sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'voice_xtts.py'), self.o['xtts_dir'],
                                   self.o.get('speaker') or '', self.o.get('lang') or 'ru'], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, env=env)
        first = self.p.stdout.readline()
        if '"ready"' not in first: self.p = None; return (json.loads(first).get('error') if first.strip().startswith('{') else None) or 'Не удалось запустить XTTS.'
    def synth(self, text, out):
        with self.lock:
            if self.p is None or self.p.poll() is not None:
                e = self._start()
                if e: return e
            self.p.stdin.write(json.dumps({'text': text, 'out': out}, ensure_ascii=False) + '\n'); self.p.stdin.flush()
            r = self.p.stdout.readline()
        try: j = json.loads(r)
        except ValueError: return 'Не удалось разобрать ответ XTTS.'
        return j.get('error')
    def close(self):
        if self.p and self.p.poll() is None: self.p.terminate()
        self.p = None

PROVIDERS = {c.id: c for c in (Espeak, Xtts)}

# ж1
class Voice:
    def __init__(self, on_error=None, play=None):
        self.err, self.cfg, self.prov, self.split = on_error or (lambda m: None), {}, None, {}
        self.qs, self.qp = queue.Queue(), queue.Queue()      # ж2
        self.gen, self.proc, self.tmp = 0, None, tempfile.mkdtemp(prefix='tierlm_voice_')
        self._play = play or self._play_default
        self.busy = 0; self.lock = threading.Lock()
        threading.Thread(target=self._synth_loop, daemon=True).start(); threading.Thread(target=self._play_loop, daemon=True).start()

    def listing(self):
        o = self.cfg
        return [{'id': i, 'name': c.name, 'why': c.ready(o) or ''} for i, c in PROVIDERS.items()]

    def configure(self, cfg):
        old = (self.cfg.get('provider'), self.cfg.get('xtts_dir'), self.cfg.get('speaker'), self.cfg.get('lang'), self.cfg.get('device'))
        self.cfg = dict(cfg)
        if old != (cfg.get('provider'), cfg.get('xtts_dir'), cfg.get('speaker'), cfg.get('lang'), cfg.get('device')):
            self.stop()
            if self.prov: self.prov.close()
            self.prov = None

    def enabled(self): return bool(self.cfg.get('provider')) and self.cfg['provider'] in PROVIDERS

    def _provider(self):
        if self.prov is None:
            c = PROVIDERS.get(self.cfg.get('provider'))
            if not c: return None
            why = c.ready(self.cfg)
            if why: self.err(why); return None
            self.prov = c(self.cfg)
        return self.prov

    def _enqueue(self, sents):
        for s in sents:
            with self.lock: self.busy += 1
            self.qs.put((self.gen, s))

    # ж3
    def feed(self, rid, piece):
        if not self.enabled(): return
        sp = self.split.setdefault(rid, Splitter()); self._enqueue(sp.feed(piece))
    def end(self, rid):
        sp = self.split.pop(rid, None)
        if sp and self.enabled(): self._enqueue(sp.flush())
    # ж4
    def say(self, text):
        if not self.enabled(): return
        sp = Splitter(); self._enqueue(sp.feed(text) + sp.flush())

    def stop(self):
        with self.lock:
            self.gen += 1; self.busy = 0
            for q in (self.qs, self.qp):
                try:
                    while True: q.get_nowait()
                except queue.Empty: pass
        self.split.clear()
        if self.proc and self.proc.poll() is None: self.proc.terminate()

    def speaking(self): return self.busy > 0

    def _synth_loop(self):
        n = 0
        while True:
            g, text = self.qs.get()
            if g != self.gen: continue
            p = self._provider()
            if p is None:
                with self.lock: self.busy = max(0, self.busy - 1)
                continue
            n += 1; out = os.path.join(self.tmp, f'{n}.wav')
            try: e = p.synth(text, out)
            except Exception as x: e = str(x)
            if g != self.gen: continue
            if e:
                self.err(e)
                with self.lock: self.busy = max(0, self.busy - 1)
                continue
            self.qp.put((g, out))

    def _play_default(self, path):
        c = player()
        if not c: self.err('Не найден проигрыватель (pw-play, paplay, aplay, ffplay).'); return
        self.proc = subprocess.Popen(c + [path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL); self.proc.wait()

    def _play_loop(self):
        while True:
            g, path = self.qp.get()
            try:
                if g == self.gen: self._play(path)
            finally:
                with self.lock: self.busy = max(0, self.busy - 1)
                try: os.remove(path)
                except OSError: pass
