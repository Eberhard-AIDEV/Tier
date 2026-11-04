# а1
import json, os

MANUAL = ('gemma4',)      # а2

def manual(model_type):
    # б1
    return str(model_type or '').lower().startswith(MANUAL)

class Fmt:
    def __init__(self, mdir, tok):
        self.tok, self.mdir = tok, mdir
        self.enc = lambda s: tok.encode(s, add_special_tokens=False).ids
        cfg = self._j('config.json'); cfg = dict(cfg, **(cfg.get('text_config') or {}))
        # в1
        self.gemma4 = manual(cfg.get('model_type')) or tok.token_to_id('<|turn>') is not None and tok.token_to_id('<|channel>') is not None
        tc = self._j('tokenizer_config.json'); sm = self._j('special_tokens_map.json')
        tp = os.path.join(mdir, 'chat_template.jinja')
        self.src = open(tp, encoding='utf-8').read() if os.path.exists(tp) else self._j('chat_template.json').get('chat_template') or tc.get('chat_template')
        if isinstance(self.src, list): self.src = next((t['template'] for t in self.src if t.get('name') == 'default'), self.src[0]['template'])
        st = lambda k: (lambda v: v.get('content') if isinstance(v, dict) else v)(tc.get(k) or sm.get(k))
        self.bos, self.eos = st('bos_token') or '', st('eos_token') or ''
        gen = self._j('generation_config.json'); e = gen.get('eos_token_id', cfg.get('eos_token_id'))
        self.stop = set(e if isinstance(e, list) else [e] if isinstance(e, int) else [])
        if self.gemma4: self.stop |= {tok.token_to_id('<turn|>'), 1}
        elif self.eos and tok.token_to_id(self.eos) is not None: self.stop.add(tok.token_to_id(self.eos))
        self._t = None
        self.think = False      # в2
        if not self.gemma4 and not self.src: raise SystemExit(f'{mdir}: Нет шаблона диалога (chat_template в chat_template.jinja, chat_template.json или tokenizer_config.json): модель базовая и не предназначена для разговора.')

    def _j(self, n):
        try: return json.load(open(os.path.join(self.mdir, n), encoding='utf-8'))
        except (OSError, ValueError): return {}

    def _render(self, msgs, gen):
        if self._t is None:
            from jinja2.sandbox import ImmutableSandboxedEnvironment
            from jinja2.ext import Extension
            from jinja2 import nodes
            import datetime

            class Generation(Extension):
                # в3
                tags = {'generation'}

                def parse(self, parser):
                    ln = next(parser.stream).lineno
                    body = parser.parse_statements(['name:endgeneration'], drop_needle=True)
                    return nodes.Scope(body).set_lineno(ln)
            env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True, extensions=['jinja2.ext.loopcontrols', Generation])
            def _raise(m): raise ValueError(m)
            env.globals.update(raise_exception=_raise, strftime_now=lambda f: datetime.datetime.now().strftime(f))
            self._t = env.from_string(self.src)
            self.nosys = False
        if self.nosys: msgs = self._fold(msgs)
        try: return self._t.render(messages=msgs, add_generation_prompt=gen, bos_token=self.bos, eos_token=self.eos, enable_thinking=self.think)
        except ValueError:
            # в4
            if self.nosys or not msgs or msgs[0]['role'] != 'system': raise
            self.nosys = True
            return self._t.render(messages=self._fold(msgs), add_generation_prompt=gen, bos_token=self.bos, eos_token=self.eos, enable_thinking=self.think)

    @staticmethod
    def _fold(msgs):
        if not msgs or msgs[0]['role'] != 'system': return msgs
        s, rest = msgs[0]['content'], [dict(m) for m in msgs[1:]]
        if rest and rest[0]['role'] == 'user': rest[0]['content'] = s + '\n\n' + rest[0]['content']
        else: rest.insert(0, {'role': 'user', 'content': s})
        return rest

    # в5
    def _g4turn(self, role, text):
        sp = self.tok.token_to_id
        return [sp('<|turn>')] + self.enc(role + '\n' + text) + [sp('<turn|>')] + self.enc('\n')
    def _g4model(self):
        sp = self.tok.token_to_id
        return [sp('<|turn>')] + self.enc('model\n') + [sp('<|channel>')] + self.enc('thought\n') + [sp('<channel|>')]

    def prompt(self, system, turns):
        # в6
        if self.gemma4:
            sp = self.tok.token_to_id; ids = [sp('<bos>')] + (self._g4turn('system', system) if system else [])
            for u, a in turns:
                ids += self._g4turn('user', u)
                if a is not None: ids += self._g4model() + self.enc(a) + [sp('<turn|>')] + self.enc('\n')
            return ids + self._g4model()
        msgs = ([{'role': 'system', 'content': system}] if system else [])
        for u, a in turns:
            msgs.append({'role': 'user', 'content': u})
            if a is not None: msgs.append({'role': 'assistant', 'content': a})
        return self.enc(self._render(msgs, True))

    def example(self, system, user, answer):
        # в7
        if self.gemma4:
            return self.prompt(system, [(user, None)]), self.enc(answer) + [self.tok.token_to_id('<turn|>')]
        msgs = ([{'role': 'system', 'content': system}] if system else []) + [{'role': 'user', 'content': user}]
        p = self._render(msgs, True); f = self._render(msgs + [{'role': 'assistant', 'content': answer}], False)
        if not f.startswith(p) or answer not in f[len(p):]:
            # в8
            return self.enc(p), self.enc(answer) + ([self.tok.token_to_id(self.eos)] if self.eos and self.tok.token_to_id(self.eos) is not None else [])
        tail = f[len(p):].rstrip('\n')
        return self.enc(p), self.enc(tail)
