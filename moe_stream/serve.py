"""OpenAI-compatible HTTP backend for streamed-expert MoE inference.

È il motore dietro l'app nativa moe-stream.app (app/): l'app lo lancia,
sorveglia /metrics e parla questi endpoint.

Endpoints:
  GET  /v1/models             -> the single served model id
  POST /v1/chat/completions   -> chat completion, streaming (SSE) or not
  GET  /metrics               -> stats runtime (cache, arena, spec, RAM)
  POST /api/orchestrate       -> {goal, n} pianificatore->coder(wave)->revisore
  POST /api/jobs              -> lancia N compiti {prompts, mode, max_tokens}
  GET  /api/jobs[/<id>]       -> elenco / dettaglio job con output
  POST /api/jobs/<id>/cancel  -> annulla i compiti non ancora partiti

La generazione usa l'auto-speculativa (come chat.py/generate.py); una
generazione alla volta — MLX/Metal non codifica in parallelo — quindi tutto
si serializza dietro un lock: è la strategia misurata migliore per
l'interattivo. La modalità job "wave" usa il batch lockstep (~104 tok/s
aggregati a 32) per il lavoro in massa.

Stdlib only (http.server).

  python -m moe_stream.serve MODEL_DIR SHARD_DIR [--port 7070]
  python -m moe_stream.serve MODEL_DIR --plain   [--port 7070]

--plain: modello caricato intero in RAM via mlx_lm (per modelli che ci
stanno, es. Gemma-4-26B 4-bit = 15 GB); niente shard, niente speculativa.
Stessi endpoint, stessa app.
"""

import argparse
import json
import os
import queue
import re
import subprocess
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

WAVE = 32  # sweet spot misurato per la modalità wave

# Gemma-4 emette il ragionamento come "<|channel>thought ... <channel|>testo".
# Agli utenti mostriamo solo il testo finale.
_CH_OPEN, _CH_CLOSE = "<|channel>", "<channel|>"


def _strip_thought_text(text):
    if text.startswith(_CH_OPEN):
        i = text.find(_CH_CLOSE)
        if i >= 0:
            return text[i + len(_CH_CLOSE):]
    return text


def _strip_thought_stream(pieces):
    """Filtra il canale 'thought' da uno stream di pezzi di testo.
    Passthrough immediato se l'output non inizia con <|channel>."""
    buf, mode = "", "sniff"
    for p in pieces:
        if mode == "pass":
            yield p
            continue
        buf += p
        if mode == "sniff":
            if len(buf) < len(_CH_OPEN):
                if _CH_OPEN.startswith(buf):
                    continue
                mode = "pass"
                yield buf
                buf = ""
                continue
            if buf.startswith(_CH_OPEN):
                mode = "strip"
            else:
                mode = "pass"
                yield buf
                buf = ""
                continue
        if mode == "strip":
            i = buf.find(_CH_CLOSE)
            if i >= 0:
                mode = "pass"
                rest = buf[i + len(_CH_CLOSE):]
                buf = ""
                if rest:
                    yield rest
    if buf:  # thought mai chiuso o sniff a fine stream: meglio mostrare
        yield buf


def _chunk(cid, model, delta, finish=None):
    return {
        "id": cid, "object": "chat.completion.chunk",
        "created": int(time.time()), "model": model,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
    }


def _full(cid, model, text, usage):
    return {
        "id": cid, "object": "chat.completion",
        "created": int(time.time()), "model": model,
        "choices": [{"index": 0, "finish_reason": "stop",
                     "message": {"role": "assistant", "content": text}}],
        "usage": usage,
    }


# MARK: agente singolo con strumenti (ispirato al prompt di pi.dev)

_PROMPT_SOLO = """Sei un agente di programmazione autonomo. Lavori nella \
cartella di lavoro dell'utente; tutti i percorsi sono relativi ad essa.

Per usare uno strumento rispondi con SOLO un blocco cosi' e poi fermati:
```tool
{"tool": "list"}
```
Strumenti disponibili:
{"tool": "list"} — elenca i file della cartella
{"tool": "read", "path": "f"} — leggi un file
{"tool": "write", "path": "f", "content": "…"} — crea o sovrascrivi un file
{"tool": "edit", "path": "f", "old": "…", "new": "…"} — cambia SOLO la porzione
  'old' (deve essere unica nel file) con 'new': preferiscilo a write per le
  modifiche, costa molti meno token
{"tool": "run", "cmd": "…"} — esegui un comando shell nella cartella
{"tool": "web", "url": "…"} — scarica una pagina web (per cercare, usa un URL
  come https://html.duckduckgo.com/html/?q=le+tue+parole)

Regole, in ordine di importanza:
1. Agisci, non chiedere: hai già tutto quello che ti serve.
2. Prima capisci (list, read), poi fai la modifica più piccola che funziona:
   per correggere un file esistente usa edit, non riscriverlo con write.
3. Verifica sempre con run (esegui il codice o i test) prima di dirti finito.
4. Sii parco: niente spiegazioni di ciò che stai per fare, un solo strumento
   per risposta, poi fermati e aspetta il risultato.
5. A obiettivo raggiunto E verificato, rispondi SENZA blocco tool: due righe
   su cosa hai fatto e come l'hai verificato."""

_PROMPT_ORCH = """Sei l'orchestratore di una squadra di agenti di \
programmazione. Organizza il lavoro in FASI sequenziali. Dentro una fase gli \
agenti lavorano in parallelo, quindi mettici SOLO compiti davvero \
indipendenti tra loro. Se un compito ha bisogno del risultato di un altro \
(es. valutare o testare qualcosa che prima va costruito), mettilo in una \
fase successiva: gli agenti di una fase ricevono i risultati delle fasi \
precedenti. Usa il minor numero di fasi e di agenti che copre l'obiettivo \
senza sovrapposizioni — al massimo {cap} agenti in totale, ognuno costa. \
Ogni token che gli agenti leggono costa: istruzioni asciutte e autosufficienti.

Formato ESATTO, niente altro:
FASE
- primo compito della fase
- secondo compito della fase
FASE
- compito che dipende dai risultati della fase precedente"""

_PROMPT_REPORT = """Rapporto degli agenti (solo esiti, non gli elaborati):
{report}

Se va tutto bene rispondi solo: TUTTO OK
Per ogni sotto-compito da rifare rispondi invece con una riga:
RIFAI <numero>: <istruzione corretta e più precisa>
Nient'altro."""

_CODER_NOTE = ("\n\nChiudi la risposta con una sola riga finale: "
               "'ESITO: OK' se hai completato, oppure "
               "'ESITO: PROBLEMA: <motivo in max 10 parole>' se no.")

# nota aggiuntiva quando c'è una cartella di lavoro: i coder CREANO file
_CODER_FILES = ("\n\nOgni file che produci va in un blocco così, con il "
                "percorso nella cartella di lavoro:\n```file:percorso/nome.ext"
                "\n<contenuto completo del file>\n```\nCrea i file, non "
                "rileggerli: al resto pensa l'orchestratore.")

_ESITO_RE = re.compile(r"^ESITO:\s*(.+)$", re.M)
_REDO_RE = re.compile(r"RIFAI\s+(\d+)\s*:\s*(.+)")


def _parse_esito(text):
    """-> (esito, testo ripulito dalla riga ESITO)."""
    m = _ESITO_RE.search(text)
    if not m:
        return "nessun esito dichiarato", text
    return m.group(1).strip(), _ESITO_RE.sub("", text).rstrip()


def _parse_redos(resp):
    return [(int(i) - 1, instr.strip()) for i, instr in
            _REDO_RE.findall(resp)]


_PHASE_RE = re.compile(r"^\s*(?:#+\s*)?fase\b", re.I)


def _parse_phases(plan, cap):
    """Il piano dell'orchestratore -> lista di fasi (ognuna lista di compiti),
    rispettando il tetto totale cap. Senza marcatori FASE: una fase sola
    (retrocompatibile con i piani a lista piatta)."""
    phases, cur = [], []
    for ln in plan.splitlines():
        if not ln.strip():
            continue
        if _PHASE_RE.match(ln):
            if cur:
                phases.append(cur)
                cur = []
            continue
        cur.append(ln.strip(" -*0123456789.").strip())
    if cur:
        phases.append(cur)
    if not phases:
        phases = [[ln.strip(" -*0123456789.").strip()
                   for ln in plan.splitlines() if ln.strip()]]
    out, tot = [], 0
    for ph in phases:
        room = cap - tot
        if room <= 0:
            break
        ph = [c for c in ph[:room] if c]
        if ph:
            out.append(ph)
            tot += len(ph)
    return out or [[""]]


# --- Lezioni: memoria condivisa tra job. Quando un RIFAI corregge un errore,
# quella correzione è una lezione riusabile; i job successivi la leggono e non
# ripetono lo stesso errore. Per-cartella se c'è una workdir (versionabile,
# visibile all'utente), altrimenti globale.
_LESSONS_PROMPT_LINES = 25   # quante ne diamo in pasto al modello
_LESSONS_FILE_LINES = 80     # quante ne teniamo su disco


def _lessons_path(job):
    if job.get("workdir"):
        return Path(job["workdir"]) / ".moe-lessons.md"
    d = Path.home() / "Library/Application Support/moe-stream"
    d.mkdir(parents=True, exist_ok=True)
    return d / "lessons.md"


def _load_lessons(job):
    """Le lezioni recenti come blocco di prompt, o '' se non ce ne sono."""
    try:
        lines = [ln for ln in
                 _lessons_path(job).read_text().splitlines() if ln.strip()]
    except OSError:
        return ""
    if not lines:
        return ""
    recent = "\n".join(lines[-_LESSONS_PROMPT_LINES:])
    return ("Lezioni da lavori precedenti — NON ripetere questi errori:\n"
            f"{recent}\n\n")


def _save_lesson(job, instr):
    """Appende una lezione (l'istruzione correttiva di un RIFAI) e tiene il
    file entro _LESSONS_FILE_LINES righe."""
    path = _lessons_path(job)
    goal = _clip_words(job.get("goal", ""), 8)
    line = f"- [{goal}] {instr.strip()}"
    try:
        old = [ln for ln in path.read_text().splitlines() if ln.strip()]
    except OSError:
        old = []
    if line in old:  # niente duplicati
        return
    kept = (old + [line])[-_LESSONS_FILE_LINES:]
    try:
        path.write_text("\n".join(kept) + "\n")
    except OSError:
        pass  # cartella non scrivibile: la lezione si perde, non è fatale


def _clip_words(s, n):
    w = s.split()
    return " ".join(w[:n]) + ("…" if len(w) > n else "")

_TOOL_RE = re.compile(r"```tool\s*(\{.*?\})\s*```", re.S)
_CLIP = 4000
_SOLO_STEPS = 16


def _clip(s):
    return s if len(s) <= _CLIP else s[:_CLIP] + "\n… (troncato)"


def _tail_loop(s, min_span=20, min_total=80, reps=4):
    """True se la CODA di s è una ripetizione degenerativa (doom loop): lo
    stesso span identico ripetuto >= reps volte, per >= min_total caratteri.
    A inferenza è l'unico rimedio senza fine-tuning (cfr. liquid.ai/antidoom):
    i modelli reasoning greedy ci cascano nel ~20% dei completamenti duri.
    Controlla solo la coda, così è O(len tail) e non O(testo intero)."""
    n = len(s)
    for span in range(min_span, n // reps + 1):
        if span * reps < min_total:
            continue
        chunk = s[n - span:]
        if all(s[n - span * (k + 1):n - span * k] == chunk
               for k in range(1, reps)):
            return True
    return False


def _parse_tool(text):
    m = _TOOL_RE.search(text)
    if not m:
        return None
    try:
        return json.loads(m.group(1))
    except json.JSONDecodeError:
        return {"tool": "_badjson"}


def _safe_path(workdir, rel):
    p = (workdir / rel).resolve()
    root = workdir.resolve()
    if p != root and root not in p.parents:
        raise ValueError(f"percorso fuori dalla cartella di lavoro: {rel}")
    return p


def _tool_list(workdir):
    skip = {"node_modules", "__pycache__", "venv", ".git", ".build"}
    lines = []
    for root, dirs, files in os.walk(workdir):
        rel = Path(root).relative_to(workdir)
        dirs[:] = sorted(d for d in dirs
                         if not d.startswith(".") and d not in skip)
        if len(rel.parts) >= 3:
            dirs[:] = []
        for f in sorted(files):
            if f.startswith("."):
                continue
            lines.append(str(rel / f) if rel.parts else f)
            if len(lines) >= 200:
                return "\n".join(lines) + "\n… (troncato)"
    return "\n".join(lines) or "(cartella vuota)"


def _exec_tool(workdir, call):
    try:
        tool = call.get("tool")
        if tool == "list":
            return _clip(_tool_list(workdir))
        if tool == "read":
            return _clip(_safe_path(workdir, call["path"])
                         .read_text(errors="replace"))
        if tool == "write":
            p = _safe_path(workdir, call["path"])
            p.parent.mkdir(parents=True, exist_ok=True)
            content = call.get("content", "")
            p.write_text(content)
            return f"scritto {call['path']} ({len(content)} caratteri)"
        if tool == "edit":
            # cambia SOLO le righe interessate: str_replace unico -> pochi token
            p = _safe_path(workdir, call["path"])
            s = p.read_text()
            old = call.get("old", "")
            cnt = s.count(old) if old else 0
            if cnt != 1:
                return (f"ERRORE: il testo da sostituire compare {cnt} volte, "
                        "deve essere unico. Allarga 'old' con più contesto.")
            p.write_text(s.replace(old, call.get("new", ""), 1))
            return f"modificato {call['path']} (una porzione)"
        if tool == "web":
            r = subprocess.run(
                ["curl", "-sL", "--max-time", "20", "-A",
                 "Mozilla/5.0 moe-stream-agent", call["url"]],
                capture_output=True, text=True, timeout=25)
            return _clip(_html_to_text(r.stdout))
        if tool == "run":
            r = subprocess.run(call["cmd"], shell=True, cwd=workdir,
                               capture_output=True, text=True, timeout=120)
            out = (r.stdout or "") + (r.stderr or "")
            return _clip(f"(exit {r.returncode})\n{out}".strip())
        return ("ERRORE: strumento sconosciuto; usa list, read, write, edit, "
                "run o web in un blocco ```tool con JSON valido")
    except subprocess.TimeoutExpired:
        return "ERRORE: comando interrotto per timeout"
    except Exception as e:
        return f"ERRORE: {e}"


_TAG_RE = re.compile(r"<(script|style)[^>]*>.*?</\1>", re.S | re.I)
_HTML_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\n\s*\n\s*\n+")


def _html_to_text(html):
    """Riduce una pagina a testo leggibile: via script/style e tag, così una
    ricerca web non spreca il contesto dell'agente in markup."""
    t = _TAG_RE.sub(" ", html)
    t = _HTML_RE.sub(" ", t)
    t = re.sub(r"[ \t]+", " ", t)
    return _WS_RE.sub("\n\n", t).strip()


def _describe_tool(call):
    what = call.get("path") or call.get("cmd") or call.get("url") or ""
    return f"{call.get('tool')} {what}"[:90]


# --- File prodotti dai coder: il coder li racchiude in ```file:percorso ...```
# e il server li scrive nella cartella di lavoro (i coder creano, non rileggono).
_FILE_RE = re.compile(r"```file:(\S+)[ \t]*\n(.*?)```", re.S)


def _write_coder_files(workdir, text):
    """Scrive i file dichiarati nel testo di un coder. -> lista dei percorsi."""
    written = []
    for rel, body in _FILE_RE.findall(text):
        try:
            p = _safe_path(Path(workdir), rel)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(body)
            written.append(rel)
        except Exception:
            pass  # percorso fuori workdir o non scrivibile: si salta
    return written


def _git_snapshot(workdir, msg):
    """Un checkpoint git nella cartella di lavoro (init pigro). Silenzioso se
    git manca o la cartella non è versionabile."""
    wd = str(workdir)
    try:
        if not (Path(workdir) / ".git").is_dir():
            subprocess.run(["git", "init", "-q"], cwd=wd, timeout=15)
            subprocess.run(["git", "config", "user.email", "agent@moe-stream"],
                           cwd=wd, timeout=15)
            subprocess.run(["git", "config", "user.name", "moe-stream agent"],
                           cwd=wd, timeout=15)
        subprocess.run(["git", "add", "-A"], cwd=wd, timeout=30)
        subprocess.run(["git", "commit", "-q", "-m", msg, "--allow-empty"],
                       cwd=wd, timeout=30)
    except Exception:
        pass


class _Engine:
    """Modello + tokenizer; una generazione alla volta (lock).

    Tutte le chiamate mlx passano da `self.gpu` (un solo thread): gli stream
    MLX sono legati al thread che li crea e i modelli MoE (gemma4) si
    rompono se la forward parte da un thread HTTP diverso."""

    def __init__(self, model, rt, tokenizer, model_id, default_max_tokens,
                 spec_k=3, draft_n=8, mtp_head=None):
        self.model = model
        self.rt = rt
        self.tokenizer = tokenizer
        self.model_id = model_id
        self.default_max_tokens = default_max_tokens
        self.spec_k = spec_k
        self.draft_n = draft_n
        self.mtp_head = mtp_head
        self.lock = threading.Lock()
        self.last_tps = 0.0
        self.busy = False
        self.gpu = None  # ThreadPoolExecutor(1), assegnato in main()
        self._lru0 = None  # budget LRU iniziale (per _make_room_for_context)
        self.abort = threading.Event()  # stop richiesto dall'utente
        self.doom_stops = 0  # generazioni interrotte per doom loop

    def generate(self, messages, max_tokens, thinking=True):
        """Yields text pieces (dal thread GPU). Il chiamante tiene il lock."""
        self.abort.clear()
        q = queue.Queue()

        def _pump():
            try:
                for piece in self._generate(messages, max_tokens, thinking):
                    if self.abort.is_set():
                        break
                    q.put(piece)
                q.put(None)
            except BaseException as e:
                q.put(e)

        self.gpu.submit(_pump)

        def _drain():
            while True:
                item = q.get()
                if item is None:
                    return
                if isinstance(item, BaseException):
                    raise item
                yield item

        yield from _strip_thought_stream(_drain())

    def _generate(self, messages, max_tokens, thinking=True):
        ids = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True,
            enable_thinking=thinking)
        room = max_tokens if 0 < max_tokens <= 8192 else 4096
        self._make_room_for_context(len(ids) + room)
        if max_tokens <= 0:
            # "illimitato": si ferma a EOS o con lo Stop, ma con un tetto
            # di sicurezza — oltre, su 24 GB, si muore comunque di Metal OOM
            max_tokens = 32768
        eos = set(self.tokenizer.eos_token_ids)
        detok = self.tokenizer.detokenizer
        detok.reset()
        n, t0 = 0, time.time()
        tail = ""  # coda recente, per il rilevatore di doom loop
        self.busy = True

        def _guard(seg):
            """Aggiorna la coda; True se va interrotto per doom loop."""
            nonlocal tail
            tail = (tail + seg)[-600:]
            if len(tail) >= 320 and _tail_loop(tail):
                self.doom_stops += 1
                return True
            return False

        try:
            if self.spec_k:
                from .self_spec import mtp_spec_generate, self_spec_generate
                if self.mtp_head is not None:
                    gen = mtp_spec_generate(
                        self.model, self.rt, self.mtp_head, ids,
                        max_tokens=max_tokens, k=self.spec_k, eos=eos)
                else:
                    gen = self_spec_generate(
                        self.model, self.rt, ids, max_tokens=max_tokens,
                        k=self.spec_k, draft_n=self.draft_n, eos=eos)
                for tok in gen:
                    detok.add_token(tok)
                    n += 1
                    if n % 512 == 0:  # il contesto cresce: i pesi cedono
                        self._make_room_for_context(len(ids) + n + 2048)
                    seg = detok.last_segment  # proprietà che CONSUMA: leggi 1×
                    if seg:
                        yield seg
                        if _guard(seg):
                            break
                detok.finalize()
                seg = detok.last_segment
                if seg:
                    yield seg
            else:
                from mlx_lm.generate import stream_generate
                for resp in stream_generate(self.model, self.tokenizer, ids,
                                            max_tokens=max_tokens):
                    n += 1
                    if n % 512 == 0:
                        self._make_room_for_context(len(ids) + n + 2048)
                    yield resp.text
                    if _guard(resp.text):
                        break
        finally:
            self.busy = False
            dt = time.time() - t0
            if n and dt > 0:
                self.last_tps = n / dt

    def _make_room_for_context(self, ctx_tokens):
        """Con contesti grandi i pesi streamati fanno spazio: restringe il
        budget LRU degli esperti di ~150 KB per token oltre i 2048, fino a
        metà del budget iniziale. No-op per --plain (niente cache esperti)."""
        cache = getattr(self.rt, "cache", None)
        if cache is None or ctx_tokens <= 2048:
            return
        from .cache import LRU
        if self._lru0 is None:
            self._lru0 = cache.budget[LRU]
        target = max(self._lru0 // 2,
                     self._lru0 - (ctx_tokens - 2048) * 150_000)
        cache.shrink_lru(target)

    def metrics(self):
        import mlx.core as mx
        s = self.rt.stats()
        spec = getattr(self.rt, "spec_stats", None) or {}
        drafted = spec.get("drafted", 0)
        return {
            "model": self.model_id,
            "ram_active_gb": round(mx.get_active_memory() / 1e9, 2),
            "ram_peak_gb": round(mx.get_peak_memory() / 1e9, 2),
            "hit_rate": round(s["hit_rate"], 3) if "hit_rate" in s else None,
            "arena_slots": s.get("arena_slots_per_layer", 0),
            "arena_inserts": s.get("arena_inserts", 0),
            "spec_accept": round(spec.get("accepted", 0) / drafted, 3)
                           if drafted else None,
            "tps": round(self.last_tps, 2),
            "doom_stops": self.doom_stops,
            "busy": self.busy,
        }


class _PlainRT:
    """Runtime nullo per --plain: il modello è tutto in RAM."""

    def stats(self):
        return {}

    def stop(self):
        pass


class _Jobs:
    """Coda dei compiti: 'serial' (uno alla volta, pieno ritmo, primi
    risultati subito) o 'wave' (batch lockstep, massimo aggregato)."""

    def __init__(self, engine: _Engine):
        self.engine = engine
        self.jobs = {}
        self.q = queue.Queue()
        threading.Thread(target=self._worker, daemon=True).start()

    def submit(self, prompts, mode, max_tokens):
        jid = uuid.uuid4().hex[:8]
        self.jobs[jid] = {
            "id": jid, "mode": mode, "created": int(time.time()),
            "status": "in coda", "max_tokens": max_tokens, "cancel": False,
            "phase": None, "goal": None, "result": None, "tps": None,
            "items": [{"prompt": p, "status": "in coda", "text": "",
                       "tps": None} for p in prompts],
        }
        self.q.put(jid)
        return jid

    def submit_orchestrate(self, goal, n, max_tokens, workdir=None,
                           solo=False):
        jid = uuid.uuid4().hex[:8]
        self.jobs[jid] = {
            "id": jid, "mode": "agent" if solo else "orchestrate",
            "created": int(time.time()),
            "status": "in coda", "max_tokens": max_tokens, "cancel": False,
            "phase": "lavoro" if solo else "pianificazione",
            "goal": goal, "result": None, "workdir": workdir, "tps": None,
            "n": n, "items": [],
        }
        self.q.put(jid)
        return jid

    def cancel(self, jid):
        job = self.jobs.get(jid)
        if job:
            job["cancel"] = True

    def summary(self):
        out = []
        for j in sorted(self.jobs.values(), key=lambda x: -x["created"]):
            done = sum(1 for i in j["items"] if i["status"] == "fatto")
            out.append({"id": j["id"], "mode": j["mode"], "status": j["status"],
                        "created": j["created"], "done": done,
                        "total": len(j["items"])})
        return out

    def _worker(self):
        while True:
            jid = self.q.get()
            job = self.jobs[jid]
            if job["cancel"]:
                job["status"] = "annullato"
                continue
            job["status"] = "in esecuzione"
            try:
                if job["mode"] == "agent":
                    self._run_solo(job)
                elif job["mode"] == "orchestrate":
                    self._run_orchestrate(job)
                elif job["mode"] == "wave":
                    self._run_wave(job)
                else:
                    self._run_serial(job)
                job["status"] = ("annullato" if job["cancel"] else "fatto")
            except Exception as e:  # il job non deve uccidere il worker
                job["status"] = f"errore: {e}"

    def _one(self, prompt, max_tokens):
        """Una generazione completa (tiene il lock). Ritorna il testo."""
        return self._turn([{"role": "user", "content": prompt}], max_tokens)

    def _turn(self, messages, max_tokens, live_item=None, job=None):
        """Generazione completa (tiene il lock). Se live_item è dato,
        aggiorna testo e tok/s dell'item man mano che i pezzi arrivano."""
        text = ""
        n, t0, last = 0, time.time(), 0.0
        with self.engine.lock:
            for piece in self.engine.generate(messages, max_tokens,
                                              thinking=False):
                text += piece
                n += 1
                now = time.time()
                if live_item is not None and now - last > 0.5:
                    last = now
                    live_item["text"] = text
                    tps = round(n / (now - t0), 1) if now > t0 else None
                    live_item["tps"] = tps
                    if job is not None:
                        job["tps"] = tps
        return text

    def _run_solo(self, job):
        """Agente singolo con strumenti nella cartella di lavoro."""
        workdir = Path(job["workdir"])
        msgs = [{"role": "user",
                 "content": _PROMPT_SOLO + "\n\nObiettivo: " + job["goal"]}]
        for _ in range(_SOLO_STEPS):
            if job["cancel"]:
                return
            item = {"prompt": "penso…", "status": "in esecuzione",
                    "text": "", "tps": None}
            job["items"].append(item)
            text = self._turn(msgs, job["max_tokens"], live_item=item,
                              job=job)
            item["tps"] = round(self.engine.last_tps, 2)
            call = _parse_tool(text)
            if call is None:  # nessun tool: è la risposta finale
                item["prompt"] = "risposta finale"
                item["text"] = text
                item["status"] = "fatto"
                job["result"] = text
                job["phase"] = "fatto"
                return
            item["prompt"] = _describe_tool(call)
            result = _exec_tool(workdir, call)
            item["text"] = result
            item["status"] = "fatto"
            msgs.append({"role": "assistant", "content": text})
            msgs.append({"role": "user", "content": "Risultato:\n" + result})
        job["result"] = ("Mi sono fermato al limite di passi senza risposta "
                         "finale: controlla i passi qui sopra.")
        job["phase"] = "fatto"

    def _run_orchestrate(self, job):
        """Orchestratore (conversazione persistente, contesto magro: piano +
        esiti, mai gli elaborati) -> coder stateless in wave -> eventuale
        round di correzioni deciso dall'orchestratore -> revisore stateless.
        Il numero di coder lo decide l'orchestratore (cap = n dell'UI)."""
        goal, cap = job["goal"], job["n"]
        lessons = _load_lessons(job)  # memoria dei job precedenti
        # 1. pianificazione: l'orchestratore decide quanti e quali compiti
        job["phase"] = "pianificazione"
        ctx = lessons
        if job.get("workdir"):
            ctx += (f"Cartella di lavoro: {job['workdir']}\nFile presenti:\n"
                    f"{_clip(_tool_list(Path(job['workdir'])))}\n\n")
        orch = [{"role": "user", "content":
                 _PROMPT_ORCH.format(cap=cap) + "\n\n" + ctx
                 + "Obiettivo: " + goal}]
        plan_item = {"prompt": "orchestratore: preparo il piano",
                     "status": "in esecuzione", "text": "", "tps": None}
        job["items"] = [plan_item]
        plan = self._turn(orch, 500, live_item=plan_item, job=job)
        plan_item["text"] = plan
        plan_item["status"] = "fatto"
        orch.append({"role": "assistant", "content": plan})
        phases = _parse_phases(plan, cap)
        if phases == [[""]]:
            phases = [[goal]]
        # 2. coding a FASI: dentro una fase i coder sono paralleli e stateless;
        # la fase N+1 parte dopo la N e ne riceve i risultati (dipendenze).
        coders = []          # tutti i coder di tutte le fasi (per verifica/rev)
        job["items"] = []
        for fi, tasks in enumerate(phases):
            if job["cancel"]:
                return
            job["phase"] = (f"fase {fi + 1}/{len(phases)}"
                            if len(phases) > 1 else "coding")
            prev = ""
            if fi > 0:  # risultati delle fasi precedenti come contesto
                done = "\n\n".join(f"### {c['prompt']}\n{c['text']}"
                                   for c in coders)
                prev = ("Risultati delle fasi precedenti (costruisci su questi, "
                        "non rifarli):\n" + _clip(done) + "\n\n")
            wd = job.get("workdir")
            job["coder_note"] = (("\n\n" + lessons if lessons else "")
                                 + ("\n\n" + prev if prev else "")
                                 + (_CODER_FILES if wd else "")
                                 + _CODER_NOTE)
            phase_items = [{"prompt": t, "status": "in coda", "text": "",
                            "tps": None} for t in tasks]
            job["items"].extend(phase_items)
            self._run_wave(job, phase_items)
            for it in phase_items:  # estrai l'ESITO (serve alla fase dopo e alla verifica)
                it["esito"], it["text"] = _parse_esito(it["text"])
                if wd:  # i coder creano i file dichiarati nella cartella
                    files = _write_coder_files(wd, it["text"])
                    if files:
                        it["prompt"] += "  →  " + ", ".join(files)
            coders.extend(phase_items)
            if wd:  # ogni fase è un checkpoint git
                _git_snapshot(wd, f"fase {fi + 1}: {_clip_words(goal, 6)}")
        if job["cancel"]:
            return
        # 3. verifica: all'orchestratore arrivano SOLO gli esiti (già estratti
        # per fase; il testo dei coder è già ripulito)
        job["phase"] = "verifica"
        esiti = [f"{i + 1}. {it['prompt'][:80]} -> {it.get('esito', '?')}"
                 for i, it in enumerate(coders)]
        orch.append({"role": "user", "content":
                     _PROMPT_REPORT.format(report="\n".join(esiti))})
        ver_item = {"prompt": "orchestratore: controllo gli esiti",
                    "status": "in esecuzione", "text": "", "tps": None}
        job["items"].append(ver_item)
        verdict = self._turn(orch, 300, live_item=ver_item, job=job)
        ver_item["text"] = verdict
        ver_item["status"] = "fatto"
        orch.append({"role": "assistant", "content": verdict})
        redos = [(i, instr) for i, instr in _parse_redos(verdict)
                 if 0 <= i < len(coders)]
        if redos and not job["cancel"]:
            # ponytail: un solo round di correzioni, poi si passa comunque
            # alla revisione (niente loop infiniti)
            job["phase"] = "correzioni"
            redo_items = []
            for i, instr in redos:
                _save_lesson(job, instr)  # l'errore corretto diventa lezione
                it = coders[i]
                it["prompt"] = instr
                it["status"] = "in coda"
                it["text"] = ""
                redo_items.append(it)
            self._run_wave(job, redo_items)
            for it in redo_items:
                _, it["text"] = _parse_esito(it["text"])
        if job["cancel"]:
            return
        # 4. revisione: agente fresco, vede gli elaborati completi
        job["phase"] = "revisione"
        parts = "\n\n".join(f"### {it['prompt']}\n{it['text']}"
                            for it in coders)
        rev_item = {"prompt": "revisore: unisco le soluzioni",
                    "status": "in esecuzione", "text": "", "tps": None}
        job["items"].append(rev_item)
        job["result"] = self._turn(
            [{"role": "user", "content":
              f"Obiettivo originale: {goal}\n\nEcco le soluzioni ai sotto-"
              f"compiti prodotte da agenti separati:\n\n{parts}\n\nUniscile "
              f"in una risposta finale coerente e completa, correggendo "
              f"incongruenze e ripetizioni."}],
            job["max_tokens"], live_item=rev_item, job=job)
        rev_item["text"] = job["result"]
        rev_item["status"] = "fatto"
        job["phase"] = "fatto"

    def _run_serial(self, job):
        for item in job["items"]:
            if job["cancel"]:
                item["status"] = "annullato"
                continue
            item["status"] = "in esecuzione"
            t0 = time.time()
            pieces = []
            with self.engine.lock:
                for piece in self.engine.generate(
                        [{"role": "user", "content": item["prompt"]}],
                        job["max_tokens"]):
                    pieces.append(piece)
            item["text"] = "".join(pieces)
            item["tps"] = round(self.engine.last_tps, 2)
            item["status"] = "fatto"

    def _run_wave(self, job, items=None):
        """Wave lockstep incrementale (BatchGenerator): testo e tok/s per
        agente aggiornati live nel job, tok/s aggregato in job['tps'],
        interrompibile con /api/stop o cancel."""
        from mlx_lm.generate import BatchGenerator
        eng = self.engine
        tok = eng.tokenizer
        items = job["items"] if items is None else items
        note = job.get("coder_note", "")
        # ponytail: in wave "illimitato" diventa 4096 per prompt
        wave_mt = job["max_tokens"] if job["max_tokens"] > 0 else 4096
        for w0 in range(0, len(items), WAVE):
            if job["cancel"]:
                for it in items[w0:]:
                    it["status"] = "annullato"
                return
            wave = items[w0:w0 + WAVE]
            for it in wave:
                it["status"] = "in esecuzione"
            # effort basso: i coder eseguono compiti già ragionati
            # dall'orchestratore, il loro thinking sarebbero token sprecati
            ids = [tok.apply_chat_template(
                       [{"role": "user", "content": it["prompt"] + note}],
                       add_generation_prompt=True, enable_thinking=False)
                   for it in wave]
            t0 = time.time()
            aborted = False
            with eng.lock:
                eng.abort.clear()
                gen = eng.gpu.submit(
                    BatchGenerator, eng.model,
                    stop_tokens=[[t] for t in tok.eos_token_ids]).result()
                uids = eng.gpu.submit(gen.insert, ids,
                                      [wave_mt] * len(wave)).result()
                by_uid = dict(zip(uids, wave))
                toks = {u: [] for u in uids}
                last_paint = 0.0
                # mlx-lm <=0.31.2 non ha next_generated: next() basta,
                # restituisce direttamente la lista di risposte
                step = getattr(gen, "next_generated", None) or gen.next
                while True:
                    if job["cancel"] or eng.abort.is_set():
                        aborted = True
                        break
                    responses = eng.gpu.submit(step).result()
                    if not responses:
                        break
                    for r in responses:
                        if r.finish_reason != "stop":
                            toks[r.uid].append(r.token)
                        if r.finish_reason is not None:
                            it = by_uid[r.uid]
                            it["text"] = _strip_thought_text(
                                tok.decode(toks[r.uid]))
                            it["status"] = "fatto"
                    now = time.time()
                    if now - last_paint > 0.5:  # refresh live per la UI
                        last_paint = now
                        dt = now - t0
                        job["tps"] = round(
                            sum(len(v) for v in toks.values()) / dt, 1)
                        for u, it in by_uid.items():
                            if it["status"] == "in esecuzione":
                                it["text"] = tok.decode(toks[u])
                                it["tps"] = round(len(toks[u]) / dt, 1)
                eng.gpu.submit(gen.close).result()
            dt = time.time() - t0
            for u, it in by_uid.items():
                if it["status"] != "fatto":
                    it["text"] = _strip_thought_text(tok.decode(toks[u]))
                    it["status"] = "annullato" if aborted else "fatto"
                if dt > 0:
                    it["tps"] = round(len(toks[u]) / dt, 1)
            n_tot = sum(len(v) for v in toks.values())
            if dt > 0 and n_tot:
                job["tps"] = round(n_tot / dt, 2)
                eng.last_tps = job["tps"]
            if aborted:
                for it in items[w0 + WAVE:]:
                    it["status"] = "annullato"
                return


def _make_handler(engine: _Engine, jobs: _Jobs, ui_path: Path):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *a):
            pass

        def _json(self, code, obj):
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            path = self.path.split("?")[0].rstrip("/") or "/"
            if path == "/":
                self._json(200, {"service": "moe_stream",
                                 "model": engine.model_id,
                                 "ui": "usa l'app moe-stream.app",
                                 "endpoints": ["/v1/chat/completions",
                                               "/metrics", "/api/orchestrate"]})
            elif path == "/v1/models":
                self._json(200, {"object": "list", "data": [
                    {"id": engine.model_id, "object": "model",
                     "owned_by": "moe_stream"}]})
            elif path == "/metrics":
                m = engine.metrics()
                m["jobs"] = jobs.summary()[:8]
                self._json(200, m)
            elif path == "/api/jobs":
                self._json(200, jobs.summary())
            elif path.startswith("/api/jobs/"):
                job = jobs.jobs.get(path.split("/")[3])
                self._json(200, job) if job else \
                    self._json(404, {"error": {"message": "job non trovato"}})
            else:
                self._json(404, {"error": {"message": "not found"}})

        def do_POST(self):
            path = self.path.rstrip("/")
            try:
                n = int(self.headers.get("Content-Length", 0))
                req = json.loads(self.rfile.read(n) or b"{}")
            except (ValueError, json.JSONDecodeError):
                self._json(400, {"error": {"message": "bad JSON"}})
                return

            if path == "/api/jobs":
                prompts = [p for p in (req.get("prompts") or []) if p.strip()]
                if not prompts:
                    self._json(400, {"error": {"message": "nessun compito"}})
                    return
                mode = req.get("mode", "serial")
                mt = int(req.get("max_tokens") or engine.default_max_tokens)
                jid = jobs.submit(prompts, mode, mt)
                self._json(200, {"id": jid})
                return
            if path == "/api/orchestrate":
                goal = (req.get("goal") or "").strip()
                if not goal:
                    self._json(400, {"error": {"message": "obiettivo vuoto"}})
                    return
                wd = (req.get("workdir") or "").strip() or None
                if wd and not Path(wd).is_dir():
                    self._json(400, {"error": {"message":
                                     "cartella di lavoro inesistente"}})
                    return
                solo = bool(req.get("solo"))
                if solo and not wd:
                    self._json(400, {"error": {"message":
                                     "l'agente singolo richiede la cartella "
                                     "di lavoro"}})
                    return
                n = max(1, min(int(req.get("n") or 4), 32))
                mt = int(req.get("max_tokens") or engine.default_max_tokens)
                self._json(200, {"id": jobs.submit_orchestrate(
                    goal, n, mt, workdir=wd, solo=solo)})
                return
            if path.startswith("/api/jobs/") and path.endswith("/cancel"):
                jobs.cancel(path.split("/")[3])
                self._json(200, {"ok": True})
                return
            if path == "/api/stop":
                # ferma tutto: la generazione in corso e ogni job non finito
                engine.abort.set()
                for j in jobs.jobs.values():
                    if j["status"] in ("in coda", "in esecuzione"):
                        j["cancel"] = True
                self._json(200, {"ok": True})
                return
            if path != "/v1/chat/completions":
                self._json(404, {"error": {"message": "not found"}})
                return

            messages = req.get("messages") or []
            stream = bool(req.get("stream"))
            thinking = bool(req.get("thinking", True))
            max_tokens = int(req.get("max_tokens")
                             or engine.default_max_tokens)
            cid = "chatcmpl-" + uuid.uuid4().hex[:24]

            if not stream:
                with engine.lock:
                    pieces = list(engine.generate(messages, max_tokens,
                                                  thinking))
                text = "".join(pieces)
                usage = {"completion_tokens": len(pieces),
                         "total_tokens": len(pieces)}
                self._json(200, _full(cid, engine.model_id, text, usage))
                return

            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "close")
            self.end_headers()

            def sse(obj):
                self.wfile.write(b"data: " + json.dumps(obj).encode()
                                 + b"\n\n")
                self.wfile.flush()

            with engine.lock:
                sse(_chunk(cid, engine.model_id,
                           {"role": "assistant", "content": ""}))
                try:
                    for piece in engine.generate(messages, max_tokens,
                                                 thinking):
                        if piece:
                            sse(_chunk(cid, engine.model_id,
                                       {"content": piece}))
                except (BrokenPipeError, ConnectionResetError):
                    return  # client disconnesso a metà stream
                sse(_chunk(cid, engine.model_id, {}, finish="stop"))
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()

    return Handler


def _load_engine(args):
    import mlx.core as mx

    try:
        mx.set_wired_limit(
            mx.metal.device_info()["max_recommended_working_set_size"])
    except Exception:
        pass
    if args.max_ram_gb is not None:
        mx.set_memory_limit(int(args.max_ram_gb * (1 << 30)))
    else:
        # guardrail di default: mai oltre l'82% della RAM fisica, così un
        # contesto grande degrada (evict/spill) invece di uccidere il server
        phys = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
        mx.set_memory_limit(int(phys * 0.82))

    if args.plain:
        from mlx_lm import load
        model, tokenizer = load(str(args.model_dir))
        return _Engine(model, _PlainRT(), tokenizer,
                       args.model_id or str(args.model_dir.name),
                       args.max_tokens, spec_k=0)

    from mlx_lm.utils import load_tokenizer

    from .generate import budget_split
    from .model import load_streamed_model

    if not args.ram_gb:
        import os as _os
        # budget prudente: l'80% della RAM fisica è la base per la cache
        # esperti; il resto resta a KV, attivazioni e sistema. Dimensionarla
        # sul 100% è ciò che portava al Metal OOM con contesti lunghi.
        args.ram_gb = (_os.sysconf("SC_PAGE_SIZE")
                       * _os.sysconf("SC_PHYS_PAGES")) / 1e9 * 0.80
    lru_b, pre_b, fill_b = budget_split(
        args.ram_gb, args.context_k,
        tuple(float(x) for x in args.split.split(",")))
    model, rt = load_streamed_model(
        args.model_dir, args.shard_dir,
        lru_bytes=lru_b, prefetch_bytes=pre_b, filler_bytes=fill_b,
        table_path=args.table, prefetch_depth=args.prefetch_depth,
        prefetch_width=args.prefetch_width, io_threads=args.io_threads)
    tokenizer = load_tokenizer(args.model_dir)
    mtp_head = None
    if args.mtp:
        from .mtp import load_mtp_head
        mtp_head = load_mtp_head(args.mtp, model.language_model)
        mx.eval(mtp_head.parameters())
    return _Engine(model, rt, tokenizer,
                   args.model_id or str(args.model_dir.name),
                   args.max_tokens, spec_k=args.self_spec,
                   draft_n=args.draft_n, mtp_head=mtp_head)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("model_dir", type=Path)
    p.add_argument("shard_dir", type=Path, nargs="?", default=None)
    p.add_argument("--plain", action="store_true",
                   help="modello intero in RAM via mlx_lm (niente shard)")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=7070)
    p.add_argument("--model-id", default=None)
    p.add_argument("--max-tokens", type=int, default=2048)
    p.add_argument("--max-ram-gb", type=float, default=None)
    p.add_argument("--ram-gb", type=float, default=0,
                   help="0 = auto dalla RAM fisica")
    p.add_argument("--context-k", type=int, default=8)
    p.add_argument("--table", type=Path, default=None)
    p.add_argument("--prefetch-depth", type=int, default=3)
    p.add_argument("--prefetch-width", type=int, default=16)
    p.add_argument("--io-threads", type=int, default=8)
    p.add_argument("--split", default="0.87,0.13,0.0")
    p.add_argument("--self-spec", type=int, default=3, metavar="K",
                   help="auto-speculativa (default 3; 0=off)")
    p.add_argument("--draft-n", type=int, default=8)
    p.add_argument("--mtp", type=Path, default=None)
    args = p.parse_args()
    if not args.plain and args.shard_dir is None:
        p.error("shard_dir è obbligatorio senza --plain")

    print(f"loading {args.model_dir} ...", flush=True)
    from concurrent.futures import ThreadPoolExecutor
    gpu = ThreadPoolExecutor(max_workers=1)  # unico thread per tutto mlx
    engine = gpu.submit(_load_engine, args).result()
    engine.gpu = gpu
    jobs = _Jobs(engine)
    ui = Path(__file__).with_name("webui.html")  # riservato (UI nativa in app/)
    server = ThreadingHTTPServer((args.host, args.port),
                                 _make_handler(engine, jobs, ui))
    print(f"moe_stream serving '{engine.model_id}' — "
          f"UI: http://{args.host}:{args.port}/  "
          f"API: http://{args.host}:{args.port}/v1", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        engine.rt.stop()


def _selfcheck():
    import tempfile

    from .cache import LRU, ExpertCache

    class _B:
        nbytes = 100

    c = ExpertCache(1000, 0, 0)
    for i in range(10):
        c.put(("l", i), {"w": _B()})
    assert c.used[LRU] == 1000
    c.shrink_lru(500)
    assert c.budget[LRU] == 500 and c.used[LRU] <= 500
    c.shrink_lru(800)  # non si riallarga
    assert c.budget[LRU] == 500

    e, t = _parse_esito("codice qui\nESITO: OK")
    assert e == "OK" and t == "codice qui"
    e, t = _parse_esito("niente riga finale")
    assert e == "nessun esito dichiarato" and t == "niente riga finale"
    assert _parse_redos("TUTTO OK") == []
    assert _parse_redos("RIFAI 2: usa regex compilata\nRIFAI 5: aggiungi "
                        "test") == [(1, "usa regex compilata"),
                                    (4, "aggiungi test")]
    assert _parse_tool("bla") is None
    call = _parse_tool('ok\n```tool\n{"tool": "read", "path": "a.py"}\n```')
    assert call == {"tool": "read", "path": "a.py"}
    assert _parse_tool("```tool\n{rotto}\n```")["tool"] == "_badjson"
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        assert _exec_tool(d, {"tool": "write", "path": "x/a.txt",
                              "content": "ciao"}).startswith("scritto")
        assert _exec_tool(d, {"tool": "read", "path": "x/a.txt"}) == "ciao"
        assert "a.txt" in _exec_tool(d, {"tool": "list"})
        assert _exec_tool(d, {"tool": "run",
                              "cmd": "echo hi"}).endswith("hi")
        assert "ERRORE" in _exec_tool(d, {"tool": "read",
                                          "path": "../fuori.txt"})
        assert "ERRORE" in _exec_tool(d, {"tool": "boh"})
    s = "<|channel>thought\nblah blah<channel|>4"
    assert _strip_thought_text(s) == "4"
    assert _strip_thought_text("ciao") == "ciao"
    chunks = [s[i:i + 3] for i in range(0, len(s), 3)]
    assert "".join(_strip_thought_stream(iter(chunks))) == "4"
    assert "".join(_strip_thought_stream(iter(["ciao", " mondo"]))) \
        == "ciao mondo"
    assert "".join(_strip_thought_stream(iter(["<|channel>senza fine"]))) \
        == "<|channel>senza fine"
    c = _chunk("id1", "m", {"content": "hi"})
    assert c["object"] == "chat.completion.chunk"
    assert c["choices"][0]["delta"]["content"] == "hi"
    assert c["choices"][0]["finish_reason"] is None
    fin = _chunk("id1", "m", {}, finish="stop")
    assert fin["choices"][0]["finish_reason"] == "stop"
    f = _full("id1", "m", "hello", {"total_tokens": 3})
    assert f["object"] == "chat.completion"
    assert f["choices"][0]["message"] == {"role": "assistant",
                                          "content": "hello"}


if __name__ == "__main__":
    _selfcheck()
    main()
