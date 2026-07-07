"""Auto-speculativa a sottocampionamento di esperti.

Bozza = lo stesso modello con fino a draft_n esperti scelti tra i residenti in
arena, selezionati su GPU (rt.draft_gpu: zero readback, zero IO, zero Python
sul percorso caldo). Verifica = il percorso pieno su k token di bozza in UN
forward, che ammortizza i 40 readback del router su k token.

Cache unica con snapshot/rollback: la bozza avanza la prompt cache con la sua
aritmetica approssimata e viene riavvolta prima della verifica (lo stato
DeltaNet è ricreato funzionalmente a ogni step, quindi lo snapshot è tenere
riferimenti: misurato ~0,02 ms). Su accept parziale si riavvolge e si
ri-avanza sui soli token confermati (il costo r~1 del conto in appendice 7).

Funziona anche con una cache persistente multi-turno (chat): passala via
`cache=`; a fine turno la cache contiene esattamente i token emessi (l'eos
resta non-fed, come nel percorso stream_generate normale).

ponytail: verifica greedy (exact-match), come spec_generate. Un sampler
richiederebbe accept probabilistico; da aggiungere solo se mai campioneremo.
"""

import mlx.core as mx

from .spec_generate import _rollback, _snapshot


def self_spec_generate(model, rt, prompt_ids, *, cache=None, max_tokens=256,
                       k=4, draft_n=4, eos=frozenset()):
    """Genera token id (greedy) con il loop bozza/verifica. Identico token per
    token alla generazione greedy normale. Statistiche su rt.spec_stats."""
    from mlx_lm.models import cache as cache_mod

    pc = cache if cache is not None else cache_mod.make_prompt_cache(model)
    rt.draft_n = draft_n
    stats = rt.spec_stats = {"cycles": 0, "drafted": 0, "accepted": 0,
                             "partial": 0}

    logits = model(mx.array(prompt_ids, mx.uint32)[None], cache=pc)
    cur = int(mx.argmax(logits[0, -1]).item())
    if cur in eos:
        return
    n = 1
    yield cur
    while n < max_tokens:
        stats["cycles"] += 1
        snap = _snapshot(pc)
        fed = 0  # posizioni avanzate in pc dallo snapshot (per il ripristino)
        try:
            # --- bozza: k forward interamente lazy (il token campionato resta
            # su GPU e rientra nell'embedding senza readback); UNA lettura.
            rt.draft_gpu = True
            toks = []
            d = mx.array([[cur]], mx.uint32)
            for _ in range(k):
                dl = model(d, cache=pc)
                fed += 1
                d = mx.argmax(dl[0, -1]).astype(mx.uint32).reshape(1, 1)
                toks.append(d)
            rt.draft_gpu = False
            mx.eval(toks)
            draft = [int(t[0, 0]) for t in toks]
            _rollback(pc, snap, fed)
            fed = 0
            stats["drafted"] += k

            # --- verifica: percorso pieno su [cur]+draft in un forward
            vl = model(mx.array([cur] + draft, mx.uint32)[None], cache=pc)
            fed = k + 1
            verified = [int(t)
                        for t in mx.argmax(vl[0, -(k + 1):], axis=-1).tolist()]

            acc = 0
            for dt, v in zip(draft, verified[:k]):
                if dt != v:
                    break
                acc += 1
            stats["accepted"] += acc

            if acc == k:
                nxt = verified[k]  # stato pc già corretto su tutti i k+1 token
            else:
                stats["partial"] += 1
                # riavvolgi la verifica, ri-avanza sui soli token confermati
                _rollback(pc, snap, fed)
                fed = 0
                mx.eval(model(mx.array([cur] + draft[:acc], mx.uint32)[None],
                              cache=pc))
                fed = acc + 1
                nxt = verified[acc]
        except KeyboardInterrupt:
            # interrotti a metà ciclo: riporta la cache allo stato confermato
            rt.draft_gpu = False
            _rollback(pc, snap, fed)
            raise

        def _truncate(nfeed):
            # multi-turno: la cache non deve contenere posizioni oltre i token
            # realmente emessi (l'eos resta non-fed come nel percorso normale)
            _rollback(pc, snap, fed)
            mx.eval(model(mx.array([[cur] + draft[:nfeed]], mx.uint32),
                          cache=pc))

        for i, t in enumerate(draft[:acc]):
            if t in eos:
                _truncate(i)
                return
            yield t
            n += 1
            if n >= max_tokens:
                _truncate(i + 1)
                return
        cur = nxt
        if cur in eos:
            return
        yield cur
        n += 1
