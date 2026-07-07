"""Prompt-lookup speculative decoding for the streamed MoE model.

Draft-model-free: drafts tokens by n-gram matching against the tokens already
seen (prompt + output), then verifies the whole draft in ONE forward pass.
Each accepted draft token is produced at ~zero marginal cost, which matters
here because every forward pays the SSD expert-streaming cost.

ponytail: greedy (argmax) verification only — exact-match accept. A sampler
path would need probability-ratio accept/reject; add it if we ever sample.

The model is hybrid (linear_attention/DeltaNet recurrent layers + periodic
full_attention). The recurrent layers are NOT trimmable and mlx_lm's
trim_prompt_cache is a no-op on a mixed cache, so on a partial accept we
snapshot the recurrent state, roll back the whole verify forward, and re-run
one forward over only the confirmed tokens to re-advance both cache kinds.
"""

from collections import defaultdict

import mlx.core as mx


class _Lookup:
    def __init__(self, ngram: int, n_draft: int):
        self.ngram = ngram
        self.n_draft = n_draft
        self.hist: list[int] = []
        self.index: dict[tuple, list[int]] = defaultdict(list)
        self.drafted = 0
        self.accepted = 0

    def add(self, token: int):
        self.hist.append(token)
        pos = len(self.hist) - 1
        for n in range(1, min(self.ngram + 1, pos + 2)):
            start = pos - n + 1
            self.index[tuple(self.hist[start:pos + 1])].append(start)

    def draft(self) -> list[int]:
        if len(self.hist) < self.ngram:
            return []
        query = tuple(self.hist[-self.ngram:])
        cur = len(self.hist) - self.ngram
        best: list[int] = []
        for start in self.index.get(query, []):
            if start == cur:
                continue
            b = start + self.ngram
            cont = self.hist[b:b + self.n_draft]
            if len(cont) > len(best):
                best = cont
        self.drafted += len(best)
        return best


def _trimmable(c):
    return hasattr(c, "is_trimmable") and c.is_trimmable() and hasattr(c, "trim")


def _snapshot(pc):
    snap = {}
    for i, c in enumerate(pc):
        if not _trimmable(c) and hasattr(c, "state"):
            s = c.state
            copied = [mx.array(x) if x is not None else None for x in s]
            snap[i] = tuple(copied) if isinstance(s, tuple) else copied
    return snap


def _rollback(pc, snap, n_trim):
    for i, s in snap.items():
        pc[i].state = s
    for c in pc:
        if _trimmable(c):
            c.trim(n_trim)


def spec_generate(model, tokenizer, prompt_ids, *, max_tokens=256,
                  ngram=3, n_draft=4):
    """Yield (token_id, from_draft) until EOS or max_tokens."""
    from mlx_lm.models import cache

    pc = cache.make_prompt_cache(model)
    lk = _Lookup(ngram, n_draft)
    for t in prompt_ids:
        lk.add(int(t))

    def step(tokens, n_predict=1):
        logits = model(mx.array(tokens, mx.uint32)[None], cache=pc)
        return mx.argmax(logits[0, -n_predict:, :], axis=-1)

    eos = set(tokenizer.eos_token_ids)
    cur = int(step(prompt_ids).item())
    n = 0
    try:
        while n < max_tokens:
            yield cur, False
            n += 1
            if cur in eos or n >= max_tokens:
                return
            lk.add(cur)

            draft = lk.draft()
            if draft:
                snap = _snapshot(pc)
                verified = step([cur] + draft, n_predict=len(draft) + 1)
                mx.eval(verified)
                verified = verified.tolist()
                acc = 0
                for d, v in zip(draft, verified[:-1]):
                    if d != v:
                        break
                    acc += 1
                    lk.add(d)
                    yield d, True
                    n += 1
                    if d in eos or n >= max_tokens:
                        return
                lk.accepted += acc
                if acc < len(draft):
                    # roll the verify forward fully back, then re-advance both
                    # KV and recurrent state over only the confirmed tokens.
                    _rollback(pc, snap, 1 + len(draft))
                    cur = int(step([cur] + draft[:acc]).item())
                else:
                    cur = verified[acc]
            else:
                cur = int(step([cur]).item())
    finally:
        if lk.drafted:
            print(f"\n[spec] {lk.accepted}/{lk.drafted} draft tokens accepted "
                  f"({lk.accepted / lk.drafted:.0%})")


def _demo():
    # Verify the n-gram lookup proposes the right continuation on a repeat.
    lk = _Lookup(ngram=2, n_draft=3)
    for t in [10, 20, 30, 40, 99, 10, 20]:
        lk.add(t)
    assert lk.draft() == [30, 40, 99], lk.draft()
    # No match -> empty draft.
    lk2 = _Lookup(ngram=2, n_draft=3)
    for t in [1, 2, 3]:
        lk2.add(t)
    assert lk2.draft() == [], lk2.draft()
    print("ok")


if __name__ == "__main__":
    _demo()
