# Latency / cost budget

## What it measures

Shipping an LLM feature has hard non-quality constraints: it must be fast enough and
cheap enough. This recipe summarizes a **run log** (per-request records) and asserts
it against **budgets**:

- **Latency percentiles** — p50/p90/p95/p99, max, mean. Tail latency (p95/p99) is
  what users actually feel, so budgets target the tail, not the average.
- **Cost** — per-request cost (supplied directly, or computed from input/output
  tokens and a price table), total, and mean.
- **Budget checks** — pass/fail with the specific violations and the ids of any
  requests that blew the per-request latency budget.

## When to use it

- **CI / pre-release gating** — fail the build if p95 latency or total cost exceeds
  budget, the same way you gate on accuracy regressions.
- **Capacity / cost planning** — project total cost from a representative sample and
  a price table before rollout.
- **SLO monitoring** — track p95/p99 over time; tail latency degrades silently.
- **Model / prompt trade-offs** — quantify the latency and cost a quality gain
  actually costs.

## Pitfalls

- **Average latency lies.** A good mean can hide a terrible p99. Always budget on
  percentiles; this module defaults to reporting the tail.
- **Percentile definition matters.** Different tools compute percentiles
  differently; this uses linear interpolation (NumPy's default "type 7"). With few
  samples percentiles are noisy — collect enough requests.
- **Cost must match your pricing.** Token-based cost assumes your price table is
  current and that you count tokens the same way your provider bills (cached tokens,
  tool tokens, and image tokens often differ). Prefer the provider's reported cost
  when available.
- **Cold starts and warmup skew tails.** First-request latency (model load,
  connection setup) can dominate p99 on small logs; segment or warm up before
  measuring steady state.
- **Concurrency and batching.** Wall-clock latency under load differs from isolated
  timing. Measure under realistic concurrency.
- **Streaming.** For streamed responses, decide whether "latency" means
  time-to-first-token or time-to-completion; they budget very differently.

## API

- `percentile(values, p)` -> interpolated percentile.
- `record_cost(record, price_per_1k=None)` -> cost of one record.
- `summarize(log, price_per_1k=None)` -> latency percentiles + cost aggregates.
- `check_budgets(log, max_latency_p95_ms=..., max_total_cost=..., max_request_latency_ms=..., price_per_1k=..., ...)`
  -> `{passed, summary, violations, over_budget_request_ids}`.
- Records are mappings with optional `latency_ms`, `cost` (or
  `input_tokens`/`output_tokens`), and `id`.

## References

- Dean & Barroso, *The Tail at Scale* (CACM 2013) — why tail latency dominates UX.
  https://research.google/pubs/the-tail-at-scale/
- Google SRE Book, *Service Level Objectives* (percentile SLOs).
  https://sre.google/sre-book/service-level-objectives/
- Hyndman & Fan, *Sample Quantiles in Statistical Packages* (1996) — percentile
  definitions. https://doi.org/10.1080/00031305.1996.10473566
- OpenAI, *Pricing* (token-based cost reference).
  https://openai.com/api/pricing/
