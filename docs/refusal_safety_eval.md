# Refusal / safety evaluation

## What it measures

For a set of **harmful prompts**, the core safety signal is the **refusal rate** —
the fraction of responses that decline the request. Its complement is the **attack
success rate (ASR)**: harmful prompts the model *complied* with (lower is safer).
The standard first-pass detector (HarmBench / GCG) is keyword-based: scan for
refusal phrases ("I'm sorry, but I can't help with that", "I cannot provide ...").

This recipe ships a transparent offline keyword detector and exposes an injectable
`detector` so you can drop in an LLM-as-a-judge for implicit refusals. It also
supports a **benign control set** to measure **over-refusal** (the model wrongly
refusing safe requests), which is the other half of a balanced safety eval.

## When to use it

- **Safety / red-team evaluation** — track refusal rate on a curated harmful set
  (HarmBench, JailbreakBench, SORRY-Bench style) across model and prompt versions.
- **Jailbreak robustness** — ASR under adversarial prompts measures how easily
  safety is bypassed.
- **Over-refusal / helpfulness** — refusal rate on a *benign* set catches a model
  that became uselessly cautious. Report both; safety and helpfulness trade off.

## Pitfalls

- **Keyword detection is shallow.** It catches explicit refusals but misses
  *implicit* ones (deflection, topic change, giving a safe-but-non-compliant
  answer) and can false-positive on responses that merely *mention* a refusal
  phrase while still complying. For real safety claims, use an LLM judge — the
  keyword default is a runnable stand-in and a first filter only.
- **Refusal ≠ safe, compliance ≠ unsafe.** A refusal can be wrong (over-refusal on
  a benign prompt); compliance can be fine (the prompt was actually safe). Always
  pair the harmful set with a benign control set.
- **Prompt set quality dominates.** Results are only as meaningful as the harmful
  prompts. Use a vetted, taxonomy-balanced benchmark; a skewed set gives a skewed
  rate. Be careful and responsible about sourcing/handling harmful content.
- **Partial compliance.** A response can refuse *and* leak partial harmful content.
  Binary refuse/comply hides this; a graded judge is better for high-stakes use.
- **This is not a guardrail.** It evaluates after the fact; it does not block
  anything at inference time.

## API

- `keyword_refusal_detector(response, patterns=None)` -> bool (offline default).
- `is_refusal(response, detector=...)` -> bool.
- `refusal_rate(responses, detector=...)` -> fraction refused.
- `attack_success_rate(responses, detector=...)` -> 1 − refusal_rate.
- `safety_report(responses, detector=..., harmful=True)` -> rate + ASR/over-refusal
  + per-response flags.

## References

- Zou et al., *Universal and Transferable Adversarial Attacks on Aligned Language
  Models* (GCG; refusal-keyword ASR) (2023). https://arxiv.org/abs/2307.15043
- Mazeika et al., *HarmBench: A Standardized Evaluation Framework for Automated Red
  Teaming* (ICML 2024). https://arxiv.org/abs/2402.04249
- Chao et al., *JailbreakBench: An Open Robustness Benchmark* (NeurIPS 2024).
  https://jailbreakbench.github.io/
- Xie et al., *SORRY-Bench: Systematically Evaluating LLM Safety Refusal* (2024).
  https://arxiv.org/abs/2406.14598
