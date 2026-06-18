# Pairwise preference → Elo / Bradley-Terry

## What it measures

When evaluation produces **head-to-head preferences** ("response A beats response
B") rather than absolute scores — the Chatbot Arena / LMArena setting — you convert
those pairwise outcomes into a single rating per model. Two standard methods, both
provided:

- **Elo** — online, sequential updates. After each match the winner gains points
  and the loser loses them, scaled by the surprise of the result
  (`expected_score`). Simple, streaming, but order-dependent and noisier.
- **Bradley-Terry (BT)** — a maximum-likelihood model: each model has a latent
  strength `p_i` and `P(i beats j) = p_i / (p_i + p_j)`. Fit here with the classic
  MM (minorize-maximize) iteration; strengths are rescaled into Elo-like points.
  Order-independent and more stable — which is why LMArena switched from online Elo
  to BT.

A light **regularization** (pseudo-matches vs a phantom average opponent) keeps BT
finite and unique even for *separable* data (a model that never loses), shrinking
ratings toward the center.

## When to use it

- **Model / prompt leaderboards** built from pairwise human or LLM-judge votes.
- **Preference data** (RLHF-style A/B comparisons) summarized into a ranking.
- When absolute rubric scores are hard but "which is better?" is easy and reliable.

## Pitfalls

- **Pairwise judges carry bias.** Position bias (A-vs-B vs B-vs-A), verbosity bias,
  and self-preference inflate ratings. Randomize and swap order; collect both
  orderings; see `llm_judge_rubric` for the bias discussion.
- **Online Elo is order-dependent.** The same matches in a different sequence give
  different ratings, and early matches are under-informed. Prefer BT for a fixed
  batch; reserve Elo for genuinely streaming settings.
- **Separable data has no finite MLE.** A model that never loses would diverge to
  +inf without regularization. The `reg` term handles this; report it.
- **Ratings are relative and anchored.** Only differences are meaningful; the
  absolute number depends on the anchor/scale. Don't compare ratings across
  different match pools.
- **Sparsity and connectivity.** If the comparison graph is disconnected (groups
  that never play each other), cross-group ratings are not identifiable. Ensure
  overlap.
- **Ties.** Handled here as half a win to each side — a common convention; richer
  models treat ties explicitly.

## API

- `elo_ratings(matches, k=32, initial=1000)` -> `{player: rating}` (online).
- `bradley_terry(matches, iterations=1000, reg=1.0, ...)` -> `{player: rating}`.
- `expected_score(rating_a, rating_b)` -> Elo win probability.
- `win_rate(matches)` -> empirical win rate per player.
- `ranking(ratings)` -> players sorted best-first.
- `matches` are `(player_a, player_b, outcome)` with outcome 1.0 / 0.5 / 0.0.

## References

- Chiang et al., *Chatbot Arena: An Open Platform for Evaluating LLMs by Human
  Preference* (2024). https://arxiv.org/abs/2403.04132
- Bradley & Terry, *Rank Analysis of Incomplete Block Designs* (Biometrika 1952).
  https://www.jstor.org/stable/2334029
- Hunter, *MM algorithms for generalized Bradley-Terry models* (Annals of
  Statistics 2004). https://sites.stat.psu.edu/~dhunter/papers/bt.pdf
- LMSYS, *Chatbot Arena Leaderboard / Elo system update*.
  https://lmsys.org/blog/2023-12-07-leaderboard/
