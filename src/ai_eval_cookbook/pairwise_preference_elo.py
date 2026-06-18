"""Turn pairwise preferences into model rankings: online Elo and Bradley-Terry.

When you collect head-to-head judgments ("response A is better than response B")
— from humans or an LLM judge — you can convert them into a single rating per model.
Two standard methods (both used by Chatbot Arena / LMArena):

- **Elo**: sequential updates. After each match the winner gains and the loser
  loses points, scaled by how surprising the result was. Simple and online, but
  order-dependent and noisier.
- **Bradley-Terry (BT)**: a maximum-likelihood model where each model has a latent
  strength and P(A beats B) is logistic in the strength difference. Order-independent
  and more stable; fit here with the classic MM (minorize-maximize) iteration. BT
  strengths are rescaled into Elo-like points for readability.

Pure standard library, offline. You supply the match records.
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Dict, Hashable, List, Optional, Sequence, Tuple

Player = Hashable
# A match: (player_a, player_b, outcome) where outcome is 1.0 (A wins),
# 0.0 (B wins), or 0.5 (tie).
Match = Tuple[Player, Player, float]


def expected_score(rating_a: float, rating_b: float) -> float:
    """Elo expected score for A vs B (probability-like, in (0, 1))."""
    return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400.0))


def elo_ratings(
    matches: Sequence[Match],
    k: float = 32.0,
    initial: float = 1000.0,
) -> Dict[Player, float]:
    """Compute online Elo ratings from a sequence of matches (processed in order).

    ``outcome`` is from A's perspective: 1.0 win, 0.0 loss, 0.5 tie. Unknown players
    start at ``initial``.
    """
    ratings: Dict[Player, float] = defaultdict(lambda: initial)
    for a, b, outcome in matches:
        if outcome not in (0.0, 0.5, 1.0):
            raise ValueError("outcome must be 0.0, 0.5, or 1.0")
        ra, rb = ratings[a], ratings[b]
        ea = expected_score(ra, rb)
        ratings[a] = ra + k * (outcome - ea)
        ratings[b] = rb + k * ((1.0 - outcome) - (1.0 - ea))
    return dict(ratings)


def _win_tie_counts(
    matches: Sequence[Match],
) -> Tuple[List[Player], Dict[Tuple[int, int], float]]:
    """Aggregate wins (ties counted as half a win to each side).

    Returns (players, wins) where wins[(i, j)] is the credited wins of player i
    over player j.
    """
    players: List[Player] = []
    index: Dict[Player, int] = {}

    def idx(p: Player) -> int:
        if p not in index:
            index[p] = len(players)
            players.append(p)
        return index[p]

    wins: Dict[Tuple[int, int], float] = defaultdict(float)
    for a, b, outcome in matches:
        if outcome not in (0.0, 0.5, 1.0):
            raise ValueError("outcome must be 0.0, 0.5, or 1.0")
        ia, ib = idx(a), idx(b)
        wins[(ia, ib)] += outcome
        wins[(ib, ia)] += 1.0 - outcome
    return players, wins


def bradley_terry(
    matches: Sequence[Match],
    iterations: int = 1000,
    tol: float = 1e-9,
    reg: float = 1.0,
    scale: float = 400.0 / math.log(10.0),
    anchor: float = 1000.0,
) -> Dict[Player, float]:
    """Fit Bradley-Terry strengths by MLE (MM algorithm) and return Elo-like ratings.

    Each player gets a positive strength ``p_i``; P(i beats j) = p_i / (p_i + p_j).
    Strengths are updated by the standard MM iteration and converted to additive
    ratings ``scale * ln(p_i)`` shifted so the mean rating equals ``anchor`` (so the
    output is on a familiar ~1000-centered Elo scale).

    ``reg`` adds a small symmetric pseudo-count of matches (one win and one loss)
    against a phantom average opponent of strength 1. This is a standard
    regularization that (a) guarantees a finite, unique solution even for
    *separable* data (a player that never loses, whose unregularized strength would
    diverge to infinity) and (b) shrinks ratings toward the center. Set ``reg=0``
    for the pure (possibly non-convergent) MLE.
    """
    players, wins = _win_tie_counts(matches)
    n = len(players)
    if n == 0:
        return {}

    # total credited wins per player (plus regularization pseudo-wins)
    W = [reg for _ in range(n)]  # one phantom win each
    for (i, j), w in wins.items():
        W[i] += w

    # games between i and j (credited wins in both directions)
    pair_games: Dict[Tuple[int, int], float] = defaultdict(float)
    for (i, j), w in wins.items():
        a, b = (i, j) if i < j else (j, i)
        pair_games[(a, b)] += w

    p = [1.0] * n
    for _ in range(iterations):
        new_p = [0.0] * n
        for i in range(n):
            # phantom opponent of strength 1 contributes 2*reg comparisons
            denom = 2.0 * reg / (p[i] + 1.0)
            for j in range(n):
                if j == i:
                    continue
                a, b = (i, j) if i < j else (j, i)
                nij = pair_games.get((a, b), 0.0)
                if nij > 0:
                    denom += nij / (p[i] + p[j])
            new_p[i] = W[i] / denom if denom > 0 else p[i]
        # normalize so the geometric mean of strengths is 1 (keeps scale bounded)
        log_mean = sum(math.log(x) for x in new_p if x > 0) / n
        norm = math.exp(log_mean)
        new_p = [x / norm if x > 0 else x for x in new_p]
        if max(abs(a - b) for a, b in zip(new_p, p)) < tol:
            p = new_p
            break
        p = new_p

    raw = [scale * math.log(x) if x > 0 else 0.0 for x in p]
    shift = anchor - (sum(raw) / n)
    return {players[i]: raw[i] + shift for i in range(n)}


def win_rate(matches: Sequence[Match]) -> Dict[Player, float]:
    """Empirical win rate per player (ties = 0.5), ignoring opponents' strength."""
    credited: Dict[Player, float] = defaultdict(float)
    played: Dict[Player, float] = defaultdict(float)
    for a, b, outcome in matches:
        credited[a] += outcome
        credited[b] += 1.0 - outcome
        played[a] += 1.0
        played[b] += 1.0
    return {p: credited[p] / played[p] for p in played}


def ranking(ratings: Dict[Player, float]) -> List[Tuple[Player, float]]:
    """Players sorted by rating, highest first."""
    return sorted(ratings.items(), key=lambda kv: kv[1], reverse=True)


if __name__ == "__main__":
    # A clearly beats B and C; B beats C.
    matches = [
        ("A", "B", 1.0), ("A", "B", 1.0), ("A", "C", 1.0),
        ("A", "C", 1.0), ("B", "C", 1.0), ("B", "C", 1.0),
    ]
    print("win rates:", {k: round(v, 2) for k, v in win_rate(matches).items()})
    print("elo:", {k: round(v) for k, v in elo_ratings(matches).items()})
    print("bradley-terry:", {k: round(v) for k, v in bradley_terry(matches).items()})
    print("ranking (BT):", [p for p, _ in ranking(bradley_terry(matches))])
