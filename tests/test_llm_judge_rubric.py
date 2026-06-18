import pytest

from ai_eval_cookbook.llm_judge_rubric import (
    Criterion,
    grade,
    grade_dataset,
    keyword_mock_judge,
)

RUBRIC = [
    Criterion("accuracy", "Factually correct", 1, 5, weight=2.0),
    Criterion("clarity", "Easy to understand", 1, 5, weight=1.0),
]
JUDGE = keyword_mock_judge(positive=["because", "clearly"], negative=["maybe", "dunno"])


def test_mock_judge_deterministic():
    crit = Criterion("x", "", 1, 5)
    s1 = JUDGE("p", "because clearly", crit)
    s2 = JUDGE("p", "because clearly", crit)
    assert s1 == s2 == 5.0  # mid 3 + 2 positives, clamped to 5


def test_mock_judge_clamps_low():
    crit = Criterion("x", "", 1, 5)
    # mid 3 - 2 negatives -> 1
    assert JUDGE("p", "maybe dunno", crit) == 1.0


def test_grade_good_response():
    good = "The sky is blue because of scattering, clearly explained."
    res = grade("q", good, RUBRIC, JUDGE)
    assert res["per_criterion"]["accuracy"] == 5.0
    assert res["overall"] == pytest.approx(5.0)
    assert res["overall_normalized"] == pytest.approx(1.0)


def test_grade_poor_response():
    poor = "Maybe the sky is blue, dunno really."
    res = grade("q", poor, RUBRIC, JUDGE)
    assert res["overall"] == pytest.approx(1.0)
    assert res["overall_normalized"] == pytest.approx(0.0)


def test_weighted_overall():
    # accuracy=5 (weight2), clarity=3 (weight1) -> (5*2+3*1)/3 = 13/3
    judge = lambda p, r, c: 5.0 if c.name == "accuracy" else 3.0
    res = grade("q", "anything", RUBRIC, judge)
    assert res["overall"] == pytest.approx(13 / 3)


def test_trials_average():
    # judge alternates is not deterministic; use counter via closure
    state = {"n": 0}

    def judge(p, r, c):
        state["n"] += 1
        return 2.0 if state["n"] % 2 else 4.0  # 2,4,2,4...

    crit = [Criterion("x", "", 1, 5)]
    res = grade("q", "r", crit, judge, trials=2)  # mean(2,4)=3
    assert res["per_criterion"]["x"] == pytest.approx(3.0)


def test_score_clamped_to_range():
    judge = lambda p, r, c: 999.0
    res = grade("q", "r", [Criterion("x", "", 1, 5)], judge)
    assert res["per_criterion"]["x"] == 5.0


def test_grade_dataset_means():
    samples = [
        {"prompt": "q1", "response": "because clearly"},  # -> 5,5
        {"prompt": "q2", "response": "maybe dunno"},  # -> 1,1
    ]
    res = grade_dataset(samples, RUBRIC, JUDGE)
    assert res["mean_per_criterion"]["accuracy"] == pytest.approx(3.0)  # mean(5,1)
    assert res["mean_overall_normalized"] == pytest.approx(0.5)  # mean(1.0, 0.0)


def test_empty_dataset():
    res = grade_dataset([], RUBRIC, JUDGE)
    assert res["results"] == []
    assert res["mean_overall_normalized"] == 0.0


def test_errors():
    with pytest.raises(ValueError):
        grade("q", "r", [], JUDGE)
    with pytest.raises(ValueError):
        grade("q", "r", RUBRIC, JUDGE, trials=0)
    with pytest.raises(ValueError):
        grade("q", "r", [Criterion("x", "", 1, 5, weight=0.0)], JUDGE)


def test_single_point_range_normalizes_to_one():
    # min==max -> normalized score defined as 1.0
    res = grade("q", "r", [Criterion("x", "", 3, 3)], lambda p, r, c: 3.0)
    assert res["overall_normalized"] == 1.0
