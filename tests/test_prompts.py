import pytest

from talos.prompts import (PromptContext, STRATEGY_TAGS, distill_prompts, edit_prompts,
                           hypothesis_prompts, parse_distillation, parse_hypothesis)


def ctx(**kw):
    base = dict(challenge="knapsack", template_rs="pub fn solve_challenge(", direction="Try tabu",
                tacit="- lesson one", files={"mod.rs": "fn x(){}"}, baseline_name="qk_v3",
                best_delta=-0.01, failed_hypotheses=[], forced_tag=None, is_gpu=False)
    base.update(kw)
    return PromptContext(**base)


def test_hypothesis_prompt_carries_direction_tacit_and_target():
    # mutation: dropping direction/tacit/baseline_name from the user prompt, or the
    # strategy tag list / template_rs from the system prompt, leaves the model blind
    system, user = hypothesis_prompts(ctx())
    assert "Try tabu" in user and "lesson one" in user and "qk_v3" in user
    assert "solve_challenge" in system
    assert all(t in system for t in STRATEGY_TAGS)


def test_hypothesis_prompt_never_contains_hash():
    # a rand_hash is 64 hex chars; the context has no field for it, so make sure
    # nothing that looks like one is interpolated from files or tacit
    # mutation: adding a rand_hash field to PromptContext and interpolating it would leak
    # a nonce identifier into the prompt, which the design forbids
    system, user = hypothesis_prompts(ctx())
    import re
    assert not re.search(r"\b[0-9a-f]{64}\b", system + user)


def test_failed_hypotheses_and_forced_tag_appear_when_given():
    # mutation: dropping the recall block loses the "do not repeat" signal
    c = ctx(failed_hypotheses=[{"title": "Bigger tabu tenure", "outcome": "failed:score"}],
            forced_tag="decomposition")
    _, user = hypothesis_prompts(c)
    assert "Bigger tabu tenure" in user and "decomposition" in user


def test_edit_prompt_shows_files_and_format():
    hyp = {"title": "Bitset tabu", "description": "Use a bitset for the tabu list",
           "strategy_tag": "local_search"}
    system, user = edit_prompts(ctx(), hyp)
    # mutation: dropping the description from the user prompt leaves the coder without the idea
    assert "<<<<<<< SEARCH" in system and "mod.rs" in user and "fn x(){}" in user
    assert "Use a bitset for the tabu list" in user


def test_parse_hypothesis_tolerates_prose_and_validates_tag():
    text = 'Sure.\n{"title": "A", "description": "B", "strategy_tag": "local_search"}\nThanks'
    h = parse_hypothesis(text)
    assert h == {"title": "A", "description": "B", "strategy_tag": "local_search"}
    # mutation: accepting an unknown tag breaks strategy_counts bookkeeping
    bad = '{"title": "A", "description": "B", "strategy_tag": "magic"}'
    assert parse_hypothesis(bad)["strategy_tag"] == "hybrid"
    with pytest.raises(ValueError):
        parse_hypothesis("no json here")


def test_distill_roundtrip():
    system, user = distill_prompts(ctx(), [{"title": "Bigger tabu tenure", "outcome": "failed:score"}])
    assert "Bigger tabu tenure" in user  # mutation: dropping the failure list from the prompt
    assert parse_distillation("LESSON: Prefer cheap moves early.") == "Prefer cheap moves early."
    assert parse_distillation("nothing useful") is None
