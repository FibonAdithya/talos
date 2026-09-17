import pytest

from talos.prompts import (PromptContext, STRATEGY_TAGS, compile_fix_prompts, distill_prompts,
                           describe_attempt, edit_prompts, edit_repair_prompts,
                           hypothesis_prompts,
                           parse_distillation, parse_hypothesis)


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


def test_parse_hypothesis_survives_braces_in_description():
    # mutation: the non-greedy brace regex cuts the object at the first `}` inside the
    # description
    text = ('Idea:\n{"title": "A", "description": "Use {x: y} sets for lookups", '
            '"strategy_tag": "data_structure"}')
    h = parse_hypothesis(text)
    assert h == {"title": "A", "description": "Use {x: y} sets for lookups",
                 "strategy_tag": "data_structure"}
    # a leading JSON object without title/description must be skipped in favour of the
    # next, valid one
    text2 = ('{"note": 1}\n{"title": "A", "description": "B", '
             '"strategy_tag": "local_search"}')
    assert parse_hypothesis(text2) == {"title": "A", "description": "B",
                                        "strategy_tag": "local_search"}


def test_compile_fix_and_repair_prompts_carry_inputs():
    # mutation: dropping the compiler output, file content, SEARCH format, or rust rules
    # from these prompts leaves the fixer without the context it needs
    system, user = compile_fix_prompts(ctx(), {"mod.rs": "fn broken("},
                                       "error[E0308]: mismatched types")
    assert "error[E0308]: mismatched types" in user
    assert "fn broken(" in user
    assert "<<<<<<< SEARCH" in system
    # distinctive substring from talos/data/rust_rules.md, unlikely to appear elsewhere
    assert "RULE 1 - NO DUPLICATE STRUCTS" in system

    system2, user2 = edit_repair_prompts(ctx(), {"mod.rs": "fn x(){}"},
                                         "- [not_found] in mod.rs: ...")
    assert "- [not_found] in mod.rs: ..." in user2
    assert "fn x(){}" in user2
    assert "<<<<<<< SEARCH" in system2


def test_compile_fix_prompts_truncates_compiler_output():
    # mutation: dropping the `[-6000:]` slice blows the prompt budget on long build logs
    # distinct head/tail content (not a repeated char) so substring checks are discriminating
    long_output = "A" * 4000 + "B" * 6000
    _, user = compile_fix_prompts(ctx(), {"mod.rs": "fn x(){}"}, long_output)
    assert long_output[-6000:] in user
    assert "A" * 4000 not in user


def test_distill_roundtrip():
    system, user = distill_prompts(ctx(), [{"title": "Bigger tabu tenure", "outcome": "failed:score"}])
    assert "Bigger tabu tenure" in user  # mutation: dropping the failure list from the prompt
    assert parse_distillation("LESSON: Prefer cheap moves early.") == "Prefer cheap moves early."
    assert parse_distillation("nothing useful") is None


def test_focused_prompts_name_the_track_and_the_guards():
    # mutation: dropping the track from the system prompt leaves the model optimising every
    # track; dropping the guard list hides that the other tracks are re-scored
    c = ctx(track="n_items=5000,budget=25", guard_tracks=["n_items=1000,budget=5"])
    system, user = hypothesis_prompts(c)
    assert "n_items=5000,budget=25" in system and "n_items=1000,budget=5" in system
    assert "regression guard" in system
    system2, user2 = edit_prompts(c, {"title": "t", "description": "d"})
    assert "n_items=5000,budget=25" in (system2 + user2)


def test_unfocused_prompts_do_not_mention_a_guard():
    # mutation: emitting the focus sentence with an empty track changes every existing prompt
    system, user = hypothesis_prompts(ctx())
    assert "regression guard" not in system and "across every active track" in system


def test_compile_fix_prompt_names_the_files_and_filters_foreign_warnings():
    # The model in iteration 3 of run 20260916-095103 addressed the candidate's files by the
    # path the compiler printed, and in iterations 1 and 5 by another algorithm's directory
    # that dominated the warning spam. mutation: dropping the file-name line or the filter
    # brings both back
    foreign = ("warning: unused variable: `x`\n"
               "  --> tig-algorithms/src/knapsack/knap_quality_opt_v11/track1.rs:1:1\n\n")
    err = ("error[E0308]: mismatched types\n"
           "  --> tig-algorithms/src/knapsack/talos_cand/track1.rs:5:5\n")
    _, user = compile_fix_prompts(ctx(), {"track1.rs": "fn x(){}", "mod.rs": ""}, foreign + err)
    assert "knap_quality_opt_v11" not in user
    assert err in user
    assert "mod.rs, track1.rs" in user


def test_failed_attempt_lines_carry_the_numbers_and_the_error():
    # A title plus "failed:score" told the model nothing about iteration 2's +0.14% on one
    # track, -0.37% on another, and 14x runtime. mutation: dropping any field silences it
    scored = {"title": "Dynamic greedy", "outcome": "failed:score", "mean_rel_delta": -0.00049,
              "worst_track": "n_items=5000,budget=10", "worst_rel_delta": -0.00374,
              "runtime_ratio": 14.07}
    line = describe_attempt(scored)
    assert line.startswith("- Dynamic greedy [failed:score]")
    assert "-0.05%" in line and "n_items=5000,budget=10" in line and "-0.37%" in line
    assert "14.1x" in line
    edit = {"title": "Beam", "outcome": "failed:edit",
            "error": "edit outside the algorithm files rejected: ['x/track2.rs']"}
    assert "rejected: ['x/track2.rs']" in describe_attempt(edit)
    _, user = hypothesis_prompts(ctx(failed_hypotheses=[scored, edit]))
    assert "14.1x" in user and "rejected: ['x/track2.rs']" in user


HP = {"t": {"b": 2, "a": 1}, "u": None}


def test_hypothesis_and_edit_prompts_show_every_tracks_hyperparameters():
    c = ctx(hyperparameters=HP)
    _, hyp_user = hypothesis_prompts(c)
    _, edit_user = edit_prompts(c, {"title": "t", "description": "d"})
    for user in (hyp_user, edit_user):
        # mutation: leaving the block out of either prompt lets the model rename a key unawares
        assert 'track t: {"a":1,"b":2}' in user
        assert "track u: none (solve_challenge receives None)" in user
        assert "do not rename or remove" in user
    import re
    assert not re.search(r"\b[0-9a-f]{64}\b", hyp_user + edit_user)


def test_no_hyperparameters_block_without_a_map():
    _, user = hypothesis_prompts(ctx())
    # mutation: printing the block for None tells the model values are passed when none are
    assert "Hyperparameters:" not in user


def test_a_focused_prompt_shows_only_the_focus_tracks_hyperparameters():
    c = ctx(hyperparameters={"t": {"x": 1}, "u": {"x": 2}}, track="t", guard_tracks=["u"])
    _, user = hypothesis_prompts(c)
    # mutation: listing every track spends the focused prompt on tracks it must not tune for
    assert 'track t: {"x":1}' in user and '{"x":2}' not in user
    assert "guard tracks run with their own values" in user
