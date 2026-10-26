from pathlib import Path
import re
from talos.challenges import CHALLENGES, STATUSES, status_text, status_warning

EXPECTED = {"knapsack": "tested", "hypergraph": "tested", "energy_arbitrage": "tested",
            "job_scheduling": "known broken", "satisfiability": "untested",
            "vehicle_routing": "untested", "vector_search": "untested",
            "neuralnet_optimizer": "untested"}

def test_every_challenge_has_the_decided_status():
    # mutation: a challenge added without a status, or a status flipped, fails this
    assert {n: s.status for n, s in CHALLENGES.items()} == EXPECTED
    assert all(s.status in STATUSES for s in CHALLENGES.values())

def test_known_broken_names_its_issue():
    # mutation: dropping issue=36 leaves the user no explanation
    assert CHALLENGES["job_scheduling"].issue == 36
    assert status_text(CHALLENGES["job_scheduling"]) == "known broken (#36)"

def test_status_warning_wording():
    # mutation: changing either warning string, or warning for a tested challenge, fails this
    assert status_warning(CHALLENGES["knapsack"]) is None
    assert status_warning(CHALLENGES["vehicle_routing"]) == \
        "warning: vehicle_routing has never been run end to end with Talos"
    assert status_warning(CHALLENGES["job_scheduling"]) == \
        "warning: job_scheduling is known broken: see issue #36"

def test_readme_status_table_matches_the_code():
    # mutation: editing a status in code or README alone fails this
    text = (Path(__file__).resolve().parent.parent / "README.md").read_text(encoding="utf-8")
    rows = re.findall(r"^\| `([a-z_]+)` \| (CPU|GPU) \| ([^|]+) \|$", text, re.M)
    table = {name: (hw, status.strip()) for name, hw, status in rows}
    assert set(table) == set(CHALLENGES)
    for name, spec in CHALLENGES.items():
        hw, status = table[name]
        assert hw == ("GPU" if spec.is_gpu else "CPU")
        assert status.startswith(spec.status)
        if spec.issue:
            assert f"#{spec.issue}" in status
