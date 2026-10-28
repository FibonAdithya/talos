"""Manual: reads GitHub only, bills nothing. Run:
`.venv/bin/python -m pytest -m live tests/test_template_start_live.py`."""
import pytest

from talos import mainnet
from talos.challenges import MONOREPO_REF, TEMPLATE_START_FILES

pytestmark = pytest.mark.live


@pytest.mark.parametrize("challenge", sorted(TEMPLATE_START_FILES))
def test_file_list_matches_the_pinned_tree(challenge):
    base = f"tig-challenges/src/{challenge}"
    paths = mainnet._walk(base, MONOREPO_REF, mainnet._get_json)
    rel = {p[len(base) + 1:] for p in paths}
    # mutation: a missed file (scenarios.rs, nn/linear.rs) or a list stale after a pin bump
    assert set(TEMPLATE_START_FILES[challenge]) == {r for r in rel
                                                    if not r.startswith("baselines/")}


def test_hypergraph_and_neuralnet_docs_are_the_real_files():
    hg = mainnet.fetch_challenge_docs("hypergraph")
    # mutation: dropping or filtering the challenge .cu
    assert "greedy_bipartition" in hg["kernels.cu"]
    nn = mainnet.fetch_challenge_docs("neuralnet_optimizer")
    assert "pub" in nn["nn/linear.rs"]
