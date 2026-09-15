import pytest

from talos import mainnet

BLOCK = {"block": {"id": "b1"}}
CHALLENGES = {"challenges": [
    {"id": "c002", "config": {"name": "vehicle_routing", "type": "cpu",
                              # inserted out of order so that forgetting sorted() is visible
                              "active_tracks": {"n_nodes=700": {}, "n_nodes=600": {}},
                              "max_fuel_budget": 5000000000000}},
    {"id": "c005", "config": {"name": "hypergraph", "type": "gpu",
                              "active_tracks": {"k=8": {}}, "max_fuel_budget": 7}},
]}
ALGOS = {"codes": [
    {"id": "a1", "details": {"challenge_id": "c002", "name": "hgs_v1"}, "block_data": {"adoption": "30"}},
    {"id": "a2", "details": {"challenge_id": "c002", "name": "fast_lane_v6"}, "block_data": {"adoption": "50"}},
    {"id": "a3", "details": {"challenge_id": "c002", "name": "broken"}, "block_data": {"adoption": "90"}},
    {"id": "a4", "details": {"challenge_id": "c005", "name": "other"}, "block_data": {"adoption": "99"}},
], "binarys": [
    {"algorithm_id": "a1", "details": {"compile_success": True}},
    {"algorithm_id": "a2", "details": {"compile_success": True}},
    {"algorithm_id": "a3", "details": {"compile_success": False}},
    {"algorithm_id": "a4", "details": {"compile_success": True}},
]}


def fake_get_json(url: str):
    if url.endswith("/get-block"):
        return BLOCK
    if "/get-challenges?" in url:
        return CHALLENGES
    if "/get-algorithms?" in url:
        return ALGOS
    if "api.github.com" in url and "/contents/" in url:
        return [{"type": "file", "path": "tig-algorithms/src/vehicle_routing/fast_lane_v6/mod.rs",
                 "name": "mod.rs"},
                {"type": "file", "path": "tig-algorithms/src/vehicle_routing/fast_lane_v6/ls.rs",
                 "name": "ls.rs"}]
    raise AssertionError(url)


def fake_get_text(url: str):
    return f"// contents of {url.rsplit('/', 1)[-1]}"


def test_challenge_info_reads_tracks_and_fuel_sorted():
    # mutation: forgetting sorted() makes track order depend on dict order
    info = mainnet.fetch_challenge_info("vehicle_routing", get_json=fake_get_json)
    assert info.id == "c002" and not info.is_gpu
    assert info.tracks == ["n_nodes=600", "n_nodes=700"]
    assert info.max_fuel == 5000000000000


def test_challenge_info_unknown_name_raises():
    # mutation: dropping the trailing raise after the loop returns None for an unknown name
    with pytest.raises(mainnet.MainnetError):
        mainnet.fetch_challenge_info("nope", get_json=fake_get_json)


def test_top_algorithm_skips_uncompiled_and_other_challenges():
    # mutation: dropping the compile_success filter picks "broken" (adoption 90)
    # mutation: dropping the challenge filter picks "other" (adoption 99)
    assert mainnet.top_algorithm("vehicle_routing", get_json=fake_get_json) == ("fast_lane_v6", 50)


def test_top_algorithm_none_when_no_adoption():
    # mutation: treating adoption 0 or a missing algo_name as a valid best returns a tuple, not None
    # (both candidates below are compiled and on the right challenge; only the filter rejects them)
    def gj(url):
        if "/get-algorithms?" in url:
            return {"codes": [
                {"id": "z1", "details": {"challenge_id": "c002", "name": "unadopted"},
                 "block_data": {"adoption": "0"}},
                {"id": "z2", "details": {"challenge_id": "c002"}, "block_data": {"adoption": "40"}},
            ], "binarys": [
                {"algorithm_id": "z1", "details": {"compile_success": True}},
                {"algorithm_id": "z2", "details": {"compile_success": True}},
            ]}
        return fake_get_json(url)
    assert mainnet.top_algorithm("vehicle_routing", get_json=gj) is None


def test_fetch_algorithm_files_relative_paths():
    # mutation: keeping the full repo path as the key breaks staging inside Modal
    files = mainnet.fetch_algorithm_files("vehicle_routing", "fast_lane_v6",
                                          get_text=fake_get_text, get_json=fake_get_json)
    assert set(files) == {"mod.rs", "ls.rs"}
    assert files["mod.rs"].startswith("// contents of mod.rs")


def test_fetch_template_url():
    # mutation: fetching from `main` instead of MONOREPO_REF drifts the template
    seen = []
    def gt(url):
        seen.append(url)
        return "pub fn solve_challenge"
    assert "solve_challenge" in mainnet.fetch_template("knapsack", get_text=gt)
    assert seen == ["https://raw.githubusercontent.com/tig-foundation/tig-monorepo/"
                    "84a5787f5b14a630bdf40f52bccf37887d3d8464/tig-algorithms/src/knapsack/template.rs"]


def test_fetch_algorithm_files_404_falls_back_to_single_file():
    # mutation: catching every MainnetError (not just status==404) would also pass this case,
    # but pins that a 404 specifically is the single-file signal, matched against test_500 below
    def gj(url):
        if "/contents/" in url:
            raise mainnet.MainnetError(f"HTTP 404 fetching {url}", status=404)
        return fake_get_json(url)
    seen = []
    def gt(url):
        seen.append(url)
        return "// solo"
    files = mainnet.fetch_algorithm_files("knapsack", "solo", get_text=gt, get_json=gj)
    assert files == {"mod.rs": "// solo"}
    assert seen == ["https://raw.githubusercontent.com/tig-foundation/tig-monorepo/"
                    "knapsack/solo/tig-algorithms/src/knapsack/solo.rs"]


def test_fetch_algorithm_files_500_reraises():
    # mutation: the broad `except MainnetError` swallows the 500 and silently returns a
    # single-file fallback result instead of aborting
    def gj(url):
        if "/contents/" in url:
            raise mainnet.MainnetError(f"HTTP 500 fetching {url}", status=500)
        return fake_get_json(url)
    with pytest.raises(mainnet.MainnetError):
        mainnet.fetch_algorithm_files("vehicle_routing", "fast_lane_v6",
                                      get_text=fake_get_text, get_json=gj)
