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
    # mutation: returning the name where the id belongs gives precommit matching nothing to match
    assert mainnet.top_algorithm("vehicle_routing", get_json=fake_get_json) == (
        "fast_lane_v6", "a2", 50)


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


FUEL = 5000000000000


def row(bid, player, algo, track, quality, hp, fuel=FUEL):
    """One benchmark as /get-benchmarks returns it: a precommit and, once submitted, a benchmark.
    `quality=None` models a precommit whose benchmark has not been submitted yet."""
    pre = {"benchmark_id": bid,
           "details": {"fuel_budget": fuel, "hyperparameters": hp, "rand_hash": "00" * 16},
           "settings": {"player_id": player, "algorithm_id": algo, "track_id": track}}
    bench = None if quality is None else {"id": bid,
                                          "details": {"average_quality_by_bundle": quality}}
    return pre, bench


def benchmarks_get_json(rows_by_player, frauds=(), failing=()):
    def gj(url):
        if url.endswith("/get-block"):
            return BLOCK
        if "/get-opow?block_id=b1" in url:
            return {"opow": [{"player_id": p} for p in rows_by_player]}
        if "/get-benchmarks?block_id=b1&player_id=" in url:
            player = url.split("player_id=", 1)[1]
            if player in failing:
                raise mainnet.MainnetError(f"HTTP 500 fetching {url}", status=500)
            rows = rows_by_player[player]
            ids = {pre["benchmark_id"] for pre, _ in rows}
            return {"precommits": [pre for pre, _ in rows],
                    "benchmarks": [b for _, b in rows if b is not None],
                    "proofs": [],
                    "frauds": [{"benchmark_id": f} for f in frauds if f in ids]}
        raise AssertionError(url)
    return gj


def test_top_hyperparameters_takes_the_best_benchmark_of_this_algorithm_across_players():
    gj = benchmarks_get_json({
        "0xp1": [row("b1", "0xp1", "a2", "T1", [100, 100], {"x": 1}),
                 row("b2", "0xp1", "a9", "T1", [900, 900], {"x": 9})],
        "0xp2": [row("b3", "0xp2", "a2", "T1", [150, 150], {"x": 2})],
    })
    got = mainnet.top_hyperparameters("a2", ["T1"], FUEL, get_json=gj)
    # mutation: dropping the algorithm_id filter picks b2 ({"x": 9}), another algorithm's keys
    # mutation: reading only the first player's benchmarks picks b1 ({"x": 1})
    assert got == {"T1": mainnet.TrackHyperparameters({"x": 2}, "b3", "0xp2", 150.0)}
    assert got["T1"].source() == {"benchmark_id": "b3", "player_id": "0xp2", "mean_quality": 150.0}


def test_top_hyperparameters_ignores_other_fuel_frauds_and_unsubmitted_benchmarks():
    gj = benchmarks_get_json({"0xp1": [
        row("ok", "0xp1", "a2", "T1", [100, 100], {"x": 1}),
        row("fuel", "0xp1", "a2", "T1", [999, 999], {"x": 2}, fuel=20000000000),
        row("fraud", "0xp1", "a2", "T1", [999, 999], {"x": 3}),
        row("pending", "0xp1", "a2", "T1", None, {"x": 4}),
        row("empty", "0xp1", "a2", "T1", [], {"x": 5}),
    ]}, frauds=("fraud",))
    got = mainnet.top_hyperparameters("a2", ["T1"], FUEL, get_json=gj)
    # mutation: dropping the fuel filter picks "fuel", values tuned for a different budget
    # mutation: dropping the fraud filter picks "fraud"
    # mutation: treating a missing or empty quality list as eligible raises or picks it
    assert got["T1"].benchmark_id == "ok" and got["T1"].hyperparameters == {"x": 1}


def test_top_hyperparameters_ranks_by_mean_over_bundles():
    gj = benchmarks_get_json({"0xp1": [
        row("first_high", "0xp1", "a2", "T1", [100, 10], {"x": 1}),
        row("mean_high", "0xp1", "a2", "T1", [60, 60], {"x": 2}),
    ]})
    got = mainnet.top_hyperparameters("a2", ["T1"], FUEL, get_json=gj)
    # mutation: ranking by the first bundle or by max() picks first_high (mean 55 < 60)
    assert got["T1"].benchmark_id == "mean_high" and got["T1"].mean_quality == 60.0


@pytest.mark.parametrize("order", [("bb", "ba"), ("ba", "bb")])
def test_top_hyperparameters_breaks_a_tie_on_the_lower_benchmark_id(order):
    gj = benchmarks_get_json({"0xp1": [row(b, "0xp1", "a2", "T1", [70, 70], {"id": b})
                                       for b in order]})
    # mutation: keeping the first (or last) seen makes the choice depend on API order
    assert mainnet.top_hyperparameters("a2", ["T1"], FUEL, get_json=gj)["T1"].benchmark_id == "ba"


def test_top_hyperparameters_covers_every_track_and_keeps_empty_apart_from_null():
    gj = benchmarks_get_json({"0xp1": [
        row("e", "0xp1", "a2", "T_empty", [10, 10], {}),
        row("n", "0xp1", "a2", "T_null", [10, 10], None),
        row("other", "0xp1", "a2", "T_not_asked", [10, 10], {"x": 1}),
    ]})
    got = mainnet.top_hyperparameters("a2", ["T_empty", "T_null", "T_missing"], FUEL, get_json=gj)
    # mutation: returning only tracks that had a benchmark raises KeyError downstream
    assert set(got) == {"T_empty", "T_null", "T_missing"}
    assert got["T_missing"] == mainnet.TrackHyperparameters(None, None, None, None)
    # mutation: `hp or None` collapses {} (run with an empty map) into None (run without the flag)
    assert got["T_empty"].hyperparameters == {} and got["T_empty"].hyperparameters is not None
    assert got["T_null"].hyperparameters is None and got["T_null"].benchmark_id == "n"


def test_top_hyperparameters_raises_when_one_player_cannot_be_read():
    gj = benchmarks_get_json({"0xp1": [row("b1", "0xp1", "a2", "T1", [1, 1], {"x": 1})],
                              "0xp2": []}, failing=("0xp2",))
    # mutation: skipping a failed player chooses from a partial view without saying so
    with pytest.raises(mainnet.MainnetError):
        mainnet.top_hyperparameters("a2", ["T1"], FUEL, get_json=gj)


def test_top_hyperparameters_raises_mainneterror_on_a_malformed_precommit():
    def gj(url):
        if url.endswith("/get-block"):
            return BLOCK
        if "/get-opow?block_id=b1" in url:
            return {"opow": [{"player_id": "0xp1"}]}
        if "/get-benchmarks?block_id=b1&player_id=" in url:
            # missing "benchmark_id": a shape mainnet has never sent, but not impossible
            return {"precommits": [{"settings": {"algorithm_id": "a2", "track_id": "T1"},
                                    "details": {"fuel_budget": FUEL}}],
                    "benchmarks": [], "proofs": [], "frauds": []}
        raise AssertionError(url)
    # mutation: leaving the KeyError unconverted sends a bare traceback out of `talos run`
    with pytest.raises(mainnet.MainnetError):
        mainnet.top_hyperparameters("a2", ["T1"], FUEL, get_json=gj)
