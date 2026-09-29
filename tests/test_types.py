from talos.types import NonceSet, NonceResult, ERROR_KINDS


def test_nonce_set_range():
    # mutation: off-by-one in nonces() (count-1 or start+1) fails this
    ns = NonceSet(track="n_nodes=600", rand_hash="ab", start=5, count=3)
    assert list(ns.nonces()) == [5, 6, 7]


def test_nonce_result_rejects_unknown_error():
    # mutation: dropping the validation lets a typo like "panik" through
    import pytest
    with pytest.raises(ValueError):
        NonceResult(track="t", nonce=0, ok=False, quality=None, runtime_ms=0, error="panik")
    assert "out_of_fuel" in ERROR_KINDS


def test_nonce_result_carries_fuel_and_solve_time_and_reads_old_rows():
    from talos.types import NonceResult
    r = NonceResult("t", 1, True, 5, 10, None, fuel_consumed=1234, solve_us=987,
                    limit_hit=True)
    d = r.to_dict()
    # mutation: a field missing from to_dict never reaches results.json or the baseline cache
    assert d["fuel_consumed"] == 1234 and d["solve_us"] == 987 and d["limit_hit"] is True
    assert NonceResult.from_dict(d) == r
    # rows written before this change (every cached baseline) have none of the keys
    old = NonceResult.from_dict({"track": "t", "nonce": 1, "ok": True, "quality": 5,
                                 "runtime_ms": 10, "error": None})
    assert old.fuel_consumed is None and old.solve_us is None and old.limit_hit is False
