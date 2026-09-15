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
