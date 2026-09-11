from talos.nonces import draw_nonce_sets, new_rand_hash, HOLDOUT_START


def test_rand_hash_is_64_hex():
    h = new_rand_hash()
    assert len(h) == 64 and int(h, 16) >= 0
    assert new_rand_hash() != h


def test_sets_are_per_track_and_disjoint():
    # mutation: holdout start == training start makes the sets overlap
    tr, ho = draw_nonce_sets(["a=1", "b=2"], "ff" * 32, training_count=4, holdout_count=3)
    assert [s.track for s in tr] == ["a=1", "b=2"] and [s.track for s in ho] == ["a=1", "b=2"]
    assert list(tr[0].nonces()) == [0, 1, 2, 3]
    assert list(ho[0].nonces()) == [HOLDOUT_START, HOLDOUT_START + 1, HOLDOUT_START + 2]
    assert set(tr[0].nonces()).isdisjoint(ho[0].nonces())
    assert all(s.rand_hash == "ff" * 32 for s in tr + ho)
