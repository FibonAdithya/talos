import pytest

from talos.policy import FixedTraining, FreshTraining
from talos.types import NonceSet

from tests.test_loop import HASH, TR2, spec


def test_fixed_returns_the_jobs_training_sets_every_iteration():
    # mutation: a fixed policy that moves its sets by 4 nonces per iteration starts at 116 for
    # iteration 30
    sp = spec(two_tracks=True)
    assert FixedTraining().training(sp, 1) == TR2
    assert FixedTraining().training(sp, 30) == TR2
    assert FixedTraining.fresh is False


def test_fresh_windows_do_not_overlap_and_start_at_zero():
    # mutation: `count * iteration` (no -1) starts at 8 and never scores nonces 0 to 7
    sp = spec(two_tracks=True)
    policy = FreshTraining(8)
    assert policy.fresh is True
    assert policy.training(sp, 1) == [NonceSet("t", HASH, 0, 8), NonceSet("u", HASH, 0, 8)]
    assert policy.training(sp, 2) == [NonceSet("t", HASH, 8, 8), NonceSet("u", HASH, 8, 8)]
    assert policy.training(sp, 30)[0] == NonceSet("t", HASH, 232, 8)
    seen = [n for i in range(1, 31) for n in policy.training(sp, i)[0].nonces()]
    assert len(seen) == 240 and len(set(seen)) == 240 and max(seen) == 239


def test_fresh_rejects_iteration_zero():
    # mutation: without the check, iteration 0 gets a window at a negative nonce
    with pytest.raises(ValueError):
        FreshTraining(8).training(spec(), 0)
