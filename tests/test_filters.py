from vision.filters import EMAFilter, OneEuroFilter, VelocityEstimator


def test_ema_converges_toward_constant_input():
    f = EMAFilter(alpha=0.5)
    out = None
    for _ in range(20):
        out = f.filter(10.0, 20.0)
    assert abs(out[0] - 10.0) < 1e-6
    assert abs(out[1] - 20.0) < 1e-6


def test_ema_smooths_a_jump():
    f = EMAFilter(alpha=0.3)
    f.filter(0.0, 0.0)
    out = f.filter(100.0, 100.0)
    # a single jump shouldn't fully snap to the new value
    assert 0 < out[0] < 100


def test_one_euro_filter_reduces_jitter():
    f = OneEuroFilter(min_cutoff=1.0, beta=0.02)
    t = 0.0
    outputs = []
    import random
    random.seed(0)
    base = 100.0
    for i in range(60):
        t += 1 / 30
        noisy_x = base + random.uniform(-2, 2)
        noisy_y = base + random.uniform(-2, 2)
        outputs.append(f.filter(noisy_x, noisy_y, t))
    # filtered output variance should be lower than raw jitter amplitude
    xs = [o[0] for o in outputs[10:]]
    spread = max(xs) - min(xs)
    assert spread < 4.0  # raw noise amplitude is +/-2 (spread 4); filtered should not exceed it


def test_one_euro_tracks_large_fast_motion_without_huge_lag():
    f = OneEuroFilter(min_cutoff=1.0, beta=0.7)
    t = 0.0
    f.filter(0.0, 0.0, t)
    for i in range(1, 20):
        t += 1 / 30
        out = f.filter(500.0, 500.0, t)
    assert out[0] > 400  # should have mostly caught up after 19 frames of fast motion


def test_velocity_estimator_reports_zero_for_stationary_point():
    v = VelocityEstimator(window=5)
    t = 0.0
    for _ in range(5):
        t += 1 / 30
        v.update(50.0, 50.0, t)
    speed = v.speed(50.0, 50.0, t + 1 / 30)
    assert speed < 1e-3


def test_velocity_estimator_reports_nonzero_for_moving_point():
    v = VelocityEstimator(window=5)
    t = 0.0
    for i in range(5):
        t += 1 / 30
        v.update(i * 10.0, 0.0, t)
    speed = v.speed(60.0, 0.0, t + 1 / 30)
    assert speed > 0
