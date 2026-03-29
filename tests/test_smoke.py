import numpy as np


def test_combiner_shapes_smoke():
    N, K, M = 50, 3, 4
    probs = np.random.RandomState(0).rand(N, K, M).astype(np.float32)
    probs = probs / probs.sum(axis=-1, keepdims=True)
    target = np.random.RandomState(1).randint(0, M, size=(N,)).astype(np.int64)

    from seg_moe.combiners.ole import OLECombiner

    ole = OLECombiner(mode="sgd_conv1x1", max_iter=10, lr=1e-2, seed=0)
    ole.fit(probs, target, num_classes=M)
    fused = ole.predict(probs)
    assert fused.shape == (N, M)

    from seg_moe.combiners.decision_template import DecisionTemplateCombiner

    dt = DecisionTemplateCombiner()
    dt.fit(probs, target, num_classes=M)
    pred = dt.predict(probs)
    assert pred.shape == (N, M), f"DT predict should return soft scores [N,M], got {pred.shape}"
    pred_hard = dt.predict_hard(probs)
    assert pred_hard.shape == (N,), f"DT predict_hard should return [N], got {pred_hard.shape}"

    from seg_moe.combiners.we_clpso import WECLPSOCombiner

    pso = WECLPSOCombiner(n_particles=5, iters=5, seed=0)
    pso.fit(probs, target, num_classes=M)
    fused2 = pso.predict(probs)
    assert fused2.shape == (N, M)

    from seg_moe.combiners.ole import fit_from_oof, fuse, predict

    W_bvls = fit_from_oof(probs, target, method="bvls", bounds=(0.0, 1.0), seed=0)
    scores = fuse(probs, W_bvls)
    seg = predict(probs, W_bvls)
    assert scores.shape == (N, M)
    assert seg.shape == (N,)

    W_nnls = fit_from_oof(probs, target, method="nnls")
    scores2 = fuse(probs, W_nnls)
    seg2 = predict(probs, W_nnls)
    assert scores2.shape == (N, M)
    assert seg2.shape == (N,)

    # --- Majority Voting ---
    from seg_moe.combiners.majority_voting import MajorityVotingCombiner

    mv = MajorityVotingCombiner()
    mv.fit(probs, target, num_classes=M)
    mv_soft = mv.predict(probs)
    assert mv_soft.shape == (N, M), f"MV predict should return [N,M], got {mv_soft.shape}"
    assert np.allclose(mv_soft.sum(axis=-1), 1.0, atol=1e-5), "MV predict should be normalized"
    mv_hard = mv.predict_hard(probs)
    assert mv_hard.shape == (N,), f"MV predict_hard should return [N], got {mv_hard.shape}"


def test_combiner_iterable_fit_smoke():
    N, K, M = 40, 3, 4
    probs = np.random.RandomState(2).rand(N, K, M).astype(np.float32)
    probs = probs / probs.sum(axis=-1, keepdims=True)
    target = np.random.RandomState(3).randint(0, M, size=(N,)).astype(np.int64)
    chunks = [
        (probs[:15], target[:15]),
        (probs[15:27], target[15:27]),
        (probs[27:], target[27:]),
    ]

    from seg_moe.combiners.ole import OLECombiner, fit_from_oof

    ole = OLECombiner(mode="lsq_bounded", seed=0)
    ole.fit(chunks, None, num_classes=M)
    fused = ole.predict(probs)
    assert fused.shape == (N, M)

    w_stream = fit_from_oof(chunks, None, method="bvls", bounds=(0.0, 1.0), seed=0)
    assert w_stream.shape == (K, M)

    from seg_moe.combiners.decision_template import DecisionTemplateCombiner

    dt = DecisionTemplateCombiner()
    dt.fit(chunks, None, num_classes=M)
    assert dt.predict(probs).shape == (N, M)

    from seg_moe.combiners.we_clpso import WECLPSOCombiner

    pso = WECLPSOCombiner(n_particles=4, iters=3, seed=0)
    pso.fit(chunks, None, num_classes=M)
    assert pso.predict(probs).shape == (N, M)
