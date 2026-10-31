"""Tests with the real PC causality discovery (gCastle). Skipped when gCastle cannot be imported
(pip install PdmContext[pc])."""
import numpy as np
import pandas as pd
import pytest
from conftest import minute

pytest.importorskip("castle.algorithms", reason="gCastle (castle.algorithms) cannot be imported")

from PdmContext.ContextGeneration import ContextGenerator, ContextGeneratorBatch  # noqa: E402
from PdmContext.utils.causal_discovery_functions import calculate_with_pc  # noqa: E402


def test_pc_finds_dependent_series_in_both_directions():
    # docs: calculate_with_pc returns the edges in both directions
    rnd = np.random.default_rng(0)
    x = rnd.normal(size=500)
    y = 2 * x + 0.1 * rnd.normal(size=500)
    z = rnd.normal(size=500)
    edges = calculate_with_pc(["x", "y", "z"], np.column_stack([x, y, z]), list(range(500)))
    assert ("x", "y") in edges and ("y", "x") in edges
    assert all("z" not in edge for edge in edges)


def stream(gen, steps, seed=0):
    rnd = np.random.default_rng(seed)
    contexts = []
    level = 0.0
    for m in range(steps):
        temp = rnd.normal()
        if m == steps - 15:
            gen.collect_data(timestamp=minute(m), source="m", name="config", type="configuration")
            level = 5.0
        gen.collect_data(timestamp=minute(m), source="m", name="temp", value=float(temp))
        gen.collect_data(timestamp=minute(m), source="m", name="noise", value=float(rnd.normal()))
        score = 2 * temp + level + 0.1 * rnd.normal()
        contexts.append(gen.collect_data(timestamp=minute(m), source="m", name="score", value=float(score)))
    return contexts


def test_default_generator_with_pc():
    # README: the default causality function is the PC of gCastle
    gen = ContextGenerator(target="score", context_horizon="60 minutes")
    context = stream(gen, 90)[-1]
    edges = context.CR["edges"]
    assert ("temp@m", "score") in edges and ("score", "temp@m") in edges
    assert len(context.CR["characterization"]) == len(edges)
    interpreted = [(i[0], i[1]) for i in context.CR["interpretation"]]
    assert ("temp@m", "score") in interpreted


def test_pc_configuration_change_increases_target():
    gen = ContextGenerator(target="score", context_horizon="60 minutes")
    context = stream(gen, 90)[-1]
    characterization = dict(zip(context.CR["edges"], context.CR["characterization"]))
    assert characterization.get(("config@m", "score")) == "increase"


def test_pc_with_causality_every():
    calls = []

    def counting_pc(names, data, timestamps):
        calls.append(names)
        return calculate_with_pc(names, data, timestamps)

    gen = ContextGenerator(target="score", context_horizon="60 minutes", Causalityfunct=counting_pc,
                           causality_every=10, characterize_it=False)
    contexts = stream(gen, 90)
    assert len(calls) == 9  # 85 contexts with causality discovery (more than 5 samples), every 10th
    assert ("temp@m", "score") in contexts[-1].CR["edges"]


def test_pc_batch_parallel_equals_sequential():
    rnd = np.random.default_rng(1)
    index = [minute(m) for m in range(80)]
    temp = rnd.normal(size=80)
    df = pd.DataFrame({"score": 2 * temp + 0.1 * rnd.normal(size=80), "temp": temp,
                       "noise": rnd.normal(size=80)}, index=index)
    types = {"score": "Univariate", "temp": "Univariate", "noise": "Univariate"}
    sequential = ContextGeneratorBatch(df, "score", types, context_horizon="30 minutes")
    parallel = ContextGeneratorBatch(df, "score", types, context_horizon="30 minutes")
    expected = sequential.generate_contexts(index[40:50])
    result = parallel.generate_contexts(index[40:50], n_jobs=2)
    assert [c.CR for c in result] == [c.CR for c in expected]
    assert ("temp", "score") in result[-1].CR["edges"]


def test_moving_pc_is_deprecated():
    pytest.importorskip("scipy")
    from PdmContext.utils.MovingPC import MovingPC
    with pytest.warns(DeprecationWarning):
        MovingPC()
