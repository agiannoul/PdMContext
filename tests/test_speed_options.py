"""Opt-in speed options: causality_every of ContextGenerator and n_jobs of ContextGeneratorBatch.generate_contexts.
The defaults must give the same results as before."""
import random

import numpy as np
import pandas as pd
from conftest import minute, FixedCausality, correlation_causality

from PdmContext.ContextGeneration import ContextGenerator, ContextGeneratorBatch


def feed(gen, steps, seed=0):
    rnd = random.Random(seed)
    contexts = []
    for m in range(steps):
        temp = rnd.random()
        gen.collect_data(timestamp=minute(m), source="m", name="temp", value=temp)
        gen.collect_data(timestamp=minute(m), source="m", name="noise", value=rnd.random())
        if m % 17 == 0:
            gen.collect_data(timestamp=minute(m), source="m", name="config", type="configuration")
        contexts.append(gen.collect_data(timestamp=minute(m), source="m", name="score", value=2 * temp))
    return contexts


def test_causality_every_default_runs_causality_for_every_context():
    causality = FixedCausality([("temp@m", "score"), ("score", "temp@m")])
    gen = ContextGenerator(target="score", context_horizon="30 minutes", Causalityfunct=causality)
    feed(gen, 100)
    # causality runs when the context has at least 2 varying series and more than 5 samples
    assert len(causality.calls) == 100 - 5


def test_causality_every_reuses_last_edges():
    every = ContextGenerator(target="score", context_horizon="30 minutes", Causalityfunct=correlation_causality,
                             causality_every=4, characterize_it=False)
    always = ContextGenerator(target="score", context_horizon="30 minutes", Causalityfunct=correlation_causality,
                              characterize_it=False)
    contexts_every, contexts_always = feed(every, 100), feed(always, 100)
    computed = contexts_always[5:]
    reused = contexts_every[5:]
    for i, (context_every, context_always) in enumerate(zip(reused, computed)):
        # the data are the same, only the edges are reused between causality discoveries
        assert context_every.CD == context_always.CD
        last_computed = computed[i - i % 4].CR["edges"]
        names = {name for name, series in context_every.CD.items()
                 if series is not None and len(set(series)) > 1}
        assert context_every.CR["edges"] == [e for e in last_computed if e[0] in names and e[1] in names]


def test_causality_every_must_be_positive():
    import pytest
    with pytest.raises(AssertionError):
        ContextGenerator(target="score", causality_every=0)


def make_batch(file_path=None):
    rnd = np.random.default_rng(0)
    index = [minute(m) for m in range(120)]
    temp = rnd.random(120)
    df = pd.DataFrame({"score": 2 * temp + 0.1 * rnd.random(120), "temp": temp, "noise": rnd.random(120)},
                      index=index)
    return ContextGeneratorBatch(df, target="score", type_of_series={"score": "Univariate", "temp": "Univariate",
                                                                     "noise": "Univariate"},
                                 context_horizon="30 minutes", Causalityfunct=correlation_causality,
                                 file_path=file_path), index


def test_batch_generate_contexts_sequential():
    batch, index = make_batch()
    contexts = batch.generate_contexts(index[10:20])
    assert [c.timestamp for c in contexts] == index[10:20]
    assert ("temp", "score") in contexts[-1].CR["edges"]
    assert len(contexts[-1].CR["characterization"]) == len(contexts[-1].CR["edges"])


def test_batch_generate_contexts_parallel_equals_sequential(tmp_path):
    sequential, index = make_batch()
    parallel, _ = make_batch(file_path=str(tmp_path / "contexts.pickle"))
    expected = sequential.generate_contexts(index[10:60])
    result = parallel.generate_contexts(index[10:60], n_jobs=2)
    assert len(result) == len(expected)
    for a, b in zip(result, expected):
        assert a.timestamp == b.timestamp
        assert {k: list(v) for k, v in a.CD.items()} == {k: list(v) for k, v in b.CD.items()}
        assert a.CR == b.CR
    assert sorted(parallel.contexts) == index[10:60]
