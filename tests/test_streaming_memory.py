"""Streaming guarantees of ContextGenerator: bounded memory, contexts history equal to the returned contexts,
NaN values ignored."""
import gc
import math
import pickle
import random

import pandas as pd
from conftest import minute, no_causality, FixedCausality

from PdmContext.ContextGeneration import ContextGenerator
from PdmContext.utils.structure import Context


def feed(gen, steps, seed=0, start=0):
    rnd = random.Random(seed)
    modes = ["run", "idle", "stop"]
    returned = []
    for m in range(start, start + steps):
        t = minute(m)
        gen.collect_data(timestamp=t, source="m", name="temp", value=rnd.random())
        if m % 3 == 0:
            gen.collect_data(timestamp=t, source="m", name="pressure", value=float(m % 11))
        if rnd.random() < 0.1:
            gen.collect_data(timestamp=t, source="m", name="alarm", type="isolated")
        if rnd.random() < 0.1:
            gen.collect_data(timestamp=t, source="m", name="config", type="configuration")
        if rnd.random() < 0.15:
            gen.collect_data(timestamp=t, source="m", name="mode", value=rnd.choice(modes), type="categorical")
        returned.append(gen.collect_data(timestamp=t, source="m", name="score", value=rnd.random() + (m % 40) / 20))
    return returned


def snapshot(context):
    return context.timestamp, {k: (None if v is None else list(v)) for k, v in context.CD.items()}, \
        {k: list(v) for k, v in context.CR.items()}


def test_stored_contexts_memory_is_constant_over_time():
    causality = FixedCausality([("temp@m", "score"), ("score", "temp@m"), ("config@m", "score")])
    gen = ContextGenerator(target="score", context_horizon="60 minutes", Causalityfunct=causality)
    feed(gen, 300)
    window = len(gen.contexts[-1].CD["score"])
    series = len(gen.contexts[-1].CD) + 1  # + timeline
    sizes = []
    for i in range(10):
        feed(gen, 200, seed=i + 1, start=300 + 200 * i)
        sizes.append(gen._context_store.size())
    # about two windows per series (stored contexts of one horizon, each covering one horizon), at most 4 windows
    # because the unused prefix of a store is dropped when it is half of the store (amortized)
    assert max(sizes) <= 4 * window * series
    # constant over time: no growth with the length of the stream
    assert max(sizes[5:]) <= 1.2 * max(sizes[:5])
    assert len(gen.buffer) < 10 * window


def test_contexts_history_equals_returned_contexts():
    causality = FixedCausality([("temp@m", "score"), ("score", "temp@m"), ("config@m", "score"),
                                ("alarm@m", "score")])
    gen = ContextGenerator(target="score", context_horizon="30 minutes", Causalityfunct=causality)
    snapshots = {}
    for context in feed(gen, 400):
        snapshots[context.timestamp] = snapshot(context)
    held = gen.contexts[-1]
    gc.collect()  # returned contexts are no longer referenced, the history rebuilds them
    # contexts of the last 30 minutes, plus the one before (as kept by the generator)
    assert len(gen.contexts) == 32
    for context in gen.contexts:
        assert snapshot(context) == snapshots[context.timestamp]
    # a returned context which is still referenced is returned itself
    assert gen.contexts[-1] is held
    assert isinstance(gen.contexts[0], Context)
    assert [c.timestamp for c in gen.contexts[-3:]] == [c.timestamp for c in list(gen.contexts)[-3:]]


def test_contexts_can_be_assigned():
    gen = ContextGenerator(target="score", context_horizon="30 minutes", Causalityfunct=no_causality)
    contexts = [c for c in feed(gen, 50)][-5:]
    other = ContextGenerator(target="score", context_horizon="30 minutes", Causalityfunct=no_causality)
    other.contexts = contexts
    assert [snapshot(c) for c in other.contexts] == [snapshot(c) for c in contexts]


def test_generator_can_be_pickled_and_continue():
    gen = ContextGenerator(target="score", context_horizon="30 minutes", Causalityfunct=no_causality)
    copy = ContextGenerator(target="score", context_horizon="30 minutes", Causalityfunct=no_causality)
    feed(gen, 100)
    feed(copy, 100)
    copy = pickle.loads(pickle.dumps(copy))
    for a, b in zip(feed(gen, 100, seed=5, start=100), feed(copy, 100, seed=5, start=100)):
        assert snapshot(a) == snapshot(b)
    assert [snapshot(c) for c in gen.contexts] == [snapshot(c) for c in copy.contexts]


def test_nan_values_are_ignored():
    gen = ContextGenerator(target="score", context_horizon="30 minutes", Causalityfunct=no_causality)
    reference = ContextGenerator(target="score", context_horizon="30 minutes", Causalityfunct=no_causality)
    for m in range(20):
        for g in (gen, reference):
            g.collect_data(timestamp=minute(m), source="m", name="temp", value=float(m))
        gen.collect_data(timestamp=minute(m), source="m", name="temp", value=float("nan"))
        gen.collect_data(timestamp=minute(m), source="m", name="mode", value=float("nan"), type="categorical")
        context = gen.collect_data(timestamp=minute(m), source="m", name="score", value=float(m % 3))
        expected = reference.collect_data(timestamp=minute(m), source="m", name="score", value=float(m % 3))
        assert snapshot(context) == snapshot(expected)
    # a NaN target sample is ignored: no context is created
    assert gen.collect_data(timestamp=minute(30), source="m", name="score", value=float("nan")) is None
    assert all(not math.isnan(v) for v in gen.contexts[-1].CD["score"])


def test_buffer_is_bounded_when_target_stops():
    gen = ContextGenerator(target="score", context_horizon="10 minutes", Causalityfunct=no_causality)
    for m in range(5000):
        gen.collect_data(timestamp=pd.Timestamp("2024-01-01") + pd.Timedelta(seconds=m), source="m",
                         name="temp", value=1.0)
    # 10 minutes of data at 1 Hz are 601 events, trimming is amortized
    assert len(gen.buffer) <= 2 * 1024
