"""ContextGenerator, as described in the README ("Instantiating a context generator") and the docstrings of
PdmContext.ContextGeneration."""

import pandas as pd
import pytest
from conftest import hour, minute, no_causality, FixedCausality

from PdmContext.ContextGeneration import ContextGenerator
from PdmContext.utils.structure import Context


def feed_hourly(gen, hours, extra=None):
    """Score every hour (and temp), extra(h) may add other data before the score. Returns the contexts."""
    contexts = []
    for h in hours:
        if extra is not None:
            extra(gen, h)
        gen.collect_data(timestamp=hour(h), source="m", name="temp", value=float(h * h % 7))
        contexts.append(gen.collect_data(timestamp=hour(h), source="m", name="score", value=float(h % 3)))
    return contexts


def test_collect_data_returns_context_only_for_target(generator):
    # docstring: return Context when the data name match to the target name or None
    assert generator.collect_data(timestamp=hour(0), source="m", name="temp", value=1.0) is None
    context = generator.collect_data(timestamp=hour(0), source="m", name="score", value=1.0)
    assert isinstance(context, Context)
    assert context.timestamp == hour(0)


def test_target_can_be_given_as_name_at_source():
    gen = ContextGenerator(target="score@m", context_horizon="8 hours", Causalityfunct=no_causality)
    assert gen.collect_data(timestamp=hour(0), source="other", name="score", value=1.0) is None
    assert isinstance(gen.collect_data(timestamp=hour(0), source="m", name="score", value=1.0), Context)


def test_context_horizon_formats():
    # docstring: "8 hours" or a single number (hours); acceptable: hours, days, minutes, seconds
    assert ContextGenerator(target="score", context_horizon="8").horizon_time == "hours"
    assert ContextGenerator(target="score", context_horizon="30 minutes").horizon == 30
    with pytest.raises(AssertionError):
        ContextGenerator(target="score", context_horizon="8 weeks")


def test_event_requires_known_type(generator):
    # docstring: when no value is passed the type must be one of the mapping functions types
    with pytest.raises(AssertionError):
        generator.collect_data(timestamp=hour(0), source="m", name="spike", type="unknown_type")


def test_cd_series_are_mapped_to_target_sample_rate(generator):
    def extra(gen, h):
        # temperature with a higher sample rate
        gen.collect_data(timestamp=hour(h) - pd.Timedelta(minutes=30), source="m", name="pressure", value=float(h))
        if h == 4:
            gen.collect_data(timestamp=hour(h), source="m", name="spike", type="isolated")
    context = feed_hourly(generator, range(10), extra)[-1]
    lengths = {name: len(series) for name, series in context.CD.items() if series is not None}
    assert set(lengths) == {"score", "temp@m", "pressure@m", "spike@m"}
    assert len(set(lengths.values())) == 1


def test_context_contains_data_of_the_horizon_inclusive(generator):
    # 8 hours horizon with hourly data: 03:00 .. 11:00
    context = feed_hourly(generator, range(12))[-1]
    assert len(context.CD["score"]) == 9
    assert context.CD["score"] == [float(h % 3) for h in range(3, 12)]


def test_memory_is_bounded_in_streaming(generator):
    def extra(gen, h):
        gen.collect_data(timestamp=hour(h), source="m", name="mode", value="ab"[h % 2], type="categorical")
        gen.collect_data(timestamp=hour(h), source="m", name="config", type="configuration")
    feed_hourly(generator, range(200), extra)
    assert len(generator.buffer) <= 4 * 9 + 1
    assert len(generator.contexts) <= 10


def test_name_matching_is_exact(generator):
    def extra(gen, h):
        gen.collect_data(timestamp=hour(h), source="m", name="data1", value=1.0)
        gen.collect_data(timestamp=hour(h), source="m", name="data10", value=float(h))
        gen.collect_data(timestamp=hour(h), source="m", name="score_avg", value=float(h))
    context = feed_hourly(generator, range(10), extra)[-1]
    assert context.CD["data1@m"] == [1.0] * 9
    assert context.CD["score_avg@m"] == [float(h) for h in range(1, 10)]
    assert context.CD["score"] == [float(h % 3) for h in range(1, 10)]


def test_returned_contexts_do_not_change_later(generator):
    # student report 1: a context must not receive values of later timestamps
    contexts, snapshots = [], []
    for h in range(36):
        if h in (1, 2, 3, 4, 5, 6, 30):
            generator.collect_data(timestamp=hour(h), source="m", name="alarm", type="isolated")
        generator.collect_data(timestamp=hour(h), source="m", name="temp", value=float(h))
        context = generator.collect_data(timestamp=hour(h), source="m", name="score", value=float(h % 3))
        contexts.append(context)
        snapshots.append({k: (None if v is None else list(v)) for k, v in context.CD.items()})
    for context, snapshot in zip(contexts, snapshots):
        assert {k: (None if v is None else list(v)) for k, v in context.CD.items()} == snapshot


def test_categorical_sent_only_on_change(generator):
    # student report 2: run at 05:00, idle at 08:00, context of 11:00
    def extra(gen, h):
        changes = {5: "run", 8: "idle"}
        if h in changes:
            gen.collect_data(timestamp=hour(h), source="m", name="mode", value=changes[h], type="categorical")
    context = feed_hourly(generator, range(12), extra)[-1]
    #                                   03 04 05 06 07 08 09 10 11
    assert context.CD["run_mode@m"] == [0, 0, 1, 1, 1, 0, 0, 0, 0]
    assert context.CD["idle_mode@m"] == [0, 0, 0, 0, 0, 1, 1, 1, 1]
    assert context.CD["state_mode@m"] == [0, 0, 0, 0, 0, 1, 1, 1, 1]


def test_categorical_value_sent_before_the_horizon_is_kept(generator):
    # README: the last value of each categorical source is kept even when older than the horizon
    def extra(gen, h):
        changes = {1: "run", 8: "idle"}
        if h in changes:
            gen.collect_data(timestamp=hour(h), source="m", name="mode", value=changes[h], type="categorical")
    contexts = feed_hourly(generator, range(30), extra)
    #                                         03 04 05 06 07 08 09 10 11
    assert contexts[11].CD["run_mode@m"] == [1, 1, 1, 1, 1, 0, 0, 0, 0]
    assert contexts[11].CD["idle_mode@m"] == [0, 0, 0, 0, 0, 1, 1, 1, 1]
    # long after the last change the category is still known
    assert contexts[29].CD["idle_mode@m"] is not None


def test_replace_policy(generator):
    # docstring: (e, b) in replace: when b is inserted, it replaces an e event with the same timestamp
    generator.collect_data(timestamp=hour(1), source="m", name="warning", type="isolated")
    generator.collect_data(timestamp=hour(1), source="m", name="alarm", type="isolated",
                           replace=[("warning", "alarm")])
    codes = [e.code for e in generator.buffer]
    assert "alarm" in codes and "warning" not in codes


def test_late_data_are_kept_time_ordered(generator):
    generator.collect_data(timestamp=hour(1), source="m", name="temp", value=1.0)
    generator.collect_data(timestamp=hour(3), source="m", name="temp", value=3.0)
    generator.collect_data(timestamp=hour(0), source="m", name="temp", value=0.0)
    generator.collect_data(timestamp=hour(2), source="m", name="temp", value=2.0)
    assert [e.timestamp for e in generator.buffer] == [hour(0), hour(1), hour(2), hour(3)]


def test_causality_function_receives_names_data_and_timestamps():
    # README: Causalityfunct gets the names and the data (2D array, one column per series)
    causality = FixedCausality([("temp@m", "score"), ("score", "temp@m")])
    gen = ContextGenerator(target="score", context_horizon="8 hours", Causalityfunct=causality)
    context = feed_hourly(gen, range(12))[-1]
    names, data, timestamps = causality.calls[-1]
    assert set(names) == {"score", "temp@m"}
    assert data.shape == (9, 2)
    assert timestamps == [hour(h) for h in range(3, 12)]
    # CR: edges and a parallel list of characterizations
    assert set(context.CR["edges"]) == {("temp@m", "score"), ("score", "temp@m")}
    assert len(context.CR["characterization"]) == len(context.CR["edges"])
    assert set(context.CR["characterization"]) <= {"increase", "decrease", "unknown"}


def test_configuration_edge_is_characterized_as_increase():
    # docstring of characterize_event_edge: median after the occurrence larger than median before + 2 std
    causality = FixedCausality([("config@m", "score"), ("score", "config@m")])
    gen = ContextGenerator(target="score", context_horizon="24 hours", Causalityfunct=causality)
    context = None
    for h in range(20):
        if h == 15:
            gen.collect_data(timestamp=hour(h), source="m", name="config", type="configuration")
        context = gen.collect_data(timestamp=hour(h), source="m", name="score",
                                   value=(1.0 + 0.01 * (h % 2)) if h < 15 else 5.0)
    characterization = dict(zip(context.CR["edges"], context.CR["characterization"]))
    assert characterization[("config@m", "score")] == "increase"


def test_isolated_events_use_event_characterization(generator):
    calls = []
    generator.characterize_event_edge = lambda context, edge: calls.append(("event", edge)) or "unknown"
    generator.characterize_event_continuous = lambda context, edge: calls.append(("continuous", edge)) or "unknown"
    generator.type_of_series = {"spike@m": "isolated", "config@m": "configuration", "temp@m": "Univariate"}
    edges = [("spike@m", "score"), ("config@m", "score"), ("temp@m", "score"), ("score", "temp@m")]
    generator.get_characterization({"edges": edges}, "score")
    assert calls == [("event", edges[0]), ("event", edges[1]), ("continuous", edges[2]), ("continuous", edges[3])]


def test_continuous_characterization_uses_sum_of_deltas(generator):
    # docstring: sum of deltas of b greater (lower) than 2 std -> increase (decrease), otherwise unknown
    def char(values):
        return generator.characterize_event_continuous({"a": [0] * len(values), "b": values}, ("a", "b"))
    assert char([0, 0, 0, 0, 0, 10]) == "increase"
    assert char([10, 0, 0, 0, 0, 0]) == "decrease"
    assert char([0, 5, 5, 5, 5, 0]) == "unknown"
    assert char([1]) == "unknown"


def test_interpretation_of_continuous_cause():
    # docstring of interpret: if seriesA is continuous and (seriesA -> target) is an edge, it is an interpretation
    causality = FixedCausality([("temp@m", "score"), ("score", "temp@m")])
    gen = ContextGenerator(target="score", context_horizon="8 hours", Causalityfunct=causality)
    context = feed_hourly(gen, range(12))[-1]
    interpretations = context.CR["interpretation"]
    assert [(i[0], i[1]) for i in interpretations] == [("temp@m", "score")]
    timestamps = [i[3] for i in interpretations]
    assert timestamps == sorted(timestamps)


def test_characterize_it_false_skips_characterization_and_interpretation():
    causality = FixedCausality([("temp@m", "score"), ("score", "temp@m")])
    gen = ContextGenerator(target="score", context_horizon="8 hours", Causalityfunct=causality, characterize_it=False)
    context = feed_hourly(gen, range(12))[-1]
    assert context.CR["edges"]
    assert context.CR["characterization"] == []
    assert context.CR["interpretation"] == []


def test_custom_mapping_functions():
    # README: user defined mapping functions with map(target_series, occurrences, name)
    class map_constant:
        def map(self, target_series, occurrences, name):
            return [[len(occurrences)] * len(target_series)], [f"count_{name}"]

    gen = ContextGenerator(target="score", context_horizon="8 hours", Causalityfunct=no_causality,
                           mapping_functions={"Univariate": map_constant(), "counter": map_constant()})
    for m in range(5):
        gen.collect_data(timestamp=minute(m), source="m", name="door", type="counter")
        context = gen.collect_data(timestamp=minute(m), source="m", name="score", value=1.0)
    assert context.CD["count_door@m"] == [5] * 5


def test_default_pc_causality_function():
    pytest.importorskip("castle.algorithms", reason="gcastle (or its torch backend) cannot be imported")
    from PdmContext.utils.causal_discovery_functions import calculate_with_pc
    gen = ContextGenerator(target="score", context_horizon="8 hours", Causalityfunct=calculate_with_pc)
    context = None
    for m in range(30):
        temp = float(m % 5)
        gen.collect_data(timestamp=minute(m), source="m", name="temp", value=temp)
        context = gen.collect_data(timestamp=minute(m), source="m", name="score", value=2 * temp + 0.01 * (m % 3))
    assert isinstance(context.CR["edges"], list)
    for edge in context.CR["edges"]:
        assert (edge[1], edge[0]) in context.CR["edges"]


def test_empty_cause_can_be_used_as_causality_function():
    from PdmContext.utils.causal_discovery_functions import empty_cause
    gen = ContextGenerator(target="score", context_horizon="8 hours", Causalityfunct=empty_cause)
    assert feed_hourly(gen, range(12))[-1].CR["edges"] == []
