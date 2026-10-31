"""Mapping functions, as described in the README ("More details regarding the default mapping_functions")
and in the docstrings of PdmContext.utils.mapping_functions."""
from conftest import minute

from PdmContext.utils.mapping_functions import (map_univariate_to_continuous, map_isolated_to_continuous,
                                                map_configuration_to_continuous, map_categorical_to_continuous)


def target(n):
    return [(0.0, minute(i)) for i in range(n)]


def test_mapper_interface_returns_vectors_and_names_of_target_length():
    # README: map(target_series, occurrences, name) returns the resulting vectors and names
    for mapper, occurrences in [(map_univariate_to_continuous(), [(1.0, minute(1))]),
                                (map_isolated_to_continuous(), [(None, minute(1))]),
                                (map_configuration_to_continuous(), [(None, minute(1))]),
                                (map_categorical_to_continuous(), [("a", minute(1))])]:
        vectors, names = mapper.map(target(4), occurrences, "src")
        assert len(vectors) == len(names) >= 1
        assert all(len(vector) == 4 for vector in vectors)


def test_univariate_higher_sample_rate_is_aggregated_with_mean():
    # README: samples between two timestamps of target series are aggregated (mean)
    occurrences = [(float(i), minute(i)) for i in range(5)]
    vectors, names = map_univariate_to_continuous().map(target(5)[::2], occurrences, "temp")
    assert names == ["temp"]
    assert vectors[0] == [0.0, 1.5, 3.5]


def test_univariate_lower_sample_rate_is_repeated():
    # README: for series with lower sample rates, repetition of their values is used
    occurrences = [(10.0, minute(0)), (20.0, minute(4))]
    vectors, _ = map_univariate_to_continuous().map(target(5), occurrences, "temp")
    assert vectors[0] == [10.0, 10.0, 10.0, 10.0, 20.0]


def test_isolated_event_is_one_at_its_occurrence():
    # docstring: series of 0s, with 1 at the position of the event timestamps
    vectors, names = map_isolated_to_continuous().map(target(5), [(None, minute(2))], "spike")
    assert names == ["spike"]
    assert vectors[0] == [0, 0, 1, 0, 0]


def test_configuration_event_adds_one_after_each_occurrence():
    # docstring: start with 0s, after each occurrence add 1 to all positions after the occurrence
    vectors, names = map_configuration_to_continuous().map(target(6), [(None, minute(2)), (None, minute(4))],
                                                           "config")
    assert names == ["config"]
    assert vectors[0] == [0, 0, 1, 1, 2, 2]


def test_configuration_values_stay_bounded_in_endless_stream():
    # streaming: values must not grow with the number of events over time
    mapper = map_configuration_to_continuous()
    window = 10
    for end in range(window, 500):
        target_series = [(0.0, minute(i)) for i in range(end - window, end)]
        occurrences = [(None, minute(i)) for i in range(end - window, end)]
        vector = mapper.map(target_series, occurrences, "config")[0][0]
        assert len(vector) == window
        assert min(vector) == 0
        assert max(vector) <= window
    # each step still adds 1 (no false drop)
    assert vector == list(range(window))


def test_categorical_one_hot_and_state_variable():
    # README: each category is treated as configuration one, and a state variable is created
    # which signal is one after the last change of category.
    occurrences = [("run", minute(1)), ("idle", minute(3))]
    vectors, names = map_categorical_to_continuous().map(target(6), occurrences, "mode")
    result = dict(zip(names, vectors))
    assert set(names) == {"run_mode", "idle_mode", "state_mode"}
    assert result["run_mode"] == [0, 1, 1, 0, 0, 0]
    assert result["idle_mode"] == [0, 0, 0, 1, 1, 1]
    assert result["state_mode"] == [0, 0, 0, 1, 1, 1]


def test_categorical_send_on_change_does_not_use_future_values():
    # a category must not appear before it was observed (student report 2)
    occurrences = [("run", minute(5)), ("idle", minute(8))]
    vectors, names = map_categorical_to_continuous().map(target(12)[3:], occurrences, "mode")
    result = dict(zip(names, vectors))
    #                            03 04 05 06 07 08 09 10 11
    assert result["run_mode"] == [0, 0, 1, 1, 1, 0, 0, 0, 0]
    assert result["idle_mode"] == [0, 0, 0, 0, 0, 1, 1, 1, 1]
    assert result["state_mode"] == [0, 0, 0, 0, 0, 1, 1, 1, 1]


def test_categorical_value_before_first_target_timestamp_is_active():
    occurrences = [("run", minute(1)), ("idle", minute(8))]
    vectors, names = map_categorical_to_continuous().map(target(12)[3:], occurrences, "mode")
    result = dict(zip(names, vectors))
    assert result["run_mode"] == [1, 1, 1, 1, 1, 0, 0, 0, 0]
    assert result["idle_mode"] == [0, 0, 0, 0, 0, 1, 1, 1, 1]


def test_stateful_mappers_return_copies():
    # a returned series must not change when the mapper is called again (student report 1)
    for mapper, occurrence in [(map_univariate_to_continuous(), 1.0), (map_isolated_to_continuous(), None),
                               (map_configuration_to_continuous(), None)]:
        first = mapper.map(target(3), [(occurrence, minute(1))], "src")[0][0]
        snapshot = list(first)
        mapper.map(target(4), [(occurrence, minute(1)), (occurrence, minute(3))], "src")
        assert first == snapshot
