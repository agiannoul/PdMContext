"""Context structure, distances, clustering, stream simulation, database and pipelines, as described in their
docstrings."""
import importlib.util

import pandas as pd
import pytest
from conftest import hour, minute, no_causality, FixedCausality

from PdmContext.ContextClustering import DBscanContextStream
from PdmContext.ContextGeneration import ContextGenerator
from PdmContext.Pipelines import ContextAndClustering, ContextAndDatabase, ContextAndClusteringAndDatabase
from PdmContext.utils import distances
from PdmContext.utils.dbconnector import SQLiteHandler
from PdmContext.utils.simulate_stream import simulate_stream, simulate_from_df
from PdmContext.utils.structure import Context


def make_context(timestamp, offset=0.0, edges=None):
    if edges is None:
        edges = [("temp@m", "score"), ("score", "temp@m")]
    return Context.context_from_dict({
        "timestamp": timestamp,
        "score": [float(i % 4) + offset for i in range(10)],
        "temp@m": [float(i % 3) for i in range(10)],
        "edges": edges,
        "characterization": ["unknown"] * len(edges),
    })


# structure

def test_context_from_dict_and_to_dict():
    context = make_context(hour(1))
    assert set(context.CD) == {"score", "temp@m"}
    assert set(context.CR) == {"edges", "characterization", "interpretation"}
    assert context.CR["interpretation"] == []
    back = Context.context_from_dict(context.to_dict())
    assert back.CD.keys() >= context.CD.keys()
    assert back.CR["edges"] == context.CR["edges"]
    assert back.timestamp == context.timestamp


# distances

def test_distance_cc_of_identical_contexts_is_maximum():
    # docstring: a similarity between 0 and 1
    c1, c2 = make_context(hour(1)), make_context(hour(2))
    similarity = distances.distance_cc(c1, c2, 0.5)
    similarity = similarity[0] if isinstance(similarity, tuple) else similarity
    assert similarity == pytest.approx(1.0)


def test_distance_cc_with_only_jaccard_returns_number():
    # used as clustering distance in the Example notebook: distance_cc(c1, c2, a=0)
    c1 = make_context(hour(1))
    c2 = make_context(hour(2), edges=[("temp@m", "score"), ("pressure@m", "score")])
    assert distances.distance_cc(c1, c2, a=0) == pytest.approx(0.5)


@pytest.mark.parametrize("function", [distances.distance_3D_sbd_jaccard, distances.distance_eu_z])
def test_distances_return_similarity_and_parts(function):
    similarity, parts = function(make_context(hour(1)), make_context(hour(2), offset=1.0), 0.5)
    assert 0 <= similarity <= 1 + 1e-9
    assert len(parts) == 2


@pytest.mark.skipif(importlib.util.find_spec("sklearn") is None, reason="scikit-learn not installed")
def test_distance_pca_with_precalculation():
    c1, c2 = make_context(hour(1)), make_context(hour(2))
    names = ["score", "temp@m"]
    first = distances.distance_PCA_jaccard(c1, c2, 0.5, names, precalc=True)
    second = distances.distance_PCA_jaccard(c1, c2, 0.5, names, precalc=True)
    assert first == second
    assert "sing" in c1.details


def test_jaccard_of_edges_ignores_direction():
    c1 = make_context(hour(1), edges=[("a", "b")])
    c2 = make_context(hour(2), edges=[("b", "a")])
    assert distances.jaccard_CR(c1, c2) == 1


def test_nearest_finds_similar_previous_context():
    targets = [make_context(hour(1)), make_context(hour(2), edges=[])]
    query = make_context(hour(3))
    assert distances.nearest(targets, query, 0.9, lambda a, b: distances.distance_3D_sbd_jaccard(a, b, 0.5)) > 0.9


# clustering

def test_dbscan_default_distance_clusters_similar_contexts():
    clustering = DBscanContextStream(cluster_similarity_limit=0.7)
    for h in range(3):
        clustering.add_sample_to_cluster(make_context(hour(h)))
    clustering.add_sample_to_cluster(make_context(hour(4), edges=[]))
    assert sorted(len(cluster) for cluster in clustering.clusters_sets) == [1, 3]


def test_dbscan_with_user_distance():
    clustering = DBscanContextStream(cluster_similarity_limit=0.7,
                                     distancefunc=lambda c1, c2: distances.distance_cc(c1, c2, a=0))
    for h in range(3):
        clustering.add_sample_to_cluster(make_context(hour(h)))
    assert len(clustering.clusters_sets) == 1
    assert clustering.efficient_neighbor(hour(2), [hour(0)])[0]


# stream simulation

def test_simulate_stream_yields_all_records_target_last():
    times = [minute(i) for i in range(3)]
    stream = simulate_stream(TimeSeries=[("score", [1.0, 2.0, 3.0], times), ("temp", [4.0, 5.0, 6.0], times)],
                             Events=[("spike", [minute(1)], "isolated")],
                             Categoricals=[("mode", [minute(0)], ["run"], "categorical")],
                             targetname="score")
    records = list(stream)
    assert len(records) == 3 + 3 + 1 + 1
    assert records[0]["timestamp"] == minute(0)
    # for each timestamp the target comes last
    for i, record in enumerate(records):
        if record["name"] == "score":
            assert all(r["timestamp"] > record["timestamp"] for r in records[i + 1:])
    spike = [r for r in records if r["name"] == "spike"][0]
    assert spike["value"] is None and spike["type"] == "isolated"


def test_simulate_from_df_with_events():
    df = pd.DataFrame({"score": [1.0, 2.0, 3.0], "spike": [0, 1, 0]}, index=[minute(i) for i in range(3)])
    records = list(simulate_from_df(df, [("spike", "isolated")], "score"))
    assert [r["name"] for r in records] == ["score", "spike", "score", "score"]


# database and pipelines

def run_stream(pipeline):
    contexts = []
    for h in range(12):
        pipeline.collect_data(timestamp=hour(h), source="m", name="temp", value=float(h % 5))
        context = pipeline.collect_data(timestamp=hour(h), source="m", name="score", value=float(h % 3))
        contexts.append(context)
    return contexts


def test_sqlite_store_and_load(tmp_path):
    database = SQLiteHandler(db_name=str(tmp_path / "contexts.db"))
    gen = ContextGenerator(target="score", context_horizon="8 hours",
                           Causalityfunct=FixedCausality([("temp@m", "score"), ("score", "temp@m")]))
    contexts = run_stream(ContextAndDatabase(context_generator_object=gen, databaseStore_object=database))
    loaded = database.get_all_context_by_target("score")
    assert len(loaded) == 12
    assert loaded[-1].CD == contexts[-1].CD
    assert loaded[-1].CR["edges"] == contexts[-1].CR["edges"]
    between = database.get_contex_between_dates(hour(2), hour(5), "score")
    assert [c.timestamp for c in between] == [hour(h) for h in range(2, 6)]
    with pytest.raises(AssertionError):
        database.get_records_by_index("target; DROP TABLE my_table", "score")
    database.close_connection()


def test_pipelines_cluster_and_store(tmp_path):
    database = SQLiteHandler(db_name=str(tmp_path / "contexts.db"))
    gen = ContextGenerator(target="score", context_horizon="8 hours", Causalityfunct=no_causality)
    clustering = DBscanContextStream(cluster_similarity_limit=0.7)
    run_stream(ContextAndClusteringAndDatabase(context_generator_object=gen, Clustring_object=clustering,
                                               databaseStore_object=database))
    assert sum(len(cluster) for cluster in clustering.clusters_sets) == 12
    assert len(database.get_all_context_by_target("score")) == 12
    database.close_connection()

    clustering = DBscanContextStream(cluster_similarity_limit=0.7)
    gen = ContextGenerator(target="score", context_horizon="8 hours", Causalityfunct=no_causality)
    run_stream(ContextAndClustering(Clustring_object=clustering, context_generator_object=gen))
    assert sum(len(cluster) for cluster in clustering.clusters_sets) == 12


def test_light_context_generation_is_deprecated():
    from PdmContext.LightContextGeneration import ContextGenerator as LightContextGenerator
    with pytest.warns(DeprecationWarning):
        LightContextGenerator(target="score", Causalityfunct=no_causality)
