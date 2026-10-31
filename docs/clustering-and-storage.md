# Clustering, distances and storage

## Distances between contexts

[`PdmContext.utils.distances`](api/distances.md) provides similarities in `[0, 1]` (1: identical), combining a
similarity of the CD series (weight `a`) with the Jaccard similarity of the CR edges (weight `1 - a`, ignoring
direction):

- `distance_cc`: shape based distance (SBD) between the common series.
- `distance_3D_sbd_jaccard`: multivariate SBD over all common series.
- `distance_eu_z`: Euclidean distance after z-normalization.
- `distance_PCA_jaccard`: singular values of PCA (`pip install PdmContext[pca]`).

```python
from PdmContext.utils.distances import distance_cc

similarity = distance_cc(context1, context2, a=0)  # only the edges
```

## Clustering

`DBscanContextStream` clusters contexts as they arrive (inspired by streaming DBSCAN). It keeps all the contexts
it receives.

```python
from PdmContext.ContextClustering import DBscanContextStream

clustering = DBscanContextStream(cluster_similarity_limit=0.7, min_points=2,
                                 distancefunc=lambda c1, c2: distance_cc(c1, c2, a=0))
clustering.add_sample_to_cluster(context)
clustering.plot()
```

## Storage

`SQLiteHandler` and `InfluxDBHandler` (`pip install PdmContext[influxdb]`) store and load contexts:

```python
from PdmContext.utils.dbconnector import SQLiteHandler

database = SQLiteHandler(db_name="ContextDatabase.db")
database.insert_record(date=context.timestamp, target="score", context=context)
contexts = database.get_contex_between_dates(start, end, "score")
```

## Pipelines

[`PdmContext.Pipelines`](api/clustering-pipelines.md) combine a generator with clustering and/or storage, with the
same `collect_data` interface:

```python
from PdmContext.Pipelines import ContextAndClusteringAndDatabase

pipeline = ContextAndClusteringAndDatabase(context_generator_object=con_gen, Clustring_object=clustering,
                                           databaseStore_object=database)
pipeline.collect_data(timestamp=t, source="m", name="score", value=0.3)
```
