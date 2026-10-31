# Streaming guarantees

`ContextGenerator` is designed for endless streams.

## Bounded memory

- **Buffer**: keeps the samples of the last horizon (plus the last sample before it of categorical sources). It is
  trimmed when a context is created and also when it grows, so it stays bounded when the target stops sending data.
- **Previous contexts** (`con_gen.contexts`): the contexts of the last horizon are kept for the interpretation.
  Consecutive contexts share most of their data, so their series are kept once per source and each stored context
  references a part of them: about two windows of data per source, constant over time.
- **Values**: configuration series start from 0 in each window, so values do not grow with the stream.

State is kept per source name, so memory grows only if new source names (or new categorical values) keep
appearing.

## `contexts`

`con_gen.contexts` is a read-only sequence of `Context` objects (indexing, slicing, `len`, iteration), oldest
first. Contexts are rebuilt when accessed; a returned context which you still reference is returned itself.

```python
last = con_gen.contexts[-1]
recent = con_gen.contexts[-10:]
```

Returned contexts are independent: later data never change a context which was already returned.

## Missing and late data

- NaN values are ignored (as if the sample did not arrive); a NaN target sample creates no context.
- Late data are placed in time order and used by the next contexts.

## Speed

Context generation is incremental where possible (each sample is preprocessed once; stored contexts share their
data). The causality discovery function is called for every target sample and its cost usually dominates.
Options, all off by default:

- `characterize_it=False` skips characterization and interpretation.
- `causality_every=k` runs the causality discovery every k contexts; the contexts in between reuse the last edges
  (only the edges between series present in the context). The CD is always computed for every context.

```python
con_gen = ContextGenerator(target="score", context_horizon="8 hours", causality_every=10)
```

- For historical data, `ContextGeneratorBatch.generate_contexts(timestamps, n_jobs=4)` generates the contexts in
  parallel processes (the causality function must be picklable, e.g. defined at module level, and scripts need an
  `if __name__ == "__main__":` guard on Windows and macOS).

```python
from PdmContext.ContextGeneration import ContextGeneratorBatch

batch = ContextGeneratorBatch(df, target="score", type_of_series={"score": "Univariate", "temp": "Univariate"},
                              context_horizon="8 hours")
contexts = batch.generate_contexts(df.index[100:], n_jobs=4)
```
