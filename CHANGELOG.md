# Changelog

## 0.3.0

<!-- --8<-- [start:notice] -->
No public function or class was removed or renamed, and no parameter was removed or reordered. Existing code keeps
working; the notes below list what you may notice.

### Check when upgrading

- **Installation**: gCastle is now an optional dependency. If you use the default causality function
  (`calculate_with_pc`), install it with `pip install PdmContext[pc]`, which also installs torch (required by
  gCastle) (`[movingpc]` for the deprecated MovingPC).
- **`ContextGenerator.contexts`** is now a read-only sequence (indexing, slicing, `len`, iteration) instead of a
  `list`, so `append` and item assignment are not available (assigning a whole list still works).
  A context changed in place after it was returned, and no longer referenced, is not kept with those changes.
- **NaN values are ignored** by `collect_data` (as if the sample did not arrive). A NaN target sample returns `None`
  (no context). Previously a NaN in the window made characterization fail.
- **`PdmContext.LightContextGeneration` is deprecated** (DeprecationWarning). Use
  `ContextGenerator(..., characterize_it=False)`.
- **`PdmContext.utils.MovingPC` is deprecated** (DeprecationWarning, experimental and not stable). Use
  `calculate_with_pc`, optionally with `ContextGenerator(..., causality_every=k)`.

### Results that change (bug fixes)

- Returned contexts no longer change afterwards (they used to receive values of later samples).
- Categorical sources: no future values in the series, a category sent before the context horizon is kept as active
  (values may be sent only when they change), no category is active before the first known value.
- The context window contains exactly the samples of the horizon (it had one extra old sample).
- Source names are matched exactly (e.g. `data1` no longer includes `data10`; a source whose name contains the
  target name is no longer dropped).
- Configuration series start from 0 in each window instead of wrapping around (bounded, without false drops).
- Characterization: isolated events use the event rule, and the continuous rule uses the sum of deltas as
  documented.
- Late data are placed in time order; the `replace` policy also works for the first buffered event.
- `simulate_stream` no longer skips the first record.

### Other fixes and additions

- Memory is bounded for endless streams: stored contexts take about two windows of data per source (it was
  quadratic in the window), and the buffer is also trimmed when the target stops sending data.
- Faster context generation with the same results: about 2.5x to 4x (larger windows gain more), with or without
  characterization, not counting the causality discovery itself.
- New, opt-in: `ContextGenerator(..., causality_every=k)` runs the causality discovery every k contexts and reuses
  the last edges in between (default 1: unchanged).
- New, opt-in: `ContextGeneratorBatch.generate_contexts(timestamps, n_jobs=1)` generates many contexts, in
  parallel processes when `n_jobs > 1`.
- NumPy 2 and pandas 3 support, `distances.PCA_pre` caching, default `DBscanContextStream` distance.
- The Pipelines classes accept the `replace` policy; `empty_cause` accepts `timestamps`.
- SQLite field names are validated and InfluxDB query values escaped.
- Test suite, including tests with the real PC (`pip install PdmContext[test,pc]`, `python -m pytest`).
<!-- --8<-- [end:notice] -->
