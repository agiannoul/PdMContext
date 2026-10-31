# Causality discovery

For every context, the causality discovery function receives the series of the context (those which are not
constant) and returns the edges of CR.

## Custom causality functions

Any function with this signature can be used:

```python
def my_causality(names, data, timestamps):
    """
    names:      list with the names of the series (columns of data)
    data:       2D numpy array, one column per series, one row per target sample
    timestamps: the timestamps of the rows
    return: list of (cause, effect) edges, using the names (or None)
    """
    edges = []
    for i, cause in enumerate(names):
        for j, effect in enumerate(names):
            if i != j and abs(np.corrcoef(data[:, i], data[:, j])[0, 1]) > 0.8:
                edges.append((cause, effect))
    return edges

con_gen = ContextGenerator(target="score", context_horizon="8 hours", Causalityfunct=my_causality)
```

Causality discovery runs for every target sample, so its cost dominates the context generation for large windows.
Use `causality_every=k` to run it every k contexts (the contexts in between reuse the last edges).

## Provided functions

- `calculate_with_pc`: the PC algorithm of [gCastle](https://github.com/huawei-noah/trustworthyAI)
  (`pip install PdmContext[pc]`, which also installs torch, required by gCastle). Edges are returned in both
  directions.
- `empty_cause`: no edges (CD only).

## MovingPC (deprecated)

!!! warning
    MovingPC is deprecated and will be removed in a future version: it is experimental and not stable on long
    streams. Use `calculate_with_pc`, optionally with `causality_every=k`.

`PdmContext.utils.MovingPC.MovingPC` updates the statistics of the PC (Fisher-z) test incrementally as samples
enter and leave the window, which is faster for consecutive contexts (`pip install PdmContext[movingpc]`).
It is experimental and not stable. It requires mapping functions whose values for past timestamps do not change.

```python
from PdmContext.utils.MovingPC import MovingPC

MPC = MovingPC()
con_gen = ContextGenerator(target="score", context_horizon="8 hours", Causalityfunct=MPC.calculate_with_pc_moving)
```
