# Generating context

## Creating a generator

```python
from PdmContext.ContextGeneration import ContextGenerator

con_gen = ContextGenerator(
    target="anomaly scores",       # name (or name@source) of the target series
    context_horizon="8 hours",     # time window of each context
    Causalityfunct=my_causality,   # causality discovery function (default: gCastle PC)
    mapping_functions=None,        # None: default mappers (Univariate, isolated, configuration, categorical)
    characterize_it=True,          # False: no characterization and interpretation (faster)
    causality_every=1,             # k > 1: causality discovery every k contexts (faster, see Streaming guarantees)
)
```

## Feeding data

```python
con_gen.collect_data(timestamp=t, source="Machine_A", name="temperature", value=40.5)            # Univariate
con_gen.collect_data(timestamp=t, source="Machine_A", name="spike", type="isolated")             # event
con_gen.collect_data(timestamp=t, source="Machine_A", name="speed change", type="configuration") # event
con_gen.collect_data(timestamp=t, source="Machine_A", name="mode", value="run", type="categorical")
context = con_gen.collect_data(timestamp=t, source="Machine_A", name="anomaly scores", value=0.3)
```

- `collect_data` returns a `Context` when the sample belongs to the target, otherwise `None`.
- Events (`value=None`) must have a `type` which is a key of the mapping functions.
- Sources may have any sample rate; they are aligned to the target when a context is created.
- Data may arrive late: they are placed in time order.
- NaN values are ignored, as if the sample did not arrive (a NaN target sample creates no context).

### Replacement policy

`replace` is a list of `(e, b)` pairs: when an event `b` arrives, it replaces an event `e` with the same timestamp.

```python
con_gen.collect_data(timestamp=t, source="m", name="alarm", type="isolated", replace=[("warning", "alarm")])
```

## Previous contexts

`con_gen.contexts` holds the contexts of the last horizon (oldest first) as a read-only sequence, used for the
interpretation. See [Streaming guarantees](streaming.md).

## Simulating a stream

For experiments on historical data, [`simulate_stream`](api/simulation-plots.md) and `simulate_from_df` produce the
records in time order (for each timestamp, the target last):

```python
from PdmContext.utils.simulate_stream import simulate_from_df

stream = simulate_from_df(df, eventTypes=[("spike", "isolated"), ("config", "configuration")], target_name="score")
for record in stream:
    con_gen.collect_data(timestamp=record["timestamp"], source="press", name=record["name"],
                         type=record["type"], value=record["value"])
```

## Plotting

```python
con_gen.plot(contexts)                 # target values and the edges of each context
con_gen.plot_interpretation(contexts)  # target values and the interpretations
context.plot()                         # CD series and CR graph of one context
```
