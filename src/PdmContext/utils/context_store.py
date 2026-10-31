"""
Internal, memory bounded storage of the contexts kept by ContextGenerator (the contexts of the last horizon).

Consecutive contexts share most of their data (each new context drops the oldest samples of the window and adds
the newest ones). Instead of keeping a full copy of every series for every context (memory ~ N^2 for N samples in
the horizon), the series of each source are kept once in a _SeriesStore covering the stored contexts (about two
horizons), and every stored context only references a part of it (memory ~ N).

When a series does not match the stored data (e.g. a mapper which recomputes past values), a new store is started
for it, so the stored contexts always reproduce exactly the series they were created with.
Contexts are rebuilt only when accessed, and the context returned by the generator is reused while it is still
referenced by the user.
"""
import weakref
from bisect import bisect_left
from collections import deque
from collections.abc import Sequence

from PdmContext.utils.structure import Context

# maximum number of ranges (with a constant value) which may differ from the stored data before a new store is
# started (e.g. the first position of categorical series)
MAX_PATCHES = 4


class _SeriesStore:
    """Values of one series for consecutive positions [base, base + len(values))."""
    __slots__ = ("base", "values", "refs")

    def __init__(self, base, values):
        self.base = base
        self.values = values
        # start positions of the stored contexts using this store (in order of insertion)
        self.refs = deque()

    @property
    def end(self):
        return self.base + len(self.values)

    def release(self):
        self.refs.popleft()
        if self.refs:
            dead = self.refs[0] - self.base
            # drop the unused prefix (amortized, when it is the larger part of the store)
            if dead > len(self.values) // 2:
                del self.values[:dead]
                self.base += dead


def _match(old, new):
    """Return (delta, patches) such that new == old (+ delta) (with patches applied), or None."""
    if old == new:
        return 0, None
    if not new:
        return None
    # integer series shifted by a constant (e.g. configuration series which start from 0 in each window)
    if type(new[0]) is int and type(old[0]) is int:
        delta = new[0] - old[0]
        if all(type(x) is int for x in new) and [x + delta for x in old] == new:
            return delta, None
    # a few changed ranges with a constant value (e.g. the first position of categorical series, or the state
    # series of categorical sources which changes from 1 to 0 up to the last change of category)
    patches = []
    for i, (a, b) in enumerate(zip(old, new)):
        if a is not b and a != b:
            if patches and patches[-1][1] == i and type(patches[-1][2]) is type(b) and patches[-1][2] == b:
                patches[-1][1] = i + 1
            elif len(patches) == MAX_PATCHES:
                return None
            else:
                patches.append([i, i + 1, b])
    return 0, [tuple(patch) for patch in patches]


def _no_context():
    return None


class _StoredContext:
    """A stored context: timestamp, CR (shared with the returned context) and references to the series."""
    __slots__ = ("timestamp", "CR", "details", "series", "timeline", "context_ref", "__weakref__")

    def __init__(self, timestamp, CR, details, series, timeline, context):
        self.timestamp = timestamp
        self.CR = CR
        self.details = details
        # list of (name, reference) in the order of the CD, reference: None or (store, start, length, delta, patches)
        # patches: (first, last, value) ranges replaced by value
        self.series = series
        # (store, start) in the target timeline
        self.timeline = timeline
        self.context_ref = weakref.ref(context)

    def context(self):
        context = self.context_ref()
        if context is None:
            CD = {}
            for name, ref in self.series:
                if ref is None:
                    CD[name] = None
                    continue
                store, start, length, delta, patches = ref
                values = store.values[start - store.base:start - store.base + length]
                if delta:
                    values = [x + delta for x in values]
                if patches:
                    for first, last, value in patches:
                        values[first:last] = [value] * (last - first)
                CD[name] = values
            context = Context(self.timestamp, CD, self.CR, self.details)
            self.context_ref = weakref.ref(context)
        return context

    def release(self):
        for name, ref in self.series:
            if ref is not None:
                ref[0].release()
        self.timeline[0].release()

    def __getstate__(self):
        # weak references can not be pickled, keep the context itself
        return {"timestamp": self.timestamp, "CR": self.CR, "details": self.details, "series": self.series,
                "timeline": self.timeline, "context": self.context_ref()}

    def __setstate__(self, state):
        self.timestamp = state["timestamp"]
        self.CR = state["CR"]
        self.details = state["details"]
        self.series = state["series"]
        self.timeline = state["timeline"]
        context = state["context"]
        self.context_ref = _no_context if context is None else weakref.ref(context)


class ContextStore:
    """Stored contexts of a ContextGenerator, with memory linear to the data of the stored contexts."""

    def __init__(self):
        self.items = []
        self.stores = {}
        self.timeline = None

    def __len__(self):
        return len(self.items)

    def size(self):
        """Number of stored values (series and timeline), for monitoring the memory."""
        stores = {id(self.timeline): self.timeline} if self.timeline is not None else {}
        for item in self.items:
            stores[id(item.timeline[0])] = item.timeline[0]
            for name, ref in item.series:
                if ref is not None:
                    stores[id(ref[0])] = ref[0]
        return sum(len(store.values) for store in stores.values())

    def append(self, context: Context, target_times):
        """Store a context, target_times: the timestamps of the target series (positions of the CD series)."""
        start = self._align(list(target_times))
        series = [(name, self._reference(name, values, start)) for name, values in context.CD.items()]
        self.items.append(_StoredContext(context.timestamp, context.CR, context.details, series,
                                         (self.timeline, start), context))

    def keep_last(self, count):
        """Keep only the last count contexts."""
        remove = len(self.items) - count
        if remove <= 0:
            return
        for item in self.items[:remove]:
            item.release()
        del self.items[:remove]
        # forget series which are not used by any stored context
        for name in [name for name, store in self.stores.items() if not store.refs]:
            del self.stores[name]

    def _align(self, times):
        """Return the position of the first target timestamp in the timeline (extending the timeline)."""
        timeline = self.timeline
        if timeline is not None and times:
            values = timeline.values
            k = bisect_left(values, times[0])
            if k < len(values) and values[k] == times[0]:
                overlap = len(values) - k
                # the buffer only loses old samples and gains new ones, so the new window is the end of the
                # timeline followed by new samples (checked at its last common position)
                if overlap <= len(times) and values[-1] == times[overlap - 1]:
                    values.extend(times[overlap:])
                    timeline.refs.append(timeline.base + k)
                    return timeline.base + k
        # new timeline (e.g. late target samples), with positions not overlapping the previous one
        base = 0 if timeline is None else timeline.end + 1
        self.timeline = _SeriesStore(base, times)
        self.timeline.refs.append(base)
        return base

    def _reference(self, name, values, start):
        if values is None:
            return None
        store = self.stores.get(name)
        if store is not None and store.base <= start <= store.end:
            overlap = store.end - start
            if overlap <= len(values):
                matched = _match(store.values[start - store.base:], list(values[:overlap]))
                if matched is not None:
                    delta, patches = matched
                    new_values = values[overlap:]
                    store.values.extend(new_values if not delta else [x - delta for x in new_values])
                    store.refs.append(start)
                    return store, start, len(values), delta, patches
        store = _SeriesStore(start, list(values))
        store.refs.append(start)
        self.stores[name] = store
        return store, start, len(values), 0, None


class ContextHistory(Sequence):
    """Read only sequence of the contexts kept by a ContextGenerator (oldest first)."""

    def __init__(self, store: ContextStore):
        self._store = store

    def __len__(self):
        return len(self._store.items)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [item.context() for item in self._store.items[index]]
        return self._store.items[index].context()

    def __eq__(self, other):
        if isinstance(other, (list, tuple, Sequence)):
            return list(self) == list(other)
        return NotImplemented

    def __repr__(self):
        return f"ContextHistory({len(self)} contexts)"
