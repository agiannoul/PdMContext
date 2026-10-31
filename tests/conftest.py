import pandas as pd
import pytest

from PdmContext.ContextGeneration import ContextGenerator

START = pd.Timestamp("2024-01-01 00:00")


def hour(h):
    return START + pd.Timedelta(hours=h)


def minute(m):
    return START + pd.Timedelta(minutes=m)


def no_causality(names, data, timestamps):
    return []


def correlation_causality(names, data, timestamps):
    """Module level (picklable) causality function: edges between strongly correlated series."""
    import numpy as np
    corr = np.corrcoef(data.T)
    return [(names[i], names[j]) for i in range(len(names)) for j in range(len(names))
            if i != j and abs(corr[i, j]) > 0.5]


class FixedCausality:
    """Causality function returning the given edges (only those whose series are passed to it)."""

    def __init__(self, edges):
        self.edges = edges
        self.calls = []

    def __call__(self, names, data, timestamps):
        self.calls.append((list(names), data, list(timestamps)))
        return [edge for edge in self.edges if edge[0] in names and edge[1] in names]


@pytest.fixture
def generator():
    return ContextGenerator(target="score", context_horizon="8 hours", Causalityfunct=no_causality)
