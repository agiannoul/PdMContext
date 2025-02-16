import networkx as nx
def empty_cause(names, data):
    return []


def calculate_with_pc(names, data,timestamps):
    """
    Peter-Clark (PC) algorithm for causal discovery from gcastle package.

    **Parameters**:

    **names**: The names of the different time series in data.

    **data**: Multivariate (2D) array containing different time series, each column represent a time series of Context CD part.


    """
    from castle.algorithms import PC
    try:
        pc = PC(variant='parallel')
        pc.learn(data)
    except Exception as e:
        print(e)
        return None

    learned_graph = nx.DiGraph(pc.causal_matrix)
    # Relabel the nodes
    MAPPING = {k: n for k, n in zip(range(len(names)), names)}
    learned_graph = nx.relabel_nodes(learned_graph, MAPPING, copy=True)
    edges =learned_graph.edges
    fedges =[]
    for tup in edges:
        if tup not in fedges:
            fedges.append(tup)
        if (tup[1] ,tup[0]) not in fedges:
            fedges.append((tup[1] ,tup[0]))
    return fedges
