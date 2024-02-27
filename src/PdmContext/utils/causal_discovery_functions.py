
import networkx as nx
from castle.algorithms import PC

def emptyCause(names, data):
    return []


def calculatewithPc(names, data):
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
    edges=learned_graph.edges
    return edges


