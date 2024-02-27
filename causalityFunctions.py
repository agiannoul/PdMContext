from causalai.data.transforms.time_series import StandardizeTransform
from causalai.models.tabular.pc import PC as pcsales
from causalai.models.common.CI_tests.partial_correlation import PartialCorrelation
from causalai.data.tabular import TabularData
from causallearn.search.ConstraintBased.FCI import fci



def calculatewith_fci(names, data):
    try:
        print(data)
        g, edges = fci(data, show_progress=False)
    except Exception as e:
        print(e)
        return None
    learned_graph = g.get_graph_edges()
    edges2 = [str(ed) for ed in learned_graph]
    fedges = []
    for edge in edges2:
        splitted = edge.split(" ")
        node1 = int(splitted[0][1:]) - 1
        connection = splitted[1]
        node2 = int(splitted[2][1:]) - 1
        fedges.append((node1, node2))
        if "<" not in connection and ">" not in connection:
            fedges.append((node2, node1))

    MAPPING = {k: n for k, n in zip(range(len(data)), names)}
    return [(MAPPING[tup[0]], MAPPING[tup[1]]) for tup in fedges]

def salesforcePC(names ,dseries):

    StandardizeTransform_ = StandardizeTransform()
    StandardizeTransform_.fit(dseries)

    dseries = StandardizeTransform_.transform(dseries)


    data_obj = TabularData(dseries, var_names=names)
    CI_test = PartialCorrelation()



    pc = pcsales(
        data=data_obj,
        prior_knowledge=None,
        CI_test=CI_test,
        use_multiprocessing=False
    )
    result = pc.run(pvalue_thres=0.05, max_condition_set_size=2)

    edges = []
    for key in result.keys():
        parents = result[key]['parents']
        for parent in parents:
            edges.append((parent, key))
    return edges