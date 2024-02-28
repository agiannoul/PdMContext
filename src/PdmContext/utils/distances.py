import statistics
import numpy as np
from numpy.linalg import norm
from numpy.fft import fft, ifft

from PdmContext.utils.structure import Context


def nearest(TargetSet :list[Context], query : Context, threshold: float,distance):
    '''
    This method searches if there is a similar context object as query in the TargetSet.
    Where the similar means with similarity at least as threshold

    **Parameters**:

    **TargetSet**: A list from context objects to search for similar ones

    **query** : The query context object

    **threshold** : The similarity threshold (real value in [0,1]
    '''
    maxdist = 0
    # starting=time.time()
    for fp in TargetSet:

        if query["timestamp"] > fp["timestamp"]:  # + dt.timedelta(hours=24):
            dist, parts = distance(query, fp)
            if dist > maxdist:
                maxdist = dist
                if maxdist > threshold:
                    break
    return maxdist


def np_pearson_cor(x, y):
    xv = x - x.mean(axis=0)
    yv = y - y.mean(axis=0)
    xvss = (xv * xv).sum(axis=0)
    yvss = (yv * yv).sum(axis=0)
    result = np.matmul(xv.transpose(), yv) / np.sqrt(np.outer(xvss, yvss))
    # bound the values to -1 to 1 in the event of precision issues
    return np.maximum(np.minimum(result, 1.0), -1.0)

def distance_eu_z(context1 : Context, context2 : Context,a,verbose=False):
    """
    Calculation of similarity between two Context objects based on two quantities:
        1) The first quantity is based on the Euclidean  distance after z_normalization
            We calculate a similarity based on the Euclidean distance between common values in the context CD,
            equal to Euclidean(c1,c2)/(norm(c1)+norm(c2) to be in [0,1]
            where each time we use the last n values (where n is the size of the shorter series)
        2) Jaccard similarity of the edges in the CR (if we ignore the direction)

    **context1**: A context object
    **context2**: A context object
    **a**: the weight of Euclidean similarity
    **verbose**:
    **return**: a similarity between 0 and 1
    """
    b=1-a
    common_values = []
    uncommon_values = []
    for key in context1.CD.keys():
        if key != "timestamp" and key != "edges" and key != "characterization" and key != "interpertation":
            if key in context2.CD.keys():
                if context1.CD[key] is not None and context2.CD[key] is not None:
                    common_values.append(key)
                else:
                    uncommon_values.append(key)
            else:
                uncommon_values.append(key)
    for key in context2.CD.keys():
        if key != "timestamp" and key != "edges" and key != "characterization" and key != "interpertation":
            if key not in context1.CD.keys():
                uncommon_values.append(key)
    if len(common_values)>0 and a>0.0000000001:
        All_common_eu=[]
        for key in common_values:
            sizee = min(len(context1.CD[key]), len(context2.CD[key]))
            if sizee < 2:
                continue
            firtsseries = context1.CD[key][-sizee:]
            secondseries = context2.CD[key][-sizee:]

            firtsseries=_z_norm(firtsseries)
            secondseries=_z_norm(secondseries)
            den=np.linalg.norm(firtsseries)+np.linalg.norm(secondseries)
            if den>0:
                dist = np.linalg.norm(np.array(firtsseries)-np.array(secondseries))/den
            else:
                dist=0
            All_common_eu.append(dist)
        in_cc_m=1-sum(All_common_eu)/len(All_common_eu)

        cc_m=in_cc_m * len(All_common_eu) / (len(All_common_eu) + len(uncommon_values))


        if verbose:
            print(f"uncommon_values: {len(uncommon_values)}")
            print(f"Final cc_m = {cc_m}")
    else:
        cc_m=0
    # cc_m ε [-1,1] -> [0,1]

    # check common causes-characterizations:
    common = 0

    edges1=ignoreOrder(context1)
    edges2=ignoreOrder(context2)

    for edge in edges1:
        for edge2  in edges2:
            if edge[0] == edge2[0] and edge[1] == edge2[1]:
                common += 1

    if (len(edges1) + len(edges2) - common) >0:
        if common == 0:
            jaccard = 0
        else:
            jaccard = common / (len(edges1) + len(edges2) - common)
        similarity = jaccard
    # there are no samples Jaccard(empty,empty) = ? , in that case we use only first part
    else:
        if a<0.0000001:
            similarity = 1
        else:
            similarity=None
    if similarity is None:
        return cc_m, (cc_m, similarity)
    else:
        return a * cc_m + b * similarity

def _z_norm(series):
    if min(series) != max(series):
        ms1 = statistics.mean(series)
        ss1 = statistics.stdev(series)
        series = [(s1 - ms1) / ss1 for s1 in series]
    else:
        series = [0 for i in range(len(series))]
    return series

def distance_cc(context1 : Context, context2 : Context,a,verbose=False):
    """
    Calculation of similarity between two Context objects based on two quantities:
        1) The first quantity is based on the sbd distance
            We calculate the minimum (average) sbd between all common series in the CD of contexts, from all possible shifts.
            The shifts apply to all series each time.
            Each time we use the last n values (where n is the size of the shorter series)
            Which is also weighted from the ratio of common values.
        2) Jaccard similarity of the edges in the CR (if we ignore the direction)

    **context1**: A context object
    **context2**: A context object
    **a**: the weight of SBD similarity
    **verbose**:
    **return**: a similarity between 0 and 1
    """
    b=1-a
    common_values = []
    uncommon_values = []
    for key in context1.CD.keys():
        if key != "timestamp" and key != "edges" and key != "characterization" and key != "interpertation":
            if key in context2.CD.keys():
                if context1.CD[key] is not None and context2.CD[key] is not None:
                    common_values.append(key)
                else:
                    uncommon_values.append(key)
            else:
                uncommon_values.append(key)
    for key in context2.CD.keys():
        if key != "timestamp" and key != "edges" and key != "characterization" and key != "interpertation":
            if key not in context1.CD.keys():
                uncommon_values.append(key)
    if len(common_values)>0 and a>0.0000000001:
        All_common_cc=[]
        for key in common_values:
            sizee = min(len(context1.CD[key]), len(context2.CD[key]))
            if sizee < 2:
                continue
            firtsseries = context1.CD[key][-sizee:]
            secondseries = context2.CD[key][-sizee:]

            firtsseries=_z_norm(firtsseries)
            secondseries=_z_norm(secondseries)

            cc_array=_ncc_c(firtsseries,secondseries)
            All_common_cc.append(cc_array)
        all_cc_means=[]
        for i in range(len(All_common_cc[0])):
            summ=0
            for j in range(len(All_common_cc)):
                summ+=All_common_cc[j][i]
            all_cc_means.append(summ/len(All_common_cc))
        in_cc_m=max(all_cc_means)
        position_max=all_cc_means.index(in_cc_m)
        in_cc_m = (in_cc_m + 1) / 2
        cc_m=in_cc_m*len(All_common_cc)/(len(All_common_cc)+len(uncommon_values))
        if verbose:
            print(f"Max position: {position_max-len(firtsseries)}")
            print(f"Common cc_m = {in_cc_m}")
            print(f"uncommon_values: {len(uncommon_values)}")
            print(f"Final cc_m = {cc_m}")
    else:
        cc_m=0
    # cc_m ε [-1,1] -> [0,1]

    # check common causes-characterizations:
    common = 0

    edges1=ignoreOrder(context1)
    edges2=ignoreOrder(context2)

    for edge in edges1:
        for edge2  in edges2:
            if edge[0] == edge2[0] and edge[1] == edge2[1]:
                common += 1

    if (len(edges1) + len(edges2) - common) >0:
        if common == 0:
            jaccard = 0
        else:
            jaccard = common / (len(edges1) + len(edges2) - common)
        similarity = jaccard
    # there are no samples Jaccard(empty,empty) = ? , in that case we use only first part
    else:
        if a<0.0000001:
            similarity = 1
        else:
            similarity=None
    if similarity is None:
        return cc_m, (cc_m, similarity)
    else:
        return a * cc_m + b * similarity

def ignoreOrder(context1: Context):
    edges1 = []

    for edge in context1.CR['edges']:
        if edge[0] > edge[1]:
            potential = (edge[0], edge[1])
        else:
            potential = (edge[1], edge[0])
        if potential not in edges1:
            edges1.append(potential)
    return edges1

def ignoreOrderList(edgeslist1):
    edges1 = []

    for edge in edgeslist1:
        if edge[0] > edge[1]:
            potential = (edge[0], edge[1])
        else:
            potential = (edge[1], edge[0])
        if potential not in edges1:
            edges1.append(potential)
    return edges1


def _sbd( x, y):
    ncc = _ncc_c(x, y)
    idx = ncc.argmax()
    dist = 1 - ncc[idx]
    return dist, None


def _ncc_c( x, y):
    den = np.array(norm(x) * norm(y))
    den[den == 0] = np.Inf

    x_len = len(x)
    fft_size = 1 << (2 * x_len - 1).bit_length()
    cc = ifft(fft(x, fft_size) * np.conj(fft(y, fft_size)))
    cc = np.concatenate((cc[-(x_len - 1):], cc[:x_len]))
    return np.real(cc) / den