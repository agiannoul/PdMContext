import statistics
import numpy as np
from numpy.linalg import norm
from numpy.fft import fft, ifft

from PdmContext.utils.structure import Context


def nearest(TargetSet: list[Context], query: Context, threshold: float, distance):
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

        if query.timestamp > fp.timestamp:  # + dt.timedelta(hours=24):
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


def distance_eu_z(context1: Context, context2: Context, a, verbose=False):
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

    **return**: a similarity between 0 and 1 , and a tuple with both z-norm and jaccard similarity
    """
    if len(context1.CD.keys()) < 1:
        return 0, (0, 0)
    if len(context2.CD.keys()) < 1:
        return 0, (0, 0)
    b = 1 - a

    common_values, uncommon_values = common_values_calc(context1, context2)

    if len(common_values) > 0 and a > 0.0000000001:
        if len(context2.CD[common_values[0]]) > 3 and len(context1.CD[common_values[0]]) > 3:
            All_common_eu = []
            for key in common_values:
                sizee = min(len(context1.CD[key]), len(context2.CD[key]))
                if sizee < 2:
                    continue
                firtsseries = context1.CD[key][-sizee:]
                secondseries = context2.CD[key][-sizee:]

                firtsseries = _z_norm(firtsseries)
                secondseries = _z_norm(secondseries)
                den = np.linalg.norm(firtsseries) + np.linalg.norm(secondseries)
                if den > 0:
                    dist = np.linalg.norm(np.array(firtsseries) - np.array(secondseries)) / den
                else:
                    dist = 0
                All_common_eu.append(dist)
            in_cc_m = 1 - sum(All_common_eu) / len(All_common_eu)

            cc_m = in_cc_m * len(All_common_eu) / (len(All_common_eu) + len(uncommon_values))

            if verbose:
                print(f"uncommon_values: {len(uncommon_values)}")
                print(f"Final cc_m = {cc_m}")
        else:
            cc_m = 0
    else:
        cc_m = 0
    # cc_m ε [-1,1] -> [0,1]
    similarity = calculate_jaccard(a, context1, context2)

    if similarity is None:
        return cc_m, (cc_m, similarity)
    else:
        return a * cc_m + b * similarity, (cc_m, similarity)


def distance_PCA_jaccard(context1: Context, context2: Context, a, seriesnames, precalc=None, verbose=False):
    """
    Calculation of similarity between two Context objects based on two quantities:
        1) The first quantity is based on the singular values from PCA.
        2) Jaccard similarity of the edges in the CR (if we ignore the direction)

    This method requires prior knowledge of the existence of all available sources in the context.

    **Parameters:**

    **context1**: A context object

    **context2**: A context object

    **a**: the weight of SBD similarity

    **seriesnames**: A list of all names from available sources in the context.

    **precalc**: If this is not None, then each time a pca fit is called, singular values are stored in details of Context in order to not be calculated next time.

    **verbose**:

    **return**: a similarity between 0 and 1 , and a tuple with both PCA and jaccard similarity
    """
    from sklearn.decomposition import PCA
    if len(context1.CD.keys()) < 1:
        return 0, (0, 0)
    if len(context2.CD.keys()) < 1:
        return 0, (0, 0)
    b = 1 - a

    common_values, uncommon_values = common_values_calc(context1, context2)

    if len(common_values) < 1:
        return 0, (0, 0)
    if len(common_values) > 0 and a > 0.0000000001 and len(context2.CD[common_values[0]]) > 5 and len(
            context1.CD[common_values[0]]) > 5:
        if precalc is not None:
            sing1 = PCA_pre(context1, seriesnames)
            sing2 = PCA_pre(context2, seriesnames)
            cc_m = 1 - np.dot(sing2, sing1) / (np.linalg.norm(sing2) * np.linalg.norm(sing1))
        else:

            c1_array = build_2D_array(seriesnames, context1)
            pca = PCA(n_components=len(seriesnames))
            pca.fit(c1_array)
            sing1 = pca.singular_values_
            c2_array = build_2D_array(seriesnames, context2)
            pca.fit(c2_array)
            sing2 = pca.singular_values_
            cc_m = 1 - np.dot(sing2, sing1) / (np.linalg.norm(sing2) * np.linalg.norm(sing1))
    else:
        cc_m = 0
    # cc_m ε [-1,1] -> [0,1]

    # check common causes-characterizations:
    similarity = calculate_jaccard(a, context1, context2)

    if similarity is None:
        return cc_m, (cc_m, similarity)
    else:
        return a * cc_m + b * similarity, (cc_m, similarity)




def distance_cc(context1: Context, context2: Context, a, verbose=False):
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

    **return**: a similarity between 0 and 1 , and a tuple with both pair-wise SBD and jaccard similarity
    """
    if len(context1.CD.keys()) < 1:
        return 0, (0, 0)
    if len(context2.CD.keys()) < 1:
        return 0, (0, 0)
    b = 1 - a

    common_values, uncommon_values = common_values_calc(context1, context2)

    if len(common_values) > 0 and a > 0.0000000001:
        if len(context2.CD[common_values[0]]) > 5 and len(context1.CD[common_values[0]]) > 5:
            All_common_cc = []
            for key in common_values:
                sizee = min(len(context1.CD[key]), len(context2.CD[key]))
                if sizee < 2:
                    continue
                firtsseries = context1.CD[key][-sizee:]
                secondseries = context2.CD[key][-sizee:]

                firtsseries = _z_norm(firtsseries)
                secondseries = _z_norm(secondseries)

                cc_array = _ncc_c(firtsseries, secondseries)
                All_common_cc.append(cc_array)
            all_cc_means = []
            for i in range(len(All_common_cc[0])):
                summ = 0
                for j in range(len(All_common_cc)):
                    summ += All_common_cc[j][i]
                all_cc_means.append(summ / len(All_common_cc))
            in_cc_m = max(all_cc_means)
            position_max = all_cc_means.index(in_cc_m)
            in_cc_m = (in_cc_m + 1) / 2
            cc_m = in_cc_m * len(All_common_cc) / (len(All_common_cc) + len(uncommon_values))
            if verbose:
                print(f"Max position: {position_max - len(firtsseries)}")
                print(f"Common cc_m = {in_cc_m}")
                print(f"uncommon_values: {len(uncommon_values)}")
                print(f"Final cc_m = {cc_m}")
        else:
            cc_m = 0
    else:
        cc_m = 0
    # cc_m ε [-1,1] -> [0,1]

    similarity=calculate_jaccard(a, context1, context2)
    if similarity is None:
        return cc_m, (cc_m, similarity)
    else:
        return a * cc_m + b * similarity




def distance_3D_sbd_jaccard(context1: Context, context2: Context, a, verbose=False):
    """
    Calculation of similarity between two Context objects based on two quantities:
        1) The first quantity is based on the 3d sbd distance upon all context data.
        2) Jaccard similarity of the edges in the CR (if we ignore the direction)

    **context1**: A context object

    **context2**: A context object

    **a**: the weight of SBD similarity

    **verbose**:

     **return**: a similarity between 0 and 1 , and a tuple with both 3D SBD and jaccard similarity
    """

    if len(context1.CD.keys()) < 1:
        return 0, (0, 0)
    if len(context2.CD.keys()) < 1:
        return 0, (0, 0)
    b = 1 - a

    common_values, uncommon_values = common_values_calc(context1, context2)

    if len(common_values)<1:
        return 0,(0,0)
    if len(common_values) > 0 and a > 0.0000000001 and len(context2.CD[common_values[0]]) > 5 and len(
            context1.CD[common_values[0]]) > 5:
        cc_m=sbd_3d(common_values,uncommon_values,context1,context2,verbose=verbose)
    else:
        cc_m = 0
    # cc_m ε [-1,1] -> [0,1]

    # check common causes-characterizations:
    similarity=calculate_jaccard(a, context1, context2)

    if similarity is None:
        return cc_m, (cc_m, similarity)
    else:
        return a * cc_m + b * similarity,(cc_m, similarity)

def common_values_calc(context1, context2):
    common_values = []
    uncommon_values = []
    for key in context1.CD.keys():
        if key in context2.CD.keys() and context1.CD[key] is not None and context2.CD[key] is not None:
            common_values.append(key)
        else:
            uncommon_values.append(key)
    for key in context2.CD.keys():
        if key not in context1.CD.keys():
            uncommon_values.append(key)

    return common_values, uncommon_values


def sbd_3d(common_values,uncommon_values,context1,context2,verbose=False):
    context1series = []
    context2series = []

    All_common_cc = []
    for key in common_values:
        # step11 = time.time()
        All_common_cc.append(key)
    # if precalc is not None: # calculate using pre calculated fft
    #     fftsize = precalc["fft_size"]
    #     names = precalc["names"]
    #
    #     Xfft,normx,x_len=get_precalculated_fft(names,fftsize, context1, common_values)
    #     Yfft,normy,y_len=get_precalculated_fft(names,fftsize, context2, common_values)
    #
    #     in_cc_m = np.max(_ncc_c_3dim_pre(Xfft, Yfft,normx,normy,x_len,y_len))
    #     cc_m = in_cc_m
    # else: # calculate normal
    for key in common_values:
        # step11 = time.time()
        firtsseries = context1.CD[key][:]
        secondseries = context2.CD[key][:]
        firtsseries = _zscore(firtsseries, ddof=1)
        secondseries = _zscore(secondseries, ddof=1)

        context1series.append(firtsseries)
        context2series.append(secondseries)

    in_cc_m = np.max(_ncc_c_3dim([np.array(context1series).transpose(), np.array(context2series).transpose()]))

    cc_m = in_cc_m * len(All_common_cc) / (len(All_common_cc) + len(uncommon_values))
    if verbose:
        print(f"Common cc_m = {in_cc_m}")
        print(f"uncommon_values: {len(uncommon_values)}")
        print(f"Final cc_m = {cc_m}")
    return cc_m


def get_precalculated_fft(seriesnames,fftsize,context1,common_values):
    if context1.details is not None and isinstance(context1.details, dict):
        if "fft" in context1.details.keys():
            return context1.details["fft"],context1.details["norm"],context1.details["len"]
    context1series=[]
    for key in context1.CD.keys():
        firtsseries = context1.CD[key][:]
        firtsseries = _zscore(firtsseries, ddof=1)
        context1series.append(firtsseries)
    for seriesname in seriesnames:
        if seriesname not in context1.CD.keys():
            firtsseries =[ 0 for i in context1series[0]]
            context1series.append(firtsseries)

    if isinstance(context1.details, dict):
        x=np.array(context1series).transpose()
        fftx=calculate_3d_fft(x, fftsize)
        context1.details["fft"]=fftx
        context1.details["norm"]=norm(x, axis=(0, 1))
        context1.details["len"]=x.shape[0]
    else:
        x = np.array(context1series).transpose()
        fftx = calculate_3d_fft(x, fftsize)
        context1.details= {"fft": fftx,
                           "norm":norm(x, axis=(0, 1)),
                           "len":x.shape[0]}
    return context1.details["fft"],context1.details["norm"],context1.details["len"]


def calculate_3d_fft(x, fft_size):
    return fft(x, fft_size, axis=0)

def _ncc_c_3dim_pre(fftX,fftY,normx,normy,x_len,y_len):
    den = normx * normy

    if den < 1e-9:
        den = np.inf

    #fft_size = 1 << (2*x_len-1).bit_length()

    cc = ifft(fftX * np.conj(fftY), axis=0)
    cc = np.concatenate((cc[-(x_len-1):], cc[:x_len]), axis=0)

    return np.real(cc).sum(axis=-1) / den

def _ncc_c_3dim(data):
    x, y = data[0], data[1]
    den = norm(x, axis=(0, 1)) * norm(y, axis=(0, 1))

    if den < 1e-9:
        den = np.inf

    x_len = x.shape[0]
    fft_size = 1 << (2*x_len-1).bit_length()

    cc = ifft(fft(x, fft_size, axis=0) * np.conj(fft(y, fft_size, axis=0)), axis=0)
    cc = np.concatenate((cc[-(x_len-1):], cc[:x_len]), axis=0)

    return np.real(cc).sum(axis=-1) / den

def _z_norm(series):
    if min(series) != max(series):
        ms1 = statistics.mean(series)
        ss1 = statistics.stdev(series)
        series = [(s1 - ms1) / ss1 for s1 in series]
    else:
        series = [0 for i in range(len(series))]
    return series

def _zscore(a, axis=0, ddof=0):
    a = np.asanyarray(a)
    mns = a.mean(axis=axis)
    sstd = a.std(axis=axis, ddof=ddof)
    if axis and mns.ndim < a.ndim:
        res = ((a - np.expand_dims(mns, axis=axis)) /
               np.expand_dims(sstd, axis=axis))
    else:
        res = (a - mns) / sstd
    return np.nan_to_num(res)

def jaccard_CR(context1,context2):
    common = 0

    edges1 = ignore_order(context1)
    edges2 = ignore_order(context2)

    for edge in edges1:
        for edge2 in edges2:
            if edge[0] == edge2[0] and edge[1] == edge2[1]:
                common += 1

    if (len(edges1) + len(edges2) - common) > 0:
        if common == 0:
            jaccard = 0
        else:
            jaccard = common / (len(edges1) + len(edges2) - common)
        similarity = jaccard
    # there are no samples Jaccard(empty,empty) = ? , in that case we return 0
    else:
        similarity = 0
    return similarity

def jaccard_distance_CR(context1,context2):
    return 1-jaccard_CR(context1,context2)

def calculate_jaccard(a,context1,context2):
    b=1-a
    if b > 0.000000001:
        # check common causes-characterizations:
        common = 0

        edges1 = ignore_order(context1)
        edges2 = ignore_order(context2)

        for edge in edges1:
            for edge2 in edges2:
                if edge[0] == edge2[0] and edge[1] == edge2[1]:
                    common += 1

        if (len(edges1) + len(edges2) - common) > 0:
            if common == 0:
                jaccard = 0
            else:
                jaccard = common / (len(edges1) + len(edges2) - common)
            similarity = jaccard
        # there are no samples Jaccard(empty,empty) = ? , in that case we use only first part
        else:
            if a < 0.0000001:
                similarity = 1
            else:
                similarity = None
    else:
        similarity = 0
    return similarity



def PCA_pre(context1,seriesnames):
    from sklearn.decomposition import PCA
    if context1.details is not None and isinstance(context1.details, dict):
        if "fft" in context1.details.keys():
            return context1.details["sing"]
    else:
        c1_array = build_2D_array(seriesnames, context1)
        pca = PCA(n_components=len(seriesnames))
        pca.fit(c1_array)
        sing1 = pca.singular_values_
        if context1.details is None:
            context1.details = {"sing":sing1}
        else:
            context1.details["sing"]= sing1
    return context1.details["sing"]
def build_2D_array(seriesnames,context1):
    context1series = []
    for key in context1.CD.keys():
        firtsseries = context1.CD[key][:]
        context1series.append(firtsseries)
    for seriesname in seriesnames:
        if seriesname not in context1.CD.keys():
            firtsseries = [0 for i in context1series[0]]
            context1series.append(firtsseries)
    return np.array(context1series).transpose()
def ignore_order(context1: Context):
    edges1 = []

    for edge in context1.CR['edges']:
        if edge[0] > edge[1]:
            potential = (edge[0], edge[1])
        else:
            potential = (edge[1], edge[0])
        if potential not in edges1:
            edges1.append(potential)
    return edges1


def ignore_order_list(edgeslist1):
    edges1 = []

    for edge in edgeslist1:
        if edge[0] > edge[1]:
            potential = (edge[0], edge[1])
        else:
            potential = (edge[1], edge[0])
        if potential not in edges1:
            edges1.append(potential)
    return edges1


def _sbd(x, y):
    ncc = _ncc_c(x, y)
    idx = ncc.argmax()
    dist = 1 - ncc[idx]
    return dist, None


def _ncc_c(x, y):
    den = np.array(norm(x) * norm(y))
    den[den == 0] = np.Inf

    x_len = len(x)
    fft_size = 1 << (2 * x_len - 1).bit_length()
    cc = ifft(fft(x, fft_size) * np.conj(fft(y, fft_size)))
    cc = np.concatenate((cc[-(x_len - 1):], cc[:x_len]))
    return np.real(cc) / den
