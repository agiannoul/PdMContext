import math

import numpy as np
import scipy.stats as stats
import networkx as nx


class RunningCovarianceTimestamps:
    def __init__(self):
        """
        Initializes running sum, sum of squares, mean, and covariance matrix
        for incremental updates.
        """
        self.datacopy=None
        self.data = None
    def init(self,data,timestamps):
        self.data = data
        self.n, self.d = data.shape
        # Sufficient statistics
        self.sum = np.sum(data, axis=0)
        self.sum_sq = np.dot(data.T, data)  # Sum of outer products
        self.mean = self.sum / self.n

        self.cov = self.coovarianve(self.sum, self.sum_sq, self.mean, self.n)
        self.corr = self.covariance_to_correlation(self.cov)
        self.timestamps = timestamps
        self.datacopy=data
        return
    def update(self, data, timestamps):

        """
        Updates the sufficient statistics when a sample is removed and a new one is added.
        Recomputes the mean and covariance exactly as np.cov would.

        """
        # Update the sums


        if self.data is None or self.data.shape[1]!=data.shape[1]:
            self.init(data, timestamps)
            return
        if timestamps[0]==self.timestamps[0] and timestamps[-1]==self.timestamps[-1]:
            return


        assert self.data.shape[1] == data.shape[1], "Data have different shapes"

        positionOld = self.timestamps.index(timestamps[0])
        oldsamples = self.data[:positionOld]

        positionNew = timestamps.index(self.timestamps[-1])+1
        newsamples = data[positionNew:]

        for old_sample in oldsamples:
            self.sum = self.sum - old_sample
            self.sum_sq = self.sum_sq - np.outer(old_sample, old_sample)
        self.datacopy=self.datacopy[positionOld:]
        for new_sample in newsamples:
            self.sum = self.sum + new_sample
            self.sum_sq = self.sum_sq + np.outer(new_sample, new_sample)
        self.datacopy = np.vstack([self.datacopy, newsamples])

        # if abs(np.sum(self.datacopy-data))>0.00000001:
        #     ok='ok'
        self.n = data.shape[0]
        # Update the mean
        self.mean = self.sum / self.n

        # Recompute the covariance (unbiased estimate with ddof=1)
        self.cov = self.coovarianve(self.sum, self.sum_sq, self.mean, self.n)
        self.corr=self.covariance_to_correlation(self.cov)


        self.timestamps = timestamps
        self.data = data

    def covariance_to_correlation(self,C):
        """Convert a covariance matrix C to a correlation matrix R."""
        diag_sqrt = np.sqrt(np.diag(C))  # Square root of diagonal elements
        R = C / np.outer(diag_sqrt, diag_sqrt)  # Element-wise division
        return R

    def coovarianve(self,sum, sum_sq, mean, n):
        return (sum_sq - (np.outer(mean, sum) + np.outer(sum, mean)) + np.outer(mean, mean) * n) / (n - 1)

    # def computer_originalfisherz(self,data, x, y, z):
    #     import scipy.stats as stats
    #
    #     n = data.shape[0]
    #     k = len(z)
    #     sub_corr=None
    #     corr=np.corrcoef(data.T)
    #     if k == 0:
    #         r = np.corrcoef(data[:, [x, y]].T)[0][1]
    #     else:
    #         sub_index = [x, y]
    #         sub_index.extend(z)
    #         sub_corr = np.corrcoef(data[:, sub_index].T)
    #         # inverse matrix
    #         try:
    #             PM = np.linalg.inv(sub_corr)
    #         except np.linalg.LinAlgError:
    #             PM = np.linalg.pinv(sub_corr)
    #         r = -1 * PM[0, 1] / math.sqrt(abs(PM[0, 0] * PM[1, 1]))
    #     cut_at = 0.99999
    #     rold=r
    #     r = min(cut_at, max(-1 * cut_at, r))  # make r between -1 and 1
    #
    #     # Fisher’s z-transform
    #     res = math.sqrt(n - k - 3) * .5 * math.log1p((2 * r) / (1 - r))
    #     p_value = 2 * (1 - stats.norm.cdf(abs(res)))
    #
    #     return corr,sub_corr,r,rold, res, p_value

    def compute_fisherz(self, x, y, z=[]):
        assert self.data is not None, "Moving Covariance Not initialized"

        k = len(z)
        if k == 0:
            r = self.corr[x, y]
        else:
            sub_index = [x, y]
            sub_index.extend(z)
            sub_corr = self.corr[sub_index][:,sub_index]
            try:
                PM = np.linalg.inv(sub_corr)
            except np.linalg.LinAlgError:
                PM = np.linalg.pinv(sub_corr)
            r = -1 * PM[0, 1] / math.sqrt(abs(PM[0, 0] * PM[1, 1]))
        cut_at = 0.99999
        r = min(cut_at, max(-1 * cut_at, r))  # make r between -1 and 1

        # Fisher’s z-transform
        res = math.sqrt(self.n - k - 3) * .5 * math.log1p((2 * r) / (1 - r))
        p_value = 2 * (1 - stats.norm.cdf(abs(res)))

        return None, None, p_value

from castle.common import Tensor
from castle.common.priori_knowledge import orient_by_priori_knowledge
from itertools import permutations
from itertools import combinations
from copy import deepcopy


class PC():

    def __init__(self, variant='stable', alpha=0.05,
                 priori_knowledge=None):
        self.variant = variant
        self.alpha = alpha
        self.ci_test = RunningCovarianceTimestamps()
        self.causal_matrix = None
        self.priori_knowledge = priori_knowledge

    def learn(self, data,timestamps, columns=None, **kwargs):
        """
        Based on the implementation of gCastle Library
        """


        self.ci_test.update(data,timestamps)
        if self.ci_test.data is None:
            self.causal_matrix=None
            return
        data = Tensor(data, columns=columns)

        skeleton, sep_set = self.find_skeleton(data,
                                          alpha=self.alpha,
                                          ci_test_class=self.ci_test,
                                          variant=self.variant,
                                          priori_knowledge=self.priori_knowledge,
                                          **kwargs)

        self._causal_matrix = Tensor(
            self.orient(skeleton, sep_set, self.priori_knowledge).astype(int),
            index=data.columns,
            columns=data.columns
        )
        self.causal_matrix=self._causal_matrix

    def orient(self, skeleton, sep_set, priori_knowledge=None):


        if priori_knowledge is not None:
            skeleton = orient_by_priori_knowledge(skeleton, priori_knowledge)

        columns = list(range(skeleton.shape[1]))
        cpdag = deepcopy(abs(skeleton))
        # pre-processing
        for ij in sep_set.keys():
            i, j = ij
            all_k = [x for x in columns if x not in ij]
            for k in all_k:
                if cpdag[i, k] + cpdag[k, i] != 0 \
                        and cpdag[k, j] + cpdag[j, k] != 0:
                    if k not in sep_set[ij]:
                        if cpdag[i, k] + cpdag[k, i] == 2:
                            cpdag[k, i] = 0
                        if cpdag[j, k] + cpdag[k, j] == 2:
                            cpdag[k, j] = 0
        while True:
            old_cpdag = deepcopy(cpdag)
            pairs = list(combinations(columns, 2))
            for ij in pairs:
                i, j = ij
                if cpdag[i, j] * cpdag[j, i] == 1:
                    # rule1
                    for i, j in permutations(ij, 2):
                        all_k = [x for x in columns if x not in ij]
                        for k in all_k:
                            if cpdag[k, i] == 1 and cpdag[i, k] == 0 \
                                    and cpdag[k, j] + cpdag[j, k] == 0:
                                cpdag[j, i] = 0
                    # rule2
                    for i, j in permutations(ij, 2):
                        all_k = [x for x in columns if x not in ij]
                        for k in all_k:
                            if (cpdag[i, k] == 1 and cpdag[k, i] == 0) \
                                    and (cpdag[k, j] == 1 and cpdag[j, k] == 0):
                                cpdag[j, i] = 0
                    # rule3
                    for i, j in permutations(ij, 2):
                        for kl in sep_set.keys():  # k and l are nonadjacent.
                            k, l = kl
                            # if i——k——>j and  i——l——>j
                            if cpdag[i, k] == 1 \
                                    and cpdag[k, i] == 1 \
                                    and cpdag[k, j] == 1 \
                                    and cpdag[j, k] == 0 \
                                    and cpdag[i, l] == 1 \
                                    and cpdag[l, i] == 1 \
                                    and cpdag[l, j] == 1 \
                                    and cpdag[j, l] == 0:
                                cpdag[j, i] = 0
                    # rule4
                    for i, j in permutations(ij, 2):
                        for kj in sep_set.keys():  # k and j are nonadjacent.
                            if j not in kj:
                                continue
                            else:
                                kj = list(kj)
                                kj.remove(j)
                                k = kj[0]
                                ls = [x for x in columns if x not in [i, j, k]]
                                for l in ls:
                                    if cpdag[k, l] == 1 \
                                            and cpdag[l, k] == 0 \
                                            and cpdag[i, k] == 1 \
                                            and cpdag[k, i] == 1 \
                                            and cpdag[l, j] == 1 \
                                            and cpdag[j, l] == 0:
                                        cpdag[j, i] = 0
            if np.all(cpdag == old_cpdag):
                break

        return cpdag

    def _loop(self,G, d):

        assert G.shape[0] == G.shape[1]

        pairs = [(x, y) for x, y in combinations(set(range(G.shape[0])), 2)]
        less_d = 0
        for i, j in pairs:
            adj_i = set(np.argwhere(G[i] != 0).reshape(-1, ))
            z = adj_i - {j}  # adj(C, i)\{j}
            if len(z) < d:
                less_d += 1
            else:
                break
        if less_d == len(pairs):
            return False
        else:
            return True





    def find_skeleton(self,data, alpha, ci_test_class, variant='original',
                      priori_knowledge=None, base_skeleton=None,
                      p_cores=1, s=None, batch=None):
        ci_test = ci_test_class.compute_fisherz
        n_feature = data.shape[1]
        if base_skeleton is None:
            skeleton = np.ones((n_feature, n_feature)) - np.eye(n_feature)
        else:
            row, col = np.diag_indices_from(base_skeleton)
            base_skeleton[row, col] = 0
            skeleton = base_skeleton
        nodes = set(range(n_feature))

        # update skeleton based on priori knowledge
        for i, j in combinations(nodes, 2):
            if priori_knowledge is not None and (
                    priori_knowledge.is_forbidden(i, j)
                    and priori_knowledge.is_forbidden(j, i)):
                skeleton[i, j] = skeleton[j, i] = 0

        sep_set = {}
        d = -1
        while self._loop(skeleton, d):  # until for each adj(C,i)\{j} < l
            d += 1
            if variant == 'stable':
                C = deepcopy(skeleton)
            else:
                C = skeleton
            if variant != 'parallel':
                for i, j in combinations(nodes, 2):
                    if skeleton[i, j] == 0:
                        continue
                    adj_i = set(np.argwhere(C[i] == 1).reshape(-1, ))
                    z = adj_i - {j}  # adj(C, i)\{j}
                    if len(z) >= d:
                        # |adj(C, i)\{j}| >= l
                        for sub_z in combinations(z, d):
                            sub_z = list(sub_z)
                            _,_,p_value, = ci_test(i, j, sub_z)
                            if p_value >= alpha:
                                skeleton[i, j] = skeleton[j, i] = 0
                                sep_set[(i, j)] = sub_z
                                break

        return skeleton, sep_set


class MovingPC():
    def __init__(self):
        self.pc = PC()

    def calculate_with_pc_moving(self,names, data,timestamps):

        sorted_indices = np.argsort(names)
        Xdata = data[:, sorted_indices]
        names = [names[i] for i in sorted_indices]


        self.pc.learn(Xdata,timestamps)
        learned_graph = nx.DiGraph(self.pc.causal_matrix)
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