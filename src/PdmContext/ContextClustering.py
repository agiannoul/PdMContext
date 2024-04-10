from PdmContext.utils.structure import Context
from PdmContext.utils import distances
from matplotlib import pyplot as plt


class DBscanContextStream():


    def __init__(self,cluster_similarity_limit=0.7,min_points=2,distancefunc=None):
        """
        A simple clustering method for context (inspired from streaming DBSCAN)

        Parameters:

        cluster_similarity_limit: The similarity upon two clusters are merged

        min_points:

        distancefunc: The function to use in order to calculate two Contexts' similarity. None value results in
        the usage of default distance function implemented in the class.
        In case of user's distance function, this has to take the two contexts as parameter and

        return a real number in [0,1] indicating their similarity (1 is maximum similarity)
        """

        self.cluster_similarity_limit=cluster_similarity_limit
        self.contexts={}
        self.min_points=min_points
        self.index= {}
        self.label= {}
        self.a=0.5
        self.b=1-self.a
        self.clusters_sets=[]
        self.count=-1
        self.D=[]
        self.Co=[] # coexist

        if distancefunc is None:
            self.distanceTouse=self.distance_default
        else:
            self.distanceTouse = distancefunc

    def cluster_is_similar(self, cluster, id):
        for j in cluster:
            if self.distance(id, j) > self.cluster_similarity_limit:
                return True
        return False

    def distance(self,id1,id2):
        return self.distanceTouse(self.contexts[id1],self.contexts[id2])

    def distance_default(self,c1,c2):
        return distances.distance_cc(c1, c2, self.a, self.b)[0]

    def add_sample_to_cluster(self, context : Context):

        self.count+=1
        self.index[context.timestamp]=self.count
        self.contexts[self.count] = context
        #self.D.append([ self.context_ob.distance_cc(context,self.contexts[i])[0] for i in range(self.count)])
        #self.Co.append([ 1 if self.context_ob.distance_cc(context,self.contexts[i])[0]>self.cluster_similarity_limit else 0 for i in range(self.count)])
        if len(self.clusters_sets)==0:
            self.clusters_sets.append([self.count])
        else:
            similarclusters=[]
            for i,clust in enumerate(self.clusters_sets):
                belong=self.cluster_is_similar(clust, self.count)
                if belong:
                    similarclusters.append(i)
            if len(similarclusters)==0:
                self.clusters_sets.append([self.count])
            elif len(similarclusters)==1:
                self.clusters_sets[similarclusters[0]].append(self.count)
            else:
                new_clusters=[]
                merge_cluster=[]
                for i,clust in enumerate(self.clusters_sets):
                    if i in similarclusters:
                        merge_cluster.extend(clust)
                    else:
                        new_clusters.append(clust)
                new_clusters.append(merge_cluster)
                self.clusters_sets=new_clusters


    # check if timestamp_target has in its neighborhood one of timestamps_query
    def efficient_neighbor(self,timestamp_target,timestamps_query):
        # get the cluster of target:
        idtarget=self.index[timestamp_target]
        cluserid=-1
        for i in range(len(self.clusters_sets)):
            if idtarget in self.clusters_sets[i]:
                cluserid=i
                break
        if cluserid==-1:
            return False,0
        all_query=timestamps_query.copy()
        in_same_cluster=[]
        for id in self.clusters_sets[cluserid]:
            if self.contexts[id].timestamp in all_query:
                in_same_cluster.append(id)
                all_query.remove(self.contexts[id].timestamp)

        if len(in_same_cluster)==0:
            return False,0

        for id in in_same_cluster:
            dist=self.distance(id,idtarget)
            if dist>self.cluster_similarity_limit:
                return True,dist
        return False,0
    def plot(self):
        num=1
        noisyclust=[]
        for clust in self.clusters_sets:
            if len(clust)<=self.min_points:
                noisyclust.extend(clust)
                continue
            self.plotsingleclust(num,clust)
            num+=1

        self.plotsingleclust(num, noisyclust,"noise")
        plt.show()


    def plotsingleclust(self,num,clust,desc=None):
        xaxistimes = [self.contexts[id].timestamp for id in clust]
        yaxistimes = [num for id in clust]
        if len(xaxistimes)==0:
            return
        edges = [self.contexts[id].CR["edges"] for id in clust]
        if len(edges)!=0:
            edgeplot = edges[len(edges) // 2]
        else:
            edgeplot=[]
        toplot = ""
        seen = []
        for ed in edgeplot:
            if (ed[0], ed[1]) not in seen:
                toplot += f"({ed[0]},{ed[1]}),"
                seen.append((ed[1], ed[0]))
                seen.append((ed[0], ed[1]))
        if desc is None:
            plt.text(xaxistimes[len(xaxistimes) // 2], yaxistimes[len(yaxistimes) // 2] + 0.1, toplot, fontsize=8)
        else:
            plt.text(xaxistimes[len(xaxistimes) // 2], yaxistimes[len(yaxistimes) // 2] + 0.1, desc, fontsize=8)

        plt.scatter(xaxistimes, yaxistimes)