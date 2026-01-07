import networkx as nx
from matplotlib import pyplot as plt


class Eventpoint():

    def __init__(self, code, source, timestamp, details=None, type=None):
        """
        This class is used as structure to help in the flow of data for Context generation, essentially each
        instance of such class represent a single data sample from specific source
        (where the code refers to the feature name), The details are used to sotre the actual data, while the type
        refer to the type of data (crucial for event data which must be on of isolated or configuration)

        """
        self.code = code
        self.source = source
        self.timestamp = timestamp
        self.details = details
        self.type = type


class Context():

    def __init__(self, timestamp, CD: dict, CR: dict, details=None):
        '''
        Representation of the Context

        **timestamp**: A timestamp of the context

        **CD**: a dictionary with names the different data sources (names)
        and values equal size of data (time series) that correspond to that source.

        **CR**: dictionary with keys "edges","characterization","interpretation"
            Where edges contain causal relationships of CD data
            characterization is a parallel list with edges which provide an characterization over the relationship (increase,decrease,uknown)

        **details**:
            String with details from user
        '''

        self.timestamp = timestamp
        self.CD = CD
        self.CR = CR
        self.details = details

    @classmethod
    def context_from_dict(cls, contextdict: dict, details=None):
        '''

        contextdict: Which contain timestamp,
         multiple keys of data sources (each one a time-series) to build CD
        and "edges","characterization" to build CR
        details:
            String with details from user
        '''

        timestamp = contextdict["timestamp"]

        CD = {}

        CR = {"edges": contextdict["edges"], "characterization": contextdict["characterization"]}
        if "interpretation" in contextdict.keys():
            CR["interpretation"] = contextdict["interpretation"]
        else:
            CR["interpretation"] = []
        for key in contextdict:
            if key not in CR.keys() and key != "timestamp":
                CD[key] = contextdict[key]

        return cls(timestamp, CD, CR, details)

    def to_dict(self):
        contextdict = {}

        contextdict["edges"] = self.CR["edges"]
        contextdict["interpretation"] = self.CR["interpretation"]
        contextdict["characterization"] = self.CR["characterization"]
        for key in self.CD.keys():
            contextdict[key] = self.CD[key]
        contextdict["timestamp"] = self.timestamp
        contextdict["details"] = self.details

        return contextdict

    def plotCD(self):
        alldata_names = [name for name in self.CD.keys()]
        alldata_data = [self.CD[key] for key in alldata_names]
        alldata_names = [nam for dat, nam in zip(alldata_data, alldata_names) if dat is not None]
        alldata_data = [dat for dat in alldata_data if dat is not None]

        if len(alldata_data) == 1:
            fig, ax = plt.subplots(len(alldata_data), 1)
            ax.plot(alldata_data[0], ".")
            ax.title(alldata_names[0])
        else:
            fig, ax = plt.subplots(len(alldata_data), 1)
            # print(alldata_data)
            for i in range(len(alldata_data)):
                ax[i].plot(alldata_data[i], ".")
                ax[i].set_title(alldata_names[i])

    def plotRD(self):
        fig, ax = plt.subplots(1, 2)
        # plt.subplots(211)
        G = nx.DiGraph()
        # print(self.CR["interpretation"])
        # Add edges to the graph
        G.add_edges_from(self.CR["edges"])
        pos = nx.spring_layout(G)  # Positions for all nodes
        nx.draw(G, pos, ax=ax[0], with_labels=True, font_weight='bold', node_size=1500, node_color='skyblue',
                font_size=10,
                edge_color='black', linewidths=3, arrowsize=20)
        # plt.subplots(212)
        for i, pair in enumerate(self.CR["interpretation"]):
            print(f"{i + 1}) {pair[0]} : {pair[3]}")
            ax[1].scatter(x=[10], y=[(i + 1) * 10])
            ax[1].scatter(x=[30], y=[(i + 1) * 10])
            ax[1].text(x=10, y=(i + 1) * 10, s=f"{i + 1}) {pair[0]} : {pair[3]}")

    def plot(self):
        self.plotCD()
        self.plotRD()
        plt.tight_layout()
        plt.show()
