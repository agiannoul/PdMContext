from PdmContext.ContextGeneration import ContextGenerator
from PdmContext.utils.causal_discovery_functions import calculatewithPc


class ContextAndClustering():

    def __init__(self,Clustring_object,target,context_horizon="8",Causalityfunct=calculatewithPc,debug=False):
        """
                This class build a pipeline of Cntext generator and Clustering technique running immediately in the results of Context Generator

                :param Clustring_object: The clustering technique to use to cluster the created context from Context generator
                The class of the clustering technique must implement the method add_sample_to_cluster(context: Context)
                :param target: The name of the target source, which will be used as the baseline in order to map different samples
                 rate to that of the target sample rate.
                :param context_horizon: The time period to look back for context data, the form of that parameter is "8 hours"
                :param Causalityfunct: the causality discovery method to use to produce causal relationships between context data,
                    This must be a function with parameters two equal size lists, one with names and the other
                    with data (a list of list or 2D numpy array).
                :param debug: If it runs on debug mode
                """


        self.clustering=Clustring_object
        self.Contexter=ContextGenerator(target,context_horizon=context_horizon,Causalityfunct=Causalityfunct,debug=debug)


    def collect_data(self,timestamp,source,name,value=None,type="Univariate"):
        """
        Call the PdmContext.ContextGeneration.ContextGenerator and pass the result to clustering technique

        :param timestamp:  The timestamp of the arrived value
        :param source: The source of the arrived value
        :param name: The name (or identifier) of the arrived value
        :param value: The value (float), in case this is None the arrived data is considered as event
        :param type: the type of the data can be one of "isolated","configuration" when no value is passed
        :return: PdmContext.utils.structure.Context object when the data name match to the target name or None.
        """

        contextTemp=self.Contexter.collect_data(timestamp,source,name,value=None,type="Univariate")
        if contextTemp is not None:
            self.clustering.add_sample_to_cluster(contextTemp)

        return contextTemp


class ContextAndClusteringAndDatabase():

    def __init__(self,context_generator_object: ContextGenerator,Clustring_object,databaseStore_object):
        """
                This class build a pipeline of Cntext generator and Clustering technique running immediately in the
                results of Context Generator, and storing context results to database

                :param context_generator_object: A PdmContext.ContextGeneration import ContextGenerator object
                :param Clustring_object: The clustering technique to use to cluster the created context from Context generator
                The class of the clustering technique must implement the method add_sample_to_cluster(context: Context)
                :param databaseStore_object: Class which implement insert_record(date : pd.datetime,target: str,context : Context, metadata: str)

                """


        self.clustering=Clustring_object
        self.Contexter=context_generator_object
        self.database=databaseStore_object


    def collect_data(self,timestamp,source,name,value=None,type="Univariate"):
        """
        Call the PdmContext.ContextGeneration.ContextGenerator and pass the result to clustering technique

        :param timestamp:  The timestamp of the arrived value
        :param source: The source of the arrived value
        :param name: The name (or identifier) of the arrived value
        :param value: The value (float), in case this is None the arrived data is considered as event
        :param type: the type of the data can be one of "isolated","configuration" when no value is passed
        :return: PdmContext.utils.structure.Context object when the data name match to the target name or None.
        """

        contextTemp=self.Contexter.collect_data(timestamp,source,name,value=value,type=type)
        if contextTemp is not None:
            self.clustering.add_sample_to_cluster(contextTemp)
            self.database.insert_record(timestamp,name,contextTemp)
        return contextTemp

class ContextAndDatabase():

    def __init__(self,context_generator_object: ContextGenerator,databaseStore_object):
        """
                This class build a pipeline of Cntext generator and storing results to database

                :param context_generator_object: A PdmContext.ContextGeneration import ContextGenerator object
                :param databaseStore_object: Class which implement insert_record(date : pd.datetime,target: str,context : Context, metadata: str)

                """


        self.Contexter=context_generator_object
        self.database=databaseStore_object


    def collect_data(self,timestamp,source,name,value=None,type="Univariate"):
        """
        Call the PdmContext.ContextGeneration.ContextGenerator and pass the result to clustering technique

        :param timestamp:  The timestamp of the arrived value
        :param source: The source of the arrived value
        :param name: The name (or identifier) of the arrived value
        :param value: The value (float), in case this is None the arrived data is considered as event
        :param type: the type of the data can be one of "isolated","configuration" when no value is passed
        :return: PdmContext.utils.structure.Context object when the data name match to the target name or None.
        """

        contextTemp=self.Contexter.collect_data(timestamp,source,name,value=value,type=type)
        if contextTemp is not None:
            self.database.insert_record(timestamp,name,contextTemp)
        return contextTemp