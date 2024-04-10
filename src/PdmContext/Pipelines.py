from PdmContext.ContextGeneration import ContextGenerator


class ContextAndClustering():

    def __init__(self, Clustring_object, context_generator_object: ContextGenerator):
        """
                This class build a pipeline of Cntext generator and Clustering technique running immediately in the results of Context Generator

                **Parameters**:

                **Clustring_object**: The clustering technique to use to cluster the created context from Context generator
                The class of the clustering technique must implement the method add_sample_to_cluster(context: Context)


                 **context_generator_object**: A PdmContext.ContextGeneration import ContextGenerator object
                """

        self.clustering = Clustring_object
        self.Contexter = context_generator_object

    def collect_data(self, timestamp, source, name, value=None, type="Univariate"):
        """
        Call the PdmContext.ContextGeneration.ContextGenerator and pass the result to clustering technique

        **Parameters**:

        **timestamp**:  The timestamp of the arrived value

        **source**: The source of the arrived value

        **name**: The name (or identifier) of the arrived value

        **value**: The value (float), in case this is None the arrived data is considered as event

        **type**: the type of the data can be one of "isolated","configuration" when no value is passed

        **return**: PdmContext.utils.structure.Context object when the data name match to the target name or None.
        """

        contextTemp = self.Contexter.collect_data(timestamp, source, name, value=value, type=type)
        if contextTemp is not None:
            self.clustering.add_sample_to_cluster(contextTemp)

        return contextTemp


class ContextAndClusteringAndDatabase():

    def __init__(self, context_generator_object: ContextGenerator, Clustring_object, databaseStore_object):
        """
                This class build a pipeline of Cntext generator and Clustering technique running immediately in the
                results of Context Generator, and storing context results to database

                **Parameters**:

                **context_generator_object**: A PdmContext.ContextGeneration import ContextGenerator object

                **Clustring_object**: The clustering technique to use to cluster the created context from Context generator
                The class of the clustering technique must implement the method add_sample_to_cluster(context: Context)
                databaseStore_object: Class which implement insert_record(date : pd.datetime,target: str,context : Context, metadata: str)


                **databaseStore_object**: An object of database connection from PdmContext.utils.dbconnector
                """

        self.clustering = Clustring_object
        self.Contexter = context_generator_object
        self.database = databaseStore_object

    def collect_data(self, timestamp, source, name, value=None, type="Univariate"):
        """
        Call the PdmContext.ContextGeneration.ContextGenerator and pass the result to clustering technique

        **Parameters**:

        **timestamp**:  The timestamp of the arrived value

        **source**: The source of the arrived value

        **name**: The name (or identifier) of the arrived value

        **value**: The value (float), in case this is None the arrived data is considered as event

        **type**: the type of the data can be one of "isolated","configuration" when no value is passed

        **return**: PdmContext.utils.structure.Context object when the data name match to the target name or None.
        """

        contextTemp = self.Contexter.collect_data(timestamp, source, name, value=value, type=type)
        if contextTemp is not None:
            self.clustering.add_sample_to_cluster(contextTemp)
            self.database.insert_record(timestamp, name, contextTemp)
        return contextTemp


class ContextAndDatabase():

    def __init__(self, context_generator_object: ContextGenerator, databaseStore_object):
        """
                This class build a pipeline of Cntext generator and storing results to database

                **Parameters**:

                **context_generator_object**: A PdmContext.ContextGeneration import ContextGenerator object

                **databaseStore_object**: Class which implement insert_record(date : pd.datetime,target: str,context : Context, metadata: str)

                """

        self.Contexter = context_generator_object
        self.database = databaseStore_object

    def collect_data(self, timestamp, source, name, value=None, type="Univariate"):
        """
        Call the PdmContext.ContextGeneration.ContextGenerator and pass the result to clustering technique

        **Parameters**:

        **timestamp**:  The timestamp of the arrived value

        **source**: The source of the arrived value

        **name**: The name (or identifier) of the arrived value

        **value**: The value (float), in case this is None the arrived data is considered as event

        **type**: the type of the data can be one of "isolated","configuration" when no value is passed

        **return**: PdmContext.utils.structure.Context object when the data name match to the target name or None.
        """

        contextTemp = self.Contexter.collect_data(timestamp, source, name, value=value, type=type)
        if contextTemp is not None:
            self.database.insert_record(timestamp, name, contextTemp)
        return contextTemp
