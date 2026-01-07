import numpy as np

import pandas as pd
from PdmContext.utils.structure import Eventpoint, Context
from PdmContext.utils.causal_discovery_functions import calculate_with_pc
from PdmContext.utils.showcontext import show_context_list

class ContextGenerator:

    def __init__(self, target, context_horizon="8 hours",
                 Causalityfunct=calculate_with_pc,
                 mapping_functions=None,debug=False):
        """
        This Class handle the Context Generation. It keeps an internal buffer to build the context for a target series based
        on the provided context_horizon. All data are passed though the collect_data method, which return a corresponding
        Context when target data are passed.


        **Parameters**:

        **target**: The name of the target source, which will be used as the baseline in order to map different samples
         rate to that of the target sample rate.

        **context_horizon**: The time period to look back for context data, the form of that parameter is "8 hours"

        **Causalityfunct**: the causality discovery method to use to produce causal relationships between context data,
            This must be a function with parameters two equal size lists, one with names and the other
            with data (a list of list or 2D numpy array).

        **mapping_functions**: Dictionary used to associate each type with a mapping function.

        User can use this dictionary to define his own mapping function and types of sources

        Default value None: use default mappers for types: isolated,configuration, categorical and univariate

        Default Sources Types Supported (in case of mapping_functions is None):

            1) Continuous type (those that have some kind of arithmetic value)

            2) Discrete events (without value) , where one of the type isolated or configuration or categorical must be assigned
                A guide on how to specify the type is, that events which assumed to have impact only on their occurrence, are called
                isolated, while others that are related to some kind of configuration with more permanent impact, are called configuration.
                Categorical values can be defined as categorical type
                Essentially the type of the events define the way that will be transformed to real values time-series.

        **debug**: If it runs on debug mode
        """
        self.debug = debug
        self.target = target

        self.causality_discovery = Causalityfunct
        self.buffer = [] # buffer 0: timestamp, 1: code, 2: value, 3:type

        # helpers
        self.type_of_series = {}
        self.horizon = context_horizon.split(" ")[0]
        if len(context_horizon.split(" ")) == 1:
            self.horizon_time = "hours"
        else:
            if context_horizon.split(" ")[1] in ["days", "hours", "minutes", "seconds"]:
                self.horizon_time = context_horizon.split(" ")[1]
            else:
                assert False, "Time horizon must be either a single number or in form of \"8 hours\" where acceptable time frames are hours,days,minutes,seconds"

        self.horizon = int(self.horizon)

        self.default_usage=False
        if mapping_functions is None:
            from PdmContext.utils.mapping_functions import map_categorical_to_continuous, \
                map_configuration_to_continuous, map_isolated_to_continuous, map_univariate_to_continuous
            self.default_usage=True
            self.mapping_functions={
                "Univariate":map_univariate_to_continuous(),
                "isolated": map_isolated_to_continuous(),
                "configuration": map_configuration_to_continuous(),
                "categorical":map_categorical_to_continuous()
            }
        else:
            self.mapping_functions=mapping_functions

    def collect_data(self, timestamp, source, name, value=None, type="Univariate", replace=[]):
        '''
        This method is used when data are passed iteratively, and stored in buffer
        When data of target source arrive, a corresponding context is produced.
        Sources can be of different sample rate (all sources are mapped to the targets sample rate when context is produced)

        Default Sources Supported (in case of mapping_functions is None) :

        1) Continuous type (those that have some kind of arithmetic value)

        2) Discrete events (without value) , where one of the type isolated or configuration or categorical must be assigned
            A guide on how to specify the type is, that events which assumed to have impact only on their occurrence, are called
            isolated, while others that are related to some kind of configuration with more permanent impact, are called configuration.
            Categorical values can be defined as categorical type
            Essentially the type of the events define the way that will be transformed to real values time-series.


        **Parameters**:

        **timestamp**:  The timestamp of the arrived value

        **source**: The source of the arrived value

        **name**: The name (or identifier) of the arrived value

        **value**: The value (float), in case this is None the arrived data is considered as event

        **type**: the type of the data associated with the mapping function

        **replace**: a list of tuples, which define a replacement policy. If a tuple (e,b) exist in replace list, then
        when a b event is inserted, it will replace an e event with the same timestamp (if exists).

        **return**:  structure.Context object when the data name match to the target name or None.
        '''

        if value is None:
            if type not in self.mapping_functions.keys():
                assert False, f"The type must be defined as one of mapping functions types: {self.mapping_functions.keys()} when no value is passed"
        eventpoint = Eventpoint(code=name, source=source, timestamp=timestamp, details=value, type=type)
        self.add_to_buffer(eventpoint, replace)
        if self.target == name or self.target == f"{name}@{source}":
            contextobject = self.generate_context(e=eventpoint)
            return contextobject
        else:
            return None

    def replace_buffer_with_event(self,pos,event):
        self.buffer[pos][0] = pd.to_datetime(event.timestamp)
        self.buffer[pos][1] = f"{event.code}@{event.source}"
        self.buffer[pos][2] = event.details
        self.buffer[pos][3] = event.type

    def add_to_buffer(self, e: Eventpoint, replace=None):
        """
        Adds an Event point to the buffer (keeping the buffer time ordered)
        """
        if e.type is None:
            e.type="Univariate"
        if len(self.buffer)>0:
            if e.timestamp<self.buffer[0][0]: # in case of very late data, memory safe.
                return
        if replace is None:
            replace = []
            index = len(self.buffer)
            for i in range(len(self.buffer) - 1, 0, -1):
                if self.buffer[i][0] < e.timestamp:
                    index = i + 1
                    break
        else:
            index = len(self.buffer)
            for i in range(len(self.buffer) - 1, 0, -1):
                # check for replacement
                if self.buffer[i][0] == e.timestamp:
                    for rep in replace:
                        if self.buffer[i][1] == rep[0] and e.code == rep[1]:
                            self.replace_buffer_with_event(i,e)
                            return
                    index = i + 1
                if self.buffer[i][0] < e.timestamp:
                    index = i + 1
                    break
        if index == len(self.buffer) and index!=0:
            self.buffer.append([pd.to_datetime(e.timestamp),f"{e.code}@{e.source}",e.details,e.type])
            pos=0
            for i in  range(len(self.buffer)):
                if self.buffer[i][0]>self.buffer[-1][0]-pd.Timedelta(self.horizon, self.horizon_time):
                    pos=i
                    break
            if pos>0:
                self.buffer=self.buffer[pos:]
        else:
            self.buffer = self.buffer[: index] + [[pd.to_datetime(e.timestamp),f"{e.code}@{e.source}",e.details,e.type]] + self.buffer[index:]
        if e.type is not None:
            self.type_of_series[f"{e.code}@{e.source}"] = e.type
    def generate_context(self, e: Eventpoint, buffer=None):
        """
        Generate context

        **Parameters**:

        **e**: Eventpoint related to the last target's data.

        **buffer**: A list with all Eventpoints in the time horizon
        """

        contextcurrent, target_series_name = self.create_context(e, buffer)

        return contextcurrent

    def create_context(self, current: Eventpoint, buffer):
        """
        Transform the data collected to the buffer in to suitable form and generates the CD part of the context along with
        the edges and characterizations:

        **Steps:**

        **Create CD**: Parallel continuous representations of all different sources in the buffer.
        This step involves, matching the different sample rates of different sources to that of the target.
        Transform the Event sources to continuous representation.

        **Calculate Causality edges**: Perform Causal Discovery using the causality function, to create edges
        (part of CR of the Context)

        **Tag each edge with characterization**: for each (a,b) in the edges, a characterization of (unknown, decrease,
         increase), based on the type of the a.

         **Parameters**:

         **current**: The current Event of target's data which triger the context creation

         **buffer**: ordered list with Eventpoint of all sources.

         **return**: Context object.
        """
        # start = time.time()
        # df with ,dt,code,source,value

        # Keep only last horizon events
        last = self.buffer[-1]
        pos = len(self.buffer) - 1
        horizon_window=pd.Timedelta(self.horizon, self.horizon_time)
        for i in range(len(self.buffer)):
            if self.buffer[i][0] >= (last[0] - horizon_window):
                pos = i
                break
        self.buffer = self.buffer[pos:]

        # end=time.time()
        # print(f"find position on buffer: {end-start}")

        # start = time.time()
        if buffer is not None:
            dataforcontext = buffer
        else:
            dataforcontext=self.buffer
        # datatodf = [[pd.to_datetime(e.timestamp) for e in dataforcontext],
        #             [str(e.code) for e in dataforcontext],
        #             [str(e.source) for e in dataforcontext],
        #             [e.details for e in dataforcontext],
        #             [e.type for e in dataforcontext]]
        npcontext=np.array(dataforcontext)
        # npcontext = np.array(datatodf)
        # npcontext = npcontext.T

        npcontext = self.Numpy_preproccess(npcontext)

        allcodes = np.unique(npcontext[:, 1])
        allcodes = [code for code in allcodes]
        allcodes = set(allcodes)

        # for uncode in allcodes:
        #     for qq in range(len(npcontext)):
        #         if uncode in npcontext[qq][1]:
        #             self.type_of_series[uncode] = npcontext[qq][3]
        #             break

        ## build target series
        target_series_name, target_series = self.build_target_series_for_context(current, npcontext)

        ## create series for each source (alldata)
        alldata = self.create_continuous_representation(target_series_name, target_series, npcontext,
                                                        self.type_of_series, allcodes,self.mapping_functions)
        # end = time.time()
        # print(f"Create series: {end-start}")

        storing = self.calculate_edges(alldata, current.timestamp,[t[1] for t in target_series])

        storing["characterization"]=[]
            # print(f"Calculate edges: {end - start}")
        # print("========================")

        contextpbject = Context.context_from_dict(storing)
        return contextpbject, target_series_name

    def calculate_edges(self, alldata, timestamp,timestamps):
        """
        Formulate the data in appropriate form to call self.calculate_causality
        which return the edges for the context.

        **Parameters**:

        **alldata**: a 2D numpy array with all series data (equivalent to the CD of the Context)

        **timestamp**: timestamp of the context from which the edges are calculated.

        **return**: a dictionary with 'edges' key (containing the calculated edges after Causality discovery).
        """
        # start = time.time()
        storing = {}
        storing["timestamp"] = timestamp


        for nn in alldata:
            storing[nn[0]] = nn[1]

        # For context with more than two series calculate PC casualities
        alldata_data=[]
        alldata_names=[]
        for nn in alldata:
            if nn[1] is not None and len(set(nn[1])) > 1:
                alldata_data.append(nn[1])
                alldata_names.append(nn[0])
        count=len(alldata_data)
        if count > 1 and len(alldata[0][1]) > 5:

            edges = self.calculate_causality(np.column_stack(alldata_data), alldata_names,timestamps)

            if edges is None:
                singleedges = []
            else:
                singleedges = edges
            # print(edges)
            storing["edges"] = singleedges
            return storing
        storing["edges"] = []
        return storing

    def create_continuous_representation(self, target_series_name, target_series, df, type_of_series, allcodes,mapping_functions):
        """
        This method handles the creation of continuous representation for all type of sources observed in context,
        and is the first part for creating the CD of the context.

        Based on the type of each source, call the appropriate method to create the continuous representaiton.

        **Parameters**:

        **target_series_name**: The name of the target series.

        **target_series**: Used to align sample rate.

        **type_of_series**: A dictionary to define the type of the different sources (the type can be Isolated, Configuration, Categorical and Univariate).

        **allcodes**: Contain all the names for the sources we want to build the context.

        **mapping_functions**: Dictionary used to associate each type with a mapping function.

        **return**: The CD part of the context.
        """

        windowvalues = df  # .values
        alldata = []
        alldata.append((target_series_name, [tag[0] for tag in target_series]))
        for name in allcodes:
            # already calculated in targetseries.
            if target_series_name in name:
                continue

            # detect the occurancies
            occurrences = [(value, time) for code, value, time in
                           zip(windowvalues[:, 1], windowvalues[:, 2], windowvalues[:, 0]) if name in code]
            # occurrences=self.select_values(windowvalues,name)

            if len(occurrences) == 0:
                vector = None
                alldata.append((name, vector))
            elif type_of_series[name] in mapping_functions.keys():
                mapper = mapping_functions[type_of_series[name]]
                vectors, names = mapper.map(target_series, occurrences, name)
                for in_vector, new_name in zip(vectors, names):
                    if max(in_vector) == 0 and min(in_vector) == 0:
                        vector = None
                    else:
                        vector = in_vector
                    if new_name not in self.type_of_series.keys():
                        self.type_of_series[new_name] = "configuration"
                    alldata.append((new_name, vector))
            else:
                assert False,f" No mapping function defined for type {type_of_series[name]}"

        return alldata


    def select_values(self,windowvalues,series_name):
        codes = windowvalues[:, 1]
        values = windowvalues[:, 2]
        times = windowvalues[:, 0]

        mask = np.char.find(codes.astype(str), series_name) >= 0
        target_series = np.column_stack((values[mask], times[mask]))
        return target_series

    def build_target_series_for_context(self, current: Eventpoint, df: np.ndarray):
        target_series_name = current.code
        windowvalues = df  # .values

        target_series=[(value, time) for code, value, time in
         zip(windowvalues[:, 1], windowvalues[:, 2], windowvalues[:, 0]) if target_series_name in code]
        # target_series=self.select_values(windowvalues,target_series_name)

        return target_series_name, target_series

    def Numpy_preproccess(self, npcontext):
        npcontext = np.where(npcontext == "nan", None, npcontext)

        return npcontext



    def calculate_causality(self, dataor, names,timestamps):

        data = np.array(dataor)
        edges = self.causality_discovery(names, data,timestamps)

        return edges

    def plot(self,contexts, filteredges=None,char=True):
        if filteredges is None:
            filteredges = [["", "", ""]]
        show_context_list(contexts, self.target, filteredges=filteredges,char=char)

