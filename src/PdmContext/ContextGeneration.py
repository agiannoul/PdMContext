import numpy as np
import statistics

import pandas as pd
from PdmContext.utils.structure import Eventpoint, Context
from PdmContext.utils.causal_discovery_functions import calculate_with_pc
from PdmContext.utils.showcontext import show_context_list, show_context_interpretations
from PdmContext.utils.mapping_functions import map_categorical_to_continuous, map_configuration_to_continuous,map_isolated_to_continuous,map_univariate_to_continuous
from pathlib import Path
import pickle

class ContextGeneratorBatch():
    def __init__(self,df_data,target,type_of_series,context_horizon="8 hours",Causalityfunct=calculate_with_pc,debug=False,file_path=None):
        """
        This version doesn't support interpretation and is created for faster experiments using Context.

        **df_data** The data to consider in context in form of Data Frame.

        ** target** The name of the target source, which will be used as the baseline in order to map different samples

        **type_of_series**: dictionary which define the type of each series (isolated,configuration or continuous)
         rate to that of the target sample rate.

        **context_horizon** The time period to look back for context data, the form of that parameter is "8 hours"

        **Causalityfunct** the causality discovery method to use to produce causal relationships between context data,
            This must be a function with parameters two equal size lists, one with names and the other
            with data (a list of list or 2D numpy array).

        **debug** If it runs on debug mode

        **file_path** Store the results of context in pickle file, considered only when it is not None.
        """

        self.file_path=file_path
        self.debug=debug
        self.target=target
        self.df_data=df_data


        self.contexts=self._load_contexts()

        self.causality_discovery=Causalityfunct
        self.buffer=[]


        #helpers
        self.type_of_series = type_of_series
        self.horizon = context_horizon.split(" ")[0]
        if len(context_horizon.split(" ")) == 1:
            self.horizon_time = "hours"
        else:
            if context_horizon.split(" ")[1] in ["days", "hours", "minutes", "seconds"]:
                self.horizon_time = context_horizon.split(" ")[1]
            else:
                assert False, "Time horizon must be either a single number or in form of \"8 hours\" where acceptable time frames are hours,days,minutes,seconds"

        self.horizon = int(self.horizon)


        self.interpret_history_pos = 0
        self.context_pos = 0


    def generate_context(self,datetime_index):
        contextpbject, target_series_name,time_index=self.create_context(datetime_index)

        self.contexts[time_index]=contextpbject

        return contextpbject

    def _save_contexts(self):
        if self.file_path is None:
            return
        with open(self.file_path, 'wb') as f:
            pickle.dump(self.contexts, f)

    def _load_contexts(self):
        if self.file_path is None:
            return {}

        my_file = Path(self.file_path)
        if my_file.is_file():
            with open(self.file_path, 'rb') as f:
                return pickle.load(f)
        else:
            return  {}
    def create_context(self, datetime_index):

        start_index = datetime_index - pd.Timedelta(self.horizon, self.horizon_time)

        context=self.df_data.loc[start_index:datetime_index]

        allcodes = np.unique(self.df_data.columns)
        allcodes = [code for code in allcodes]
        allcodes = set(allcodes)

        ## build target series
        target_series_name= self.target
        target_series=context[self.target].values

        ## create series for each source (alldata)
        alldata = []
        for col in context.columns:
            alldata.append((col, context[col].values))
        # end = time.time()
        # print(f"Create series: {end-start}")

        storing = self.calculate_edges(alldata, context.index[-1])
        storing["characterization"] = self.getcaracterize(storing)

        # print(f"Calculate edges: {end - start}")
        # print("========================")

        contextpbject = Context.context_from_dict(storing)
        return contextpbject, target_series_name,context.index[-1]

    def calculate_edges(self,alldata, timestamp):
        #start = time.time()
        storing = {}
        storing["timestamp"] = timestamp

        alldata_names = [nn[0] for nn in alldata]
        alldata_data = [nn[1] for nn in alldata]

        for namedd, datadd in zip(alldata_names, alldata_data):
            storing[namedd] = datadd
        alldata_names = [nn[0] for nn in alldata if nn[1] is not None and len(set(nn[1])) > 1]
        alldata_data = [nn[1] for nn in alldata if nn[1] is not None and len(set(nn[1])) > 1]
        # For context with more than two series calculate PC casualities
        count = len([1 for lista in alldata_data if lista is not None])
        if count > 1:
            if len(alldata[0][1]) > 5:
                # alldata_names = [nn[0] for nn in alldata if nn[1] is not None and len(set(nn[1]))>1]
                # alldata_data = [nn[1] for nn in alldata if nn[1] is not None and len(set(nn[1]))>1]

                #end = time.time()
                #print(f"before {end - start}")
                #start=time.time()

                if len(alldata) <= 1:
                    edges = []
                else:
                    edges = self.calculate_causality(np.column_stack(alldata_data), alldata_names)
                #end=time.time()
                #print(f"actual edge calculation {end-start}")
                if edges is None:
                    singleedges = []
                else:
                    singleedges = edges
                # print(edges)
                storing["edges"] = singleedges
                return storing
        storing["edges"] = []
        return storing

    def calculate_causality(self, dataor, names,timestamps):
        num_time_series = len(dataor)
        data = np.array(dataor)
        edges = self.causality_discovery(names, data,timestamps)
        # edges=self.calculatewithPc(names,data)
        # edges=self.calculatewith_fci(names,data)
        # edges=self.salesforcePC(names,data)
        return edges
    def getcaracterize(self,context):
        edges = context["edges"]
        characterizations = []
        for edge in edges:
            name1 = edge[0]
            name2 = edge[1]
            values1 = context[name1]
            values2 = [float(kati) for kati in context[name2]]

            occurence=len(values1)-1
            for i in range(len(values1)-2,0,-1):
                if values1[i] != values1[-1]:
                    occurence = i+1
                    break
            previusoccurence = 0
            for i in range(occurence-2,0,-1):
                if values1[i] !=  values1[occurence-1]:
                    previusoccurence = i

            if occurence - previusoccurence < 2:  # or len(values2)-occurence<2:
                characterizations.append("uknown")
                continue
            values2before = values2[previusoccurence:occurence]
            # stdv = statistics.stdev(values2before)
            # mean = statistics.stdev(values2before)
            # values2before=[v if v<mean+5*stdv else mean for v in values2before]

            # values2after=values2[occurence:]
            values2after = [values2[-1]]
            # stdv = statistics.stdev(values2after)
            # mean = statistics.stdev(values2after)
            # values2after = [v if v < mean + 3 * stdv else mean for v in values2after]

            stdv = statistics.stdev(values2before)
            if len(values2before) == 0:
                char = "uknown"
            elif statistics.median(values2before) - statistics.median(values2after) > 2 * stdv:
                char = "decrease"
            elif statistics.median(values2after) - statistics.median(values2before) > 2 * stdv:
                char = "increase"
            else:
                char = "uknown"
            characterizations.append(char)
        return characterizations


    def __del__(self):
        if self.file_path is None:
            return
        my_file = Path(self.file_path)
        if my_file.is_file():
            return
        else:
            self._save_contexts()

class ContextGenerator:

    def __init__(self, target, context_horizon="8 hours",
                 Causalityfunct=calculate_with_pc,
                 mapping_functions=None,debug=False,characterize_it=True):
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
        self.characterize_it=characterize_it
        self.debug = debug
        self.target = target

        self.contexts = []

        self.causality_discovery = Causalityfunct
        self.buffer = []

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
        self.interpret_history_pos = 0
        self.context_pos = 0
        self.default_usage=False
        if mapping_functions is None:
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
            contextobject = self.generate_context(e=eventpoint, buffer=self.buffer)
            return contextobject
        else:
            return None

    def add_to_buffer(self, e: Eventpoint, replace=None):
        """
        Adds an Event point to the buffer (keeping the buffer time ordered)
        """
        if e.type is None:
            e.type="Univariate"
        if replace is None:
            replace = []
            index = len(self.buffer)
            for i in range(len(self.buffer) - 1, 0, -1):
                if self.buffer[i].timestamp < e.timestamp:
                    index = i + 1
                    break
        else:
            index = len(self.buffer)
            for i in range(len(self.buffer) - 1, 0, -1):
                # check for replacement
                if self.buffer[i].timestamp == e.timestamp:
                    for rep in replace:
                        if self.buffer[i].code == rep[0] and e.code == rep[1]:
                            self.buffer[i] = e
                            return
                    index = i + 1
                if self.buffer[i].timestamp < e.timestamp:
                    index = i + 1
                    break
        if index == len(self.buffer):
            self.buffer.append(e)
        else:
            self.buffer = self.buffer[: index] + [e] + self.buffer[index:]

    def generate_context(self, e: Eventpoint, buffer):
        """
        Generate context (and interpretation only when mapping function is the default).

        **Parameters**:

        **e**: Eventpoint related to the last target's data.

        **buffer**: A list with all Eventpoints in the time horizon
        """

        contextcurrent, target_series_name = self.create_context(e, buffer)
        if self.default_usage and self.characterize_it:
            contextcurrent=self.create_interpretation(contextcurrent, target_series_name)
        else:
            contextcurrent.CR["interpretation"]=[]
        self.contexts.append(contextcurrent)

        i=0
        pos=0
        while self.contexts[i].timestamp < self.contexts[-1].timestamp - pd.Timedelta(self.horizon, self.horizon_time):
            pos=i
            i+=1
        self.contexts=self.contexts[pos:]
        return contextcurrent

    def create_interpretation(self, contextcurrent: Context, target_series_name):
        """
        This method collect all the contexts in the horizon and calls the interpretation method.

        **Parameters**:

        **contextcurrent**: The Context for which the interpretation is calculated

        **target_series_name**: Then name of the target variable
        """
        causeswindow = self.contexts
        interpr = self.interpret(contextcurrent, causeswindow, target_series_name, self.type_of_series)
        contextcurrent.CR["interpretation"] = interpr
        return contextcurrent

    def interpret(self, context: Context, causeswindow, target, type_of_series):
        """
        This method enhance the CR part of the Context with interpretation relating to the target variable.
        The interpretation are based on the edges extracted from the Casualty discovery.
        For each edge of type (seriesA -> target) in the edges we test if the seriesA interprets
        the target using the following rules:

        **if seriesA is isolated**: We check if its last occurrence is the current timestamp or the previous one.

        **if seriesA is configuration**: We check if the edges (seriesA -> target) appears
        in at least 80% contexts in the horizon (i.e. in causeswindow list).

        **if seriesA is continuous**: Then that means the target is related with seriesA
        and we add it to the interpretation.

        All the interpretations are tagged with a timestamp which refers to the first time
        of appearense of consecutive interpretations. Then the interpretation are sorted using
        this timestamp. This is done to provide a hierarchy to the interpretation since it may be the case,
        that when seriesA cause target , and SeriesB cause target, the oldest one is stronger since the SeriesB may be
        effect of seriesA.

        **Parameters**:

        **context**: Context Object to be interpreted

        **causeswindow**: list with the last horizon contexts

        **target**: the name of the target variable to interpret

        **type_of_series**: dictionary which define the type of each series (isolated,configuration or continuous)
        """

        pairs = [(pair[0], pair[1], car) for pair, car in zip(context.CR['edges'], context.CR['characterization']) if
                 target in pair[1]]

        # pairshop2 = []
        # intermediates=[pair[0] for pair in pairs]
        # allothes=[pair for pair in context.CD.keys() if pair not in intermediates]
        #
        # for inter_pair in intermediates:
        #     for name in allothes:
        #         if (name,inter_pair) in context.CR["edges"]:
        #             pairshop2.append((name,inter_pair))

        interpretation = []
        typeconection = []
        if len(pairs) == 0:
            return []
        for pair in pairs:
            if type_of_series[pair[0]] == "isolated":
                values1 = context.CD[pair[0]]
                if values1[-1] == 1:
                    interpretation.append((pair[0], pair[1], pair[2], context.timestamp))
                elif values1[-2] == 1:
                    temptimestamp = context.timestamp
                    lastcontext = self.contexts[max(len(self.contexts) - 2, 0)]
                    for interpair in lastcontext.CR["interpretation"]:
                        if interpair[0] == pair[0] and interpair[1] == pair[1]:
                            temptimestamp = interpair[3]
                    interpretation.append((pair[0], pair[1], pair[2], temptimestamp))
            elif type_of_series[pair[0]] == "configuration":
                values1 = context.CD[pair[0]]
                occurence = 0
                for q in range(len(values1)):
                    if values1[q] == values1[-1] and values1[q] > 0:
                        occurence = q
                        break
                if occurence == len(values1) - 1:
                    interpretation.append((pair[0], pair[1], pair[2], context.timestamp))
                else:
                    lead = len(values1) - occurence
                    counter = 0
                    leadcontext = causeswindow[-lead:]
                    for conte in leadcontext:
                        try:
                            pos = list(conte.CR["edges"]).index((pair[0], pair[1]))
                        except:
                            pos = -1
                        if pos != -1:
                            if conte.CR["characterization"][pos] == pair[2]:
                                counter += 1
                    if counter >= 0.8 * lead:
                        temptimestamp = context.timestamp
                        lastcontext = self.contexts[max(len(self.contexts) - 2, 0)]
                        for interpair in lastcontext.CR["interpretation"]:
                            if interpair[0] == pair[0] and interpair[1] == pair[1]:
                                temptimestamp = interpair[3]
                        interpretation.append((pair[0], pair[1], pair[2], temptimestamp))
            # for real value series
            elif len(set(context.CD[pair[0]])) > 2:
                temptimestamp = context.timestamp
                lastcontext = self.contexts[max(len(self.contexts) - 2, 0)]
                for interpair in lastcontext.CR["interpretation"]:
                    if interpair[0] == pair[0] and interpair[1] == pair[1]:
                        temptimestamp = interpair[3]
                interpretation.append((pair[0], pair[1], pair[2], temptimestamp))
                continue
        # sort with time
        finterpret = []
        for pair in interpretation:
            finterpret.append((pair[0], pair[1], pair[2], pair[3]))
        finterpret.sort(key=lambda tup: tup[3])  # sorts in place
        return finterpret

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
        for i in range(len(self.buffer)):
            if self.buffer[i].timestamp >= (last.timestamp - pd.Timedelta(self.horizon, self.horizon_time)):
                pos = i
                break
        self.buffer = self.buffer[pos:]

        # end=time.time()
        # print(f"find position on buffer: {end-start}")

        # start = time.time()
        dataforcontext = buffer
        datatodf = [[pd.to_datetime(e.timestamp) for e in dataforcontext],
                    [str(e.code) for e in dataforcontext],
                    [str(e.source) for e in dataforcontext],
                    [e.details for e in dataforcontext],
                    [e.type for e in dataforcontext]]

        npcontext = np.array(datatodf)
        npcontext = npcontext.T

        npcontext = self.Numpy_preproccess(npcontext)

        # end = time.time()
        # print(f"preprocess df: {end - start}")
        ############# create context ############################
        # start = time.time()
        ## collect uniqeu data series
        # allcodes = df['code'].unique()
        allcodes = np.unique(npcontext[:, 1])
        allcodes = [code for code in allcodes]
        allcodes = set(allcodes)

        for uncode in allcodes:
            for qq in range(len(npcontext)):
                if uncode in npcontext[qq][1]:
                    self.type_of_series[uncode] = npcontext[qq][4]
                    break

        ## build target series
        target_series_name, target_series = self.build_target_series_for_context(current, npcontext)

        ## create series for each source (alldata)
        alldata = self.create_continuous_representation(target_series_name, target_series, npcontext,
                                                        self.type_of_series, allcodes,self.mapping_functions)
        # end = time.time()
        # print(f"Create series: {end-start}")

        storing = self.calculate_edges(alldata, current.timestamp,[t[1] for t in target_series])
        if self.characterize_it:
            storing["characterization"] = self.get_characterization(storing)
        else:
            storing["characterization"]=[]
            # print(f"Calculate edges: {end - start}")
        # print("========================")

        contextpbject = Context.context_from_dict(storing)
        return contextpbject, target_series_name

    def get_characterization(self, context):
        """
        This method calculate the characterizations of (a,b) edges to characterize the influence of a in b by one of the
        three characterizations: unknown, decrease, increase.

        The characterization is calculated differently for event and continuous data.
        **Parameters**:

        **context**: dictionary with a kye 'edges' for which a parallel list of characterizations will be created.
        """
        edges = context["edges"]
        characterizations = []
        for edge in edges:
            if self.target in edge[0]:
                char = self.characterize_event_continuous(context, edge)
            elif self.type_of_series[edge[0]] == "isolation" or self.type_of_series[edge[0]] == "configuration":
                char = self.characterize_event_edge(context, edge)
            else:
                char = self.characterize_event_continuous(context, edge)
            characterizations.append(char)
        return characterizations

    def characterize_event_edge(self, context, edge):
        """
        This method characterizes the edge (a,b) when a is isolated or configuration event.
        To do this, it detects the last occurrence of a and split the data of series a and series b based on that.
        Then checks if the median of b after the occurrence is at larger than the median b before
        plus two times the standard deviation of b series before the occurrence.
        If that is true then is characterized as increase. Else it is checked for the opposite, (if it is smaller
        for at least 2 times the standard deviation of b) and characterized as decrease. Otherwise, is characterized as
         unknown.

        **Parameters**:

        **context**: that the edge to be characterized belongs.

        **edge**: A tuple (a,b) where a is isolated or configuration event.

        **return**: A characterization for the edge (unknown, decrease or increase)
        """
        name1 = edge[0]
        name2 = edge[1]
        values1 = context[name1]
        values2 = [float(kati) for kati in context[name2]]

        occurence = len(values1) - 1
        for i in range(len(values1) - 2, 0, -1):
            if values1[i] != values1[-1]:
                occurence = i + 1
                break
        previusoccurence = 0
        for i in range(occurence - 2, 0, -1):
            if values1[i] != values1[occurence - 1]:
                previusoccurence = i

        if occurence - previusoccurence < 2:  # or len(values2)-occurence<2:
            return "unknown"
        values2before = values2[previusoccurence:occurence]
        # stdv = statistics.stdev(values2before)
        # mean = statistics.stdev(values2before)
        # values2before=[v if v<mean+5*stdv else mean for v in values2before]

        # values2after=values2[occurence:]
        values2after = [values2[-1]]
        # stdv = statistics.stdev(values2after)
        # mean = statistics.stdev(values2after)
        # values2after = [v if v < mean + 3 * stdv else mean for v in values2after]

        stdv = statistics.stdev(values2before)
        if len(values2before) == 0:
            char = "unknown"
        elif statistics.median(values2before) - statistics.median(values2after) > 2 * stdv:
            char = "decrease"
        elif statistics.median(values2after) - statistics.median(values2before) > 2 * stdv:
            char = "increase"
        else:
            char = "unknown"
        return char

    def characterize_event_continuous(self, context, edge):
        """
            This method characterizes the edge (a,b) when a is continuous.
            To do this, the delta between the timestamps of the series b is calculated
            (i.e. the difference between current and next timestamps). If the summ of the deltas between b series data, is
            greater than 2 times the standard deviation of the b, then the increase characterization is returned. If the sum
            is lower than 2 times the standard deviation the decreased characterization is returned. Otherwise, the unknown
            characterization is returned.

            **Parameters**:

            **context**: that the edge to be characterized belongs.

            **edge**: A tuple (a,b) where a is isolated or configuration event.

            **return**: A characterization for the edge (unknown, decrease or increase)
        """
        name1 = edge[0]
        name2 = edge[1]
        values1 = [float(kati) for kati in context[name1]]
        values2 = [float(kati) for kati in context[name2]]

        prev = values2[0]
        diff = 0
        for v in values2[1:]:
            diff += (v - prev)

        if len(values2) == 0:
            char = "unknown"
        stdv = statistics.stdev(values2)
        if diff > 2 * stdv:
            char = "increase"
        elif diff < -2 * stdv:
            char = "decrease"
        else:
            char = "unknown"
        return char

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

        alldata_names = [nn[0] for nn in alldata]
        alldata_data = [nn[1] for nn in alldata]

        for namedd, datadd in zip(alldata_names, alldata_data):
            storing[namedd] = datadd

        # For context with more than two series calculate PC casualities
        count = len([1 for lista in alldata_data if lista is not None and len(set(lista)) > 1])
        if count > 1 and len(alldata[0][1]) > 5:
            alldata_names = [nn[0] for nn in alldata if nn[1] is not None and len(set(nn[1])) > 1]
            alldata_data = [nn[1] for nn in alldata if nn[1] is not None and len(set(nn[1])) > 1]

            # end = time.time()
            # print(f"before {end - start}")
            # start=time.time()

            edges = self.calculate_causality(np.column_stack(alldata_data), alldata_names,timestamps)
            # end=time.time()
            # print(f"actual edge calculation {end-start}")
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
                           zip(windowvalues[:, 1], windowvalues[:, 3], windowvalues[:, 0]) if name in code]

            # # Filter rows where `name` is in column 1
            # mask = np.char.find(windowvalues[:, 1].astype(str), name) >= 0
            # # Extract the required columns using NumPy indexing
            # occurrences = list(zip(windowvalues[mask, 3], windowvalues[mask, 0]))

            vector = [0 for i in range(len(target_series))]
            if len(occurrences) == 0:
                vector = [0 for i in range(len(target_series))]
                if max(vector) == 0 and min(vector) == 0:
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




    def build_target_series_for_context(self, current: Eventpoint, df: np.ndarray):
        target_series_name = current.code
        windowvalues = df  # .values
        target_series = [(value, time) for code, value, time in
                         zip(windowvalues[:, 1], windowvalues[:, 3], windowvalues[:, 0]) if target_series_name in code]

        return target_series_name, target_series

    def Numpy_preproccess(self, npcontext):
        npcontext = np.where(npcontext == "nan", None, npcontext)

        mask = [('0_' in code and value is None) for code, value in zip(npcontext[:, 1], npcontext[:, 3])]

        # if isinstance(mask,collections.abc.Sequence)==False:
        #     mask=[mask]
        npcontext[mask, 3] = 0
        mask = [('1_' in code and value is None) for code, value in zip(npcontext[:, 1], npcontext[:, 3])]
        npcontext[mask, 3] = 1
        code_with_source = [f"{code}{source}" if source[0] == "@" else f"{code}@{source}" for code, source in
                            zip(npcontext[:, 1], npcontext[:, 2])]
        npcontext[:, 1] = code_with_source

        return npcontext



    def calculate_causality(self, dataor, names,timestamps):
        num_time_series = len(dataor)
        data = np.array(dataor)
        edges = self.causality_discovery(names, data,timestamps)
        # edges=self.calculatewithPc(names,data)
        # edges=self.calculatewith_fci(names,data)
        # edges=self.salesforcePC(names,data)
        return edges

    def plot(self,contexts, filteredges=None,char=True):
        if filteredges is None:
            filteredges = [["", "", ""]]
        show_context_list(contexts, self.target, filteredges=filteredges,char=char)

    def plot_interpretation(self,contexts, filteredges=None):
        if filteredges is None:
            filteredges = [["", "", ""]]
        show_context_interpretations(contexts, self.target, filteredges=filteredges)
