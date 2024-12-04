

class map_categorical_to_continuous:
    """
    Wrapper Class for mapping categorical to continuous time-series.

    """
    def __init__(self):
        pass
    def map(self,target_series, occurrences, name):
        """
        This method is used to generate context time series of categorical type.

        The way we do this is by generating a time series of each different category, by creating a zero and one series,
        similar to "isolated" type, by filling ones in the timestamps that each category appears. Finally, we create an
        additional series with the name state_{name}, having zeros until the occurrence of the last category, and filled
        with ones afterward.

        **Parameters**:

        **target_series**: Used to align sample rate.

        **occurrences**: a list of tuple with timestamps and categorical value, refering to the observed value of a
        categorical source.

        **name**: name of the categorical source.


        **return**: A list of time-series to populate CD part of the context.
        """
        vector = [[0] for i in range(len(target_series))]
        pos = 0
        unique_categories = set([occ[0] for occ in occurrences])
        # this is to aling the series in case of different sample Rate
        for i in range(len(target_series)):
            timestamp = target_series[i][1]
            current_pos = pos
            for q in range(pos, len(occurrences)):
                if occurrences[q][1] > timestamp:
                    current_pos = q
                    break
            # no data found

            if current_pos == pos:
                # if no data in betwwen values use the previus value
                if i > 0:
                    vector[i] = [occurrences[-1][0]]
                # if no data until i timestamp use the first occurence as value
                else:
                    vector[i] = [occurrences[0][0]]
            # if multiple values in between two timestamps use the last as value
            else:
                dataInBetween = [value for value, time in occurrences[pos:current_pos]]
                vector[i] = [v for v in set(dataInBetween)]
            # if no other occurrences just repeat the last value
            if current_pos == len(occurrences):
                for k in range(i + 1, len(vector)):
                    vector[k] = [occurrences[-1][0]]
                break
            pos = current_pos
        all_vectors = []
        all_names = []
        # one-hot encoding of unique categories
        for value in unique_categories:
            in_vector = [1 if value in v else 0 for v in vector]
            if len(set(in_vector)) == 1:
                in_vector[0] = 0
            all_vectors.append(in_vector)
            all_names.append(f"{value}_{name}")
        # create of the state variable
        state_vector = [0 for i in range(len(target_series))]
        lastv = occurrences[-1][0]
        ## not stable
        for i in range(len(state_vector) - 1, -1, -1):
            if lastv in vector[i]:
                state_vector[i] = 1
            else:
                break
        all_vectors.append(state_vector)
        all_names.append(f"state_{name}")
        return all_vectors, all_names

class map_configuration_to_continuous:
    """
        Wrapper Class for mapping configuration events (defined as events with constant impact) to continuous time-series.

    """
    def __init__(self):
        pass
    def map(self,target_series, occurrences,name):
        """
         Configuration events, refers to configuration changes or events that alter the state of the monitored asset.
          To transform these events into continuous signals, we start with a series of 0s, and after each occurrence of
          such an event, we add 1 to all the positions after the occurrence's timestamp

         **Parameters**:

        **target_series**: Used to align sample rate of the continuous series.

        **occurrences**: Contain time stamps of the occurrences of an isolated type source.

        **return**: A binary time series with same size as target_series, that models the occurrences
        of the provided Configuration source, to populate CD part of the context.
        """
        vector = [0 for i in range(len(target_series))]
        for occ in occurrences:
            for q in range(len(target_series)):
                if target_series[q][1] >= occ[1]:
                    for k in range(q, len(vector)):
                        vector[k] += 1
                    break
        ## not stable
        if len(set(vector)) == 1:
            vector[0] = 0
        return [vector],[name]


class map_isolated_to_continuous:
    """
           Wrapper Class for mapping isolated events (defined as events with instant impact) to continuous time-series.

    """
    def __init__(self):
        pass
    def map(self,target_series, occurrences,name):
        """
         Isolated events are discrete events that have an immediate impact on the behavior of the asset.
         To transform such events into a continuous representation, we start with a series of 0s as an initial signal
         and assign 1 to the position corresponding to the timestamps of the events. If the event timestamp does
         not match any target_series timestamps, it is mapped to the closest timestamp in target_series.

         **Parameters**:

        **target_series**: Used to align sample rate of the continuous series.

        **occurrences**: Contain time stamps of the occurrences of an isolated type source.

        **return**: A binary time series with same size as target_series, that models the occurrences
        of the provided isolated source, to populate CD part of the context.
        """
        vector = [0 for i in range(len(target_series))]
        for occ in occurrences:
            for q in range(len(target_series)):
                if target_series[q][1] > occ[1]:
                    vector[q] = 1
                    break
        return [vector],[name]

class map_univariate_to_continuous:
    """
        Wrapper Class for mapping univariate time-series (defined as events with instant impact) to continuous time-series with same frequency as a target time-series.
    """
    def __init__(self):
        self.existing_results={}
    def map(self,target_series, occurrences,name):
        """
        For continuous data sources, we simply collect the values within the time window.
        Although the time window is the same for all sources, each source may have a different sample rate.
        To create a signal of the same size as target_series, we perform mean aggregation if a source has a higher
        sample rate than target_series, using the mean value of the data between each timestamp of the target_series.

         **Parameters**:

        **target_series**: Used to align sample rate of the continuous series.

        **occurrences**: The univariate time series.

        **return**: A time series with same size as target_series, to populate CD part of context.
        """

        if name not in self.existing_results.keys():
            self.existing_results[name]=[]
        # position of first timestamp of target series in aggregated data
        spos=-1
        for tup in self.existing_results[name]:
            spos += 1
            if tup[1]>=target_series[0][1]:
                break

        self.existing_results[name]=self.existing_results[name][spos:]

        pos = 0
        if len(self.existing_results[name]) > 0:
            for q in range(len(occurrences)):
                pos = q
                if occurrences[q][1] > self.existing_results[name][-1][1]:
                    break

        vector = [tup[0] for tup in self.existing_results[name]]+[0 for i in range(len(target_series)-len(self.existing_results[name]))]

        for i in range(len(self.existing_results[name]),len(target_series)):
            timestamp = target_series[i][1]
            current_pos = pos
            for q in range(pos, len(occurrences)):
                if occurrences[q][1] > timestamp:
                    current_pos = q
                    break
            if i==len(target_series)-1:
                current_pos=len(occurrences)+1
            # no data found

            if current_pos == pos:
                # if no data in betwwen values use the previus value
                if i > 0:
                    vector[i] = vector[i - 1]
                # if no data until i timestamp use the first occurence as value
                else:
                    vector[i] = occurrences[0][0]
            # if multiple values in between two timestamps use the mean of them as value
            else:
                dataInBetween = [value for value, time in occurrences[pos:current_pos]]
                vector[i] = sum(dataInBetween) / len(dataInBetween)
            # if no other occurrences just repeat the last value
            if current_pos == len(occurrences):
                for k in range(i + 1, len(vector)):
                    vector[k] = vector[k - 1]
                break
            pos = current_pos
        self.existing_results[name]=[(v,tup[1]) for v,tup in zip(vector,target_series)]
        return [vector],[name]










