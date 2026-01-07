class map_base():
    def __init__(self):
        self.existing_results = {}
        self.existing_timestamps = {}
        self.last_occurance = {}

    def find_pos(self, name, occurrences, target_series):
        """
        This method is used to find the position of the first timestamp of target_series in aggregated data.

        **Parameters**:
        **name**: name of the source.
        **occurrences**: a list of tuple with timestamps and categorical value, referring to the observed value of a
        categorical source.
        **target_series**: Used to align sample rate.
        **return**: The position of the first timestamp of target_series in aggregated data.
        """
        if name not in self.existing_results.keys():
            self.existing_results[name] = []
            self.existing_timestamps[name] = []
            self.last_occurance[name] = None
            pos_occ = 0
            pos_target = 0
        else:
            pos_occ = len(occurrences)
            for i in range(len(occurrences) - 1, -1, -1):
                pos_occ = i
                if occurrences[i][1] <= self.last_occurance[name]:
                    pos_occ += 1
                    break
            pos_target = len(target_series)
            for i in range(len(target_series) - 1, -1, -1):
                pos_target = i
                if target_series[i][1] <= self.existing_timestamps[name][-1]:
                    pos_target += 1
                    break
        self.last_occurance[name] = occurrences[-1][1]
        return pos_target, pos_occ

    def trim(self, name, target_series):
        self.existing_timestamps[name] = self.existing_timestamps[name][-len(target_series):]
        self.existing_results[name] = self.existing_results[name][-len(target_series):]


class map_categorical_to_continuous:
    """
    Wrapper Class for mapping categorical to continuous time-series.

    """

    def __init__(self):
        pass

    def map(self, target_series, occurrences, name):
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


class map_configuration_to_continuous(map_base):
    """
        Wrapper Class for mapping configuration events (defined as events with constant impact) to continuous time-series.

    """

    def __init__(self):
        super().__init__()
        self.existing_results = {}
        self.base = {}

    def map(self, target_series, occurrences, name):
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
        if name not in self.base.keys():
            self.base[name] = 0
        pos_target, pos_occ = self.find_pos(name, occurrences, target_series)

        for i in range(pos_target, len(target_series)):
            if pos_occ < len(occurrences) and occurrences[pos_occ][1] <= target_series[i][1]:
                self.base[name] += 1
                self.base[name] = self.base[name] % (len(target_series) + 1)
                self.existing_results[name].append(self.base[name])
                self.existing_timestamps[name].append(target_series[i][1])
                # finde next occurance after target_series[ti][1] timestamp
                while occurrences[pos_occ][1] <= target_series[i][1] and pos_occ < len(occurrences) - 1:
                    pos_occ += 1
                if occurrences[pos_occ][1] <= target_series[i][
                    1]:  # no other occurencies after target_series[ti][1] timestamp
                    for innert1 in range(i + 1, len(target_series)):
                        self.existing_results[name].append(self.base[name])
                        self.existing_timestamps[name].append(target_series[innert1][1])
                    break
            else:
                self.existing_results[name].append(self.base[name])
                self.existing_timestamps[name].append(target_series[i][1])
        self.trim(name, target_series)

        return [self.existing_results[name]], [name]


class map_configuration_to_continuous_deprecated:
    """
        Wrapper Class for mapping configuration events (defined as events with constant impact) to continuous time-series.

    """

    def __init__(self):
        pass

    def map(self, target_series, occurrences, name):
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
        ci = 0
        base = 0
        for ti in range(len(target_series)):
            if occurrences[ci][1] <= target_series[ti][1]:
                base += 1
                # finde next occurance after target_series[ti][1] timestamp
                while occurrences[ci][1] <= target_series[ti][1] and ci < len(occurrences) - 1:
                    ci += 1
                if occurrences[ci][1] <= target_series[ti][
                    1]:  # no other occurencies after target_series[ti][1] timestamp
                    for innert1 in range(ti, len(target_series)):
                        vector[innert1] = base
                    break
            vector[ti] = base
        ## not stable
        if len(set(vector)) == 1:
            vector[0] = 0
        return [vector], [name]


class map_isolated_to_continuous(map_base):
    """
    Wrapper Class for mapping isolated events (defined as events with instant impact) to continuous time-series.
    """

    def __init__(self):
        super().__init__()

    def map(self, target_series, occurrences, name):
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
        pos_target, pos_occ = self.find_pos(name, occurrences, target_series)

        for i in range(pos_target, len(target_series)):
            if pos_occ < len(occurrences) and occurrences[pos_occ][1] <= target_series[i][1]:
                self.existing_results[name].append(1)
                self.existing_timestamps[name].append(target_series[i][1])
                # finde next occurance after target_series[ti][1] timestamp
                while occurrences[pos_occ][1] <= target_series[i][1] and pos_occ < len(occurrences) - 1:
                    pos_occ += 1
                if occurrences[pos_occ][1] <= target_series[i][
                    1]:  # no other occurencies after target_series[ti][1] timestamp
                    for innert1 in range(i + 1, len(target_series)):
                        self.existing_results[name].append(0)
                        self.existing_timestamps[name].append(target_series[innert1][1])
                    break
            else:
                self.existing_results[name].append(0)
                self.existing_timestamps[name].append(target_series[i][1])
        self.trim(name, target_series)

        return [self.existing_results[name]], [name]


class map_isolated_to_continuous_deprecated:
    """
           Wrapper Class for mapping isolated events (defined as events with instant impact) to continuous time-series.

    """

    def __init__(self):
        pass

    def map(self, target_series, occurrences, name):
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
        ci = 0
        vector = [0 for i in range(len(target_series))]
        for ti in range(len(target_series)):
            if occurrences[ci][1] <= target_series[ti][1]:
                vector[ti] = 1
                # finde next occurance after target_series[ti][1] timestamp
                while occurrences[ci][1] <= target_series[ti][1] and ci < len(occurrences) - 1:
                    ci += 1
                if occurrences[ci][1] <= target_series[ti][
                    1]:  # no other occurencies after target_series[ti][1] timestamp
                    break
        return [vector], [name]


class map_univariate_to_continuous_deprecated:
    """
        Wrapper Class for mapping univariate time-series (defined as events with instant impact) to continuous time-series with same frequency as a target time-series.
    """

    def __init__(self):
        self.existing_results = {}

    def map(self, target_series, occurrences, name):
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
            self.existing_results[name] = []
        # position of first timestamp of target series in aggregated data
        spos = -1
        for tup in self.existing_results[name]:
            spos += 1
            if tup[1] >= target_series[0][1]:
                break

        self.existing_results[name] = self.existing_results[name][spos:]

        pos = 0
        if len(self.existing_results[name]) > 0:
            for q in range(len(occurrences)):
                pos = q
                if occurrences[q][1] > self.existing_results[name][-1][1]:
                    break

        vector = [tup[0] for tup in self.existing_results[name]] + [0 for i in range(
            len(target_series) - len(self.existing_results[name]))]

        for i in range(len(self.existing_results[name]), len(target_series)):
            timestamp = target_series[i][1]
            current_pos = pos
            for q in range(pos, len(occurrences)):
                if occurrences[q][1] > timestamp:
                    current_pos = q
                    break
            if i == len(target_series) - 1:
                current_pos = len(occurrences) + 1
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
        self.existing_results[name] = [(v, tup[1]) for v, tup in zip(vector, target_series)]
        return [vector], [name]


class map_univariate_to_continuous(map_base):
    """
        Wrapper Class for mapping univariate time-series (defined as events with instant impact) to continuous time-series with same frequency as a target time-series.
    """

    def __init__(self):
        super().__init__()

    def map(self, target_series, occurrences, name):
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
        pos_target, pos_occ = self.find_pos(name, occurrences, target_series)

        for i in range(pos_target, len(target_series)):

            current_pos = len(occurrences)
            for q in range(pos_occ, len(occurrences)):
                if occurrences[q][1] > target_series[i][1]:
                    current_pos = q
                    break
            if current_pos == pos_occ:
                if len(self.existing_results[name]) > 0:
                    self.existing_results[name].append(self.existing_results[name][-1])
                # if no data until i timestamp use the first occurence as value
                else:
                    self.existing_results[name].append(occurrences[pos_occ][0])
            # if multiple values in between two timestamps use the mean of them as value
            else:
                if i == len(target_series) - 1:
                    dataInBetween = [value for value, time in occurrences[pos_occ:]]
                    self.existing_results[name].append(sum(dataInBetween) / len(dataInBetween))
                else:
                    dataInBetween = [value for value, time in occurrences[pos_occ:current_pos]]
                    self.existing_results[name].append(sum(dataInBetween) / len(dataInBetween))
            self.existing_timestamps[name].append(target_series[i][1])

            # if no other occurrences just repeat the last value
            if current_pos == len(occurrences):
                for k in range(i + 1, len(target_series)):
                    self.existing_results[name].append(self.existing_results[name][-1])
                    self.existing_timestamps[name].append(target_series[k][1])
                break
            pos_occ = current_pos
        self.trim(name, target_series)

        return [self.existing_results[name]], [name]

mapping_functions = {

    "Univariate":map_univariate_to_continuous_deprecated(), # represent continuous sources

    "isolated": map_isolated_to_continuous_deprecated(), # represent sources of events with instant impact

    "configuration": map_configuration_to_continuous_deprecated(), # represent sources of events with constant impact

    "categorical":map_categorical_to_continuous() # represent sources of catigorical values

}

