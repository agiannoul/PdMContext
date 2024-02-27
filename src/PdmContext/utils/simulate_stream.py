import pandas as pd

class simulate_stream():
    def __init__(self,TimeSeries:[(str,list[float],list[pd.Timestamp])],Events:[(str,list[pd.Timestamp],str)],targetname):
        """
        This method simulate the traffic of the data

        **Parameters**:

        **TimeSeries**: A list of time series data in form of (name,list with data, list of timestamp)

        **Events**: A list of events in form of (name, list of timestamp when the event occured,type of the Event (isolated or configuration)) as expected in ContextGenerator.
        """
        # build dataframe with columns dt, name, value, type
        dict_df={
            "dt":[],
            "name":[],
            "value":[],
            "type":[]
        }
        for series in TimeSeries:
            name = series[0]
            data = series[1]
            times = series[2]
            for dd,dtt in zip(data,times):
                dict_df["dt"].append(dtt)
                dict_df["name"].append(name)
                dict_df["value"].append(dd)
                dict_df["type"].append("Univariate")

        for series in Events:
            name = series[0]
            times = series[1]
            type = series[2]
            for dtt in times:
                dict_df["dt"].append(dtt)
                dict_df["name"].append(name)
                dict_df["value"].append(None)
                dict_df["type"].append(type)

        self.df = pd.DataFrame(dict_df)
        self.df.sort_values(by="dt",inplace=True)
        self.uniquedates=self.df["dt"].unique()
        list(self.uniquedates).sort()
        self.targetname=targetname
        self.liststream=[]
        self.current=0
        for date in self.uniquedates:
            rows = self.df[self.df["dt"] == date]
            keep = {}
            for ind, row in rows.iterrows():
                if row["name"] == self.targetname:
                    keep["name"] = row["name"]
                    keep["type"] = row["type"]
                    keep["dt"] = row["dt"]
                    keep["value"] = row["value"]
                else:
                    if row["type"] != "Univariate":
                        tempvalue = None
                    else:
                        tempvalue = row["value"]
                    self.liststream.append({"name": row["name"], "type": row["type"], "timestamp": row["dt"], "value": tempvalue})
            if len(keep.keys()):
                self.liststream.append({"name": keep["name"], "type": keep["type"], "timestamp": keep["dt"], "value": keep["value"]})

    def __iter__(self):
        return self
    def __next__(self):
        self.current += 1
        if self.current < len(self.liststream):
            return self.liststream[self.current]
        raise StopIteration


def simulate_from_df(df: pd.DataFrame,target_name: str):
    '''
    **Parameters**:

    **df**: Dataframe with the data to simulate stream, the index has to be of Date type.

    **target_name**: the target name data

    **return**: an itterator
    '''
    #TimeSeries:[(str,list[float],list[pd.Timestamp])]
    timeseries=[]
    for col in df.columns:
        timeseries.append((col, df[col].values, [dt for dt in df.index]))
        print(col)
    return simulate_stream(TimeSeries=timeseries,Events=[],targetname=target_name)

