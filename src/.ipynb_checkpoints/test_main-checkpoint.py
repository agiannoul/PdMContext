from random import random
import pandas as pd
from PdmContext.ContextGeneration import ContextGenerator
from PdmContext.ContextClustering import DBscanContextStream
from src.PdmContext.utils.causal_discovery_functions import calculatewithPc,empty_cause
from src.PdmContext.utils.structure import Context
from src.PdmContext.utils.distances import distance_cc
from src.PdmContext.utils.simulate_stream import simulate_stream, simulate_from_df
from src.PdmContext.utils.dbconnector import SQLiteHandler,InfluxDBHandler
from src.PdmContext.Pipelines import ContextAndClusteringAndDatabase,ContextAndDatabase
from PdmContext.utils.showcontext import show_Context_list
from tqdm import tqdm
# Press the green button in the gutter to run the script.

def my_distance(c1:Context,c2:Context):

    return distance_cc(c1,c2,a=0,b=1)

def test_context_Generation_dummy():
    con_gen = ContextGenerator(target="anomaly1", context_horizon="8", Causalityfunct=calculatewithPc, debug=False)

    configur = [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
    data1 = [random() for i in range(len(configur))]
    start = pd.to_datetime("2023-01-01 00:00:00")
    timestamps = [start + pd.Timedelta(minutes=i) for i in range(len(data1))]
    anomaly1 = [0.2, 0.3, 0.2, 0.1, 0.2, 0.2, 0.1, 0.4, 0.8, 0.7, 0.7, 0.8, 0.7, 0.8, 1, 0.6, 0.7]

    spikes = [0 for i in range(len(data1))]
    spikes[1] = 1
    spikes[13] = 1

    source = "press"
    for d1, an1, t1, sp1, con1 in zip(data1, anomaly1, timestamps, spikes, configur):
        con_gen.collect_data(timestamp=t1, source=source, name="data1", value=d1)
        if sp1 == 1:
            con_gen.collect_data(timestamp=t1, source=source, name="spike", type="isolated")
        if con1 == 1:
            con_gen.collect_data(timestamp=t1, source=source, name="config", type="configuration")
        contextTemp = con_gen.collect_data(timestamp=t1, source=source, name="anomaly1", value=an1)
    con_gen.plot()

def test_context_Generation_dummy_and_Database_Sqlite():
    con_gen = ContextGenerator(target="anomaly1", context_horizon="8", Causalityfunct=calculatewithPc, debug=False)
    database = SQLiteHandler(db_name="ContextDatabase")
    contextpipeline = ContextAndDatabase(context_generator_object=con_gen, databaseStore_object=database)

    configur = [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
    data1 = [random() for i in range(len(configur))]
    start = pd.to_datetime("2023-01-01 00:00:00")
    timestamps = [start + pd.Timedelta(minutes=i) for i in range(len(data1))]
    anomaly1 = [0.2, 0.3, 0.2, 0.1, 0.2, 0.2, 0.1, 0.4, 0.8, 0.7, 0.7, 0.8, 0.7, 0.8, 1, 0.6, 0.7]

    spikes = [0 for i in range(len(data1))]
    spikes[1] = 1
    spikes[13] = 1

    source = "press"
    for d1, an1, t1, sp1, con1 in zip(data1, anomaly1, timestamps, spikes, configur):
        contextpipeline.collect_data(timestamp=t1, source=source, name="data1", value=d1)
        if sp1 == 1:
            contextpipeline.collect_data(timestamp=t1, source=source, name="spike", type="isolated")
        if con1 == 1:
            contextpipeline.collect_data(timestamp=t1, source=source, name="config", type="configuration")
        contextTemp = contextpipeline.collect_data(timestamp=t1, source=source, name="anomaly1", value=an1)
    contextpipeline.Contexter.plot()

def test_context_Generation_with_clustering_and_Database():
    con_gen = ContextGenerator(target="anomaly1", context_horizon="8", Causalityfunct=calculatewithPc, debug=False)
    database = SQLiteHandler(db_name="ContextDatabase.db")
    clustering=DBscanContextStream(cluster_similarity_limit=0.7,min_points=2,distancefunc=my_distance)
    contextpipeline = ContextAndClusteringAndDatabase(context_generator_object=con_gen,Clustring_object=clustering, databaseStore_object=database)

    configur = [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
    data1 = [random() for i in range(len(configur))]
    start = pd.to_datetime("2023-01-01 00:00:00")
    timestamps = [start + pd.Timedelta(minutes=i) for i in range(len(data1))]
    anomaly1 = [0.2, 0.3, 0.2, 0.1, 0.2, 0.2, 0.1, 0.4, 0.8, 0.7, 0.7, 0.8, 0.7, 0.8, 1, 0.6, 0.7]

    spikes = [0 for i in range(len(data1))]
    spikes[1] = 1
    spikes[13] = 1

    source = "press"
    for d1, an1, t1, sp1, con1 in zip(data1, anomaly1, timestamps, spikes, configur):
        contextpipeline.collect_data(timestamp=t1, source=source, name="data1", value=d1)
        if sp1 == 1:
            contextpipeline.collect_data(timestamp=t1, source=source, name="spike", type="isolated")
        if con1 == 1:
            contextpipeline.collect_data(timestamp=t1, source=source, name="config", type="configuration")
        contextTemp = contextpipeline.collect_data(timestamp=t1, source=source, name="anomaly1", value=an1)
    contextpipeline.Contexter.plot()
    clustering.plot()

def Test_Existing_database():
    database = SQLiteHandler(db_name="ContextDatabase.db")
    contextlist=database.get_all_context_by_target("anomaly1")
    show_Context_list(contextlist,"anomaly1")

def Test_simulator():
    start = pd.to_datetime("2023-01-01 00:00:00")
    timestamps = [start + pd.Timedelta(minutes=i) for i in range(17)]

    eventconf=("config",[pd.to_datetime("2023-01-01 00:09:00")],"configuration")
    spiketuples=("spikes",[pd.to_datetime("2023-01-01 00:01:00"),pd.to_datetime("2023-01-01 00:13:00")],"isolated")

    data1tuples=("data1",[random() for i in range(17)],timestamps)
    anomaly1tuples=("anomaly1", [0.2, 0.3, 0.2, 0.1, 0.2, 0.2, 0.1, 0.4, 0.8, 0.7, 0.7, 0.8, 0.7, 0.8, 1, 0.6, 0.7],timestamps)

    stream=simulate_stream([data1tuples,anomaly1tuples],[eventconf,spiketuples],"anomaly1")

    for record in stream:
        print(record)
import os
def Test_simulator_given_DataFrame():
    df = get_df()

    print(df.head())

    stream=simulate_from_df(df,"actual_pressure_container_PtP")

    limit=20
    count=0
    for record in stream:
        print(record)
        count+=1
        if count>limit:
            break

def Test_simulator_dataframe_with_pipeline_and_Influx():
    traget_name="actual_pressure_container_PtP"
    con_gen = ContextGenerator(target=traget_name, context_horizon="8", Causalityfunct=empty_cause, debug=False)
    database = SQLiteHandler(db_name="ContextDatabase.db")
    #database = InfluxDBHandler(host='localhost', port=8086, db_name='my_database',measurment_name="my_table")
    clustering = DBscanContextStream(cluster_similarity_limit=0.7, min_points=2, distancefunc=my_distance)
    #contextpipeline = ContextAndClusteringAndDatabase(context_generator_object=con_gen, Clustring_object=clustering, databaseStore_object=database)
    #contextpipeline = ContextAndClusteringAndDatabase(context_generator_object=con_gen, Clustring_object=clustering, databaseStore_object=database)
    contextpipeline = ContextAndDatabase(context_generator_object=con_gen, databaseStore_object=database)

    df = get_df()

    print(df.head())

    stream = simulate_from_df(df, traget_name)


    count=1
    source = "press"
    for record in stream:
        count+=1
        if count%3301==0:
            print(count)
            break
        contextpipeline.collect_data(timestamp=record["timestamp"], source=source, name=record["name"], type=record["type"],value=record["value"])
    contextpipeline.Contexter.plot(filteredges=[[traget_name, "", ""],["", traget_name, ""]])
    #contextpipeline.clustering.plot()
def showfrombase():
    database = SQLiteHandler(db_name="ContextDatabase.db")
    traget_name = "actual_pressure_container_PtP"
    contextlist = database.get_all_context_by_target(traget_name)
    show_Context_list(contextlist, target_text=traget_name, filteredges=[[traget_name, "", ""], ["", traget_name, ""]])
def showfrombaseInflux():
    database = InfluxDBHandler(host='localhost', port=8086, db_name='my_database',measurment_name="my_table")
    traget_name = "actual_pressure_container_PtP"
    contextlist = database.get_all_context_by_target(traget_name)
    show_Context_list(contextlist, target_text=traget_name, filteredges=[[traget_name, "", ""], ["", traget_name, ""]])

def get_df():
    df = pd.read_csv("data.csvbig", index_col=0)
    df.index = pd.to_datetime(df.index)
    meta_data_cols = ["die_change", "wait_times_sum"]
    meta_data_cols.extend([col for col in df.columns if "diff_std" in col])
    meta_data_cols.extend([col for col in df.columns if "ampere" in col])
    df = df.drop(meta_data_cols, axis=1)
    return df

if __name__ == '__main__':


    #Test_simulator_dataframe_with_pipeline_and_Influx()
    showfrombaseInflux()
    #showfrombase()




