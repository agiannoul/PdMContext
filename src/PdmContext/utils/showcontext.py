import numpy as np
from PdmContext.utils.structure import Context
import matplotlib.pyplot as plt
import pandas as pd


def filter_edges(edge, char, filteredges):
    for filter in filteredges:
        if filter[0] in edge[0] and filter[1] in edge[1] and filter[2] in char:
            return True
    return False


def show_context_list(contextlist: list[Context], target_text, filteredges=[["", "", ""]],char=True):
    """
    Visualization of Contexts in the contextlist

    **Parameters**:

    **contextlist**: List of Context objects

    **target_text**: The target data that the context built with

    **filteredges**: A list of list, where each list define which of the edges in Context.Cr should be displayed
        where each time we check if the fields of the list are part of the edges text and characterization.
    """

    fig, ax = plt.subplots()
    ax.set_title("Complete Timespan")
    ax.set_xlabel("Date")
    # ax.set_ylabel("Data Present")

    local_timezone = None

    date_list2 = []
    target_values = []

    querytimes = []
    querycolors = []
    queryvalues = []

    for context_obj in contextlist:
        timestamptemp2 = context_obj.timestamp
        date_list2.append(timestamptemp2)

        target_values.append(context_obj.CD[target_text][-1])

        if char:
            for edge, char in zip(context_obj.CR["edges"], context_obj.CR["characterization"]):
                if filter_edges(edge, char, filteredges):
                    querytimes.append(timestamptemp2)
                    queryvalues.append(target_values[-1])
                    querycolors.append(f"{edge[0].split('@')[0]}->{edge[1].split('@')[0]}")
        else:
            for edge in context_obj.CR["edges"]:
                if filter_edges(edge, " ", filteredges):
                    querytimes.append(timestamptemp2)
                    queryvalues.append(target_values[-1])
                    querycolors.append(f"{edge[0].split('@')[0]}->{edge[1].split('@')[0]}")
    width = 3
    # Plot the complete timespan
    ax.clear()
    ax.plot(date_list2, target_values, 'o', color="black", markersize=7,
            label=target_text)  # The 'ro' format will display red dots for each data point

    ############ PLOT EDGES #######################################################
    color_mapping = {}
    for description in set(querycolors):
        if description not in color_mapping:
            # Generate a random RGB color
            color = np.random.rand(3, )
            color_mapping[description] = color
    color_toplot = []
    for i in range(len(querycolors)):
        description = querycolors[i]
        color_toplot.append(color_mapping[description])
        ax.plot(querytimes[i], queryvalues[i], 'o', markersize=7,
                color=color_mapping[description], alpha=0.7, label=description)

    handles, labels = ax.get_legend_handles_labels()
    unique = [(h, l) for i, (h, l) in enumerate(zip(handles, labels)) if l not in labels[:i]]
    ax.legend(*zip(*unique), loc="upper left")
    ax.grid()
    plt.show()


def show_context_interpretations(contextlist: list[Context], target_text, filteredges=[["", "", ""]]):
    """
    Visualization of Interpretations from Contexts in the contextlist

    **Parameters**:

    **contextlist**: List of Context objects

    **target_text**: The target data that the context built with

    **filteredges**: A list of list, where each list define which of the edges in Context.Cr should be displayed
        where each time we check if the fields of the list are part of the edges text and characterization.
    """

    fig, ax = plt.subplots()
    ax.set_title("Complete Timespan")
    ax.set_xlabel("Date")
    # ax.set_ylabel("Data Present")

    local_timezone = None

    date_list2 = []
    target_values = []

    querytimes = []
    querycolors = []
    queryvalues = []

    for context_obj in contextlist:
        timestamptemp2 = context_obj.timestamp
        date_list2.append(timestamptemp2)

        target_values.append(context_obj.CD[target_text][-1])

        for triplet in context_obj.CR["interpretation"]:
            edge = (triplet[0], triplet[1])
            char = triplet[2]
            if filter_edges(edge, char, filteredges):
                querytimes.append(timestamptemp2)
                queryvalues.append(target_values[-1])
                querycolors.append(f"{edge[0].split('@')[0]}->{edge[1].split('@')[0]}")

    width = 3
    # Plot the complete timespan
    ax.clear()
    ax.plot(date_list2, target_values, 'o', color="black", markersize=7,
            label=target_text)  # The 'ro' format will display red dots for each data point

    ############ PLOT EDGES #######################################################
    color_mapping = {}
    for description in set(querycolors):
        if description not in color_mapping:
            # Generate a random RGB color
            color = np.random.rand(3, )
            color_mapping[description] = color
    color_toplot = []
    for i in range(len(querycolors)):
        description = querycolors[i]
        color_toplot.append(color_mapping[description])
        ax.plot(querytimes[i], queryvalues[i], 'o', markersize=7,
                color=color_mapping[description], alpha=0.7, label=description)

    handles, labels = ax.get_legend_handles_labels()
    unique = [(h, l) for i, (h, l) in enumerate(zip(handles, labels)) if l not in labels[:i]]
    ax.legend(*zip(*unique), loc="upper left")
    ax.grid()
    plt.show()
