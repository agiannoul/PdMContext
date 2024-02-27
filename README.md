# PdMContext

A python package for ectracting context in streaming application (related to Predictive Maintenance and Anomaly Detection)

### Documentation can be found in Documentation folder

### See src/Example.ipynb for usage

# Context and Data types 

**Context** is used here to provide a better understanding of the difference cases the data are each time.

In esense Context represent the data (**CD**), existing in a time window, and their relationships (**CR**), where the relationships are extracted using **causal discovery** between the data (the causal discovery method can be user defiend).

**PdmContext.utils.structure.Context** is used to Represent such a context.

### Data Types

#### Continiuous (analog, real, Univariate series ...):

To this point **CD** contain data from different sources, and support different sample rates of signals, and event discrete data. The difference in sample rate is handled internally in the context generation process where all the series are maped to a single series sample rate callse **target** series (also reffered to the code and documentation as such): 

1) For series with sample rate higher than that of target, the samples between two timestamps of targets series, are aggregated (mean)
2) For series with lower sample rate, repetition of their values is used.


#### Event Data: 

The context suppor also data which are not numeric, but related to some kind of event (events that occur in time). These are oftenly refered as discrete data. To this end the Context support two types of such events:

1) isolated: Event that have instant impact when they occur.
2) configuration: Events that refer to a configuration change that has impact and after its occurance.

The type of events is used to tranform the in to contiuous space and add them to **CD**.

![alt text](src/images/CDextraction.png)


# Related works: 
