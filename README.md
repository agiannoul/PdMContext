# PdMContext

A Python package for extracting context in streaming applications (related to Predictive Maintenance and Anomaly Detection)

### Documentation can be found in the Documentation folder

### See src/Example.ipynb for usage

# Context and Data types 

**Context** is used here to provide a better understanding of the different cases the data are each time.

Essentially Context represents the data (**CD**), existing in a time window, and their relationships (**CR**), where the relationships are extracted using **causal discovery** between the data (the causal discovery method can be user defined).

**PdmContext.utils.structure.Context** is used to represent such a context.

### Data Types

#### Continuous (analog, real, Univariate series ...):

To this point **CD** contains data from different sources, and supports different sample rates of signals, and event discrete data. The difference in sample rate is handled internally in the context generation process where all the series are mapped to a single series sample rate called **target** series (also referred to the code and documentation as such): 

1) For series with a sample rate higher than that of the target, the samples between two timestamps of the targets series, are aggregated (mean)
2) For series with lower sample rates, repetition of their values is used.


#### Event Data: 

The context supports also data that are not numeric, but related to some kind of event (events that occur in time). These are often referred to as discrete data. To this end, the Context supports two types of such events:

1) isolated: Events that have an instant impact when they occur.
2) configuration: Events that refer to a configuration change that has an impact after its occurrence.

The type of events is used to transform them into continuous space and add them to **CD**.

![alt text](src/images/CDextraction.png)


# Related works: 
