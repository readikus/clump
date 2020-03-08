# clump

Easy-to-use library for grouping, tagging and finding similar documents, using advanced natural language processing techniques.

The initial purpose of this package is for finding similar or related documents. The typical use case for this is news apps that want to display related content for a particular news story.

Many developers don’t have the time to invest in learning the best practices for this, so this module provides a simple package for loading all the content to consider and then a function that given a passage of text, will find all the related stories.

## How to use:

Simply install the library with pip and import the VectorSpaceModel 

```python
from clumb import VectorSpaceModel
training_docs = ['Some pieces of text', 'More text']

# build the model
model = VectorSpaceModel(training_docs)

# find the three most similar documents
similar = model.find_similar('Another text', n=3)
print(similar)

```

## Current Limitations

* Small contextual consideration.
* Performance on large datasets 

## Road Map

* Document clustering
* Automatic tagging
* Topic classification
* Performance improvements by pre clustering documents into large groups, then searching just the similar clusters.