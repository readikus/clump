# clump

Natural Language Processing made simple: text clustering, "related posts" and topic tagging

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
