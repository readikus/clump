import json
from model import VectorSpaceModel

# load the JSON data for the documents
with open('./sample_data/mtb.json') as data_file:
    training_docs = json.load(data_file)

# build the model
model = VectorSpaceModel(training_docs)

# find the most similar documents on sample above
test_docs = ["Yeti SB150 with SRAM AXS",
    "2019 Enduro World Series | Round 6 | Crankworx Whistler Highlights.",
    "Kirt Voreis Owns Whistler Mountain Bike Park",
    "I like sausages"]


#model = VectorSpaceModel(test_docs)
model.cluster(training_docs)

# display the similar documents
for test_doc in test_docs:
    similar = model.find_similar(test_doc)

    print('input: ' + test_doc)
    print(similar)
 

