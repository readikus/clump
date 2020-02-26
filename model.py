from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# used for sort
class Similarity:
    def __init__(self, index, distance):
        self.index = index
        self.distance = distance

class VectorSpaceModel:

    def __init__(self, docs = None):
        if docs != None:
            self.train(docs)

    def train(self, docs):
        print('training on ' + str(len(docs)))
        self.docs = docs
        # create the tf-idf model
        self.vectorizer = TfidfVectorizer(max_features=250000, use_idf=True, stop_words='english')
        self.vectors = self.vectorizer.fit_transform(docs)
        return self.vectorizer


    def find_similar(self, doc, *args, **kwargs):
        """Find similar documents from the training data

        Arguments:
        doc -- the document to search against
        Keyword arguments:
        n -- the number of results to return (default 5)
        distance_threshold -- the minium distance threshold to consider, a value 
        between 0 and 1, with 0 being no similarity, 1 being identical. (default: 0.4)
        
        Todo:
        Rewrite to reprocess everything into clusters
        Include more text-preprocessing
        """
        n = kwargs.get('n', 5)
        distance_threshold = kwargs.get('distance_threshold', 0.4)


        # for each doc, find the most similar one...
        distances = []
        doc_vector = self.vectorizer.transform([doc])

        for i in range(self.vectors.shape[0]):
            # find distance to the jth doc
            distance = cosine_similarity(doc_vector, self.vectors[i])
            # ignore elements that are too far away
            if distance[0] > distance_threshold:
                distances.append(Similarity(i, distance[0]))

        sorted_list = sorted(distances, key=lambda x: (x.distance), reverse=True)
        sorted_list = sorted_list[:n]
        similar_docs = []

        for j in range(len(sorted_list)):
            index = sorted_list[j].index
            similar_docs.append(self.docs[index])
        return similar_docs
