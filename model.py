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
        self.docs = docs
        # create the tf-idf model
        self.vectorizer = TfidfVectorizer(max_features=250000, use_idf=True, stop_words='english')
        self.vectors = self.vectorizer.fit_transform(docs)
        return self.vectorizer

    def compute_distance_matrix(self):
        len_vectors = self.vectors.shape[0]
        D = [[0 for x in range(len_vectors)] for y in range(len_vectors)]
        # create the row and symetrical column values
        for i in range(len_vectors):
            for j in range(0, i):
                D[i][j] = D[j][i] = cosine_similarity(self.vectors[i], self.vectors[j])
        return D
    
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
            # find distance to the ith doc
            distance = cosine_similarity(doc_vector, self.vectors[i])
            # ignore elements that are too far away
            if distance[0] > distance_threshold:
                distances.append(Similarity(i, distance[0]))
        
        # sort the list and pick the top n records
        sorted_list = sorted(distances, key=lambda x: (x.distance), reverse=True)[:n]
        return [self.docs[similar.index] for similar in sorted_list]

    def cluster(self, docs, *args, **kwargs):
        """

        need to make sure something doesn't easily fit into another cluster (could get confusing)
        """
        distance_threshold = kwargs.get('distance_threshold', 0.4)

        # create the distance matrix
        D = self.compute_distance_matrix()
        c = self.init_clusters(self.vectors.toarray())

        # initially prune stuff that is too far from anything else



        print(D)

        match = self.find_max_match(D)
        print(match)

        exit()

        # find the best match, then find all the unclusteredx

    # initialise clusters
    def init_clusters(self, vectors):
        c = []
        for i, vi in enumerate(vectors):
            cluster = { 'centroid': vi.copy(), 'vectors': [vi] }
            c.append(cluster)
        return c


    # find the first match
    def find_max_match(self, D):

        # assume 2 elements to start with....
        max_distance = D[0][1]
        max_pos = (0, 1)

        print (len(D))
        # for each doc, find the most similar one...
        for i in range(0, len(D)):
            for j in range(0, i):
                dj = D[i][j]
                if (i != j and dj > max_distance):
                    max_distance = dj
                    max_pos = (i, j)

        print ('max_distance')
        print (max_distance)
        return max_pos




"""



def hier_cluster(c, D, distance_func, match_func):

    match = match_func(D)
    print('find_best_match:')
    print(match)
    print(match[0])

    # merge the rows/columns of the matrix
    v1 = c[match[0]]['vectors']
    print('v1')
    print(v1)

    print('Merge element i')
    print(c[match[0]]['vectors'][0])
    print('with element i2')
    print(c[match[1]]['vectors'][0])
    vectors = c[match[0]]['vectors'] + c[match[1]]['vectors']
    
    # calculate the centroids
    vector_0 = c[0]['vectors'][0] #.toarray()

    print('shape(vector_0)')
    print(vector_0.shape)

#    vector_0 = [0] * vector_0.shape[1] #len(vector_0[0])
#    print('vector_0')
#    print(vector_0)
    #print(vector_0.shape)


    centroid = csr_matrix([[0] * vector_0.shape[1]])
    centroid = centroid[0]
    for i, vi in enumerate(vectors):
        v_i = vi #.toarray()
        # enum may do the matrix anyway???
        for j, v_j in enumerate(v_i):

            print('v_j')
            print(v_j)
            print(v_j.shape)
            print(centroid.shape)
            #centroid[j] += v_j
            centroid += v_j


    denominator = float(len(vectors)) # csr_matrix([float(len(vectors))])
    denominator = csr_matrix([[float(len(vectors))] * vector_0.shape[1]])
    print('centroid (pre divide)')
    print(centroid)

    print('denominator')
    print(denominator)
    centroid /= denominator
    print('centroid')
    print(centroid)

    exit()
    total_vectors = len(vectors)
    for i, value_i in enumerate(centroid):
        centroid[i] = centroid[i]/total_vectors

    # set the ith row
    c[match[0]] = { 'centroid': centroid, 'vectors': vectors }
    del c[match[1]]
    match_i = match[0]
    match_j = match[1]

    # recompute all distances to ith cluster
    for i in range(0, len(D)):
        D[i][match_i] = distance_func(c[i]['centroid'], c[match_i]['centroid'])
        # remove the jth entry
        del D[i][match_j]

    # remove jth row/column
    del D[match_j]

    return { 'c': c, 'D': D }

    # total / 2
    #merge the ith vector with the jth vector 

    # remove the jth row and column from D

    # recompute the ith row and column

    #recompute distances between all other clusters and the 


"""

