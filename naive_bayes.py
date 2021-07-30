import numpy as np


ATTRIBUTES = 1 #size of the feature vector
PICKS = 6

MIN = 1
MAX = 49

LAPLACE_M = 3


class Pick:#3-8-14-22-23-45
    def __init__(self, pos, data, f_vec, classes):
        self.pos = pos
        self.data = data
        self.f_vec = f_vec
        self.classes = classes
        


def read_data():
    with open("data.txt", 'r') as file:
        return np.array([v.split('-') for v in file.read().splitlines()])


def process_data(pos, numbers):
    data_vecs = np.empty(shape=(len(numbers)-ATTRIBUTES, ATTRIBUTES+1), dtype='i')
    for k, j in enumerate(range(ATTRIBUTES, len(numbers))):
        data_vecs[k] = np.array(numbers[j-ATTRIBUTES:j+1])
        
    return Pick(pos, data_vecs, np.array(numbers[-ATTRIBUTES:], dtype='i'), np.stack(data_vecs[:,-1]))


def learn(pick):
    probabilities = {} #Key = (Attribute, Value, Class)
    for i in range(ATTRIBUTES):
        for c in np.unique(pick.classes):
            for j in range(MIN, MAX+1):
                attr_occs = np.count_nonzero(np.logical_and(pick.data[:, i] == j, pick.data[:, -1] == c))
                if attr_occs > 0:
                    class_ocs = np.count_nonzero(pick.classes == c)
                    probabilities[(i, j, c)] = attr_occs/class_ocs
                else:
                    probabilities[(i, j, c)] = LAPLACE_M * (1/(MAX-(PICKS-1)))
    return probabilities


def classify(pick, probs):
    argmax = 0
    currmax = 0
    c_len = len(pick.classes)
    for c in np.unique(pick.classes):
        prob = np.count_nonzero(pick.classes == c)/c_len
        for i in range(ATTRIBUTES):
            prob *= probs[(i, pick.f_vec[i], c)]
        if prob > currmax:
            argmax, currmax = c, prob
    return argmax

    
def get_next_numbers():
    numbers = np.stack(read_data(), axis=1)
    result = np.empty(shape=(PICKS), dtype='i')
    for i in range(PICKS):
        pick = process_data(i, numbers[i])
        probabilities = learn(pick)
        result[i] = classify(pick, probabilities)

    print(result)


get_next_numbers()


    
