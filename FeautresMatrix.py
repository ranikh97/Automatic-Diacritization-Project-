import numpy as np
from scipy.sparse import csr_matrix


class FeaturesMatrix:
    def __init__(self, features_num, tags_len):
        self.matrix = csr_matrix((features_num, tags_len), dtype=np.int8)
