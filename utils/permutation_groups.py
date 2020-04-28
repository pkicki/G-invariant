import numpy as np
from itertools import permutations

Z4 = [[0, 1, 2, 3], [1, 2, 3, 0], [2, 3, 0, 1], [3, 0, 1, 2]]
Z5 = [[0, 1, 2, 3, 4], [1, 2, 3, 4, 0], [2, 3, 4, 0, 1], [3, 4, 0, 1, 2], [4, 0, 1, 2, 3]]
perm = [[0, 1, 2, 3], [1, 2, 3, 0], [2, 3, 0, 1], [3, 0, 1, 2],
        [3, 2, 1, 0], [2, 1, 0, 3], [1, 0, 3, 2], [0, 3, 2, 1]]
D8 = [list(x) + [4] for x in perm]
perm = [[0, 1, 2, 3], [1, 2, 0, 3], [2, 0, 1, 3], [3, 0, 2, 1], [1, 3, 2, 0], [3, 1, 0, 2], [2, 1, 3, 0],
        [0, 3, 1, 2], [0, 2, 3, 1], [1, 0, 3, 2], [2, 3, 0, 1], [3, 2, 1, 0]]
A4 = [list(x) + [4] for x in perm]
perm = list(permutations(range(4)))
S4 = [list(x) + [4] for x in perm]
perm = list(permutations(range(3)))
S3 = [list(x) + [3, 4] for x in perm]

perm = list(permutations(range(3)))
S3_in_S4 = [list(x) + [3] for x in perm]

H3O_perm = [[0, 1, 2, 3, 4, 5], [1, 0, 2, 3, 5, 4], [2, 1, 0, 5, 4, 3], [0, 2, 1, 4, 3, 5], [1, 2, 0, 5, 3, 4], [2, 0, 1, 4, 5, 3]]
#H2CO_perm = [[0, 1, 2, 3, 4, 5], [0, 2, 1, 4, 3, 5]]
H2CO_perm = [[0, 1, 2, 3], [1, 0, 2, 3]]
