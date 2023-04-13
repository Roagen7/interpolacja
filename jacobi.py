from copy import deepcopy


def matsub(mat1: list, mat2: list) -> list:
    mat = [[0 for _ in range(len(mat1[0]))] for _ in range(len(mat1))]
    for y in range(len(mat1)):
        for x in range(len(mat1[0])):
            mat[y][x] = mat1[y][x] - mat2[y][x]
    return mat


def norm(vec: list):
    return sum([y[0]**2 for y in vec])**0.5


def matmul(mat1: list, mat2: list):
    y1 = len(mat1)
    x1 = len(mat1[0])
    x2 = len(mat2[0])
    mat = [[0 for _ in range(x2)] for _ in range(y1)]

    for y in range(len(mat)):
        for x in range(len(mat[0])):
            mat[y][x] = sum([mat1[y][l] * mat2[l][x] for l in range(x1)])

    return mat


def residuum(mat_a: list, b: list, x: list) -> list:
    return matsub(matmul(mat_a, x), b)


def jacobi(mat_a: list, b: list,  diverges=False, epsilon=10**-9) -> (list, int):
    iters = 0
    n = len(mat_a)
    vec_x = [[0] for _ in range(n)]
    norms = []

    while norm(residuum(mat_a, b, vec_x)) > epsilon:
        if diverges and iters >= 50:
            break
        new_vec_x = [[0] for _ in range(n)]
        norms.append(norm(residuum(mat_a, b, vec_x)))
        iters += 1
        for i in range(0, n):
            sig = sum([0 if i == j else vec_x[j][0]*mat_a[i][j] for j in range(n)])
            new_vec_x[i][0] = (b[i][0] - sig)/mat_a[i][i]
        for i in range(len(vec_x)):
            vec_x[i][0] = new_vec_x[i][0]

    norms.append(norm(residuum(mat_a, b, vec_x)))
    return vec_x, iters, norms