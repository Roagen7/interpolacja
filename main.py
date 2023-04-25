import operator

import matplotlib
import matplotlib.pyplot as plt

from functools import reduce
from pandas import read_csv
from jacobi import jacobi

matplotlib.use("MacOSX")

DATA_DIR = "dane/"
MEDIA_DIR = "sprawozdanie/media/"
PROFILE1 = "fuji_yoshida_trail"
PROFILE2 = "uj_laz_sol"
PROFILE3 = "slupsk"


def read_profile(filename):
    records = read_csv(DATA_DIR + filename + ".csv").values
    distances = [record[0] for record in records]
    elevations = [record[1] for record in records]
    return distances, elevations


def draw_profile(filename):
    distances, elevations = read_profile(filename)
    plt.plot(distances, elevations)
    plt.show()


def linspace(start, stop, n):
    if n == 1:
        yield stop
        return
    h = (stop - start) / (n - 1)
    for i in range(n):
        yield start + h * i


def lagrange(X, Y, num_interpolation=9, num_evaluated=1000, indexes=None):

    if indexes is None:
        indexes = [int(i) for i in linspace(0, len(X) - 1, num_interpolation)]

    def F(x):
        return sum(
            reduce(
                operator.mul,
                [(x - X[j]) / (X[i] - X[j]) for j in indexes if i != j],
                1) * Y[i]
            for i in indexes)

    interpolated_X = list(linspace(X[0], X[-1], num_evaluated))
    interpolated_Y = [F(x) for x in interpolated_X]

    return interpolated_X, interpolated_Y, indexes


def splines(X, Y, num_interpolation=15, num_evaluated=1000, indexes=None):
    if indexes is None:
        indexes = [int(i) for i in linspace(0, len(X) - 1, num_interpolation)]
    n = len(indexes)
    a = [Y[ix] for ix in indexes]
    b = []
    d = []
    h = [X[indexes[i+1]] - X[indexes[i]] for i in range(n-1)]


    A = [[0 for _ in range(n)] for _ in range(n)]
    vec = [[0] for _ in range(n)]

    #solve for c
    for i in range(1, n-1):
        A[i][i] = 2 * (h[i-1] + h[i])
        A[i][i-1] = h[i-1]
        A[i][i+1] = h[i]
        vec[i][0] = 3 * ((Y[indexes[i+1]] - Y[indexes[i]])/h[i] - (Y[indexes[i]] - Y[indexes[i-1]])/h[i-1])

    A[0][0] = 1
    A[n-1][n-1] = 1

    c, _, _ = jacobi(A, vec)
    c = [x[0] for x in c]

    for i in range(n-1):
        d.append((c[i+1] - c[i])/(3 * h[i]))
        b.append((Y[indexes[i+1]] - Y[indexes[i]])/h[i] - h[i]/3 * (2 * c[i] + c[i+1]))

    b.append(0)
    d.append(0)

    def F(x):
        ix = n-1
        for ix_num in range(len(indexes) - 1):
            if X[indexes[ix_num]] <= x < X[indexes[ix_num + 1]]:
                ix = ix_num
                break

        h = x-X[indexes[ix]]
        return a[ix] + b[ix] * h + c[ix] * h**2 + d[ix] * h ** 3

    interpolated_X = list(linspace(X[0], X[-1], num_evaluated))
    interpolated_Y = [F(x) for x in interpolated_X]

    return interpolated_X, interpolated_Y, indexes


def evenly_spaced_plots(filename, title, interpolations=(6, 9, 15), interp_function=lagrange):
    X, Y = read_profile(filename)

    _, axis = plt.subplots(len(interpolations))

    plt.suptitle(f"{title}")

    for num_axis,i in enumerate(interpolations):
        x, y, ixs = interp_function(X, Y, num_interpolation=i)

        axis[num_axis].set_title(f"{i} punktów węzłowych")
        axis[num_axis].set_xlabel("odległość [m]")
        axis[num_axis].set_ylabel("wysokość [m]")
        axis[num_axis].plot(X, Y)
        axis[num_axis].plot(x, y)
        axis[num_axis].scatter([X[i] for i in ixs], [Y[i] for i in ixs], c='g')
        axis[num_axis].legend(["dane", "interpolacja", "punkty węzłowe"])

    plt.show()


def plots_specific_indexes(filename, title, indexes, interp_function=lagrange):
    X, Y = read_profile(filename)
    x, y, ixs = interp_function(X, Y, indexes=indexes)


    plt.title(f"nierównomierne punkty węzłowe: {title}")
    plt.xlabel("odległość [m]")
    plt.ylabel("wysokość [m]")
    plt.plot(X, Y)
    plt.plot(x, y)
    plt.scatter([X[i] for i in ixs], [Y[i] for i in ixs], c='g')
    plt.legend(["dane", "interpolacja", "punkty węzłowe"])

    plt.show()

def generate_czebyszew(N):
    import math

    def fu(k):
        return 511/2 * math.cos((2 * k + 1)/(2 * N) * math.pi) + 511/2
    return [int(fu(k)) for k in range(N-1, -1, -1)] + [511]

# evenly_spaced_plots(PROFILE1, "Lagrange: Trasa Yoshidy na górę Fuji", interp_function=lagrange)
# evenly_spaced_plots(PROFILE1, "Spline'y: Trasa Yoshidy na górę Fuji", interp_function=splines)
# plots_specific_indexes(PROFILE1, "Lagrange:Trasa Yoshidy na górę Fuji",
#                        indexes=[0, 15, 40, 50, 100, 150, 200, 250, 300, 350, 400, 450, 470, 500, 511],
#                        interp_function=lagrange)
# plots_specific_indexes(PROFILE1, "Spline'y:Trasa Yoshidy na górę Fuji",
#                        indexes=[0, 100, 150, 210, 220, 235, 250,260, 350, 400, 511],
#                        interp_function=splines)

# evenly_spaced_plots(PROFILE2, "Lagrange: Al. Ujazdowskie-Łazienki-Solec", interp_function=lagrange, interpolations=(9, 12, 15))
# evenly_spaced_plots(PROFILE2, "Spline'y: Al. Ujazdowskie-Łazienki-Solec", interp_function=splines, interpolations=(15, 45))
# plots_specific_indexes(PROFILE2, "Lagrange: Al. Ujazdowskie-Łazienki-Solec",
#                        indexes=[0, 100,150,  220, 235, 250, 300, 350, 450, 511],
#                        interp_function=lagrange)
# plots_specific_indexes(PROFILE2, "Lagrange: Al. Ujazdowskie-Łazienki-Solec",
#                        indexes=[0, 15, 40, 50, 100, 150, 200, 250, 300, 350, 400, 450, 470, 500, 511],
#                        interp_function=lagrange)


# evenly_spaced_plots(PROFILE3, "Lagrange: Wokół centrum Słupska", interp_function=lagrange, interpolations=(5, 9))

# plots_specific_indexes(PROFILE3, "Lagrange: Wokół centrum Słupska",
#                        indexes=[0, 15, 40, 50, 100, 150, 200, 250, 300, 350, 400, 450, 470, 500, 511],
#                        interp_function=lagrange)
#
# plots_specific_indexes(PROFILE3, "Lagrange: Wokół centrum Słupska",
#                        indexes=[0, 10, 20, 40, 50, 100, 150, 200, 250, 300, 350, 400, 450, 470, 480, 490,500, 511],
#                        interp_function=lagrange)
# plots_specific_indexes(PROFILE3, "Spline'y: Wokół centrum Słupska",
#                        indexes=[0, 10, 20, 40, 50, 100, 150, 200, 250, 300, 350, 400, 450, 470, 480, 490,500, 511],
#                        interp_function=splines)
evenly_spaced_plots(PROFILE3, "Spline'y: Wokół centrum Słupska", interp_function=splines, interpolations=(5, 15, 35))

# plots_specific_indexes(PROFILE3, "Lagrange: eliminacja efektu Runge'go",
#                        indexes=generate_czebyszew(30),
#                        interp_function=lagrange)
# plots_specific_indexes(PROFILE3, "Spline: porównanie dla 31 punktów",
#                        indexes=None,
#                        interp_function=splines)
