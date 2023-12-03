import numpy as np

def A(n = 7):
    matrix = np.zeros((n, n))
    matrix[0,-1] = 2
    matrix[:,0] = 2
    for i in range(1, n):
        matrix[i, i] = 4+0.3*i
    for i in range(1,n-1):
        matrix[i+1, i] = 2
    return matrix

def b(n = 7):
    return np.array([3*np.cos(np.pi/(n-i)) for i in range(0, n)]).transpose()

def Gauss(dict: dict):
    n = len(dict["b"])
    A1, x, E1 = dict["A"].copy(), dict["b"].copy(), dict["INV"].copy()
    det = 1
    for k in range(0, n):
        i_max = np.argmax(abs(A1[k:n, k])) + k
        if A1[i_max, k] == 0:
            raise ValueError("Matrix is singular")
        if i_max != k:
            A1[[i_max, k]] = A1[[k, i_max]]
            E1[[i_max, k]] = E1[[k, i_max]]
            x[[i_max, k]] = x[[k, i_max]]
        akk = A1[k, k]
        for i in range(0, n):
            A1[k, i] = A1[k, i]/akk
            E1[k, i] = E1[k, i]/akk
        x[k] = x[k]/akk
        det = det * akk
        for i in range(0, k-1):
            x[i] = x[i] - A1[i, k] * x[k]
            aik = A1[i, k]
            for j in range(0, n):
                E1[i, j] = E1[i, j] - aik*E1[k, j]
                A1[i, j] = A1[i, j] - aik*A1[k, j]
        for i in range(k+1, n):
            x[i] = x[i] - A1[i, k]*x[k]
            aik = A1[i, k]
            for j in range(0, n):
                E1[i, j] = E1[i, j] - aik*E1[k, j]
                A1[i, j] = A1[i, j] - aik*A1[k, j]
    dict["x_gauss"], dict["det"], dict["INV"] = x, det, E1
    return dict


def seidel(res):
    A1 = res["A"].copy()
    b = res["b"].copy()
    A1[0][-1] = 2
    x0 = np.zeros(len(res["b"]))
    x = np.zeros(len(x0))
    LD = np.tril(A1)
    U = A1 - LD
    А2 = np.dot(np.linalg.inv(LD), U)
    i =0
    res["A_seidel"] = A1
    #if np.linalg.norm(А2) >= 1:
    #   print(np.linalg.norm(А2))
    #   raise ValueError("||(L+D)^-1*U|| >- 1, method is not convergent")
    while True:
        i += 1
        x = np.dot(np.linalg.inv(LD), b - np.dot(U, x0))
        if np.linalg.norm(x - x0) < res["eps"]:
            break
        x0 = x
    res["x_seidel"] = x
    res["it"] = i
    return res

def resul_a(res: dict):
    print("Gauss method")
    res = Gauss(res)
    print("1. result x_gauss = {}".format(res["x_gauss"]))
    print("2. residual r = {}".format(np.dot(res["A"], res["x_gauss"]) - res["b"]))
    print("3. condition number = {}".format(np.linalg.cond(res["A"])))
    print("4. det = {}".format(res["det"]))
    print("5. inverse matrix = {}".format(res["INV"]))
    print("5.1 A*A^-1 = {}".format(np.dot(res["A"], res["INV"])))
    print("6 A = {}".format(res["A"]))
    print("7 b = {}".format(res["b"]))


def resul_b(res: dict):
    print("Seidel method")
    res = seidel(res)
    print("0. new matrix A = {}".format(res["A_seidel"]))
    print("1. result x_seidel = {}".format(res["x_seidel"]))
    print("2. residual r = {}".format(np.dot(res["A_seidel"], res["x_seidel"]) - res["b"]))
    print("3. number of iterations = {}".format(res["it"]))

def main():
    res = {"A": A(), "b": b(), "INV": np.eye(7), "det": 0, "eps": 1e-4, "it": 0}
    resul_a(res)
    print("\n\n")
    resul_b(res)



if __name__ == "__main__":
    main()


