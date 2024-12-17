def LU_fact(A,b):

    """
    Solve a linear system of equations using Gaussian elimination method.
    Author: Guilherme Brandão
    """
    import numpy as np
    np.set_printoptions(precision=4) 

    # Ensure A and b are NumPy arrays
    A = np.array(A, dtype=np.float64)
    b = np.array(b, dtype=np.float64).flatten()

    if A.shape[0] != A.shape[1]: # Check if [A] is square
        raise ValueError ("Matrix A must be square.")
    
    if np.linalg.det(A) == 0: # Check if the system assume an unique solution
        raise ValueError("Matrix [A] is singular.")
    
    n = A.shape[0]
    L = np.eye(n)  # Initialize L as an identity matrix
    U = A.copy()   # Initialize U as a copy of A

    for i in range(n): # LU decomposition
        for j in range(i + 1, n):
            factor = U[j, i] / U[i, i]
            L[j, i] = factor
            U[j, i:] -= factor * U[i, i:]

    print(f"Lower triangular matrix L:\n{L}\n")
    print(f"Upper triangular matrix U:\n{U}\n")

    # Forward substitution to solve L * y = b
    y = np.zeros(n)
    for i in range(n):
        y[i] = (b[i] - np.dot(L[i, :i], y[:i])) / L[i, i]

    print(f"Intermediate vector y:\n{y}\n")
    
    # Backward substitution to solve U * x = y
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (y[i] - np.dot(U[i, i + 1:], x[i + 1:])) / U[i, i]

    print(f"Solution vector x:\n{x}\n")

    return x

import numpy as np

# Example usage
if __name__ == "__main__":
    A = np.array([[3, 1, -1], [2, 4, 1], [-1, 2, 5]])
    b = np.array([4, 1, 1])
    x = LU_fact(A, b)
    print(f"The solution is: {x}")