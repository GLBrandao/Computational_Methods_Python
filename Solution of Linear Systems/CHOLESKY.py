def cholesky (A,b):

    """
    Solve a linear system of equations using the Cholesky Decomposition Method
    Author: Guilherme Brandão
    """

    import numpy as np
    np.set_printoptions(precision=4)  

    # Ensure that A and b are Numpy arrays
    A = np.array(A, dtype=np.float64)
    b = np.array(b, dtype=np.float64).flatten()

    if A.shape[0] != A.shape[1]: # Check if [A] is square
        raise ValueError ("Matrix A must be square.")

    if not np.allclose(A, A.T): # Check if [A] is symmetric
        raise ValueError ("Matrix A must be symmetric.")
    
    if np.linalg.det(A) == 0: # Check if the system assume an unique solution
        raise ValueError("Matrix [A] is singular.")
    
    n = A.shape[0]
    L = np.zeros(A.shape)

    # Compute L (Cholesky Decomposition)
    for i in range(n):
        for j in range(i+1):  
            if i == j: # Diagonal Elements
                L[i, j] = np.sqrt(A[i, i] - np.sum(L[i, :j] ** 2))
            else: 
                L[i, j] = (A[i, j] - np.sum(L[i, :j] * L[j, :j])) / L[j, j]
    
    print(f"Matrix L:\n{L}\n")

    # Forward Substitution to solve Ly = b
    y = np.zeros(n)
    for i in range (0,n):
        y[i] = (b[i] - np.dot(L[i, :i],y[:i]))/L[i,i]
        
    print(f"Intermediate vector y:\n{y}\n")

    # Backward substitution to solve L.T x = y
    x = np.zeros(n)
    for i in range (n-1,-1,-1):
        x[i] = (y[i] - np.dot(L.T[i,i+1:],x[i+1:]))/L.T[i,i]

    print(f"The solution vector x is: \n{x}")

    return x

import numpy as np

# Example usage
if __name__ == "__main__":
    A = np.array([[6, 15, 55], [15, 55, 225], [55, 225, 979]])
    b = np.array([76, 295, 1259])
    x = cholesky(A, b)
    print(f"The solution is: {x}")
