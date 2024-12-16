def jacobi(A,b,x0,max_ite,tol):
    """
    Solve a linear system of equations using the Jacobi Iterative Method.
    Author: Guilherme Brandão
    """
    import numpy as np
    np.set_printoptions(precision=6)  

    # Ensure A, b, x0 are NumPy arrays
    A = np.array(A, dtype=np.float64)
    b = np.array(b, dtype=np.float64).flatten()
    x0 = np.array(x0, dtype=np.float64)

    if A.shape[0] != A.shape[1]: # Check if [A] is square
        raise ValueError ("Matrix A must be square.")
    
    if np.linalg.det(A) == 0: # Check if the system assume an unique solution
        raise ValueError("Matrix [A] is singular.")
    
    n = A.shape[0]

    # Extract diagonal matrix D and compute D_inv
    D = np.diag(np.diag(A))
    D_inv = np.linalg.inv(D)

    # Extract upper and lower triangular matrices
    U = np.triu(A, 1)
    L = np.tril(A, -1)

    # Compute iteration matrices
    Tj = -np.dot(D_inv, L + U)
    Cj = np.dot(D_inv, b)

    # Initialize variables for iteration
    iteration_count = 0
    error = tol + 1
    x = x0.copy()

    print("Starting Jacobi iterations...")

    # Iterative process
    while iteration_count < max_ite and error >= tol:
        x_new = np.dot(Tj, x0) + Cj
        error = np.linalg.norm(x_new - x0) / np.linalg.norm(x_new)
        iteration_count += 1
        x0 = x_new.copy()

        print(f"Iteration {iteration_count}: Solution = {x_new}, Error = {error}")

    if error < tol:
        print("Convergence achieved.")
    else:
        print("Maximum iterations reached without convergence.")

    print(f"Final solution vector: {x_new}")
    print(f"Total iterations: {iteration_count}")

    return x_new

import numpy as np
# Example usage
if __name__ == "__main__":
    A = np.array([[10, 2, 1], [1, 5, 1], [2, 3, 10]])
    b = np.array([7, -8, 6])
    x0 = np.ones(3)
    max_ite = 300
    tol = 1e-5

    solution = jacobi(A, b, x0, max_ite, tol)