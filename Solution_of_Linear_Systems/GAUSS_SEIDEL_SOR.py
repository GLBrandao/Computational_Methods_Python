def SOR(A, b, x0, max_ite, tol, w):

    """
    Solve a linear system of equations using the Gauss-Seidel method with Successive Over-Relaxation (SOR).
    Author: Guilherme Brandão
    Inputs:
        A: Coefficient matrix (must be square).
        b: Right-hand side vector.
        x0: Initial guess vector.
        max_ite: Maximum number of iterations.
        tol: Convergence tolerance.
        w: Relaxation factor (0 < w <= 2).
    Returns:
        x: Solution vector (if converged).
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
    
    n = len(b)

    # Extract diagonal, upper, and lower matrices
    D = np.diag(np.diag(A))
    U = np.triu(-A, 1)
    L = np.tril(-A, -1)

    # Compute iteration matrices
    Tw = np.linalg.inv(D - w * L) @ ((1 - w) * D + w * U)
    Cw = w * np.linalg.inv(D - w * L) @ b

    x_k = x0.copy()

    print("Starting SOR iterations...")

    for k in range(max_ite):
        x_new = Tw @ x_k + Cw
        conv = np.linalg.norm(x_new - x_k) / np.linalg.norm(x_new)
        print(f"Iteration {k + 1}: Solution = {x_new}, Convergence = {conv}")

        if conv < tol:
            print(f"Convergence achieved after {k + 1} iterations.")
            return x_new

        x_k = x_new.copy()

    print("Maximum iterations reached without convergence.")
    return x_k

import numpy as np
# Example usage
if __name__ == "__main__":
    A = np.array([[10, 2, 1], [1, 5, 1], [2, 3, 10]])
    b = np.array([7, -8, 6])
    x0 = np.ones(3)
    max_ite = 300
    tol = 1e-5
    w = 1  # Relaxation factor

    solution = SOR(A, b, x0, max_ite, tol, w)
    print(f"The solution is: {solution}")