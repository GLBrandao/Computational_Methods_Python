def EIG_Jacobi_Generalized(K, M, threshold=None, tol=None, max_iterations=1000):
    """
    Computes eigenvalues and eigenvectors of a generalized eigenvalue problem Kx = λMx
    using the Jacobi method.

    Input:
        - K: Stiffness matrix (symmetric).
        - M: Mass matrix (symmetric and positive definite).
        - threshold: Threshold for applying rotations (default: 1e-6).
        - tol: Convergence tolerance (default: 1e-6).
        - max_iterations: Maximum iterations to avoid infinite loops.

    Returns:
        - eigenvalues: Array of computed eigenvalues.
        - eigenvectors: Matrix with eigenvectors as columns.
    """
    
    import numpy as np

    # Initialize parameters
    if threshold is None:
        threshold = 1e-6
    if tol is None:
        tol = 1e-6

    n = K.shape[0]
    eigenvectors = np.eye(n)
    iteration = 0

    def compute_convergence_error(K, M, p, q):
        """Calculate convergence errors for the off-diagonal elements."""
        error_K = np.sqrt(K[p, q]**2 / (np.abs(K[p, p] * K[q, q])))
        error_M = np.sqrt(M[p, q]**2 / (np.abs(M[p, p] * M[q, q])))
        return error_K, error_M

    def compute_rotation_params(K, M, p, q):
        """Calculate the rotation parameters gamma and alpha."""
        if np.isclose(K[p, p] * M[q, q], M[p, p] * K[q, q]):
            # Special case handling
            gamma = np.sqrt(
                (K[p, p] * M[p, q] - M[p, p] * K[p, q]) / 
                (K[q, q] * M[p, q] - M[q, q] * K[p, q])
            )
            alpha = -1 / gamma
        else:
            # General case
            z1 = (K[q, q] * M[p, q] - M[q, q] * K[p, q]) / (K[p, p] * M[q, q] - M[p, p] * K[q, q])
            z2 = (K[p, p] * M[p, q] - M[p, p] * K[p, q]) / (K[p, p] * M[q, q] - M[p, p] * K[q, q])
            gamma = (1 / z1) * (0.5 - np.sqrt(0.25 + z1 * z2))
            alpha = (-z1 / z2) * gamma
        return alpha, gamma

    while iteration < max_iterations:
        iteration += 1
        converged = True

        for p in range(n):
            for q in range(p + 1, n):
                # Calculate errors for convergence
                error_K, error_M = compute_convergence_error(K, M, p, q)

                if error_K > threshold or error_M > threshold:
                    converged = False

                    # Compute rotation parameters
                    alpha, gamma = compute_rotation_params(K, M, p, q)

                    # Construct rotation matrix
                    P = np.eye(n)
                    P[p, q] = alpha
                    P[q, p] = gamma

                    # Update K and M using similarity transformations
                    K = P.T @ K @ P
                    M = P.T @ M @ P

                    # Update eigenvectors
                    eigenvectors = eigenvectors @ P

        # Check convergence
        if converged:
            break
    else:
        print("Warning: Maximum iterations reached. Convergence may not have been achieved.")

    # Normalize eigenvectors
    normalization_factors = np.sqrt(np.diag(M))
    eigenvectors = eigenvectors @ np.diag(1 / normalization_factors)

    # Compute eigenvalues
    eigenvalues = np.diag(K) / np.diag(M)

    return eigenvalues, eigenvectors

import numpy as np

# Example usage
if __name__ == "__main__":
    K = np.array([[2, -1, 0],
                  [-1, 4, -1],
                  [0, -1, 2]])

    M = np.array([[0.5, 0.5, 0],
                  [0.5, 1, 0.5],
                  [0, 0.5, 1]])

    eigenvalues, eigenvectors = EIG_Jacobi_Generalized(K, M)
    print("Eigenvalues:\n", eigenvalues)
    print("Eigenvectors:\n", eigenvectors)
