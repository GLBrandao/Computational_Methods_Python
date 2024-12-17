def EIG_Jacobi_Standard(K, threshold=None, tol=None, max_iterations=1000):
    """
    Computes all eigenvalues and eigenvectors of a symmetric matrix using the Jacobi method.

    Input:
        - K: Symmetric matrix.
        - threshold: Threshold for applying the rotation (default: 1e-6).
        - tol: Tolerance for convergence (default: 1e-6).
        - max_iterations: Maximum allowed iterations.

    Returns:
        - sorted_eigenvalues: Eigenvalues sorted in ascending order.
        - sorted_eigenvectors: Corresponding eigenvectors as columns.
    """

    import numpy as np

    if threshold is None:
        threshold = 1e-6
    if tol is None:
        tol = 1e-6

    n = K.shape[0]
    eigenvectors = np.eye(n)
    iteration = 0

    def rotation_matrix(n, p, q, c, s):
        """Jacobi rotation matrix."""
        P = np.eye(n)
        P[p, p] = c
        P[q, q] = c
        P[p, q] = -s
        P[q, p] = s
        return P

    while iteration < max_iterations:
        max_off_diag = 0
        for p in range(n):
            for q in range(p + 1, n):
                if np.abs(K[p, q]) > threshold:
                    max_off_diag = max(max_off_diag, np.abs(K[p, q]))

                    # Compute rotation parameters
                    if K[p, p] == K[q, q]:
                        theta = np.pi / 4
                    else:
                        theta = 0.5 * np.arctan2(2 * K[p, q], K[p, p] - K[q, q])
                    c = np.cos(theta)
                    s = np.sin(theta)

                    # Construct the rotation matrix
                    P = rotation_matrix(n, p, q, c, s)

                    # Update the stiffness matrix and eigenvectors
                    K = P.T @ K @ P
                    eigenvectors = eigenvectors @ P

        # Check convergence based on the maximum off-diagonal element
        if max_off_diag < tol:
            break
        iteration += 1
    else:
        print("Warning: Maximum iterations reached. Convergence may not have been achieved.")

    # Extract eigenvalues and sort them
    eigenvalues = np.diag(K)
    sorted_indices = np.argsort(eigenvalues)
    sorted_eigenvalues = eigenvalues[sorted_indices]
    sorted_eigenvectors = eigenvectors[:, sorted_indices]

    return sorted_eigenvalues, sorted_eigenvectors

import numpy as np

# Example usage
if __name__ == "__main__":
    K = np.array([[12, 6, -6],
                  [6, 16, 2],
                  [-6, 2, 16]])

    eigenvalues, eigenvectors = EIG_Jacobi_Standard(K)
    print("Eigenvalues:\n", eigenvalues)
    print("Eigenvectors:\n", eigenvectors)
