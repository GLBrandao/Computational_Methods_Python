def inverse_iteration(K, M, x0, Phi_m, tol=None, ite_max=None):
    """
    Inverse Iteration Method with Orthogonalization to find the smallest eigenvalue and eigenvector.

    PiNPUTS:
        - K: Stiffness matrix (symmetric and positive definite).
        - M: Mass matrix (symmetric and positive definite).
        - x0: Initial guess vector.
        - Phi_m: Matrix of previously computed eigenvectors for orthogonalization.
        - tol: Tolerance for convergence (default: 1e-6).
        - ite_max: Maximum number of iterations (default: 500).

    Returns:
        - lambda_k: Computed eigenvalue.
        - phi: Computed eigenvector.
    """

    import numpy as np


    # Defaults for tolerance and maximum iterations
    if tol is None:
        tol = 1e-6
    if ite_max is None:
        ite_max = 500

    # Normalize the initial guess with respect to M
    x_norm_k = x0 / np.sqrt(x0.T @ M @ x0)
    lambda_k = 0
    nite = 0
    error = tol + 1  # Initialize error larger than tolerance

    while error >= tol and nite < ite_max:
        nite += 1

        # Solve (K - λM)y = Mx
        y_k = np.linalg.solve(K, M @ x_norm_k)

        # Orthogonalize with respect to previous eigenvectors
        if Phi_m.size != 0:
            for j in range(Phi_m.shape[1]):
                coeff = (Phi_m[:, j].T @ M @ y_k)
                y_k -= coeff * Phi_m[:, j]

        # Normalize y_k with respect to M
        x_norm_k1 = y_k / np.sqrt(y_k.T @ M @ y_k)

        # Rayleigh quotient to compute lambda
        lambda_k1 = (x_norm_k1.T @ K @ x_norm_k1) / (x_norm_k1.T @ M @ x_norm_k1)

        # Relative error
        error = abs(lambda_k1 - lambda_k) / abs(lambda_k1) if lambda_k1 != 0 else np.inf

        # Update variables for next iteration
        lambda_k = lambda_k1
        x_norm_k = x_norm_k1

    if error >= tol:
        print("Warning: Inverse iteration did not converge.")
    return lambda_k, x_norm_k


def G_S_ortho(K, M, tol=None, ite_max=None):
    """
    Computes the smallest eigenvalues and eigenvectors using Inverse Iteration 
    and Gram-Schmidt Orthogonalization.

    Parameters:
        - K: Stiffness matrix (symmetric and positive definite).
        - M: Mass matrix (symmetric and positive definite).
        - tol: Tolerance for convergence (default: 1e-6).
        - ite_max: Maximum number of iterations (default: 500).

    Returns:
        - eigenvalues: Array of computed eigenvalues.
        - Phi: Matrix of computed eigenvectors.
    """

    import numpy as np

    # Defaults
    if tol is None:
        tol = 1e-6
    if ite_max is None:
        ite_max = 500

    n = K.shape[0]
    Phi = np.zeros((n, n))  # Eigenvector matrix
    eigenvalues = np.zeros(n)  # Eigenvalue array

    # Initial guess: Ones vector
    x0 = np.ones(n)
    x0 = x0 / np.sqrt(x0.T @ M @ x0)

    # Compute the smallest eigenvalue and eigenvector using Inverse Iteration
    lambda1, phi1 = inverse_iteration(K, M, x0, np.array([]), tol, ite_max)
    eigenvalues[0] = lambda1
    Phi[:, 0] = phi1

    # Compute the remaining eigenvalues and eigenvectors
    for i in range(1, n):
        # Extract previous eigenvectors
        Phi_m = Phi[:, :i]

        # Enforce orthogonality
        S_m = np.eye(n) - Phi_m @ (Phi_m.T @ M)
        M_hat = M @ S_m

        # Next eigenpair
        lambda_i, phi_i = inverse_iteration(K, M_hat, x0, Phi_m, tol, ite_max)

        # Store results
        eigenvalues[i] = lambda_i
        Phi[:, i] = phi_i

    return eigenvalues, Phi

import numpy as np


if __name__ == "__main__":

    K = np.array([[2, -1, 0],
                  [-1, 4, -1],
                  [0, -1, 2]])

    M = np.array([[0.5, 0.5, 0],
                  [0.5, 1, 0.5],
                  [0, 0.5, 1]])

    # Solve for eigenvalues and eigenvectors
    eigenvalues, eigenvectors = G_S_ortho(K, M)

    # Print results
    print("Eigenvalues:\n", eigenvalues)
    print("Eigenvectors:\n", eigenvectors)
