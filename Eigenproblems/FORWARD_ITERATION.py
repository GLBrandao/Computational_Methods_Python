import numpy as np
import matplotlib.pyplot as plt

def forward_iteration(K, M, n_max, tol, x1=None):
    """
    Find the first eigenvalue and eigenvector using the Forward Iteration Method.

    Args:
        K: Stiffness matrix.
        M: Mass matrix.
        n_max: Maximum number of iterations.
        tol (float): Convergence tolerance.
        x1 (array, optional): Initial guess for the eigenvector. Defaults to a unit vector.

    Returns:
        eig, eigenvector : Eigenvalue and eigenvector.
    """
    # Inverse of the mass matrix
    M_inv = np.linalg.inv(M)

    # Default initial guess if none provided
    if x1 is None:
        x1 = np.ones(M.shape[0])

    # Initialize variables for eigenvalue and iteration
    eig_value = []  # Store eigenvalues
    hist_eig = []   # Track eigenvalue history
    errors = []     # Track errors
    error_eig = 1   # Initial error
    i = 0           # Iteration counter

    # Normalize initial guess
    y = x1 / np.linalg.norm(x1)

    # Forward iteration
    while error_eig >= tol and i < n_max:
        i += 1

        # Update eigenvector approximation
        x_k = M_inv @ (K @ y)

        # Compute new eigenvalue (Rayleigh quotient)
        eig_new = (y.T @ K @ y) / (y.T @ M @ y)

        # Normalize the new eigenvector
        y = x_k / np.linalg.norm(x_k)

        # Calculate error if not the first iteration
        if len(eig_value) > 0:
            error_eig = np.abs(eig_new - eig_value[-1]) / np.abs(eig_new)
        else:
            error_eig = tol  # Set error to a high value for the first iteration

        # Store the current eigenvalue
        eig_value.append(eig_new)
        hist_eig.append(eig_new)

        # Check convergence
        if error_eig < tol:
            break

    # Final eigenvalue and eigenvector
    eig = eig_value[-1]
    eigenvector = y

    # Plot Eigenvalue Convergence
    plt.figure(figsize=(6, 4))
    plt.plot(hist_eig, marker='o', label="Eigenvalue Estimate")
    plt.xlabel("Iteration")
    plt.ylabel("Eigenvalue")
    plt.title("Eigenvalue Convergence")
    plt.grid(True)
    plt.legend()
    plt.show()

    return eig, eigenvector

# Example usage
if __name__ == "__main__": 
    # Example matrices
    K = np.array([[5, -4, 1 ,0], [-4, 6, -4, 1], [1, -4, 6, -4],[0, 1, -4, 5]])
    M = np.array([[2, 0, 0, 0], [0, 2, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])

    # Parameters
    n_max = 20
    tol = 10e-6

    # Run Forward Iteration
    eig, eigenvector = forward_iteration(K, M, n_max, tol)

    print(f"Eigenvalue: {eig}")
    print(f"Eigenvector: {eigenvector}")
