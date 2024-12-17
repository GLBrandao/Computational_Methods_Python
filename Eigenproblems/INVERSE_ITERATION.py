def EIG_Inverse_Iteration(K, M, n_max, tol, x0=None):
    """
    Inverse Iteration Method to find the smallest eigenvalue and corresponding eigenvector
    for the generalized eigenvalue problem Kx = λMx.

    Parameters:
        K: Stiffness matrix 
        M: Mass matrix
        n_max: Maximum number of iterations
        tol: Tolerance for convergence
        x0: Initial guess for eigenvector (optional)

    Returns:
        eig_value : Smallest eigenvalue
        eig_vector: Corresponding normalized eigenvector
    """

    import numpy as np
    import matplotlib.pyplot as plt

    # If no initial guess is provided, start with a vector of ones
    if x0 is None:
        x0 = np.ones(K.shape[0])
    
    # Normalize the initial guess
    x = x0 / np.linalg.norm(x0)
    
    # Precompute the inverse of K
    K_inv = np.linalg.inv(K)

    # Initialize variables
    eig_values = [0]  # Store eigenvalue estimates
    error_eig = np.inf  # Initial error
    iteration = 0

    print("Starting Inverse Iteration Method...")
    
    # Iterative process
    while error_eig > tol and iteration < n_max:
        iteration += 1

        # Solve for next eigenvector approximation
        y = np.dot(K_inv, np.dot(M, x))

        # Normalize the new vector
        x_new = y / np.linalg.norm(y)

        # Estimate the Rayleigh quotient (eigenvalue)
        eig_new = np.dot(x_new.T, np.dot(K, x_new)) / np.dot(x_new.T, np.dot(M, x_new))

        # Compute relative error for stopping condition
        if iteration > 1:
            error_eig = np.abs(eig_new - eig_values[-1]) / np.abs(eig_new)
        
        # Store new eigenvalue and update the vector
        eig_values.append(eig_new)
        x = x_new

        # Print progress
        print(f"Iteration {iteration}: Eigenvalue = {eig_new:.6f}, Error = {error_eig:.2e}")
    
    # Final eigenvalue and eigenvector
    eig_value = eig_values[-1]
    eig_vector = x
    
    print("Inverse Iteration Method Completed.")
    print(f"Smallest Eigenvalue: {eig_value:.6f}, Iterations: {iteration}")
    return eig_value, eig_vector

import numpy as np

# Example usage
if __name__ == "__main__":
    # Example matrices (replace with your own matrices)
    K = np.array([[2, -1, 0 ,0], [-1, 2, -1, 0], [0, -1, 2, -1], [0, 0, -1, 1]])
    M = np.array([[0, 0, 0, 0], [0, 2, 0, 0], [0, 0, 0, 0], [0, 0, 0, 1]])
    
    eig_value, eig_vector = EIG_Inverse_Iteration(K, M, n_max=100, tol=1e-6)
    print("Corresponding Eigenvector:\n", eig_vector)
