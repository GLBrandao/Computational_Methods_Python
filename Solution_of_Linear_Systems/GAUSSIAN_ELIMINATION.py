import numpy as np
np.set_printoptions(precision=4)  

def gauss (A,b):

    """
    Solve a linear system of equations using Gaussian elimination method.
    Author: Guilherme Brandão
    """
    # Ensure A and b are NumPy arrays
    A = np.array(A, dtype=np.float64)
    b = np.array(b, dtype=np.float64).flatten()

    if A.shape[0] != A.shape[1]: # Check if [A] is square
        raise ValueError ("Matrix A must be square.")
    
    if np.linalg.det(A) == 0: # Check if the system assume an unique solution
        raise ValueError("Matrix [A] is singular.")
    
    # Augment the matrix A with b
    augmented = np.hstack((A, b.reshape(-1, 1)))
    n = A.shape[0]

    # Gaussian elimination
    for i in range(n):
        # Pivoting: Swap rows if the pivot is zero
        if augmented[i, i] == 0:
            for k in range(i + 1, n):
                if augmented[k, i] != 0:
                    augmented[[i, k]] = augmented[[k, i]]
                    break
            else:
                raise ValueError("The system does not have a unique solution.")

        # Eliminate entries below the pivot
        for j in range(i + 1, n):
            factor = augmented[j, i] / augmented[i, i]
            augmented[j, i:] -= factor * augmented[i, i:]

    print(f"Augmented matrix after elimination:\n{augmented}\n")
    
    # Backward substitution
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (augmented[i, -1] - np.dot(augmented[i, i + 1:n], x[i + 1:])) / augmented[i, i]

    print(f"Solution vector x:\n{x}\n")
    return x

# Example usage
if __name__ == "__main__":
    A = np.array([[3, 1, -1], [2, 4, 1], [-1, 2, 5]])
    b = np.array([4, 1, 1])
    x = gauss(A, b)
    print(f"The solution is: {x}")