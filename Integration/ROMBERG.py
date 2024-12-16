def romberg_integration(f, a, b, n):
    """
    Perform numerical integration using Romberg's method.
    Author: Guilherme Brandão.
    
    Input:
        f: Function to integrate.
        a: Lower boundary of integration.
        b: Upper boundary of integration.
        n: Number of iterations (levels of refinement).

    Returns:
        result: Approximation of the integral.
    """

    import numpy as np

    if n <= 0:
        raise ValueError("Number of iterations 'n' must be positive.")

    # Create a table to store results
    r = np.zeros((n + 1, n + 1))
    h = b - a

    # Trapezoidal approximation
    r[0, 0] = (f(a) + f(b)) * h / 2

    # Romberg integration table
    for i in range(1, n + 1):
        # Compute the sum for the trapezoidal rule
        num_points = 2**(i - 1)  # Number of new points to add
        sum_val = sum(f(a + (k - 0.5) * h) for k in range(1, num_points + 1))

        # Update the current trapezoidal estimate
        r[i, 0] = (r[i - 1, 0] + h * sum_val) / 2

        # Richardson extrapolation
        for j in range(1, i + 1):
            factor = 4**j
            r[i, j] = (factor * r[i, j - 1] - r[i - 1, j - 1]) / (factor - 1)

        # Print the current row of the table
        print(" ".join(f"{r[i, j]:.6f}" for j in range(i + 1)))

        # Update the step size
        h /= 2

    result = r[n, n]
    print(f"\nIntegration value: {result:.4f}")

    return result

import numpy as np

# Example usage
if __name__ == "__main__":
    def f(x): # Define the function to integrate
        return np.sin(x)

    a = 0  # Start of interval
    b = np.pi  # End of interval
    n = 4  # Number of iterations

    result = romberg_integration(f, a, b, n)
    print(f"\nThe integral from {a} to {b} is approximately {result:.4f}")
