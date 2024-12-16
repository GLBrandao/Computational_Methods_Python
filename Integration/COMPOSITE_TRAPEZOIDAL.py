def composite_trapezoidal(a, b, n, f):
    """
    Approximate the integral of a function using the Composite Trapezoidal Rule.
    Author: Guilherme Brandão

    Input:
        a: Start of the interval.
        b: End of the interval.
        n: Number of subintervals (must be even).
        f: Function to integrate.

    Returns:
        integral: Approximation of the integral.
    """
    import numpy as np

    # Ensure n is positive
    if n <= 0:
        raise ValueError("Error! n must be a positive integer.")

    # Calculate step size
    h = (b - a) / n

    # Integration points
    X = np.linspace(a, b, n + 1)
    Y = f(X)

    print(f"Integration points: \n{X}")
    print(f"\nFunction values at the integration points: \n{Y}")

    # Composite Trapezoidal rule formula
    integral = (h / 2) * (Y[0] + 2 * np.sum(Y[1:-1]) + Y[-1])
    
    print(f"\nApproximate integral: {integral}")
    
    return integral

# Example usage
if __name__ == "__main__":
    # Define the function to integrate
    def f(x):
        return 1/x  # Example function

    a = 1  # Start of interval
    b = 2  # End of interval
    n = 1  # Number of subintervals

    result = composite_trapezoidal(a, b, n, f)
    print(f"\nThe integral from {a} to {b} is approximately {result}")
