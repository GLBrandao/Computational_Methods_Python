def simpson_1_3(a, b, n, f):
    """
    Approximate the integral of a function using the Composite Simpson's 1/3 Rule.
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
    
    # Ensure n is even
    if n % 2 != 0:
        raise ValueError("Error! n must be even.")

    # Calculate step size
    h = (b - a) / n

    # Integration points
    X = np.linspace(a, b, n + 1)
    Y = f(X)

    print(f"Integration points: \n{X}")
    print(f"\nFunction values at the integration points: \n{Y}")

    # Composite Simpson's 1/3 rule formula
    integral = (h / 3) * (Y[0] + 4 * np.sum(Y[1:n:2]) + 2 * np.sum(Y[2:n-1:2]) + Y[n])
    
    print(f"\nApproximate integral: {integral}")
    
    return integral

# Example usage
if __name__ == "__main__":
    # Define the function to integrate
    def f(x):
        return 1/x  # Example function

    a = 1  # Start of interval
    b = 2 # End of interval
    n = 2  # Number of subintervals (must be even)

    result = simpson_1_3(a, b, n, f)
    print(f"\nThe integral from {a} to {b} is approximately {result}")
