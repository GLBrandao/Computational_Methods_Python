def gauss_legendre(fun, xmin, xmax, n):
    """
    Perform numerical integration using Gauss-Legendre Quadrature.
    Author: Guilherme Brandão.

    Args:
        fun: Function to integrate.
        xmin: Lower boundary of integration.
        xmax: Upper boundary of integration.
        n: Number of integration points (order of the quadrature).

    Returns:
        integral: Approximation of the integral.
    """

    import numpy as np
    from scipy.special import roots_legendre

    if n <= 0:
        raise ValueError("Number of integration points 'n' must be positive.")

    # Get the integration points and weights for the standard interval [-1, 1]
    x_IP, weights = roots_legendre(n)

    # Transform the integration points to the interval [xmin, xmax]
    x_eval = 0.5 * (x_IP * (xmax - xmin) + (xmax + xmin))

    # Apply the Gauss-Legendre quadrature formula
    integral = 0.5 * (xmax - xmin) * np.sum(weights * fun(x_eval))

    return integral

# Example usage
if __name__ == "__main__":
    def f(x): # Define the function to integrate
        return 1/x 

    xmin = 1  # Lower limit
    xmax = 2  # Upper limit
    n = 4     # Number of integration points

    result = gauss_legendre(f, xmin, xmax, n)
    print(f"The integral of the function from {xmin} to {xmax} is approximately {result}")
