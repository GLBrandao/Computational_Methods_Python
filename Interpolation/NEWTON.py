def newton_interpolation(x, y, P):
    """
    Code used to approximate a function value using Newton polynomial - Divided difference.

    Input:
        x: Input data points (x-coordinates).
        y: Corresponding function values (y-coordinates).
        P: x value at which y should be approximated.

    Returns:
        f_x: Approximated value at P.
        poly_expr: Newton polynomial in symbolic form.
    """

    import numpy as np
    import matplotlib.pyplot as plt
    import sympy as sp
    np.set_printoptions(precision=4)  

    # Ensure input arrays are NumPy arrays
    x = np.array(x, dtype=np.float64)
    y = np.array(y, dtype=np.float64)

    # Input validation
    if len(x) != len(y):
        raise ValueError("x and y must have the same length.")
    if len(x) < 2:
        raise ValueError("At least two data points are required.")

    n = len(x)
    F = np.zeros((n, n))

    # First column of matrix F should be the y known points
    F[:, 0] = y

    # Divided difference table
    for i in range(1, n):
        for j in range(1, i + 1):
            F[i, j] = (F[i, j-1] - F[i-1, j-1]) / (x[i] - x[i-j])

    # Newton polynomial coefficients
    C = np.array([F[n-1, n-1]])
    for k in range(n-2, -1, -1):
        C = np.convolve(C, np.poly([x[k]]))
        C[-1] += F[k, k]

    print(f"Newton polynomial coefficients: \n{C}\n")

    # Evaluate the approximate value at P
    f_x = np.polyval(C, P)
    print(f"The approximated value at x = {P} is: {f_x}")

    # Symbolic representation of the polynomial
    x_sym = sp.Symbol('x')
    poly_expr = sp.Poly(C[::-1], x_sym).as_expr()
    print(f"\nNewton polynomial: {poly_expr}")

    # Plot the polynomial and data points
    x_vals = np.linspace(x[0], x[-1], 1000)
    y_vals = np.polyval(C, x_vals)

    plt.figure(figsize=(10, 6))
    plt.plot(x_vals, y_vals, label="Newton Interpolation Polynomial")
    plt.scatter(x, y, color='red', label="Data Points")
    plt.scatter(P, f_x, color='green', marker="X", s=100, label=f"Interpolated value at x = {P}")
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title("Newton Interpolation")
    plt.legend()
    plt.grid(True)
    plt.show()

    return f_x, poly_expr

# Example usage
if __name__ == "__main__":
    x = [0, 1, 2, 3, 4]
    y = [1, 2, 0, 2, 3]
    P = 2.5
    result, polynomial = newton_interpolation(x, y, P)
    print(f"Interpolated value at x = {P}: {result}")
    print(f"Newton Polynomial: {polynomial}")