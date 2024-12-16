def cubic_spline(x, y, X):
    """
    Perform cubic spline interpolation for a given dataset and evaluate at a specified point.
    Author: Guilherme Brandão
    Input:
        x: Input data points (x-coordinates).
        y: Corresponding function values (y-coordinates).
        X: The point where the spline should be evaluated.
    Returns:
        float: Interpolated value at X.
    """

    import numpy as np
    import matplotlib.pyplot as plt

    # Ensure x and y are NumPy arrays
    x = np.array(x, dtype=np.float64)
    y = np.array(y, dtype=np.float64)

    # Input validation
    if len(x) != len(y):
        raise ValueError("Arrays x and y must have the same length.")
    if len(x) < 2:
        raise ValueError("At least two data points are required.")

    n = len(x)
    h = np.diff(x)  # Step sizes between x[i] and x[i+1]
    alpha = np.zeros(n)

    # Compute alpha
    for i in range(1, n-1):
        alpha[i] = (3/h[i] * (y[i+1] - y[i])) - (3/h[i-1] * (y[i] - y[i-1]))

    # Setup the coefficient matrix A and the right-hand side vector alpha
    A = np.zeros((n, n))
    A[0, 0] = A[n-1, n-1] = 1  # Natural spline boundary conditions

    for i in range(1, n-1):
        A[i, i-1] = h[i-1]
        A[i, i] = 2 * (h[i-1] + h[i])
        A[i, i+1] = h[i]

    # Solve for c coefficients
    c = np.linalg.solve(A, alpha)

    # Compute b and d coefficients
    b = np.zeros(n-1)
    d = np.zeros(n-1)

    for i in range(n-1):
        b[i] = (y[i+1] - y[i]) / h[i] - (h[i] * (2*c[i] + c[i+1])) / 3
        d[i] = (c[i+1] - c[i]) / (3*h[i])

    # Evaluate the spline at X
    spline_value = None
    for i in range(n-1):
        if x[i] <= X <= x[i+1]:
            dx = X - x[i]
            spline_value = y[i] + b[i]*dx + c[i]*dx**2 + d[i]*dx**3
            break

    if spline_value is None:
        raise ValueError(f"X={X} is outside the interpolation range.")

    print(f"Interpolated value at X={X}: {spline_value}")

    # Plot the cubic spline
    plt.scatter(x, y, color='red', label="Data Points")

    x_vals = np.linspace(x[0], x[-1], 1000)
    y_vals = np.zeros_like(x_vals)

    for i in range(n-1):
        indices = np.where((x_vals >= x[i]) & (x_vals <= x[i+1]))
        dx = x_vals[indices] - x[i]
        y_vals[indices] = y[i] + b[i]*dx + c[i]*dx**2 + d[i]*dx**3

    plt.plot(x_vals, y_vals, label="Cubic Spline Interpolation")

    # Highlight the interpolated point
    plt.scatter(X, spline_value, color='green', marker='X', s=100, label=f"Interpolated value at X = {X}")

    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title("Cubic Spline Interpolation")
    plt.legend()
    plt.grid(True)
    plt.show()

    return spline_value

# Example Usage
if __name__ == "__main__":
    x = [0, 1, 2, 3, 4]
    y = [1, 2, 0, 2, 3]
    X = 2.5
    cubic_spline(x, y, X)
