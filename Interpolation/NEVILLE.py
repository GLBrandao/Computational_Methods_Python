def neville(x, y, X=None):
    """
    Perform Neville's interpolation for a given dataset and optionally plot the results.
    Author: Guilherme Brandão 

    Args:
        x: Input data points (x-coordinates).
        y: Corresponding function values (y-coordinates).
        X (optional): Point at which to evaluate the interpolation. Defaults to None.

    Returns:
        interpolated_value: Interpolated value at X.
        Q: Neville matrix.
    """
    import numpy as np
    import matplotlib.pyplot as plt

    # Ensure input arrays are NumPy arrays
    x = np.array(x, dtype=np.float64)
    y = np.array(y, dtype=np.float64)

    # Input validation
    if len(x) != len(y):
        raise ValueError("x and y must have the same length.")
    if len(x) < 2:
        raise ValueError("At least two data points are required.")
    
    # Initialize Neville's Q matrix
    N = len(x)
    Q = np.zeros((N, N))
    for i in range(N):
        Q[i, 0] = y[i]

    # Fill Neville's Q matrix
    for i in range(1, N):
        for j in range(1, i + 1):
            Q[i, j] = ((X - x[i-j]) * Q[i, j-1] - (X - x[i]) * Q[i-1, j-1]) / (x[i] - x[i-j])

    interpolated_value = Q[-1, -1]

    # Print the Q matrix and interpolated value (if X is provided)
    if X is not None:
        print(f"\nMatrix [Q] is:\n{Q}")
        print(f"\nApproximation at point X = {X} is {interpolated_value}")

    # Plot the interpolation
    x_plot = np.linspace(min(x), max(x), 500)
    y_plot = []
    for xi in x_plot:
        # Evaluate Neville's interpolation for each xi in x_plot
        Q_temp = np.zeros((N, N))
        for i in range(N):
            Q_temp[i, 0] = y[i]
        for i in range(1, N):
            for j in range(1, i + 1):
                Q_temp[i, j] = ((xi - x[i-j]) * Q_temp[i, j-1] - (xi - x[i]) * Q_temp[i-1, j-1]) / (x[i] - x[i-j])
        y_plot.append(Q_temp[-1, -1])

    plt.figure(figsize=(10, 6))
    plt.plot(x, y, 'ro', label='Data Points')
    plt.plot(x_plot, y_plot, 'b-', label='Neville Interpolation')
    if X is not None:
        plt.plot(X, interpolated_value, 'go', label=f'Interpolated Point X = {X}')
    plt.title("Neville's Interpolation")
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.grid(True)
    plt.show()

    return interpolated_value, Q

# Example usage
if __name__ == "__main__":
    x = [0, 1, 2, 3, 4]
    y = [1, 2, 0, 2, 3]
    X = 2.5
    result = neville(x, y, X)