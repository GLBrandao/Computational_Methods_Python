def newmark(M, C, K, initial_t, final_t, n, u0, v0, F_func, plot_dofs=None):
    """
    Solve a system of ODEs using the Newmark-beta method.

    Args:
        M: Mass matrix.
        C: Damping matrix.
        K: Stiffness matrix.
        initial_t: Start time.
        final_t: End time.
        n: Number of time steps.
        u0: Initial displacements.
        v0: Initial velocities.
        F_func: Function to compute the external force vector as a function of time.
        plot_dofs (list, optional): List of DOFs to plot. Defaults to None (plots all DOFs).

    Returns:
        u, v, a, time_array:: Displacement, velocity, acceleration arrays, and time array.
    """

    import numpy as np
    import matplotlib.pyplot as plt

    if n <= 0:
        raise ValueError("Number of time steps 'n' must be positive.")

    # Newmark-beta parameters
    beta = 0.25
    gamma = 0.5

    # Time step size
    dt = (final_t - initial_t) / n
    print(f"Time Step: {dt:.4f}\n")

    # Number of degrees of freedom (DOFs)
    ndof = M.shape[0]

    # Precompute constants
    K_hat = (1 / (beta * dt**2)) * M + (gamma / (beta * dt)) * C + K
    K_hat_inv = np.linalg.inv(K_hat)

    # Initialize arrays to store displacement, velocity, and acceleration
    u = np.zeros((ndof, n + 1))
    v = np.zeros((ndof, n + 1))
    a = np.zeros((ndof, n + 1))

    # Set initial conditions
    u[:, 0] = u0
    v[:, 0] = v0
    a[:, 0] = np.linalg.solve(M, F_func(initial_t) - C @ v0 - K @ u0)

    # Constants for time stepping
    alpha_M = 1 / (beta * dt**2)
    alpha_C = gamma / (beta * dt)

    # Time step loop
    time_array = np.linspace(initial_t, final_t, n + 1)
    for i in range(1, n + 1):
        t = time_array[i]
        F_t = F_func(t)

        # Effective force vector
        F_hat = F_t + M @ (alpha_M * u[:, i - 1] + (1 / (beta * dt)) * v[:, i - 1] + (1 / (2 * beta) - 1) * a[:, i - 1])
        F_hat += C @ (alpha_C * u[:, i - 1] + (gamma / beta - 1) * v[:, i - 1] + dt * (gamma / (2 * beta) - 1) * a[:, i - 1])

        # Solve for displacement
        u[:, i] = K_hat_inv @ F_hat

        # Update acceleration and velocity
        a[:, i] = alpha_M * (u[:, i] - u[:, i - 1]) - (1 / (beta * dt)) * v[:, i - 1] - ((1 / (2 * beta)) - 1) * a[:, i - 1]
        v[:, i] = v[:, i - 1] + dt * ((1 - gamma) * a[:, i - 1] + gamma * a[:, i])

    # If no DOFs are specified for plotting, plot all
    if plot_dofs is None:
        plot_dofs = list(range(ndof))

    # Plot results
    for quantity, label in zip([u, v, a], ["Displacement", "Velocity", "Acceleration"]):
        plt.figure(figsize=(8, 5))
        for dof in plot_dofs:
            plt.plot(time_array, quantity[dof, :], label=f"DOF-{dof + 1}")
        plt.title(f"{label} vs Time")
        plt.xlabel("Time (s)")
        plt.ylabel(label)
        plt.legend()
        plt.grid(True)
        plt.show()

    return u, v, a, time_array

import numpy as np

# Example usage
if __name__ == "__main__":

    M = np.array([[2.0, 0.0], [0.0, 1.0]])
    K = np.array([[6.0, -2.0], [-2.0, 4.0]])
    C = np.zeros((2, 2))  # No damping

    initial_t = 0.0
    final_t = 3.36
    n = 12

    u0 = np.array([0.0, 0.0])
    v0 = np.array([0.0, 0.0])

    # Define the external force function
    def F_func(t):
        return np.array([0.0, 10.0])  # Constant force applied to DOF-2

    plot_dofs = [0, 1]

    u, v, a, time_array = newmark(M, C, K, initial_t, final_t, n, u0, v0, F_func, plot_dofs)
