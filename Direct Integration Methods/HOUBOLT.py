def Houbolt(M, C, K, initial_t, final_t, n, u0, v0, F_func, plot_dofs=None):
    """
    Solve a system of ODEs using the Houbolt method.
    Central Difference Method (CDM) is used to initialize the first two steps.

    Input:
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

    # Time step size
    dt = (final_t - initial_t) / n

    # Number of degrees of freedom (DOFs)
    ndof = M.shape[0]

    # Initialize arrays to store results
    u = np.zeros((ndof, n + 1))
    v = np.zeros((ndof, n + 1))
    a = np.zeros((ndof, n + 1))

    # Set initial conditions
    u[:, 0] = u0
    v[:, 0] = v0

    # Initial acceleration at t = 0
    u0_dd = np.linalg.solve(M, F_func(initial_t) - C @ v0 - K @ u0)
    a[:, 0] = u0_dd

    # Step 1: Use Central Difference Method for initialization
    u[:, 1] = u0 + dt * v0 + 0.5 * dt**2 * u0_dd
    v[:, 1] = (u[:, 1] - u[:, 0]) / dt
    a[:, 1] = np.linalg.solve(M, F_func(initial_t + dt) - C @ v[:, 1] - K @ u[:, 1])

    # Step 2: CDM second step
    u[:, 2] = 2 * u[:, 1] - u[:, 0] + dt**2 * a[:, 1]
    v[:, 2] = (u[:, 2] - u[:, 0]) / (2 * dt)
    a[:, 2] = np.linalg.solve(M, F_func(initial_t + 2 * dt) - C @ v[:, 2] - K @ u[:, 2])

    # Precompute constants for Houbolt method
    A = (4 / dt**2) * M + (3 / (2 * dt)) * C
    B = (5 / dt**2) * M + (3 / dt) * C
    D = (1 / dt**2) * M + (1 / (3 * dt)) * C
    K_hat = (2 / dt**2) * M + (11 / (6 * dt)) * C + K
    K_hat_inv = np.linalg.inv(K_hat)

    # Time step loop
    for i in range(3, n + 1):
        t = initial_t + i * dt

        # Effective force
        F_hat = F_func(t) + B @ u[:, i - 1] - A @ u[:, i - 2] + D @ u[:, i - 3]

        # Solve for displacement at t + dt
        u[:, i] = K_hat_inv @ F_hat

        # Velocity at t + dt
        v[:, i] = (1 / (6 * dt)) * (11 * u[:, i] - 18 * u[:, i - 1] + 9 * u[:, i - 2] - 2 * u[:, i - 3])

        # Acceleration at t + dt
        a[:, i] = (1 / dt**2) * (2 * u[:, i] - 5 * u[:, i - 1] + 4 * u[:, i - 2] - u[:, i - 3])

    # Create time array
    time_array = np.linspace(initial_t, final_t, n + 1)

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

    u, v, a, time_array = Houbolt(M, C, K, initial_t, final_t, n, u0, v0, F_func, plot_dofs)
