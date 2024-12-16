def CDM(M, C, K, initial_t, final_t, n, u0, v0, F_func, plot_dofs=None):
    """
    Solve a system of ODEs using the Central Difference Method.

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
    
    # Initial acceleration at t = 0
    u0_dd = np.linalg.solve(M, F_func(initial_t) - C @ v0 - K @ u0)

    # Initial displacement at t = -dt
    u_start = u0 - dt * v0 + 0.5 * dt**2 * u0_dd

    # Precompute matrices
    A = (1 / dt**2) * M - (1 / (2 * dt)) * C
    B = K - (2 / dt**2) * M
    K_hat = (1 / dt**2) * M + (1 / (2 * dt)) * C
    K_hat_inv = np.linalg.inv(K_hat)

    # Initialize arrays to store results
    ndof = M.shape[0]
    u = np.zeros((ndof, n + 1))
    v = np.zeros((ndof, n + 1))
    a = np.zeros((ndof, n + 1))

    # Set initial conditions
    u[:, 0] = u0
    v[:, 0] = v0
    a[:, 0] = u0_dd

    # Time step loop
    for i in range(1, n + 1):
        t = initial_t + i * dt

        # Force vector at time t
        F_t = F_func(t)

        # Force at time t
        F_hat = F_t - B @ u[:, i - 1] - A @ u_start

        # Displacement at time t + dt
        u_t = K_hat_inv @ F_hat

        # Update displacement and calculate velocity/acceleration
        u_start = u[:, i - 1]
        u[:, i] = u_t

        if i == 1:
            v[:, i] = (u[:, i] - u[:, i - 1]) / dt
            a[:, i] = (u[:, i] - 2 * u[:, i - 1] + u_start) / dt**2
        else:
            v[:, i] = (u[:, i] - u[:, i - 2]) / (2 * dt)
            a[:, i] = (u[:, i] - 2 * u[:, i - 1] + u[:, i - 2]) / dt**2

    # Time array
    time_array = np.linspace(initial_t, final_t, n + 1)

    # Plot results if required
    if plot_dofs is None:
        plot_dofs = list(range(ndof))

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

if __name__ == "__main__":
    # Example system
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

    u, v, a, time_array = CDM(M, C, K, initial_t, final_t, n, u0, v0, F_func, plot_dofs)