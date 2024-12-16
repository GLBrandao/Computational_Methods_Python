import numpy as np
import matplotlib.pyplot as plt

def Bathe(M, C, K, initial_t, final_t, n, u0, v0, F_func, plot_dofs=None):
    """
    Solve a system of ODEs using the Bathe time integration method.
    Author: Guilherme Brandão.
    
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
        tuple: Displacement, velocity, acceleration arrays, and time array.
    """
    if n <= 0:
        raise ValueError("Number of time steps 'n' must be positive.")

    # Time step size
    dt = (final_t - initial_t) / n

    # Ensure initial conditions are numpy arrays
    u0 = np.asarray(u0)
    v0 = np.asarray(v0)

    # Initial acceleration
    a0 = np.linalg.solve(M, F_func(initial_t) - C @ v0 - K @ u0)

    # Number of degrees of freedom (DOFs)
    ndof = M.shape[0]

    # Initialize arrays to store results
    u = np.zeros((ndof, n + 1))
    v = np.zeros((ndof, n + 1))
    a = np.zeros((ndof, n + 1))

    # Set initial conditions
    u[:, 0] = u0
    v[:, 0] = v0
    a[:, 0] = a0

    # Precompute modified stiffness matrices
    K_hat1 = (16 / dt**2) * M + (4 / dt) * C + K
    K_hat2 = (9 / dt**2) * M + (3 / dt) * C + K

    for t_step in range(n):
        t = initial_t + t_step * dt

        # First sub-step at t + 0.5*dt
        F_half = F_func(t + 0.5 * dt)
        a_M1 = (16 / dt**2) * u[:, t_step] + (8 / dt) * v[:, t_step] + a[:, t_step]
        a_C1 = (4 / dt) * u[:, t_step] + v[:, t_step]
        F_hat_half = F_half + M @ a_M1 + C @ a_C1
        u_half = np.linalg.solve(K_hat1, F_hat_half)
        v_half = (4 / dt) * (u_half - u[:, t_step]) - v[:, t_step]

        # Second sub-step at t + dt
        F_full = F_func(t + dt)
        a_M2 = (12 / dt**2) * u_half + (4 / dt) * v_half - (3 / dt**2) * u[:, t_step] - (1 / dt) * v[:, t_step]
        a_C2 = (4 / dt) * u_half - (1 / dt) * u[:, t_step]
        F_hat_full = F_full + M @ a_M2 + C @ a_C2
        u_full = np.linalg.solve(K_hat2, F_hat_full)
        v_full = (4 / dt) * (u_full - u_half) - v_half
        a_full = np.linalg.solve(M, F_func(t + dt) - C @ v_full - K @ u_full)

        # Store results
        u[:, t_step + 1] = u_full
        v[:, t_step + 1] = v_full
        a[:, t_step + 1] = a_full

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

# Example usage
if __name__ == "__main__":
    # Example system
    M = np.array([[2.0, 0.0], [0.0, 1.0]])
    K = np.array([[6.0, -2.0], [-2.0, 4.0]])
    C = np.zeros((2, 2))

    initial_t = 0.0
    final_t = 3.36
    n = 12

    u0 = [0.0, 0.0]
    v0 = [0.0, 0.0]

    def F_func(t):
        return np.array([0.0, 10.0])  # Constant force applied to DOF-2

    plot_dofs = [0, 1]

    u, v, a, time_array = Bathe(M, C, K, initial_t, final_t, n, u0, v0, F_func, plot_dofs)