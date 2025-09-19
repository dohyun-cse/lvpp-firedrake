from firedrake import *

import matplotlib.pyplot as plt
plt.ion() # Turn on interactive mode

alpha_max = 1e05

# Define the domain
mesh = PeriodicRectangleMesh(128, 128, 1.0, 1.0)
dt = 1.0 / 128 / 5
T = 1.0
# Define function spaces
primal_space = FunctionSpace(mesh, "CG", 2)
latent_space = FunctionSpace(mesh, "DG", 0)
V = primal_space * latent_space
W = VectorFunctionSpace(mesh, "CG", 2, dim=2) # For velocity field

# Define the right-hand side
x, y = SpatialCoordinate(mesh)
b = Function(W).interpolate(5*as_vector((y - 0.5, -x + 0.5))) # Velocity field
K = Constant(0.01) # Diffusion coefficient
u_init = Function(primal_space).interpolate(exp(-50*((x - 0.25)**2 + (y - 0.5)**2)))

# Solution (Symbolic + Discrete) 
sol = Function(V)
u, psi = split(sol) # symbolic representation
uh, psih = sol.subfunctions # discrete representation
sol.assign(0.0) # Initialize to zero
uh.assign(u_init) # Initial condition

# Previous time step solution
sol_prev = Function(V)
u_prev, psi_prev = sol_prev.subfunctions

# Previous iteration
sol_k = Function(V)
u_k, psi_k = sol_k.subfunctions

# For visualization of the alternative solution, u = grad R^*(psi)
mapped_uh = Function(latent_space)

# Mirror map (\nabla R^*)
mirror_map = exp(psi)

# Define the saddle-point Lagrangian
v, phi = TestFunctions(V)
alpha = Constant(1.0) # Prox step size
A = (u - u_prev)*v + dt*(dot(K*grad(u), grad(v)) + dot(b, grad(u_prev))*v)
prox_diff = (psi - psi_k)*v 
consistency = (u - mirror_map)*phi
F = (alpha*A + prox_diff + consistency)*dx

# Plotting setup
fig_primal = plt.figure()
fig_dual = plt.figure()
def plot_solution(fig, uh):
    fig.clear()
    axes = fig.add_subplot(111)
    colors = tricontourf(uh, axes=axes)
    fig.colorbar(colors, ax=axes)
    fig.canvas.draw()
    fig.canvas.flush_events()

t = 0.0
alpha.assign(1.0)
# Map the dual variable to the primal space for plotting
mapped_uh.interpolate(mirror_map)
plot_solution(fig_primal, uh)
plot_solution(fig_dual, mapped_uh)
while t < T:
    t += dt
    sol_prev.assign(sol)
    print(f'Time = {t:.3f}')
    for i in range(1, 101):
        sol_k.assign(sol)
        for attempt in range(5):
            print(f'\tIteration {i:3d}, alpha = {alpha.values()[0]:.2e}', end='')
            try:
                solve(F == 0, sol,
                      solver_parameters={"snes_rtol": 0.0, "snes_atol": 1e-8, "ksp_type": "gmres"})
                break
            except ConvergenceError:
                sol.assign(sol_k) # Reset to previous iterate
                alpha.assign(0.5*alpha) # Reduce step size
                print(' ... failed to converge, reducing alpha and retrying')
                continue


        primal_succ_norm = norm(u - u_k)
        print(f': primal success norm = {primal_succ_norm:.4e}')
        if (primal_succ_norm < 1e-07):
            break
        alpha.assign(min(2.0*alpha.values()[0], alpha_max))
    mapped_uh.interpolate(mirror_map)
    plot_solution(fig_primal, uh)
    plot_solution(fig_dual, mapped_uh)
