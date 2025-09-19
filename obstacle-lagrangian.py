from firedrake import *

import matplotlib.pyplot as plt
plt.ion() # Turn on interactive mode

# Define the domain
mesh = UnitDiskMesh(64)

# Define function spaces
primal_space = FunctionSpace(mesh, "CG", 2)
latent_space = FunctionSpace(mesh, "DG", 0)
V = primal_space * latent_space

# Define the right-hand side
bc = DirichletBC(V.sub(0), 0.0, "on_boundary") # u = 0 on the boundary
x, y = SpatialCoordinate(mesh)
f = Function(primal_space).interpolate(4.0*2*pi**2*sin(pi*x)*sin(pi*y))

# Solution (Symbolic + Discrete) 
sol = Function(V)
u, psi = split(sol) # symbolic representation
uh, psih = sol.subfunctions # discrete representation
sol.assign(0.0) # Initialize to zero

# Previous iteration
sol_k = Function(V)
u_k, psi_k = sol_k.subfunctions

# For visualization of the alternative solution, u = grad R^*(psi)
mapped_uh = Function(latent_space)

# Fenchel conjugate (R^*) and mirror map (\nabla R^*)
conj_R = ufl.conditional(ufl.ge(psi, 0.0),
                         psi + ufl.ln(1.0 + ufl.exp(-psi)),
                         ufl.ln(1.0 + ufl.exp(psi)))
# Derivative of R^*
one = Function(latent_space).interpolate(Constant(1.0))
mirror_map = derivative(conj_R, psi, one)

# Define the saddle-point Lagrangian
alpha = Constant(1.0) # Prox step size
J = dot(grad(u), grad(u))/2.0 - f*u
prox_diff = (psi - psi_k)*u 
L = (alpha*J + prox_diff - conj_R)*dx
dL = derivative(L, sol)

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

alpha.assign(1.0)
for i in range(1, 101):
    sol_k.assign(sol)
    for attempt in range(5):
        print(f'Iteration {i:3d}, alpha = {alpha.values()[0]:.2e}', end='')
        try:
            solve(dL == 0, sol, bcs=[bc],
                  solver_parameters={"snes_rtol": 0.0, "snes_atol": 1e-8})
            break
        except ConvergenceError:
            sol.assign(sol_k) # Reset to previous iterate
            alpha.assign(0.5*alpha) # Reduce step size
            print(' ... failed to converge, reducing alpha and retrying')
            continue
    alpha.assign(2.0*alpha)

    # Map the dual variable to the primal space for plotting
    plot_solution(fig_primal, uh)
    mapped_uh.interpolate(mirror_map)
    plot_solution(fig_dual, mapped_uh)

    primal_succ_norm = norm(u - u_k)
    print(f': primal success norm = {primal_succ_norm:.4e}')
    if (primal_succ_norm < 1e-07):
        break
