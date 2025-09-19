from firedrake import *

import matplotlib.pyplot as plt
plt.ion() # Turn on interactive mode

# Define the domain
mesh = UnitDiskMesh(6)

# Define function spaces
primal_space = FunctionSpace(mesh, "CG", 1)
latent_space = FunctionSpace(mesh, "RT", 1)
V = primal_space * latent_space

# Define the right-hand side
bc = DirichletBC(V.sub(0), 0.0, "on_boundary") # u = 0 on the boundary

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
conj_R = ufl.sqrt(1.0 + dot(psi, psi))

# Define the saddle-point Lagrangian
alpha = Constant(1.0) # Prox step size
J = -u
prox_diff = (div(psi) - div(psi_k))*u 
L = (alpha*J + prox_diff - conj_R)*dx
dL = derivative(L, sol)

# Plotting setup
fig = plt.figure()

alpha.assign(1.0)
for i in range(1, 101):
    sol_k.assign(sol)
    for attempt in range(10):
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
    else:
        raise ConvergenceError('Nonlinear solver failed to converge')

    alpha.assign(2.0*alpha)

    # Visualize
    fig.clear()
    axes = fig.add_subplot(111)
    colors = tricontourf(uh, axes=axes)
    fig.colorbar(colors, ax=axes)
    fig.canvas.draw()
    fig.canvas.flush_events()

    primal_succ_norm = norm(u - u_k)
    print(f': primal success norm = {primal_succ_norm:.4e}')
    if (primal_succ_norm < 1e-07):
        break
