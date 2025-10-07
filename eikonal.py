from firedrake import *

def pg_residual(dE, B, mirror_map, psi_k, alpha_k, u, v, psi, w, dOmega_d=dx):
    F = alpha_k*dE(u,v) # Energy derivative with step size
    F += B(v, psi) - B(v, psi_k) # Proximal term
    F += B(u, w) - inner(mirror_map(psi),w)*dOmega_d # Mirror map constraint
    return F

if __name__ == "__main__":
    # Mesh and parameters
    mesh = UnitSquareMesh(128, 128)
    order = 1
    prox_max_it, prox_tol = 100, 1e-08

    # Obstacle
    x, y = SpatialCoordinate(mesh)
    obstacle = ufl.sqrt(ufl.max_value(0.25**2 - (x-0.5)**2 - (y-0.5)**2, 0.0))

    # Define function spaces
    V = FunctionSpace(mesh, "DG", order)
    W = FunctionSpace(mesh, "RT", order + 1)
    spaces = V * W
    # energy, constraint operator, and mirror map
    dE = lambda u, v: -v*dx
    B = lambda u, phi: -u*div(phi)*dx # B=Id
    mirror_map = lambda psi: psi / sqrt(1 + inner(psi, psi))
    # Define the essential boundary condition
    bc = DirichletBC(spaces[0], 0.0, "on_boundary") # u = 0 on the boundary

    # Solution (Symbolic + Discrete) 
    sol = Function(spaces)
    u, psi = split(sol) # symbolic representation
    uh, psih = sol.subfunctions # discrete representation
    sol.assign(0.0) # Initialize to zero

    # Previous iteration
    sol_k = Function(spaces)
    u_k, psi_k = sol_k.subfunctions

    # Test functions
    v, w = TestFunctions(spaces)

    # Step size
    alpha_k = Constant(1.0)
    F = pg_residual(dE, B, mirror_map, psi_k, alpha_k, u, v, psi, w)
    for prox_it in range(1, prox_max_it):
        alpha_k.assign(prox_it)
        sol_k.assign(sol)
        solve(F == 0, sol, bcs=[bc],
              solver_parameters={"snes_rtol": 0.0, "snes_atol": 1e-8})

        primal_succ_diff = norm(u - u_k)
        print(f"Prox It {prox_it:3d}: {primal_succ_diff:.4e}")
        if (primal_succ_diff < prox_tol):
            break
