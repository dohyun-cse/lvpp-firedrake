from firedrake import *

def pg(dE, B, mirror_map, psi_k, alpha_k, u, v, psi, w, dOmega_d=dx):
    F = alpha_k*dE(u,v) # Energy derivative with step size
    F += B(v, psi) - B(v, psi_k) # Proximal term
    F += B(u, w) - mirror_map(psi)*w*dOmega_d # Mirror map constraint
    print(F)
    return F

if __name__ == "__main__":
    mesh = UnitSquareMesh(32, 32)

    x, y = SpatialCoordinate(mesh)
    n, h = FacetNormal(mesh), CellDiameter(mesh)

    # To be modified
    f = Constant(-1.0) # constant pull force
    obstacle = x

    # energy, constraint operator, and mirror map
    dE = lambda u, v: inner(grad(u), grad(v))*dx - inner(f, v)*dx
    B = lambda u, phi: u*phi*dx # B=Id
    mirror_map = lambda psi: exp(psi) + obstacle # C = [0, inf)

    # Finite Element Spaces, and Discrete Functions
    V = FunctionSpace(mesh, "CG", 5)
    W = FunctionSpace(mesh, "DG", 3)
    spaces = V*W
    bc = DirichletBC(spaces.sub(0), 0.0, "on_boundary") # u = 0 on the boundary

    v, w = TestFunctions(spaces)

    sol = Function(spaces)
    u, psi = split(sol)
    uh, psih = sol.subfunctions
    sol.assign(0.0)

    sol_k = Function(spaces) # previous solution
    u_k, psi_k = sol_k.subfunctions

    # PG Iteration
    alpha_k = Constant(1.0) # step size
    max_prox_it = 100
    tol_prox = 1e-5
    F = pg(dE, B, mirror_map,psi_k, alpha_k, u, v, psi, w)
    for prox_it in range(max_prox_it):
        sol_k.assign(sol)
        solve(F == 0, sol, bcs=[bc],
              solver_parameters={"snes_rtol": 0.0, "snes_atol": 1e-8})
        # measure L2 successive difference
        diff_sol = Integrate((uh - u_k)**2*dx, mesh)**0.5
        if diff_sol < tol_prox:
            break
