import numpy as np
from dolfin import *
import logging
import pnptransport.utils as utils
# import traceback
# import sys
# from logwriter import LoggerWriter
import h5py
import os
from typing import Union
import scipy.constants as constants
from scipy import integrate
# import configparser


# parameters['form_compiler']['optimize'] = True
# parameters['form_compiler']['cpp_optimize'] = True
# parameters['form_compiler']['cpp_optimize_flags'] = '-O3'

q_red = 1.6021766208  # x 1E-19 C
e0_red = 8.854187817620389  # x 1E-12 C^2 / J m
CM3TOUM3 = 1E-12


def two_layers_constant_source(D1cms: float, D2cms: float, Cs: float, h: float,
                               m: float, thickness_sinx: float,
                               thickness_si: float, tempC: float,
                               voltage: float, time_s: Union[float, int],
                               recovery_time_s: Union[float, int] = 0,
                               recovery_voltage: float = 0,
                               **kwargs):
    """
    This function simulates the flatband voltage as a function of time for a
    MIS device where Na is migrating into the cell. It also returns a matrix
    with the concentration profiles as a function of time.

    The system solves Poisson-Nernst-Planck equation for a single species.

    Parameters
    ----------
    D1cms: float
        The diffusion coefficient of Na in the dielectric (cm^2/s)
    D2cms: float
        The diffusion coefficient of Na in silicon (cm^2/s)
    Cs: float
        The source concentration (1/cm^3)
    h: float
        The surface mass transfer coefficient (cm/s)
    m: float
        The segregation coefficient (unitless)
    thickness_sinx: float
        The thickness of the simulated dielectric layer (um)
    thickness_si: float
        The thickness of the simulated silicon layer (um)
    tempC: Union[float, int]
        The temperature in °C
    voltage: Union[float, int]
        The voltage applied to the dielectric (V)
    time_s: Union[float, int]
        The simulation time in seconds
    recovery_time_s: Union[float, int]
        An additional simulation time during which, no electrical stress is applied (s).
    recovery_voltage: float
        If provided a recovery time, the bias at which it will recover (V).
    **kwargs:
        cbulk: double
            The base concentration cm-3
        xpoints_sinx: int
            The number of cells in the sinx layer
        xpoints_si: int
            The number of cells in the si layer
        z: integer
            The valency of the ion
            default: 1
        er: double
            The relative permittivity of the dielectric
        xpoints: int
            The number of x points to simulate
        fcall: int
            The number of times the function has been called to solve the same
            problem
        tsteps: int
            The number of time steps to simulate
        max_calls: int
            The maximum number of times the function can be recursively call if the convergence fails.
        max_iter: int
            The maximum number of iterations for the solver
        relaxation_parameter: float
            The relaxation w for the Newton algorithm
        t_smear: int, float
            The time in seconds taken to "smooth" the initial profile (assuming a constant concentration diffused
            over t_smear).
        h5fn: str
            The path to the h5 file to store the simulation results

        debug: bool
            True if debugging the function


    Returns
    -------
    Vfb: np.ndarray
        An array containing the flat band voltage shift as a function of time
        in (V)
    tsim: np.ndarray
        The time for each flatband voltage point
    x1: np.ndarray
        The depth of the concentration profile in SiNx
    c1: np.ndarray
        The final concentration profile as a function of depth in SiNx
    potential: np.ndarray
        The final potential profile as a function of depth in SiNx
    x2: np.ndarray
        The depth of the concentration profile in Si
    c2: np.ndarray
        The final concentration profile in Si
    cmax: float
        The maximum concentration in silicon nitride
    """
    Cbulk = kwargs.get('cbulk', 1E-20)
    xpoints_sinx = kwargs.get('xpoints_sinx', 1000)
    xpoints_si = kwargs.get('xpoints_si', 1000)
    fcall = kwargs.get('fcall', 1)
    tsteps = kwargs.get('tsteps', 400)
    max_calls = kwargs.get('max_rcalls', 3)
    max_iter = kwargs.get('max_iter', 500)
    er = kwargs.get('er', 7.0)
    z = kwargs.get('z', 1.0)
    t_smear = kwargs.get('t_smear', 60)
    h5fn = kwargs.get('h5_storage', None)
    debug = kwargs.get('debug', False)
    relaxation_parameter = kwargs.get('relaxation_parameter', 1.0)

    fcallLogger = logging.getLogger('simlog')

    # Chose the backend type
    if has_linear_algebra_backend("PETSc"):
        parameters["linear_algebra_backend"] = "PETSc"
    #        print('PETSc linear algebra backend found.')
    elif has_linear_algebra_backend("Eigen"):
        parameters["linear_algebra_backend"] = "Eigen"
    else:
        fcallLogger.warning("DOLFIN has not been configured with PETSc or Eigen.")
        exit()

    L1 = thickness_sinx  # thickness*1.05
    L2 = thickness_si
    L = L1 + L2
    M1 = xpoints_sinx
    M2 = xpoints_si
    N = tsteps
    E = voltage / thickness_sinx / 100 / er
    dt = time_s / N
    # Estimate the diffusion coefficients for the given temperature
    tempK = tempC + 273.15

    # Transform everything to um, s
    D1ums = D1cms * 1E8
    D2ums = D2cms * 1E8

    hums = h * 1E4  # ums/s

    # If the diffusion coefficient is negative, something is wrong! Return an array with zeros
    if D1cms <= 0:
        vfb = np.zeros(N + 1)
        x1 = np.linspace(0, thickness_sinx, M1)
        x2 = np.linspace(thickness_sinx, thickness_sinx + thickness_si, M2)
        c1 = np.ones(M1) * Cbulk
        c2 = np.ones(M1) * Cbulk * m
        potential = voltage - x1 * voltage / thickness_sinx
        t_sim = np.linspace(0, time_s, (N + 1))
        return vfb, t_sim, x1, c1, potential, x2, c2, Cbulk

    # The constant mobility in um/s/V
    mu1 = z * constants.elementary_charge * D1ums / (constants.Boltzmann * tempK)
    # mu2 = 0.0
    # The constant ze/(epsilon0,*er) in V*um
    qee = z * constants.elementary_charge / (er * constants.epsilon_0) * 1E6

    # MV/cm x (10^6 V / 1 MV) x ( 10^2 cm / 1 m) = 10^8 V/m = 10^8 J/C/m
    # cm2/s  x (1 m / 10^2 cm)^2 = 10^-4 m^2/s
    # J/C/m x C / J * m^2/s = m/s x (10^6 um / 1 m) = 10^6 um/s
    vd1 = constants.elementary_charge * (E * 1E8) * (D1cms * 1E-4) * 1E6 / (constants.Boltzmann * tempK)
    # vd2 = 0.0  # constants.elementary_charge*(E2*1E8)*(D2cms*1E-4)*1E6/(constants.Boltzmann*TempK)

    set_log_level(50)
    logging.getLogger('FFC').setLevel(logging.WARNING)

    # config = configparser.ConfigParser()
    # config['partial_fit'] = {
    #     'Cs': Cs,
    #     'D1cms': D1cms,
    #     'D2cms': D2cms,
    #     'h': h,
    #     'm': m
    # }
    # fh = fcallLogger.handlers[0]
    # logger_filename = fh.baseFilename
    # partial_fit_filename = os.path.splitext(logger_filename)[0] + '_partial_fit.ini'
    # with open(partial_fit_filename, 'w') as config_file:
    #     config.write(config_file)

    if debug:
        fcallLogger.info('********* Global parameters *********')
        fcallLogger.info('-------------------------------------')
        fcallLogger.info('Time: {0}'.format(utils.format_time_str(time_s)))
        fcallLogger.info('Time step: {0}'.format(utils.format_time_str(time_s / tsteps)))
        fcallLogger.info('Temperature: {0:.1f} °C'.format(tempC))
        fcallLogger.info('Cs: {0:.4E} cm^-3'.format(Cs))
        fcallLogger.info('h: {0:.4E} cm/s'.format(h))
        fcallLogger.info('m: {0:.4E}'.format(m))
        fcallLogger.info('Recovery time: {0}.'.format(utils.format_time_str(recovery_time_s)))
        fcallLogger.info('*************** SiNx ******************')
        fcallLogger.info('Thickness: {0:.1E} um'.format(thickness_sinx))
        fcallLogger.info('er: {0:.1f}'.format(er))
        fcallLogger.info('Voltage: {0:.1f} V'.format(voltage))
        fcallLogger.info('Electric Field: {0:.3E} MV/cm'.format(E * er))
        fcallLogger.info('Electric Field (Effective): {0:.3E} MV/cm'.format(E))
        fcallLogger.info('D: {0:.3E} cm^2/s'.format(D1cms))
        fcallLogger.info('Ionic mobility: {0:.3E} um^2/ V*s'.format(mu1))
        fcallLogger.info('Drift velocity: {0:.3E} um/s'.format(vd1))
        fcallLogger.info('**************** Si *******************')
        fcallLogger.info('Thickness: {0:.1E} um'.format(thickness_si))
        fcallLogger.info('er: {0:.1f}'.format(11.9))
        fcallLogger.info('Voltage: {0:.1f} V'.format(0.0))
        fcallLogger.info('Electric Field: {0:.3E} MV/cm'.format(0.0))
        fcallLogger.info('Electric Field (Effective): {0:.3E} MV/cm'.format(0.0))
        fcallLogger.info('D: {0:.3E} cm^2/s'.format(D2cms))
        fcallLogger.info('Ionic mobility: {0:.3E} cm^2/ V*s'.format(0.0))
        fcallLogger.info('Drift velocity: {0:.3E} cm/s'.format(0.0))

    # Create classes for defining parts of the boundaries and the interior
    # of the domain

    tol = 1E-14

    class Top(SubDomain):
        def inside(self, x_, on_boundary):
            return near(x_[0], 0.0, tol) and on_boundary

    class InnerBoundary(SubDomain):
        def inside(self, x_, on_boundary):
            return near(x_[0], L1, tol) and on_boundary

    class Bottom(SubDomain):
        def inside(self, x_, on_boundary):
            return near(x_[0], L, tol) and on_boundary

    def get_solution_array1(mesh, sol):
        c_, phi = sol.split()
        xu = mesh.coordinates()
        cu = c_.compute_vertex_values(mesh) * 1E12
        pu = phi.compute_vertex_values(mesh)
        xyz = np.array([(xu[j], cu[j], pu[j]) for j in range(len(xu))], dtype=[('x', 'd'), ('c', 'd'), ('phi', 'd')])
        xyz.sort(order='x')
        return xyz['x'], xyz['c'], xyz['phi']

    def get_solution_array2(mesh, sol):
        xu = mesh.coordinates()
        yu = sol.compute_vertex_values(mesh) * 1E12
        xy = np.array([(xu[j], yu[j]) for j in range(len(xu))], dtype=[('x', 'd'), ('y', 'd')])
        xy.sort(order='x')
        return xy['x'], xy['y']

    top = Top()
    bottom = Bottom()
    innerBoundaryL = InnerBoundary()
    innerBoundaryR = InnerBoundary()

    # Create mesh and define function space
    mesh1 = IntervalMesh(M1, 0.0, L1)
    mesh2 = IntervalMesh(M2, L1, L)

    nor = 2
    dr = L1 * 0.2
    for i in range(nor):
        cell_markers = MeshFunction("bool", mesh1, mesh1.topology().dim(), False)
        for cell in cells(mesh1):
            p = cell.midpoint()
            if p[0] >= L1 - dr or p[0] <= dr:
                cell_markers[cell] = True
        mesh1 = refine(mesh1, cell_markers)
        dr = dr / 1.5

    nor = 3
    dr = L2 * 1.1
    for i in range(nor):
        cell_markers = MeshFunction("bool", mesh2, mesh2.topology().dim(), False)
        for cell in cells(mesh2):
            p = cell.midpoint()
            if p[0] <= dr:
                cell_markers[cell] = True
        mesh2 = refine(mesh2, cell_markers)
        dr = dr / 1.5

    if debug:
        fcallLogger.info('Refined meshes.')
        gdim1 = len(mesh1.coordinates())
        gdim2 = len(mesh2.coordinates())
        fcallLogger.info('********** Mesh 1 **********')
        fcallLogger.info('Elements: {0}'.format(gdim1))
        fcallLogger.info('MIN DX: {0:.3E} um, MAX DX {1:.3E}'.format(mesh1.hmin(), mesh1.hmax()))
        fcallLogger.info('********** Mesh 2 **********')
        fcallLogger.info('Elements: {0}'.format(gdim2))
        fcallLogger.info('MIN DX: {0:.3E} um, MAX DX {1:.3E}'.format(mesh2.hmin(), mesh2.hmax()))

    # Initialize mesh function for boundary domains
    boundaries1 = MeshFunction("size_t", mesh1, mesh1.topology().dim() - 1)
    boundaries2 = MeshFunction("size_t", mesh2, mesh2.topology().dim() - 1)
    boundaries1.set_all(0)
    boundaries2.set_all(0)

    top.mark(boundaries1, 1)
    innerBoundaryL.mark(boundaries1, 2)
    innerBoundaryR.mark(boundaries2, 1)
    bottom.mark(boundaries2, 2)

    # Define the measures
    ds1 = Measure('ds', domain=mesh1, subdomain_data=boundaries1)
    ds2 = Measure('ds', domain=mesh2, subdomain_data=boundaries2)
    dx1 = Measure('dx', domain=mesh1, subdomain_data=boundaries1)
    dx2 = Measure('dx', domain=mesh2, subdomain_data=boundaries2)

    # Add some initial Na profile
    # Define the initial concentration in both layers
    # Dsmear = D1ums / 1000
    # u1i = Expression(('c1+(cs-c1)*(1-erf(x[0]/(2*sqrt(D*t))))', '(1-x[0]/L)*Vapp/er'),
    #                  c1=Cbulk * CM3TOUM3, L=L1, cs=Cs * 1E-12,
    #                  D=Dsmear, t=t_smear,
    #                  Vapp=float(voltage), er=er, degree=1)
    # Just add background concentration
    u1i = Expression(('cb', '(1-x[0]/L)*Vapp/er'),
                     cb=Cbulk * CM3TOUM3, L=L1,
                     Vapp=float(voltage), er=er, degree=1)
    u2i = Expression('cb', cb=Cbulk * CM3TOUM3 * m, degree=0)

    # Defining the mixed function space
    CG1 = FiniteElement("CG", mesh1.ufl_cell(), 1)
    W_elem = MixedElement([CG1, CG1])
    W = FunctionSpace(mesh1, W_elem)

    V2 = FunctionSpace(mesh2, 'CG', 1)

    # Defining the "Trial" functions
    u1 = interpolate(u1i, W)  # For time i+1
    c1, phi1 = split(u1)
    u1_G = interpolate(u1i, W)  # For time i+1/2
    c1_G, phi1_G = split(u1_G)
    u1_n = interpolate(u1i, W)  # For time i
    c1_n, phi1_n = split(u1_n)

    u2 = interpolate(u2i, V2)
    u2_n = interpolate(u2i, V2)
    u2_G = interpolate(u2i, V2)

    # Define the test functions
    v1 = TestFunction(W)
    (v1c, v1p) = split(v1)
    v2 = TestFunction(V2)

    du1 = TrialFunction(W)
    du2 = TrialFunction(V2)

    u1.set_allow_extrapolation(True)
    u2.set_allow_extrapolation(True)
    u1_G.set_allow_extrapolation(True)
    u2_G.set_allow_extrapolation(True)
    u1_n.set_allow_extrapolation(True)
    u2_n.set_allow_extrapolation(True)

    tol = 1E-16

    def update_bcs1(bias):
        return [DirichletBC(W.sub(1), bias / er, boundaries1, 1)]

    bcs1 = [DirichletBC(W.sub(0), Cs * 1E-12, boundaries1, 1),
            DirichletBC(W.sub(1), voltage / er, boundaries1, 1)]
    bcs2 = None  # [DirichletBC(V2,Cbulk*CM3TOUM3,boundaries2,2)]


    def get_variational_form1(uc, up, gp1_, gp2_, u2c):
        sf2 = segregation_flux(hums, uc, u2c, m)
        gc01 = 0.0  # -(mu1*uc*gp1 - sf2)
        gc12 = -(mu1 * uc * gp2_ + sf2)

        a = -D1ums * inner(grad(uc), grad(v1c)) * dx1
        a += gc01 * v1c * ds1(1) + gc12 * v1c * ds1(2)
        a -= mu1 * uc * inner(grad(up), grad(v1c)) * dx1
        a += mu1 * gp1_ * uc * v1c * ds1(1) + mu1 * gp2_ * uc * v1c * ds1(2)
        a -= (inner(grad(up), grad(v1p)) - qee * uc * v1p) * dx1
        a += gp1_ * v1p * ds1(1) + gp2_ * v1p * ds1(2)
        return a

    def get_variational_form2(uc, u1c):
        sf2 = segregation_flux(hums, u1c, uc, m)
        gc21 = sf2
        a = -D2ums * inner(grad(uc), grad(v2)) * dx2
        a += gc21 * v2 * ds2(1)
        return a

    def getTRBDF2ta(uc, up):
        r = D1ums * div(grad(uc)) + div(grad(up)) \
            + mu1 * uc * div(grad(up)) + mu1 * inner(grad(up), grad(uc)) \
            + qee * uc
        return r

    # def segregation_flux(h_, cc1, cc2, m_: Union[float, int] = 1):
    #     ux1, uc1, _ = get_solution_array1(mesh1, cc1)
    #     ux2, uc2, _ = get_solution_array1(mesh1, cc2)
    #     J = h_ * (uc1[-1] - uc2[-1] / m_)
    #     return J

    def segregation_flux(h_, cc1, cc2, m_: Union[float, int] = 1):
        J = h_ * (cc1 - cc2 / m_)
        return J

    def update_potential_bc(uui, bias: float = voltage):
        # The total concentration in the oxide (um-2)
        #        uc,up = ui.split()
        #        Ctot = assemble(uc*dx)
        # The integral in Poisson's equation solution (Nicollian & Brews p.426)
        # units: 1/um
        #        Cint = assemble(uc*Expression('x[0]',degree=1)*dx)

        # Get the solution in an array form
        uxi, uci, upi = get_solution_array1(mesh1, uui)

        # The total concentration in the oxide (cm-2)
        Ctot_ = integrate.simps(uci, uxi) * 1E-4
        # The integral in Poisson's equation solution (Nicollian & Brews p.426)
        # units: 1/cm
        Cint_ = integrate.simps(uxi * uci, uxi) * 1E-8
        # The centroid of the charge distribution
        xbar_ = Cint_ / Ctot_ * 1E4  # um

        # The surface charge density at silicon C/cm2
        scd_si = -constants.e * (xbar_ / L1) * Ctot_
        # The surface charge density at the gate C/cm2
        scd_g = -constants.e * (1.0 - xbar_ / L1) * Ctot_

        # The electric field at the gate interface
        E_g_ = bias / L1 / er + 1E-2 * scd_g / constants.epsilon_0 / er  # x
        # The electric field at the Si interface
        E_si_ = bias / L1 / er - 1E-2 * scd_si / constants.epsilon_0 / er  # x

        # Since grad(phi) = -E
        # s1: <-
        # s2: ->
        #        gp1 = E_g
        gp1_ = E_g_
        gp2_ = -E_si_

        vfb_ = -q_red * Cint_ / (er * e0_red) * 1E-5

        return gp1_, gp2_, Cint_, Ctot_, E_g_, E_si_, xbar_, vfb_

    hk1 = CellDiameter(mesh1)

    GAMMA = 2.0 - np.sqrt(2)  # 0.59
    TRF = Constant(0.5 * GAMMA)
    BDF2_T1 = Constant(1.0 / (GAMMA * (2.0 - GAMMA)))
    BDF2_T2 = Constant((1.0 - GAMMA) * (1.0 - GAMMA) / (GAMMA * (2.0 - GAMMA)))
    BDF2_T3 = Constant((1.0 - GAMMA) / (2.0 - GAMMA))

    ffc_options = {"optimize": True,
                   'cpp_optimize': True,
                   "quadrature_degree": 5}

    newton_solver_parameters = {"nonlinear_solver": "newton",
                                "newton_solver": {
                                    "linear_solver": "lu",
                                    # "preconditioner": 'ilu',  # 'hypre_euclid',
                                    "convergence_criterion": "incremental",
                                    "absolute_tolerance": 1E-5,
                                    "relative_tolerance": 1E-4,
                                    "maximum_iterations": max_iter,
                                    "relaxation_parameter": relaxation_parameter,
                                    # 'krylov_solver': {
                                    #     'absolute_tolerance': 1E-8,
                                    #     'relative_tolerance': 1E-6,
                                    #     'maximum_iterations': 100}
                                }}

    def get_solvers_1(gp1_, gp2_, dt_):

        a10 = get_variational_form1(c1_n, phi1_n, gp1_, gp2_, u2_n)
        a1G = get_variational_form1(c1_G, phi1_G, gp1_, gp2_, u2_G)
        a11 = get_variational_form1(c1, phi1, gp1_, gp2_, u2)

        F1G = (1 / dt_) * (c1_G - c1_n) * v1c * dx1 - TRF * (a1G + a10)
        F1N = (1 / dt_) * (c1 - BDF2_T1 * c1_G + BDF2_T2 * c1_n) * v1c * dx1 - BDF2_T3 * a11

        # SUPG stabilization
        b1_ = mu1 * Dx(phi1_n, 0)
        nb1 = sqrt(dot(b1_, b1_) + DOLFIN_EPS)
        Pek1 = nb1 * hk1 / (2.0 * D1ums)

        b2_ = mu1 * Dx(phi1_G, 0)
        nb2 = sqrt(dot(b2_, b2_) + DOLFIN_EPS)
        Pek2 = nb2 * hk1 / (2.0 * D1ums)

        tau1 = conditional(gt(Pek1, DOLFIN_EPS),
                           (hk1 / (2.0 * nb1)) * (((exp(2.0 * Pek1) + 1.0) / (exp(2.0 * Pek1) - 1.0)) - 1.0 / Pek1),
                           0.0)
        tau2 = conditional(gt(Pek2, DOLFIN_EPS),
                           (hk1 / (2.0 * nb2)) * (((exp(2.0 * Pek2) + 1.0) / (exp(2.0 * Pek2) - 1.0)) - 1.0 / Pek2),
                           0.0)

        # get the skew symmetric part of the L operator
        # LSSNP = dot(vel2,Dx(v2,0))
        Lss1 = (mu1 * inner(grad(phi1_G), grad(v1c)) + (mu1 / 2) * div(grad(phi1_G)) * v1c)
        Lss2 = (mu1 * inner(grad(phi1), grad(v1c)) + (mu1 / 2) * div(grad(phi1)) * v1c)
        # SUPG Stabilization term
        ta = getTRBDF2ta(c1_G, phi1_G)
        tb = getTRBDF2ta(c1_n, phi1_n)
        tc = getTRBDF2ta(c1, phi1)
        ra = inner(((1 / dt_) * (c1_G - c1_n) - TRF * (ta + tb)), tau1 * Lss1) * dx1
        rb = inner((c1 / dt_ - BDF2_T1 * c1_G / dt_ + BDF2_T2 * c1_n / dt_ - BDF2_T3 * tc), tau2 * Lss2) * dx1

        F1G += ra
        F1N += rb

        J1G = derivative(F1G, u1_G, du1)
        J1N = derivative(F1N, u1, du1)  # J1G

        problem1N = NonlinearVariationalProblem(F1N, u1, bcs1, J1N, form_compiler_parameters=ffc_options)
        problem1G = NonlinearVariationalProblem(F1G, u1_G, bcs1, J1G, form_compiler_parameters=ffc_options)
        solver1N_ = NonlinearVariationalSolver(problem1N)
        solver1N_.parameters.update(newton_solver_parameters)
        solver1G_ = NonlinearVariationalSolver(problem1G)
        solver1G_.parameters.update(newton_solver_parameters)

        return solver1N_, solver1G_

    def get_solvers_2(dt_):
        a20 = get_variational_form2(u2_n, c1_n)
        a2G = get_variational_form2(u2_G, c1_G)
        a21 = get_variational_form2(u2, c1)

        F2G = (1 / dt_) * (u2_G - u2_n) * v2 * dx2 - TRF * (a2G + a20)
        F2N = (1 / dt_) * (u2 - BDF2_T1 * u2_G + BDF2_T2 * u2_n) * v2 * dx2 - BDF2_T3 * a21

        J2G = derivative(F2G, u2_G, du2)
        J2N = derivative(F2N, u2, du2)  # J2G

        problem2N = NonlinearVariationalProblem(F2N, u2, bcs2, J2N, form_compiler_parameters=ffc_options)
        problem2G = NonlinearVariationalProblem(F2G, u2_G, bcs2, J2G, form_compiler_parameters=ffc_options)
        solver2N_ = NonlinearVariationalSolver(problem2N)
        solver2N_.parameters.update(newton_solver_parameters)
        solver2G_ = NonlinearVariationalSolver(problem2G)
        solver2G_.parameters.update(newton_solver_parameters)

        return solver2N_, solver2G_

    # The time for each concentration profile
    # Get tau_c
    tauc = utils.tau_c(D=D1cms, E=E, L=L1 * 1E-4, T=tempC)
    delta_t = time_s / (N + 1)
    if time_s <= 86400 * 4 or int(tauc / delta_t) < 50:
        size_n = N + 1
        t_sim = np.array([k * dt for k in range(size_n)], dtype=np.float64)
        dtt = np.concatenate([np.diff(t_sim), [dt]])
    else:
        base = 1.5
        dt_min = 1
        dt_max = dt
        num_t = 30
        b1 = np.log(dt_min) / np.log(base)
        b2 = np.log(dt_max) / np.log(base)
        t1 = np.logspace(b1, b2, num=num_t, base=base)
        t2 = np.array([k * dt for k in range(1, N + 1)], dtype=np.float64)
        t_sim = np.concatenate([[0], t1, t2])
        dtt = np.concatenate([np.diff(t_sim), [dt]])
        size_n = len(dtt)
        del base, dt_min, dt_max, num_t, b1, b2, t1, t2

    if recovery_time_s > 0:
        recovery_n_points = int(recovery_time_s / dt)
        t_recovery = np.array([np.amax(t_sim) + k*dt for k in range(1, recovery_n_points)])
        t_sim = np.insert(t_sim, len(t_sim), t_recovery)
        dtt = np.concatenate([np.diff(t_sim), [dt]])
        size_n += recovery_n_points


    if debug:
        fcallLogger.info('**** Time stepping *****')
        fcallLogger.info('Min dt: {0:.3E} s, Max dt: {1:.3E} s.'.format(np.amin(dtt), np.amax(dtt)))
        fcallLogger.info('Simulation time: {0}.'.format(utils.format_time_str(time_s)))
        fcallLogger.info('Number of time steps: {0}'.format(len(t_sim)))

    # Allocate memory for the flatband voltage
    vfb = np.zeros(len(t_sim), dtype=np.float64)

    x1i, c1i, p1i = get_solution_array1(mesh1, u1_n)
    x2i, c2i = get_solution_array2(mesh2, u2_n)
    c_max = -np.inf

    if h5fn is not None:
        if os.path.exists(h5fn):
            os.remove(h5fn)
        with h5py.File(h5fn, 'w') as hf:
            # filetag = os.path.splitext(os.path.basename(h5fn))[0]
            if debug:
                fcallLogger.info('Created file for storage \'{}\''.format(h5fn))

            dst = hf.create_dataset('/time', (len(t_sim),))
            dst[...] = t_sim
            dst.attrs['temp_c'] = tempC
            dst.attrs['Csource'] = Cs
            dst.attrs['Cbulk'] = Cbulk
            dst.attrs['h'] = h
            dst.attrs['m'] = m

        with h5py.File(h5fn, 'a') as hf:
            grp_l1 = hf.create_group('L1')
            grp_l2 = hf.create_group('L2')

            dsx1 = grp_l1.create_dataset('x', (len(x1i),))
            dsx1[...] = x1i
            grp_l1.attrs['D'] = D1cms
            grp_l1.attrs['stress_voltage'] = voltage
            grp_l1.attrs['er'] = er
            grp_l1.attrs['electric_field_eff'] = E
            grp_l1.attrs['electric_field_app'] = E * er
            grp_l1.attrs['ion_valency'] = 1
            grp_l1.attrs['ion_mobility'] = mu1 * 1E-4
            grp_l1.attrs['drift_velocity'] = vd1 * 1E-4

            dsx2 = grp_l2.create_dataset('x', (len(x2i),))
            dsx2[...] = x2i
            grp_l2.attrs['D'] = D2cms
            grp_l2.attrs['stress_voltage'] = 0
            grp_l2.attrs['er'] = 11.9
            grp_l2.attrs['electric_field_eff'] = 0
            grp_l2.attrs['electric_field_app'] = 0
            grp_l2.attrs['ion_valency'] = 0
            grp_l2.attrs['ion_mobility'] = 0
            grp_l2.attrs['drift_velocity'] = 0

            grp_l1.create_group('concentration')
            grp_l1.create_group('potential')
            grp_l2.create_group('concentration')

    if debug:
        fcallLogger.info('Starting time integration loop...')

    for n, t in enumerate(t_sim):
        bias_t = voltage if t <= time_s else recovery_voltage
        gp1, gp2, Cint, Ctot, E_g, E_si, xbar, vfb[n] = update_potential_bc(u1_n, bias=bias_t)
        x1i, c1i, p1i = get_solution_array1(mesh1, u1_n)
        x2i, c2i = get_solution_array2(mesh2, u2_n)
        c_max = max(c_max, np.amax(c1i))
        dti = dtt[n]

        if h5fn is not None:
            # Store the data in h5py
            with h5py.File(h5fn, 'a') as hf:
                grp_l1_c = hf['L1/concentration']
                grp_l1_p = hf['L1/potential']
                grp_l2_c = hf['L2/concentration']
                dsc_str = 'ct_{0:d}'.format(n)
                dsv_str = 'vt_{0:d}'.format(n)
                if dsc_str not in grp_l1_c:
                    dsc1 = grp_l1_c.create_dataset(dsc_str, (len(x1i),), compression="gzip")
                    dsc1[...] = c1i
                    dsc1.attrs['time'] = t
                if dsc_str not in grp_l2_c:
                    dsc1 = grp_l2_c.create_dataset(dsc_str, (len(x2i),), compression="gzip")
                    dsc1[...] = c2i
                    dsc1.attrs['time'] = t
                if dsv_str not in grp_l1_p:
                    dsv1 = grp_l1_p.create_dataset(dsv_str, (len(x1i),), compression="gzip")
                    dsv1[...] = p1i
                    dsv1.attrs['time'] = t

        if n == (size_n - 1) or debug:
            js1 = h * (c1i[-1] - c2i[0] / m) * 1E-4
            prog_str = "%s, " % utils.format_time_str(time_s=t)
            #        progStr = ' i={0:4d}/{1:4d} '.format(n,num_steps)
            prog_str += 'C0R={0:.4E}, C1L={1:.1E}, C1R={2:.1E}, Js1={3:.1E} '.format(c1i[0], c1i[-1], c2i[0], js1)
            prog_str += "vfb={0:.2E} V ".format(vfb[n])
            prog_str += 'D1 = {0:2.1E}, '.format(D1cms)
            prog_str += 'D2 = {0:2.1E} cm2/s '.format(D2cms)
            prog_str += 'h = {0:2.4E} cm/s '.format(h)
            prog_str += 'Eg = {0:2.4E} V/cm '.format(E_g*er)

            fcallLogger.info(prog_str)

        try:
            solver1N, solver1G = get_solvers_1(gp1, gp2, dti)
            solver2N, solver2G = get_solvers_2(dti)
            solver1G.solve()
            solver2G.solve()

            # Update the electric field
            gp1, gp2, _, _, _, _, _, _ = update_potential_bc(u1_n, bias=bias_t)
            solver1N, solver1G = get_solvers_1(gp1, gp2, dti)

            solver1N.solve()
            solver2N.solve()
            # Update previous solution
            u1_n.assign(u1)
            # Update previous solution
            u2_n.assign(u2)
        except RuntimeError:
            message = 'Could not solve for time {0:.1f} h. D2 = {1:.3E} cm2/s CSi = {2:.1E} 1/cm^3,\t'.format(
                t / 3600, D2cms, c1i[-1] * 1E1
            )
            message += 'T = {0:3.1f} °C, E = {1:.1E} MV/cm, tmax: {2:3.2f} hr, XPOINTS = {3:d}, TSTEPS: {4:d}'.format(
                tempC, E * er, time_s / 3600, M1, N
            )
            fcallLogger.warning(message )
            if fcall <= max_calls:
                tsteps = int(N * 2)
                # xpoints_sinx = int(xpoints_sinx*1.5)
                # xpoints_si = int(xpoints_si * 1.5)
                #                relaxation_parameter=relaxation_parameter*0.5
                fcallLogger.warning(
                    'Trying with a larger number of time steps: {0:d}, refinement step: {1:d}'.format(N, fcall))
                fcall += 1

                return two_layers_constant_source(
                    D1cms=D1cms, D2cms=D2cms, Cs=Cs,
                    h=h, m=m, thickness_sinx=thickness_sinx,
                    thickness_si=thickness_si,
                    tempC=tempC, voltage=voltage,
                    time_s=time_s,
                    cbulk=Cbulk,
                    fcall=fcall,
                    tsteps=tsteps,
                    fcallLogger=fcallLogger,
                    xpoints_sinx=xpoints_sinx,
                    xpoints_si=xpoints_si,
                    max_rcalls=max_calls,
                    max_iter=max_iter,
                    er=er,
                    z=z,
                    t_smear=t_smear,
                    relaxation_parameter=relaxation_parameter,
                    debug=debug,
                    h5_storage=h5fn
                )
            else:
                fcallLogger.error('Reached max refinement without success...')
                x1i, c1i, p1i = get_solution_array1(mesh1, u1_n)
                #                    print('failed: len(tsim): {0:d}, len(vfb): {1:d}'.format(len(tsim),len(Vfb)))
                vfb = 1E50 * np.ones(size_n, dtype=np.float64)
                return vfb, t_sim, x1i, c1i, p1i, x2i, c2i, c_max

    if h5fn is not None:
        with h5py.File(h5fn, 'a') as hf:
            ds_vfb = hf.create_dataset('vfb', (len(vfb),))
            ds_vfb[...] = vfb
            hf['/time'].attrs['Cmax'] = c_max
            hf.close()

    return vfb, t_sim, x1i, c1i, p1i, x2i, c2i, c_max