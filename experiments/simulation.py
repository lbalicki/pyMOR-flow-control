import sys
# add parent directory to path in order to access modules
sys.path.append('..')
from data_utils import *
from models import load_fom

import numpy as np
import os
import pickle
import scipy.io as spio
import scipy.linalg as spla
import scipy.sparse as sps
import scipy.sparse.linalg as spsla

from pymor.algorithms.newton_lradi import solve_ricc_lrcf
from pymor.algorithms.to_matrix import to_matrix
from pymor.operators.constructions import LowRankOperator, LerayProjectedOperator, IdentityOperator
from pymor.operators.numpy import NumpyMatrixOperator
from pymor.vectorarrays.constructions import cat_arrays
from pymor.vectorarrays.numpy import NumpyVectorArray


def run_no_control_simulation(Re=110, level=2, palpha=1e-3):
    """Run the simulation with no control."""

    # define strings, directories and paths for data storage
    setup_str = 'lvl_' + str(level) + '_re_' + str(Re)
    data_path = '../data/' + 'lvl_' + str(level)
    setup_path = data_path + '/re_' + str(Re)

    if not os.path.exists(setup_str):
        os.makedirs(setup_str)

    with open(setup_str + '/no_control_simulation.csv', 'w') as file:
        file.write('t, yerr \n')

    # use fom to project initial velocity onto hidden manifold later
    fom = load_fom(Re=110, level=2, palpha=1e-3, control=None)

    mats = spio.loadmat(data_path + '/mats')

    M = mats['M']
    J = mats['J']
    hmat = mats['H']
    fv = mats['fv'] + 1./Re*mats['fv_diff'] + mats['fv_conv']
    fp = mats['fp'] + mats['fp_div']
    vcmat = mats['Cv']
    NV, NP = fv.shape[0], fp.shape[0]

    A = 1./Re*mats['A'] + mats['L1'] + mats['L2']

    # compute steady-state solution and linearized convection
    if not os.path.isfile(setup_path + '/ss_nse_sol'):
        ss_nse_v, _ = solve_steadystate_nse(mats, Re, None, palpha=palpha)
        conv_mat = linearized_convection(mats['H'], ss_nse_v)
        spio.savemat(setup_path + '/ss_nse_sol', {'ss_nse_v': ss_nse_v, 'conv_mat': conv_mat})
    else:
        ss_nse_sol = spio.loadmat(setup_path + '/ss_nse_sol')
        ss_nse_v, conv_mat = ss_nse_sol['ss_nse_v'], ss_nse_sol['conv_mat']

    # Define parameters for time stepping
    t0 = 0.
    tE = 8.
    Nts = 2**12
    DT = (tE-t0)/Nts
    trange = np.linspace(t0, tE, Nts+1)

    Css = vcmat @ ss_nse_v

    # introduce small perturbation to steady-state solution as initial value
    pert = fom.A.source.project_onto_subspace(fom.A.operator.source.ones(), trans=True).to_numpy().T
    old_v = ss_nse_v + 1e-3 * pert

    sysmat = sps.vstack([sps.hstack([M+DT*A, -J.T]),
                         sps.hstack([J, sps.csc_matrix((NP, NP))])]).tocsc()
    sysmati = spsla.factorized(sysmat)

    for k, t in enumerate(trange):
        crhsv = M*old_v + DT*(fv - eva_quadterm(hmat, old_v))
        crhs = np.vstack([crhsv, fp])
        vp_new = np.atleast_2d(sysmati(np.squeeze(np.asarray(crhs)))).T
        old_v = vp_new[:NV]
        Cv = vcmat @ old_v

        print(k, '/', Nts)
        print(spla.norm(Cv - Css, 2))

        with open(setup_str + '/no_control_simulation.csv', 'a') as file:
            file.write(str(t) + ',' + str(spla.norm(Cv - Css, 2)) + '\n')


def run_fom_cl_simulation(Re=110, level=2, palpha=1e-3, control='bc'):
    """Run the closed-loop simulation with full order LQG controller."""

    # define strings, directories and paths for data storage
    setup_str = 'lvl_' + str(level) + ('_' + control if control is not None else '') \
                + '_re_' + str(Re) + ('_palpha_' + str(palpha) if control == 'bc' else '')
    data_path = '../data/' + 'lvl_' + str(level) + ('_' + control if control is not None else '')
    setup_path = data_path + '/re_' + str(Re) + ('_palpha_' + str(palpha) if control == 'bc' else '')

    with open(setup_str + '/fom_simulation.csv', 'w') as file:
        file.write('t, yerr \n')

    # define first order model and matrices for simulation
    fom = load_fom(Re=110, level=2, palpha=1e-3, control='bc')

    mats = spio.loadmat(data_path + '/mats')

    M = mats['M']
    J = mats['J']
    hmat = mats['H']
    fv = mats['fv'] + 1./Re*mats['fv_diff'] + mats['fv_conv']
    fp = mats['fp'] + mats['fp_div']
    vcmat = mats['Cv']
    NV, NP = fv.shape[0], fp.shape[0]

    if control == 'bc':
        A = 1./Re*mats['A'] + mats['L1'] + mats['L2'] + 1./palpha*mats['Arob']
        B = 1./palpha*mats['Brob']
    else:
        A = 1./Re*mats['A'] + mats['L1'] + mats['L2']
        B = mats['B']
        # restrict to less dofs in the input
        NU = B.shape[1]
        B  = B[:, [0, NU//2]]

    # compute steady-state solution and linearized convection
    if not os.path.isfile(setup_path + '/ss_nse_sol'):
        ss_nse_v, _ = solve_steadystate_nse(mats, Re, control, palpha=palpha)
        conv_mat = linearized_convection(mats['H'], ss_nse_v)
        spio.savemat(setup_path + '/ss_nse_sol', {'ss_nse_v': ss_nse_v, 'conv_mat': conv_mat})
    else:
        ss_nse_sol = spio.loadmat(setup_path + '/ss_nse_sol')
        ss_nse_v, conv_mat = ss_nse_sol['ss_nse_v'], ss_nse_sol['conv_mat']

    # Define parameters for time stepping
    t0 = 0.
    tE = 8.
    Nts = 2**12
    DT = (tE-t0)/Nts
    trange = np.linspace(t0, tE, Nts+1)

    # Define functions that represent the system inputs
    if control is None:
        def bbcu(ko_state):
            return np.zeros((NV, 1))

        def update_ko_state(ko_state, Cv, DT):
            return ko_state

    else:
        _, Kc = solve_ricc_lrcf(fom.A, fom.E, fom.B.as_range_array(), fom.C.as_source_array(),
                                trans=True, return_K=True)
        _, Ko = solve_ricc_lrcf(fom.A, fom.E, fom.B.as_range_array(), fom.C.as_source_array(),
                                trans=False, return_K=True)

        Kc = Kc.to_numpy().T
        Ko = Ko.to_numpy().T

        Aconv = -1./Re * mats['A'] - 1./palpha * mats['Arob'] - conv_mat
        BK = sps.bmat([
                      [B, Ko],
                      [sps.csc_matrix((NP, len(B.T))), sps.csc_matrix((NP, len(Ko.T)))]
                      ]).todense()

        KC = sps.bmat([
                      [Kc.T, sps.csc_matrix((len(Kc.T), NP))],
                      [vcmat, sps.csc_matrix((len(vcmat.todense()), NP))]
                      ]).todense()

        def bbcu(ko_state):
            uvec = -Kc.T @ ko_state
            return B @ uvec

        EmA = sps.vstack([
                         sps.hstack([M-DT*Aconv, J.T]),
                         sps.hstack([J, sps.csc_matrix((NP, NP))])
                         ]).tocsc()

        EmAlu = spsla.factorized(EmA)
        S = EmAlu(BK)
        IKS = spla.solve(np.eye(len(KC)) + DT * KC @ S, KC)

        Css = vcmat @ ss_nse_v

        # function that determines the next state of the Kalman observer via implicit euler step
        def update_ko_state(ko_state, Cv):
            rhs = M @ ko_state + DT * (Ko @ (Cv - Css))
            rhs_block = np.vstack([rhs, np.zeros((NP, 1))])
            EmAluko = EmAlu(rhs_block)
            ko_new_block = EmAluko - (DT * (S @ (IKS @ EmAluko)))
            return ko_new_block[:NV]

    # introduce small perturbation to steady-state solution as initial value
    pert = fom.A.source.project_onto_subspace(fom.A.operator.source.ones(), trans=True).to_numpy().T
    old_v = ss_nse_v + 1e-3 * pert

    # initialize state for observer
    ko_state = np.zeros((NV, 1))

    sysmat = sps.vstack([sps.hstack([M+DT*A, -J.T]),
                         sps.hstack([J, sps.csc_matrix((NP, NP))])]).tocsc()
    sysmati = spsla.factorized(sysmat)

    for k, t in enumerate(trange):
        crhsv = M*old_v + DT*(fv - eva_quadterm(hmat, old_v) + bbcu(ko_state))
        crhs = np.vstack([crhsv, fp])
        vp_new = np.atleast_2d(sysmati(np.squeeze(np.asarray(crhs)))).T
        old_v = vp_new[:NV]
        Cv = vcmat @ old_v
        ko_state = update_ko_state(ko_state, Cv)

        print(k, '/', Nts)
        print(spla.norm(Cv - Css, 2))

        with open(setup_str + '/fom_simulation.csv', 'a') as file:
            file.write(str(t) + ',' + str(spla.norm(Cv - Css, 2)) + '\n')


def run_cl_simulation(rom, name, Re=110, level=2, palpha=1e-3, control='bc'):
    """Run the closed-loop simulation with reduced LQG controller."""

    # define strings, directories and paths for data storage
    setup_str = 'lvl_' + str(level) + ('_' + control if control is not None else '') \
                + '_re_' + str(Re) + ('_palpha_' + str(palpha) if control == 'bc' else '')
    data_path = '../data/' + 'lvl_' + str(level) + ('_' + control if control is not None else '')
    setup_path = data_path + '/re_' + str(Re) + ('_palpha_' + str(palpha) if control == 'bc' else '')
    simulation_path = setup_str + '/' + name + '_simulation'

    if not os.path.exists(simulation_path):
        os.makedirs(simulation_path)

    with open(simulation_path + '/rom_' + str(rom.order) + '.csv', 'w') as file:
        file.write('t, yerr \n')

    # define first order model and matrices for simulation
    fom = load_fom(Re=110, level=2, palpha=1e-3, control='bc')

    mats = spio.loadmat(data_path + '/mats')

    M = mats['M']
    J = mats['J']
    hmat = mats['H']
    fv = mats['fv'] + 1./Re*mats['fv_diff'] + mats['fv_conv']
    fp = mats['fp'] + mats['fp_div']
    vcmat = mats['Cv']
    NV, NP = fv.shape[0], fp.shape[0]

    if control == 'bc':
        A = 1./Re*mats['A'] + mats['L1'] + mats['L2'] + 1./palpha*mats['Arob']
        B = 1./palpha*mats['Brob']
    else:
        A = 1./Re*mats['A'] + mats['L1'] + mats['L2']
        B = mats['B']
        # restrict to less dofs in the input
        NU = B.shape[1]
        B  = B[:, [0, NU//2]]

    # compute steady-state solution and linearized convection
    if not os.path.isfile(setup_path + '/ss_nse_sol'):
        ss_nse_v, _ = solve_steadystate_nse(mats, Re, control, palpha=palpha)
        conv_mat = linearized_convection(mats['H'], ss_nse_v)
        spio.savemat(setup_path + '/ss_nse_sol', {'ss_nse_v': ss_nse_v, 'conv_mat': conv_mat})
    else:
        ss_nse_sol = spio.loadmat(setup_path + '/ss_nse_sol')
        ss_nse_v, conv_mat = ss_nse_sol['ss_nse_v'], ss_nse_sol['conv_mat']

    # Define parameters for time stepping
    t0 = 0.
    tE = 8.
    Nts = 2**12
    DT = (tE-t0)/Nts
    trange = np.linspace(t0, tE, Nts+1)

    # Define functions that represent the system inputs
    if control is None:
        def bbcu(ko_state):
            return np.zeros((NV, 1))

        def update_ko_state(ko_state, Cv, DT):
            return ko_state

    else:
        Arom = to_matrix(rom.A, format='dense')  # .real?
        Brom = to_matrix(rom.B, format='dense')
        Crom = to_matrix(rom.C, format='dense')

        XCARE = spla.solve_continuous_are(Arom, Brom, Crom.T @ Crom, np.eye(Brom.shape[1]), balanced=False)
        XFARE = spla.solve_continuous_are(Arom.T, Crom.T, Brom @ Brom.T, np.eye(Crom.shape[0]), balanced=False)

        # define control based on Kalman observer state
        def bbcu(ko_state):
            uvec = -Brom.T @ XCARE @ ko_state
            return B @ uvec

        ko1_mat = Arom - XFARE @ Crom.T @ Crom - Brom @ Brom.T @ XCARE
        ko2_mat = XFARE @ Crom.T
        lu_piv = spla.lu_factor(np.eye(rom.order) - DT * ko1_mat)

        Css = vcmat @ ss_nse_v

        # function that determines the next state of the Kalman observer via implicit euler step
        def update_ko_state(ko_state, Cv, DT):
            return spla.lu_solve(lu_piv, ko_state + DT * ko2_mat @ (Cv - Css))

    # introduce small perturbation to steady-state solution as initial value
    pert = fom.A.source.project_onto_subspace(fom.A.operator.source.ones(), trans=True).to_numpy().T
    old_v = ss_nse_v + 1e-3 * pert

    # initialize state for observer
    ko_state = np.zeros((rom.order, 1))

    sysmat = sps.vstack([sps.hstack([M+DT*A, -J.T]),
                         sps.hstack([J, sps.csc_matrix((NP, NP))])]).tocsc()
    sysmati = spsla.factorized(sysmat)

    try:
        for k, t in enumerate(trange):
            crhsv = M*old_v + DT*(fv - eva_quadterm(hmat, old_v) + bbcu(ko_state))
            crhs = np.vstack([crhsv, fp])
            vp_new = np.atleast_2d(sysmati(crhs.flatten())).T
            old_v = vp_new[:NV]
            Cv = vcmat @ old_v
            ko_state = update_ko_state(ko_state, Cv, DT)

            print(k, '/', Nts)
            print(spla.norm(Cv - Css, 2))

            with open(simulation_path + '/rom_' + str(rom.order) + '.csv', 'a') as file:
                file.write(str(t) + ',' + str(spla.norm(Cv - Css, 2)) + '\n')
    except:
        with open('simulation_error_log.txt', 'a') as file:
            file.write(name + '_' + str(rom.order) + '\n')
