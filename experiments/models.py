import sys
sys.path.append('..')
from data_utils import *

import numpy as np
import os
import pickle
import scipy.io as spio
import scipy.linalg as spla
import scipy.sparse as sps
import scipy.sparse.linalg as spsla

from pymor.algorithms.newton_lradi import solve_ricc_lrcf
from pymor.algorithms.to_matrix import to_matrix
from pymor.models.iosys import StokesDescriptorModel, LTIModel
from pymor.operators.constructions import LowRankOperator, LerayProjectedOperator, IdentityOperator
from pymor.operators.numpy import NumpyMatrixOperator
from pymor.vectorarrays.constructions import cat_arrays
from pymor.vectorarrays.numpy import NumpyVectorArray


def create_cl_fom(Re=110, level=2, palpha=1e-3, control='bc'):
    """Create model which is used to evaluate the H2-Gap norm."""
    setup_str = 'lvl_' + str(level) + ('_' + control if control is not None else '') \
                + '_re_' + str(Re) + ('_palpha_' + str(palpha) if control == 'bc' else '')

    fom = load_fom(Re, level, palpha, control)

    Bra = fom.B.as_range_array()
    Cva = fom.C.as_source_array()

    Z = solve_ricc_lrcf(fom.A, fom.E, Bra, Cva, trans=False)
    K = fom.E.apply(Z).lincomb(Z.dot(Cva).T)

    KC = LowRankOperator(K, np.eye(len(K)), Cva)
    mKB = cat_arrays([-K, Bra]).to_numpy().T
    mKBop = NumpyMatrixOperator(mKB)

    mKBop_proj = LerayProjectedOperator(mKBop, fom.A.source.G, fom.A.source.E, projection_space='range')

    cl_fom = LTIModel(fom.A - KC, mKBop_proj, fom.C, None, fom.E)

    with open(setup_str + '/cl_fom', 'wb') as cl_fom_file:
        pickle.dump({'cl_fom': cl_fom}, cl_fom_file)


def load_cl_fom(Re=110, level=2, palpha=1e-3, control='bc'):
    """Load the model created by create_cl_fom."""
    setup_str = 'lvl_' + str(level) + ('_' + control if control is not None else '') \
                + '_re_' + str(Re) + ('_palpha_' + str(palpha) if control == 'bc' else '')

    if not os.path.isfile(setup_str + '/cl_fom'):
        create_cl_fom(Re=110, level=2, palpha=1e-3, control='bc')

    with open(setup_str + '/cl_fom', 'rb') as fom_file:
        d = pickle.load(fom_file)

    return d['cl_fom']


def create_fom(Re=110, level=2, palpha=1e-3, control='bc'):
    """Create and save StokesDescriptorModel."""
    setup_str = 'lvl_' + str(level) + ('_' + control if control is not None else '') \
                + '_re_' + str(Re) + ('_palpha_' + str(palpha) if control == 'bc' else '')
    data_path = '../data/' + 'lvl_' + str(level) + ('_' + control if control is not None else '')
    setup_path = data_path + '/re_' + str(Re) + ('_palpha_' + str(palpha) if control == 'bc' else '')

    if not os.path.exists(setup_str):
        os.makedirs(setup_str)

    if not os.path.exists(setup_path):
        os.makedirs(setup_path)

    mats = spio.loadmat(data_path + '/mats')

    M = mats['M']
    J = mats['J']
    vcmat = mats['Cv'].todense()

    if control == 'bc':
        B = 1./palpha*mats['Brob']
    else:
        B = mats['B']
        # restrict to less dofs in the input
        NU = B.shape[1]
        B = B[:, [0, NU//2]]

    if not os.path.isfile(setup_path + '/ss_nse_sol'):
        ss_nse_v, _ = solve_steadystate_nse(mats, Re, control, palpha=palpha)
        conv_mat = linearized_convection(mats['H'], ss_nse_v)
        spio.savemat(setup_path + '/ss_nse_sol', {'ss_nse_v': ss_nse_v, 'conv_mat': conv_mat})
    else:
        ss_nse_sol = spio.loadmat(setup_path + '/ss_nse_sol')
        ss_nse_v, conv_mat = ss_nse_sol['ss_nse_v'], ss_nse_sol['conv_mat']

    if control == 'bc':
        Aop = NumpyMatrixOperator(-1./Re * mats['A'] - 1./palpha*mats['Arob'] - conv_mat)
    else:
        Aop = NumpyMatrixOperator(-1./Re * mats['A'] - conv_mat)

    Eop = NumpyMatrixOperator(M)
    Gop = NumpyMatrixOperator(J.T)
    Bop = NumpyMatrixOperator(B)
    Cop = NumpyMatrixOperator(vcmat)
    fom = StokesDescriptorModel(Aop, Gop, Bop, Cop, None, Eop)

    with open(setup_str + '/fom', 'wb') as fom_file:
        pickle.dump({'fom': fom}, fom_file)


def load_fom(Re=110, level=2, palpha=1e-3, control='bc'):
    """Load the model created by create_fom."""
    setup_str = 'lvl_' + str(level) + ('_' + control if control is not None else '') \
                + '_re_' + str(Re) + ('_palpha_' + str(palpha) if control == 'bc' else '')

    if not os.path.isfile(setup_str + '/fom'):
        create_fom(Re=Re, level=level, palpha=palpha, control=control)

    with open(setup_str + '/fom', 'rb') as fom_file:
        d = pickle.load(fom_file)

    return d['fom']


def get_gap_rom(rom):
    """Based on a rom, create model which is used to evaluate H2-Gap norm."""
    A = to_matrix(rom.A, format='dense')
    B = to_matrix(rom.B, format='dense')
    C = to_matrix(rom.C, format='dense')

    if isinstance(rom.E, IdentityOperator):
        P = spla.solve_continuous_are(A.T, C.T, B.dot(B.T), np.eye(len(C)), balanced=False)
        F = P @ C.T
    else:
        E = to_matrix(rom.E, format='dense')
        P = spla.solve_continuous_are(A.T, C.T, B.dot(B.T), np.eye(len(C)), e=E.T, balanced=False)
        F = E @ P @ C.T

    AF = A - F @ C
    mFB = np.concatenate((-F, B), axis=1)
    return LTIModel.from_matrices(AF, mFB, C, E=None if isinstance(rom.E, IdentityOperator) else E)


def write_freq_errors(fom, rom, name, setup_str, r, w=np.logspace(-4, 4, 50)):
    """Evaluate errors of a rom in the frequency domain."""
    if not os.path.exists(setup_str + '/' + name + '_freq'):
        os.makedirs(setup_str + '/' + name + '_freq')

    with open(setup_str + '/' + name + '_freq/r_' + str(r) + '.csv', 'w') as file:
        file.write('w, relerror, abserror \n')

    err = fom - rom

    for freq in w:
        abs_err = spla.norm(err.eval_tf(1j * freq), 2)
        rel_err = abs_err / spla.norm(fom.eval_tf(1j * freq), 2)
        with open(setup_str + '/' + name + '_freq/r_' + str(r) + '.csv', 'a') as file:
            file.write(str(freq) + ',' + str(rel_err) + ',' + str(abs_err) + '\n')
