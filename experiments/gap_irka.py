from models import load_fom, load_cl_fom, get_gap_rom, write_freq_errors
from simulation import run_cl_simulation

from pymor.core.cache import disable_caching, enable_caching
from pymor.reductors.h2 import GapIRKAReductor

import numpy as np
import scipy.linalg as spla

import os
import timeit


def gap_irka_convergence(Re=110, level=2, palpha=1e-3, control='bc', r_list=[10, 20, 30]):
    setup_str = 'lvl_' + str(level) + ('_' + control if control is not None else '') \
                + '_re_' + str(Re) + ('_palpha_' + str(palpha) if control == 'bc' else '')

    disable_caching()
    fom = load_fom(Re, level, palpha, control)
    cl_fom = load_cl_fom(Re, level, palpha, control)

    for r in r_list:
        reductor = GapIRKAReductor(fom)
        _ = reductor.reduce(r, tol=-1, maxit=50, num_prev=1, conv_crit='htwogap',
                            projection='Eorth', compute_errors=True, closed_loop_fom=cl_fom)

        iter_num = len(reductor.conv_crit)
        sigmas = reductor.sigma_list

        with open(setup_str + '/conv_crit_' + str(r) + '.csv', 'w') as file:
            file.write('iter,sigma,sigmatwo,romhtwogap,htwogap,absolute\n')

        for i in range(iter_num):
            sigma_change_1 = spla.norm((sigmas[i+1] - sigmas[i]) / sigmas[i+1], ord=np.inf)

            if i > 0:
                sigma_change_2 = min(spla.norm((sigmas[i-1] - sigmas[i+1]) / sigmas[i+1], ord=np.inf),
                                     spla.norm((sigmas[i] - sigmas[i+1]) / sigmas[i+1], ord=np.inf))
            else:
                sigma_change_2 = sigma_change_1

            if i > 0:
                error_change = spla.norm(reductor.errors[i] - reductor.errors[i-1]) / spla.norm(reductor.errors[0])
            else:
                error_change = ''

            with open(setup_str + '/conv_crit_' + str(r) + '.csv', 'a') as file:
                file.write(str(i+1) + ',' + str(sigma_change_1) + ',' + str(sigma_change_2)
                           + ',' + str(reductor.conv_crit[i]) + ',' + str(error_change)
                           + ',' + str(reductor.errors[i]) + '\n')
    enable_caching()


def gap_irka_errors(Re=110, level=2, palpha=1e-3, control='bc', r_list=[5, 10, 15, 20, 25, 30, 35, 40]):
    setup_str = 'lvl_' + str(level) + ('_' + control if control is not None else '') \
                + '_re_' + str(Re) + ('_palpha_' + str(palpha) if control == 'bc' else '')

    fom = load_fom(Re, level, palpha, control)

    for r in r_list:
        reductor = GapIRKAReductor(fom)
        rom = reductor.reduce(r, tol=1e-3, maxit=100, num_prev=1, conv_crit='htwogap',
                              projection='Eorth', compute_errors=False)

        write_freq_errors(fom, rom, 'gap_irka', setup_str, r)


def gap_irka_simulation(Re=110, level=2, palpha=1e-3, control='bc', r_list=[5]):
    fom = load_fom(Re, level, palpha, control)

    reductor = GapIRKAReductor(fom)

    for r in r_list:
        rom = reductor.reduce(r, tol=1e-3, maxit=100, num_prev=1, conv_crit='htwogap',
                              projection='Eorth', compute_errors=False)
        run_cl_simulation(rom, 'gap_irka', Re=110, level=2, palpha=1e-3, control='bc')


def gap_irka_time(Re=110, level=2, palpha=1e-3, control='bc', r_list=[5]):
    setup_str = 'lvl_' + str(level) + ('_' + control if control is not None else '') \
                + '_re_' + str(Re) + ('_palpha_' + str(palpha) if control == 'bc' else '')

    fom = load_fom(Re, level, palpha, control)

    disable_caching()

    reductor = GapIRKAReductor(fom)

    with open(setup_str + '/gap_irka_runtime.csv', 'w') as file:
        file.write('r,time\n')

    for r in r_list:
        times = timeit.repeat(
            stmt="""reductor.reduce(r, tol=1e-3, maxit=100, num_prev=1, conv_crit='htwogap',
                                  projection='Eorth', compute_errors=False)""",
            globals=locals(),
            repeat=3,
            number=1,
        )
        with open(setup_str + '/gap_irka_runtime.csv', 'a') as file:
            file.write(str(r) + ',' + '{:.3f}'.format(min(times)) + '\n')

    enable_caching()
