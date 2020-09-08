from models import load_fom, write_freq_errors
from simulation import run_cl_simulation

from pymor.algorithms.newton_lradi import ricc_lrcf_solver_options
from pymor.reductors.bt import LQGBTReductor

from pymor.core.cache import disable_caching, enable_caching

import timeit


def lqgbt_errors(Re=110, level=2, palpha=1e-3, control='bc', r_list=[5, 10, 15, 20, 25, 30, 35, 40]):
    setup_str = 'lvl_' + str(level) + ('_' + control if control is not None else '') \
                + '_re_' + str(Re) + ('_palpha_' + str(palpha) if control == 'bc' else '')

    fom = load_fom(Re, level, palpha, control)

    reductor = LQGBTReductor(fom, solver_options=ricc_lrcf_solver_options()['lrnadi'])

    for r in r_list:
        rom = reductor.reduce(r=r, projection='biorth')

        write_freq_errors(fom, rom, 'lqgbt', setup_str, r)


def lqgbt_simulation(Re=110, level=2, palpha=1e-3, control='bc', r_list=[5]):
    fom = load_fom(Re, level, palpha, control)

    reductor = LQGBTReductor(fom, solver_options=ricc_lrcf_solver_options()['lrnadi'])

    for r in r_list:
        rom = reductor.reduce(r=r, projection='biorth')
        run_cl_simulation(rom, 'lqgbt', Re=110, level=2, palpha=1e-3, control='bc')


def lqgbt_time(Re=110, level=2, palpha=1e-3, control='bc', r_list=[5]):
    setup_str = 'lvl_' + str(level) + ('_' + control if control is not None else '') \
                + '_re_' + str(Re) + ('_palpha_' + str(palpha) if control == 'bc' else '')

    fom = load_fom(Re, level, palpha, control)

    disable_caching()

    reductor = LQGBTReductor(fom, solver_options=ricc_lrcf_solver_options()['lrnadi'])

    with open(setup_str + '/lqgbt_runtime.csv', 'w') as file:
        file.write('r,time\n')

    for r in r_list:
        times = timeit.repeat(
            stmt="""reductor.reduce(r=r, projection='biorth')""",
            globals=locals(),
            repeat=3,
            number=1,
        )
        with open(setup_str + '/lqgbt_runtime.csv', 'a') as file:
            file.write(str(r) + ',' + '{:.3f}'.format(min(times)) + '\n')

    enable_caching()
