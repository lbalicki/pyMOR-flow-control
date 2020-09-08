from models import load_fom, write_freq_errors
from pymor.reductors.bt import StabilizingBTReductor

from simulation import run_cl_simulation

from pymor.core.cache import disable_caching, enable_caching

import timeit


def bernoulli_bt_errors(Re=110, level=2, palpha=1e-3, control='bc', r_list=[5, 10, 15, 20, 25, 30, 35, 40]):
    setup_str = 'lvl_' + str(level) + ('_' + control if control is not None else '') \
                + '_re_' + str(Re) + ('_palpha_' + str(palpha) if control == 'bc' else '')

    fom = load_fom(Re, level, palpha, control)

    reductor = StabilizingBTReductor(fom)

    for r in r_list:
        rom = reductor.reduce(r=r, projection='biorth')

        write_freq_errors(fom, rom, 'bernoulli_bt', setup_str, r)


def bernoulli_bt_simulation(Re=110, level=2, palpha=1e-3, control='bc', r_list=[5]):
    fom = load_fom(Re, level, palpha, control)

    reductor = StabilizingBTReductor(fom)

    for r in r_list:
        rom = reductor.reduce(r=r, projection='biorth')
        run_cl_simulation(rom, 'bernoulli_bt', Re=110, level=2, palpha=1e-3, control='bc')


def bernoulli_bt_time(Re=110, level=2, palpha=1e-3, control='bc', r_list=[5]):
    setup_str = 'lvl_' + str(level) + ('_' + control if control is not None else '') \
                + '_re_' + str(Re) + ('_palpha_' + str(palpha) if control == 'bc' else '')

    fom = load_fom(Re, level, palpha, control)

    disable_caching()

    reductor = StabilizingBTReductor(fom)

    with open(setup_str + '/bernoulli_bt_runtime.csv', 'w') as file:
        file.write('r,time\n')

    for r in r_list:
        times = timeit.repeat(
            stmt="""reductor.reduce(r=r, projection='biorth')""",
            globals=locals(),
            repeat=3,
            number=1,
        )
        with open(setup_str + '/bernoulli_bt_runtime.csv', 'a') as file:
            file.write(str(r) + ',' + '{:.3f}'.format(min(times)) + '\n')

    enable_caching()
