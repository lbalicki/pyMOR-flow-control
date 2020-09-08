from simulation import run_fom_cl_simulation, run_no_control_simulation
from lqgbt import lqgbt_simulation, lqgbt_errors, lqgbt_time
from bernoulli_bt import bernoulli_bt_simulation, bernoulli_bt_errors, bernoulli_bt_time
from gap_irka import gap_irka_simulation, gap_irka_convergence, gap_irka_errors, gap_irka_time
from models import merge_error_results

r_list = [10, 20, 30]
lqgbt_errors(r_list=r_list)
bernoulli_bt_errors(r_list=r_list)
gap_irka_errors(r_list=r_list)


r_list = [10, 20, 30]
gap_irka_convergence(r_list=r_list)

r_list = [3, 4, 5, 10, 20, 30]
lqgbt_simulation(r_list=r_list)
bernoulli_bt_simulation(r_list=r_list)
gap_irka_simulation(r_list=r_list)

bernoulli_bt_time(r_list=[30])
lqgbt_time(r_list=[30])
gap_irka_time(r_list=[5, 10, 15, 20, 25, 30])
