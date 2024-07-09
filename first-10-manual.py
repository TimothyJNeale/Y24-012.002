import os
import logging

from plotting import check_solution, get_task_data, plot_task
from solutions import *


INPUT_DIR = 'data'

LOGGING_FORMAT = '%(asctime)s - %(levelname)s - %(message)s'
LOGGING_LEVEL = logging.INFO

# Setup logging messages
logging.basicConfig(level=LOGGING_LEVEL, format=LOGGING_FORMAT)

# Log start message using config message
logging.info('Start')

# get the list of input files
training_tasks = sorted(os.listdir(INPUT_DIR))

task = get_task_data(os.path.join(INPUT_DIR,training_tasks[0]))
logging.info(f'File: {training_tasks[0]}')
check_solution(task, task_train000, training_tasks[0])

task = get_task_data(os.path.join(INPUT_DIR,training_tasks[1]))
logging.info(f'File: {training_tasks[1]}')
check_solution(task, task_train001, training_tasks[1])

task = get_task_data(os.path.join(INPUT_DIR,training_tasks[2]))
logging.info(f'File: {training_tasks[2]}')
check_solution(task, task_train002, training_tasks[2])

# # No solution for task_train003
# # It slides the top of any shape one step to the right - this is still not right
task = get_task_data(os.path.join(INPUT_DIR,training_tasks[3]))
logging.info(f'File: {training_tasks[3]}')
check_solution(task, task_train003, training_tasks[3])

task = get_task_data(os.path.join(INPUT_DIR,training_tasks[4]))
logging.info(f'File: {training_tasks[4]}')
check_solution(task, task_train004, training_tasks[4])

task = get_task_data(os.path.join(INPUT_DIR,training_tasks[5]))
logging.info(f'File: {training_tasks[5]}')
check_solution(task, task_train005, training_tasks[5])

task = get_task_data(os.path.join(INPUT_DIR,training_tasks[6]))
logging.info(f'File: {training_tasks[6]}')
check_solution(task, task_train006, training_tasks[6])

task = get_task_data(os.path.join(INPUT_DIR,training_tasks[7]))
logging.info(f'File: {training_tasks[7]}')
check_solution(task, task_train007, training_tasks[7])

task = get_task_data(os.path.join(INPUT_DIR,training_tasks[8]))
logging.info(f'File: {training_tasks[8]}')
check_solution(task, task_train008, training_tasks[8])

task = get_task_data(os.path.join(INPUT_DIR,training_tasks[9]))
logging.info(f'File: {training_tasks[9]}')
check_solution(task, task_train009, training_tasks[9])