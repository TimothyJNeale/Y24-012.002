import os
import json
import logging

from plotting import colour_chart, plot_task

INPUT_DIR = 'data'

LOGGING_FORMAT = '%(asctime)s - %(levelname)s - %(message)s'
LOGGING_LEVEL = logging.INFO

# Setup logging messages
logging.basicConfig(level=LOGGING_LEVEL, format=LOGGING_FORMAT)

# Log start message using config message
logging.info('Start')

# get the list of input files
training_tasks = sorted(os.listdir(INPUT_DIR))

# build a new list with just the first 10 training tasks
training_tasks = training_tasks[:10]

# Log the number of training files
logging.info(f'Number of training files: {len(training_tasks)}')

# Plot the colour chart
colour_chart()

# Display graphic of each task in the first 10 training
for file in training_tasks:
    with open(os.path.join(INPUT_DIR, file), 'r') as f:
        task = json.load(f)
        logging.info(f'File: {file}')
        plot_task(task, file)
