import datetime
import os
import logging
from datetime import datetime

def set_logger(args):
    """Write logs to checkpoint and console"""
    cur_time = datetime.now()
    if args.do_train:
        log_file = os.path.join(args.save_path, f"{0}-{1}-train.log".format(args.data_path, cur_time))
    else:
        log_file = os.path.join(args.save_path, f"{0}-{1}-test.log".format(args.data_path, cur_time))

    logging.basicConfig(
        format='%(asctime)s %(levelname)-8s %(message)s',
        level=logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S',
        filename=log_file,
        filemode='w'
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)

def log_metrics(mode, step, metrics):
    """Print the evaluation logs"""
    for metric in metrics:
        logging.info('%s %s at epoch %d: %f' % (mode, metric, step, metrics[metric]))