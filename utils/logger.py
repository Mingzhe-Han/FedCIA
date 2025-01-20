import logging
import os

def logger(exp_name, exp_output):
    exp_save_name = 'log/' + exp_name + '/' + exp_output
    if os.path.exists(exp_save_name):
        raise ValueError(("Exp name and output has already existed"))
    os.makedirs(exp_save_name)

    logger = logging.getLogger(exp_name)

    fh = logging.FileHandler(filename = exp_save_name + "/output.log", mode='w')
    fh.setLevel(logging.INFO)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)

    logger.setLevel(logging.INFO)

    return logger, exp_save_name