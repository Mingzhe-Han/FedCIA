import argparse
import configparser

from flalgorithm.base import Base

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./config/ml100k/base_mf.ini')
    args= parser.parse_args()

    config = configparser.ConfigParser()
    config.read(args.config)
    Testfed = Base(config)
    Testfed.fit()