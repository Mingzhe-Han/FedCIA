import argparse
import configparser

from flalgorithm.base_gsp import Base_gsp

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./config/ml100k/base_gsp.ini')
    args= parser.parse_args()

    config = configparser.ConfigParser()
    config.read(args.config)
    Testfed = Base_gsp(config)
    Testfed.fit()