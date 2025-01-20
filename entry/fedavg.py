import argparse
import configparser

from flalgorithm.fedavg import Fedavg

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./config/ml100k/fedmf.ini')
    args= parser.parse_args()

    config = configparser.ConfigParser()
    config.read(args.config)
    Testfed = Fedavg(config)
    Testfed.fit()