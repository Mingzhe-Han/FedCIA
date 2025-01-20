import argparse
import configparser

from flalgorithm.fedcia_gsp import Fedcia

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./config/ml100k/fedgsp.ini')
    args= parser.parse_args()

    config = configparser.ConfigParser()
    config.read(args.config)
    Testfed = Fedcia(config)
    Testfed.fit()