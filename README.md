# FedCIA
A federated recommendation framework for collaborative information aggregation

## Running
A `log` file is needed in our file directory.

Run experiments on the baselines:
`python -m entry.fedavg --config=config/ml100k/fedmf.ini`.

Run experiments on our FedCIA:
`python -m entry.fedcia_param --config=config/ml100k/fedmf.ini`.

For GSP models:
`python -m entry.fedcia_base --config=config/ml100k/fedgsp.ini`.

For the ml100k dataset where only one user in a client:
`python -m entry.fedcia_param --config=config/ml100k_all/fedmf.ini`.

## Repeat Experiments
If the `output file` in the config already exists, the experiment will not start.
You can change the output in config in [exp] to start a second experiment.
