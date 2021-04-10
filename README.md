# Time-Series-Anomaly-Detection
This is the open-source code for the paper titled "Attention-based Bi-LSTM for Anomaly Detection on Time-Series Data"

Detailed comparative results have been given in the paper. This repository has been made for reproducibility of the paper.

## Do this before running anything
1. Clone the [DeepADoTS](https://github.com/KDD-OpenSource/DeepADoTS) in the same directory
2. Clone the [Numenta Anomaly Benchmark](https://github.com/numenta/NAB) repo in the same directory

## Comparison of the proposed model with existing and previous state-of-the-art models
|         Dataset         | Our Model | DeepAnT |   WG  | AdVec | Skyline | NumentaTM | Numenta | KNN CAD | HTM Java |
|:-----------------------:|:---------:|:-------:|:-----:|:-----:|:-------:|:---------:|:-------:|:-------:|:--------:|
| artificialWithNoAnomaly |     0     |    0    |   0   |   0   |    0    |     0     |    0    |    0    |     0    |
|  artificialWithAnomaly  |   0.402   |  0.156  | 0.013 | 0.017 |  0.043  |   0.017   |  0.012  |  0.003  |   0.017  |
|      realAdExchange     |   0.214   |  0.132  | 0.026 | 0.018 |  0.005  |   0.035   |  0.040  |  0.024  |   0.034  |
|    realAWSCloudwatch    |   0.269   |  0.146  | 0.060 | 0.013 |  0.053  |   0.018   |  0.017  |  0.006  |   0.018  |
|      realKnownCause     |   0.331   |  0.200  | 0.006 | 0.017 |  0.008  |   0.012   |  0.015  |  0.008  |   0.013  |
|       realTraffic       |   0.398   |  0.223  | 0.045 | 0.020 |  0.091  |   0.036   |  0.033  |  0.013  |   0.032  |
|        realTweets       |   0.165   |  0.075  | 0.026 | 0.018 |  0.035  |   0.010   |  0.009  |  0.004  |   0.010  |

## Reproducing baseline results
The paper discusses and introduces the following four models as baselines:
* LSTMED
* DAGMM
* REBM
* Donut

You can reproduce the results by the following command : 

`python3 baselines.py /path/to/NAB/directory/ model_name`

For example, if you want to reproduce the results of LSTMED on the realAdExchange dataset, the command would be : 

`python3 baselines.py ./NAB/data/readAdExchange/ LSTMED`

The results will be written in a file by the name : `baseline_results.csv`

## Reproducing model results
You can reproduce the results of the model on any of the datasets by the following command : 

`python3 model.py /path/to/NAB/directory/`

For example, if you want to reproduce the results of LSTMED on the realAdExchange dataset, the command would be : 

`python3 model.py ./NAB/data/readAdExchange/`

The results will be written in a file by the name : `model_results.csv`
