# Time-Series-Anomaly-Detection
This is the open-source code for the paper titled "Attention-based Bi-LSTM for Anomaly Detection on Time-Series Data"

Detailed comparative results have been given in the paper. This repository has been made for reproducibility of the paper.

## Do this before running anything
1. Clone the [DeepADoTS](https://github.com/KDD-OpenSource/DeepADoTS) in the same directory
2. Clone the [Numenta Anomaly Benchmark](https://github.com/numenta/NAB) repo in the same directory

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
