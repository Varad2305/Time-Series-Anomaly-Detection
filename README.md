# Time-Series-Anomaly-Detection
This repository contains the open-source code for the paper titled *"Attention-based Bi-LSTM for Anomaly Detection on Time-Series Data"* by Sanket Mishra, Varad Kshirsagar, Rohit Dwivedula and Chittaranjan Hota.

## Model diagram
![frame](https://user-images.githubusercontent.com/43151489/114315014-bbf85080-9b1a-11eb-9c53-4df20c9468e2.png)
*

## Comparison of the proposed model with existing and previous state-of-the-art models
1. On the basis of average F-Score:

|         Dataset         | Our Model | DeepAnT |   WG  | AdVec | Skyline | NumentaTM | Numenta | KNN CAD | HTM Java |
|:-----------------------:|:---------:|:-------:|:-----:|:-----:|:-------:|:---------:|:-------:|:-------:|:--------:|
| artificialWithNoAnomaly |     0     |    0    |   0   |   0   |    0    |     0     |    0    |    0    |     0    |
|  artificialWithAnomaly  |   0.402   |  0.156  | 0.013 | 0.017 |  0.043  |   0.017   |  0.012  |  0.003  |   0.017  |
|      realAdExchange     |   0.214   |  0.132  | 0.026 | 0.018 |  0.005  |   0.035   |  0.040  |  0.024  |   0.034  |
|    realAWSCloudwatch    |   0.269   |  0.146  | 0.060 | 0.013 |  0.053  |   0.018   |  0.017  |  0.006  |   0.018  |
|      realKnownCause     |   0.331   |  0.200  | 0.006 | 0.017 |  0.008  |   0.012   |  0.015  |  0.008  |   0.013  |
|       realTraffic       |   0.398   |  0.223  | 0.045 | 0.020 |  0.091  |   0.036   |  0.033  |  0.013  |   0.032  |
|        realTweets       |   0.165   |  0.075  | 0.026 | 0.018 |  0.035  |   0.010   |  0.009  |  0.004  |   0.010  |

2. On the basis of average AUC:

| Dataset                 | Our Model | FuseAD | DeepAnT | WG    | AdVec | Skyline | Numenta | HTM Java |
|-------------------------|-----------|--------|---------|-------|-------|---------|---------|----------|
| artificialWithNoAnomaly | 0         | 0      | 0       | 0     | 0     | 0       | 0       | 0        |
| artificialWithAnomaly   | 0.678     | 0.544  | 0.555   | 0.406 | 0.503 | 0.558   | 0.531   | 0.653    |
| reaAdExchange           | 0.673     | 0.588  | 0.563   | 0.538 | 0.504 | 0.534   | 0.576   | 0.568    |
| realAWSCloudwatch       | 0.640     | 0.572  | 0.583   | 0.614 | 0.503 | 0.602   | 0.542   | 0.587    |
| realKnownCause          | 0.909     | 0.587  | 0.601   | 0.572 | 0.504 | 0.610   | 0.590   | 0.584    |
| realTraffic             | 0.737     | 0.619  | 0.637   | 0.553 | 0.505 | 0.556   | 0.679   | 0.691    |
| realTweets              | 0.729     | 0.546  | 0.554   | 0.560 | 0.505 | 0.559   | 0.586   | 0.549    |

## Comparison of the proposed model with new baselines introduced by us
1. On the basis of average F-Score:

| Dataset                 | Our Model | DAGMM | REBM  | Donut | LSTM-ED |
|-------------------------|-----------|-------|-------|-------|---------|
| artificialWithNoAnomaly | 0         | 0     | 0     | 0     | 0       |
| artificialWithAnomaly   | 0.402     | 0.400 | 0.325 | 0.399 | 0.346   |
| reaAdExchange           | 0.214     | 0.279 | 0.167 | 0.173 | 0.222   |
| realAWSCloudwatch       | 0.269     | 0.226 | 0.209 | 0.207 | 0.208   |
| realKnownCause          | 0.331     | 0.326 | 0.155 | 0.197 | 0.326   |
| realTraffic             | 0.398     | 0.327 | 0.288 | 0.315 | 0.365   |
| realTweets              | 0.165     | 0.132 | 0.117 | 0.127 | 0.182   |

## Reproducing baseline results
The paper discusses and introduces the following four models as baselines for this dataset:
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

## Note
1. This project uses the [NAB](https://github.com/numenta/NAB) dataset. We have used a slightly modified version of the dataset for convenience. The data is exaclty the same as the one referred to earlier in this point.
2. The project also uses code from the [DeepADoTS](https://github.com/KDD-OpenSource/DeepADoTS) repository for the baseline models. This repository is officially maintained by KDD. We have made small changes in the repository to resolve dependency issues. Hence, the modified code is added as a submodule in this repo. A few changes have also been made in the source code of some packages used in this project. While we have included a requirements.txt file, you'll either have to install tensorflow 1.13 to run the baselines or you'll have to change `import tensorflow as tf` to `import tensorflow.compat.v1 as tf` wherever you face an error, virtual environment included. This is because the DeepADoTS repo was written when tensorflow was still in version 1 and it hasn't been updated since.

## License

MIT License

Copyright (c) 2021 Varad Kshirsagar

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
