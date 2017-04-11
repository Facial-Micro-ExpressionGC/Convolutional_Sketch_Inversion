# [Keras](https://keras.io/) Implementation of Convolutional Sketch Inversion

## Instruction
### 1. Prepare Dataset
Follow [these instructions](https://github.com/TengdaHan/Convolutional_Sketch_Inversion/tree/master/src/data).

### 2. Training
Follow [these instructions](https://github.com/TengdaHan/Convolutional_Sketch_Inversion/tree/master/src).

## Introduction
* Network Structure

Layer|Type|in_channels|out_channels|ksize|stride|pad|normalization|activation
---|---|---|---|---|---|---|---|---
1|con.|1 or 3|32|9|1|4|BN|ReLU
2|con.|32|64|3|2|1|BN|ReLU
3|con.|64|128|3|2|1|BN|ReLU
4|res.|128/128|128/128|3/3|1/1|1/1|BN/BN|ReLU
5|res.|128/128|128/128|3/3|1/1|1/1|BN/BN|ReLU/+x
6|res.|128/128|128/128|3/3|1/1|1/1|BN/BN|ReLU/+x
7|res.|128/128|128/128|3/3|1/1|1/1|BN/BN|ReLU/+x
8|res.|128/128|128/128|3/3|1/1|1/1|BN/BN|ReLU/+x
9|dec.|128|64|3|2|1|BN|ReLU
10|dec.|64|32|3|2|1|BN|ReLU
11|con.|32|3|9|1|4|BN|tanh

* Requirement
```
keras
sys
os
time
cv2
h5py
parmap
argparse
numpy
pandas
tqdm
matplotlib
```

## Reference: 
[Convolutional Sketch Inversion](https://arxiv.org/abs/1606.03073) by Yağmur Güçlütürk, Umut Güçlü, Rob van Lier, Marcel A. J. van Gerven, (2016).
