<br><br><br>

# RED-CNN

Tensorflow 2.0 implementation for Low-dose CT with a Residual Encoder-Decoder Convolutional Neural Network (RED-CNN).
Chen et al. proposed this method in https://arxiv.org/ftp/arxiv/papers/1702/1702.00288.pdf.

## Data Preparation
Download Mayo Clinic low-dose CT dataset.
Firstly, convert dcm file to numpy array or mat. This operation is defined in dcm_convert_mat.py. 
The put CT image patch is 64 x 64. 
## Prerequisites
- tensorflow r2.3.1
- numpy 1.11.0
- scipy 1.6.0
- scikit-image 0.18.1

## Getting Started
### Installation
- Install tensorflow from https://www.tensorflow.org

```

### Train

```
- Train a model:
```bash
python main.py
```
- Use tensorboard to visualize the training details:
```bash
tensorboard --logdir=./logs
```

