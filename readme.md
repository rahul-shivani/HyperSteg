# HyperSteg

## Introduction 

This is an open-source implementation of the paper **HyperSteg: Hyperbolic Learning for Deep Stegnography**. The paper is available [here]()

**Abstract**: Steganography is the art of hiding a secret message signal inside a publicly visible carrier with minimum perceptual loss in the carrier. In order to better hide information, it is critical to optimally represent the message-carrier wave interference while blending the message with the carrier. We propose HyperSteg: a novel steganography method in the hyperbolic space grounded in the hyperbolic properties of wave interference. Through hyperbolic learning, HyperSteg learns to better represent the hyperbolic properties of message-carrier interference with minimal additional computational cost. Through extensive exploratory and quantitative experiments over image and audio datasets, we introduce HyperSteg as a practical, model and modality agnostic approach for information hiding.

# Setup

### Dataset
ImageNet : Download the dataset from [here](https://drive.google.com/drive/folders/1oSJFQ1BgTbD8Ya9ub-sno3yJLUvIDpwQ?usp=sharing) and put in "imagenet-data" folder.
ESC50 : Download the dataset from [here](https://drive.google.com/drive/folders/1gf8PxHpDikYINT9ZjiQyH-wy40Z22gVQ?usp=sharing) and put in "esc-50-data" folder.

### Install dependencies
```
pip install -r requirements.txt
```

### Update Hyperparameters
Update modules/args.py

### To run experiments for ImageNet Dataset
```
python train.py
```

### To run experiments for ESC-50 Dataset
```
python train.py --dataset ESC-50
```