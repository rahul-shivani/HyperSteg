import torch

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

# Hyper Parameters
num_epochs = 100
batch_size = 64
learning_rate = 0.0001
beta_ = 1
add_noise = False
add_jpeg_compression = False

# Mean and std deviation of imagenet dataset. Source: http://cs231n.stanford.edu/reports/2017/pdfs/101.pdf
std = [0.229, 0.224, 0.225]
mean = [0.485, 0.456, 0.406]

# Checkpoints path
MODELS_PATH = 'checkpoints/HypStegNet/'

# For ImageNet
VALID_PATH = 'imagenet-data/val/'
TRAIN_PATH = 'imagenet-data/train/'
TEST_PATH = 'imagenet-data/test/'

# For ESC50
AUDIO_PATH = 'esc-50-data/audio'
META_PATH = 'esc-50-data/meta'
