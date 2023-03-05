import numpy as np
import librosa
import torch
import os
import soundfile
import torchvision.transforms as transforms

def normalize(spec, eps=1e-6, mean=-26.2485, std=14.7750):
  spec_norm = (spec - mean) / (std + eps)
  return spec_norm

def get_melspectrogram_db(file_path):
  wav, sr = librosa.load(file_path)
  spec=librosa.feature.melspectrogram(wav, hop_length=1150)
  spec_db=librosa.power_to_db(spec)
  return spec_db

def denormalize(spec_norm, eps=1e-6, mean=-26.2485, std=14.7750):
  spec = (np.asarray(spec_norm) * (std + eps)) + mean
  return spec

def spec_to_audio(spec, path):
  spec = librosa.db_to_power(spec)
  spec =  librosa.feature.inverse.mel_to_audio(spec, hop_length=1150)
  soundfile.write(path, spec, samplerate=22050)

class ESC50Data(torch.utils.data.Dataset):
  def __init__(self, base, df, in_col, out_col):
    self.df = df
    self.data = []
    self.labels = []
    self.c2i={}
    self.i2c={}
    self.categories = sorted(df[out_col].unique())
    for i, category in enumerate(self.categories):
      self.c2i[category]=i
      self.i2c[i]=category
    for ind in range(len(df)):
      row = df.iloc[ind]
      file_path = os.path.join(base,row[in_col])
      img = normalize(get_melspectrogram_db(file_path))
      img = np.resize(img, (64,64,3))
      convert_tensor = transforms.Compose([transforms.ToTensor()])
      # img = torch.unsqueeze(convert_tensor(img), 0)
      img = convert_tensor(img)
      # # img = np.stack([img, img, img], axis=0)
      # img = np.expand_dims(img, axis=0)
      # print(img.shape)
      # img = np.asarray([img])
      self.data.append(img)
      self.labels.append(self.c2i[row['category']])
  def __len__(self):
    return len(self.data)
  def __getitem__(self, idx):
    return self.data[idx], self.labels[idx]
