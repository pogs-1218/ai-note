import os
import pandas as pd

# TODO: 
# Dataset class's member, method
# Details of Dataset class 
# Details of transform and target_transform
# iloc ?? 
# https://pytorch.org/docs/stable/data.html#torch.utils.data.Dataset
class CustomImageDataset(Dataset, annotations_file, img_dir, transform=None, target_transform=None):
  def __init__(self):
    # TODO: read_csv !!
    self.img_labels = pd.read_csv(annotations_file)
    self.img_dir = img_dir
    self.transform = transform
    self.target_transform = target_transform

  def __len__(self):
    return len(self.img_labels)

  def __getitem__(self, idx):
    img_path = os.path.join(self.img_dir, self.img_labels.iloc([idx, 0])
    # TODO:deatils of torchvision.io.read_image, ie. what is the return type?
    img = torchvision.io.read_image(img_path)
    label = self.img_labels.iloc[idx, 1]
    if self.transform:
      image = self.transform(image)
    if self.target_transform:
      label = self.target_transform(label)
    return image, label
