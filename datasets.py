# Copyright 2019 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Provides data for training and testing."""
import numpy as np
import PIL
import skimage.io
import torch
import json
import torch.utils.data
import torchvision
import warnings
import random
import os


class BaseDataset(torch.utils.data.Dataset):
  """Base class for a dataset."""

  def __init__(self):
    super(BaseDataset, self).__init__()
    self.imgs = []
    self.test_queries = []

  def get_loader(self,
                 batch_size,
                 shuffle=False,
                 drop_last=False,
                 num_workers=0):
    return torch.utils.data.DataLoader(
        self,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=drop_last,
        collate_fn=lambda i: i)

  def get_test_queries(self):
    return self.test_queries

  def get_all_texts(self):
    raise NotImplementedError

  def __getitem__(self, idx):
    return self.generate_random_query_target()

  def generate_random_query_target(self):
    raise NotImplementedError

  def get_img(self, idx, raw_img=False):
    raise NotImplementedError

class VisualGenome(BaseDataset):
  # Visual Genome Dataset
  def __init__(self, path, split = 'train', transform = None):
    super(VisualGenome, self).__init__()

    self.split = split
    self.transform = transform
    self.img_path = path + '/'
    #print('Hello --- ', split, transform, path)
    with open('/DATA/dataset/Visual_Genome/data/cat_attr_img_roi.json') as f:
        data = json.load(f)
        self.cat = []
        self.txt = []
        self.imgid = []

        for category in data.keys():
#            k=0
            for txt_query in data[category].keys():
#                k+=1
#                if k>10:
#                    break
#                jk=0
                for img_object in data[category][txt_query]:
#                    jk+=1
#                    if jk>10:
#                        break
#                    print(category, txt_query, img_id)
                    self.cat.append(category)
                    self.txt.append(txt_query)
                    self.imgid.append(img_object)

  def get_all_texts(self):
    #print("get_all_texts ----- ", len(self.txt), self.txt[0], type(self.txt[0]))
    return(self.txt)

  def __len__(self):
    return len(self.cat)

  def __getitem__(self, idx):
    out = {}
    #print('INDEX -----', idx)
    out['source_img_data'] = self.get_sketch(self.cat[idx])
    out['target_img_data'] = self.get_img(self.imgid[idx])
    out['mod'] = {'str': self.txt[idx]}
    out['category']=self.cat[idx]
    out['target_id']=self.imgid[idx]
    return out

  def get_img(self, idx, raw_img=False):
    img_path = '/DATA/dataset/Visual_Genome/images/' + str(idx['id']) + '.jpg'
    #print('path ----- ', img_path)
    with open(img_path, 'rb') as f:
      img = PIL.Image.open(f)
      #img = img.crop((idx['x'], idx['y'], idx['x'] + idx['w'], idx['y'] + idx['h']))
      img = img.convert('RGB')
    if raw_img:
      return img
    if self.transform:
      img = self.transform(img)
    return img

  def get_sketch(self, cat):
    if cat == 'car':
        cat = 'car_(sedan)'
    sketch_folder = '/DATA/dataset/SketchyExtended/rendered_256x256/256x256/photo/tx_000000000000/' + cat #cat = category 
#    /DATA/dataset/SketchyExtended/rendered_256x256/256x256/photo/tx_000000000000/
    sketch_path = os.path.join(sketch_folder,random.choice(os.listdir(sketch_folder)))
    with open(sketch_path, 'rb') as f:
      img = PIL.Image.open(f)
      img = img.convert('RGB')
    if self.transform:
      img = self.transform(img)
    return img
