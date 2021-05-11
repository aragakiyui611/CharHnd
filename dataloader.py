import h5py
from PIL import Image
import PIL
import numpy as np
from torch.utils.data import DataLoader,Dataset
from torchvision import transforms
import torch
import re


class CharTrainVal(Dataset):
    def __init__(self,val=False):
        super().__init__()
        if val:
            datalist = "test.txt"
        else:
            datalist = "train.txt"

        with open(datalist, "r") as f:
            self.datalist = f.read().splitlines()

        self.transform = transforms.Compose([
                        
                        transforms.Grayscale(num_output_channels=1), # transform to gray image
                        transforms.Resize((300, 300)),               # resize to 100x100 image
                        transforms.ToTensor(),                       # transform to pytorch tensor
                        #transforms.Normalize((0.1307,), (0.3081,)),
                        ])


    def __len__(self):
        # this return the length of a dataset
        return len(self.datalist)


    def __getitem__(self, index):
        # this func returns a datum

        # --generate a label according to datum name
        a=self.datalist[index].split("/")[1]
        a=re.findall("\d+",a)[0]
        a=a.lstrip("0")
        a=int(a)
        label = torch.tensor(a-1)
        # --generate a label according to datum name

        with Image.open("./"+self.datalist[index]+".png") as datum:
            datum = PIL.ImageOps.invert(datum)
            datum = self.transform(datum)
            datum = datum-0.5
        return datum, label
