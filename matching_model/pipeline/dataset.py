from sklearn.preprocessing import LabelEncoder
from torchvision.io import read_image
from torch.utils.data import Dataset
from torchvision import transforms
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os





class DataPreparation():
    """Класс, в котором предполагалась предобработка сырых данных для подачи их в модели"""

    def __init__(self, data):
        pass

    def prepare(self):
        pass



class CustomMatchingDataset(Dataset):
    """Кастомный датасет для использования встроенного загрузчика torch.utils.data.DataLoader"""
    def __init__(self, data, img_dir):
        self.le_cat1 = LabelEncoder()
        self.le_cat2 = LabelEncoder()


        data['cat1_enc'] = self.le_cat1.fit_transform(data['cat1'].values)
        data['cat2_enc'] = self.le_cat2.fit_transform(data['cat2'].values)

        self.data = data
        self.images = data['image']
        self.img_dir = img_dir
        self.transform = transforms.Compose([
                                                transforms.ToPILImage(),
                                                transforms.Resize((380, 380)),
                                                transforms.ToTensor()

                                            ])
        
        self.names = data['name_prepared']
        self.attributes = data['attributes_prepared']
        self.labels_cat1 = data['cat1_enc']
        self.labels_cat2 = data['cat2_enc']


    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        label_cat1 = self.labels_cat1.iloc[idx]
        label_cat2 = self.labels_cat2.iloc[idx]


        img_path = os.path.join(self.img_dir, self.images.iloc[idx])
        image = read_image(img_path)
        if self.transform:
            image = self.transform(image).float()

        name = self.names.iloc[idx]
        attribute = self.attributes.iloc[idx]

        return name, attribute, image, idx, label_cat1, label_cat2
    


