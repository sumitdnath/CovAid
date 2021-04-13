import pandas as pd
import os
import numpy as np
import cv2


datapath1='D:/Minor Project/COVID-19 Detection/Files/Covid-19 prediction using X-Ray images/covid-chestxray-dataset-master/'
dataset_path='D:/Minor Project/COVID-19 Detection/Files/Covid-19 prediction using X-Ray images/dataset'

categories=os.listdir(dataset_path)
print(categories)

dataset=pd.read_csv(os.path.join(datapath1,'metadata.csv'))
findings=dataset['finding']
image_names=dataset['filename']

positives_index=np.concatenate((np.where(findings=='COVID-19')[0],np.where(findings=='SARS')[0]))
positive_image_names=image_names[positives_index]


for positive_image_name in positive_image_names:
    image=cv2.imread(os.path.join(datapath1,'images',positive_image_name))
    try:
        cv2.imwrite(os.path.join(dataset_path,categories[1],positive_image_name),image)
    except Exception as e:
        print(e)
        
datapath2='D:/Minor Project/COVID-19 Detection/Covid-19 prediction using X-Ray images/archive'
dataset=pd.read_csv(os.path.join(datapath2,'Chest_xray_Corona_Metadata.csv'))

findings=dataset['Label']
image_names=dataset['X_ray_image_name']

negative_index=np.where(findings=='Normal')[0]
negative_image_names=image_names[negative_index]

for negative_image_name in negative_image_names:
    image=cv2.imread(os.path.join(datapath1,'images',negative_image_name))
    try:
        cv2.imwrite(os.path.join(dataset_path,categories[0],negative_image_name),image)
    except Exception as e:
        print(e)
negative_image_names.shape