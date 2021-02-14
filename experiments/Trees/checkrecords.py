import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from DeepTreeAttention.generators import boxes

#metadata
created_records = glob.glob("/orange/idtrees-collab/DeepTreeAttention/tfrecords/train/*.tfrecord")
dataset = boxes.tf_dataset(created_records, mode = "HSI_submodel", batch_size=10)
counter=0
labels=[]
data =[]
for data, label in dataset:
    counter+=data[0].shape[0]
    print(data.shape)
    
labels = np.concatenate(labels)
labels = np.argmax(labels,1)
pd.DataFrame({"label":labels}).groupby("label").size()