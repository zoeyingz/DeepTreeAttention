#Extract highly confused classes for visualization
import tensorflow as tf
import numpy as np
import pandas as pd
from scipy import io

#THe matlab instructions were taken from https://stackoverflow.com/questions/10997254/converting-numpy-arrays-to-matlab-and-vice-versa
    
def save_images_to_matlab(DeepTreeAttention, savedir, classes):
    """Load a DeepTreeAttention object and save example HSI images to file
    Args:
        DeepTreeAttention: AttentionModel class object
        classes: list which classes to save
        savedir
    """

    DeepTreeAttention.read_data("HSI")
    
    labeldf = pd.read_csv(DeepTreeAttention.classes_file)
    selected_labels = labeldf[labeldf.taxonID.isin(classes)].taxonID.values
    label_names = dict(zip(labeldf.label, labeldf.taxonID))
    
    # load model
    
    #Keep track of index per species
    counter = {}
    for taxon in classes:
        counter[taxon] = 0 
        
    #Loop through the data and export data
    for box_id, batch in DeepTreeAttention.val_split_with_ids:
        data, label = batch
        images = data.numpy()
        labels = label.numpy()
        labels = np.argmax(labels,1)
        ids = box_id.numpy()
        for box_index, image, label in zip(ids, images, labels):
            taxon=label_names[label]
            if taxon in selected_labels:
                counter[taxon] +=1
                filename = "{}/{}_{}.mat".format(savedir, taxon, box_index)
                io.savemat(filename,  dict({"image":image}))
    
    print("Saved {}".format(counter))

        
