import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

def predict_pixels(model, eval_dataset_with_ids, submodel=None):
    """For a tf.dataset which yields index, (data, label) return pixel level predictions
    Args:
       model: a keras model object
       eval_dataset_with_ids: tf.dataset that yields index, (data,label)
       submodel: whether to select a index of data for multi-input models
    Returns:
       majority: a pandas dataframe with true class, predicted class and crown index
    """
    pixel_predictions = []
    box_index = []
    true_label = []
    
    for index, batch in eval_dataset_with_ids:
        data, label = batch
        pixels = model.predict(data)
        
        if submodel:
            pixels = pixels[0]
            label = label[0]
            
        pixel_predictions.append(pixels)
        box_index.append(index)
        true_label.append(label)
    
    pixel_predictions = np.concatenate(pixel_predictions)
    box_index = np.concatenate(box_index)
    true_label = np.concatenate(true_label)

    pixel_predictions = np.argmax(pixel_predictions, 1)
    true_label = np.argmax(true_label, 1)
    
    return true_label, pixel_predictions, box_index

def predict_crowns(model, eval_dataset_with_ids, submodel):
    true_pixels, predicted_pixels, box_index = predict_pixels(model, eval_dataset_with_ids, submodel)
    results = pd.DataFrame({"true": true_pixels, "predicted": predicted_pixels,"crown":box_index})
    
    #majority vole on class per crown
    majority_predicted = results.groupby("crown").apply(lambda x: x.predicted.value_counts().index[0]).reset_index(name="predicted")
    majority_true = results.groupby("crown").apply(lambda x: x.true.value_counts().index[0]).reset_index(name="true")
    majority = pd.merge(majority_predicted,majority_true)
    
    return majority
    
def site_confusion(y_true, y_pred, site_lists):
    """What proportion of misidentified species come from the same site?
    Args: 
        y_true: string values of true labels
        y_pred: string values or predicted labels
        site_lists: list of site labels for each string label taxonID -> sites
    Returns:
        Within site confusion score
    """
    within_site = 0
    cross_site = 0    
    for index, value in enumerate(y_pred):
        #If not correctly predicted
        if not value == y_true[index]:
            correct_sites = site_lists[y_true[index]]
            incorrect_site = site_lists[y_pred[index]]
        
            #Do they co-occur?
            site_overlap = any([site in incorrect_site for site in correct_sites])
            if site_overlap:
                within_site +=1
            else:
                cross_site +=1   
        else:
            pass
    
    #don't divide by zero
    if within_site + cross_site == 0:
        return 0
    
    #Get proportion of within site error
    proportion_within = within_site/(within_site + cross_site)
    
    return proportion_within

def genus_confusion(y_true, y_pred, scientific_dict):
    """What proportion of misidentified species come from the same genus?
    Args: 
        y_true: taxonID of true labels
        y_pred: taxonID of predicted labels
        scientific_dict: a dict of taxonID -> scientific name
    Returns:
        Within site confusion score
    """
    within_genus = 0
    cross_genus = 0    
    for index, value in enumerate(y_pred):
        #If not correctly predicted
        if not value == y_true[index]:
            true_genus = scientific_dict[y_true[index]][0].split()[0]
            pred_genus = scientific_dict[y_pred[index]][0].split()[0]
            
            if true_genus == pred_genus:
                within_genus +=1
            else:
                cross_genus +=1
    
    #don't divide by zero
    if within_genus + cross_genus == 0:
        return 0
    
    #Get proportion of within site error
    proportion_within = within_genus/(within_genus + cross_genus)
    
    return proportion_within

        
def f1_scores(y_true, y_pred):
    """Calculate micro, macro
    Args:
        y_true: one_hot ground truth labels
        y_pred: softmax classes 
    Returns:
        macro: macro average fscore
        micro: micro averge fscore
    """
    #F1 scores

    macro = f1_score(y_true, y_pred, average='macro')
    micro = f1_score(y_true, y_pred, average='micro')

    return macro, micro
