#Callbacks
"""Create training callbacks"""

import os
import numpy as np
import pandas as pd

from DeepTreeAttention.utils import metrics
from DeepTreeAttention.visualization import visualize
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.callbacks import Callback, TensorBoard

class F1Callback(Callback):

    def __init__(self, experiment, eval_dataset,  y_true, label_names, submodel, train_shp, n=10):
        """F1 callback
        Args:
            n: number of epochs to run. If n=4, function will run every 4 epochs
            y_true: instead of iterating through the dataset every time, just do it once and pass the true labels to the function
            eval_dataset_with_ids: a data generator to yeild, index, data, where index is the position to group trees.
        """
        self.experiment = experiment
        self.eval_dataset_with_ids = eval_dataset
        self.label_names = label_names
        self.submodel = submodel
        self.n = n
        self.train_shp = train_shp
        self.y_true = y_true
 
    def on_train_end(self, logs={}):
            
        results = metrics.predict_crowns(self.model, self.eval_dataset_with_ids)
        
        #F1
        macro, micro = metrics.f1_scores(results.true, results.predicted)
        self.experiment.log_metric("Final MicroF1", micro)
        self.experiment.log_metric("Final MacroF1", macro)
        
        #Log number of predictions to make sure its constant
        self.experiment.log_metric("Prediction samples",results.shape[0])
        results = pd.DataFrame({"true":np.argmax(results.true, 1),"predicted":np.argmax(y_pred, 1)})
        #assign labels
        if self.label_names:
            results["true_taxonID"] = results.true.apply(lambda x: self.label_names[x])
            results["predicted_taxonID"] = results.predicted.apply(lambda x: self.label_names[x])
            
            #Within site confusion
            site_lists = self.train_shp.groupby("taxonID").siteID.unique()
            site_confusion = metrics.site_confusion(y_true = results.true_taxonID, y_pred = results.predicted_taxonID, site_lists=site_lists)
            self.experiment.log_metric(name = "Within_site confusion[training]", value = site_confusion)
        
            plot_lists = self.train_shp.groupby("taxonID").plotID.unique()        
            plot_confusion = metrics.site_confusion(y_true = results.true_taxonID, y_pred = results.predicted_taxonID, site_lists=plot_lists)
            self.experiment.log_metric(name = "Within_plot confusion[training]", value = plot_confusion)        
        
            domain_lists = self.train_shp.groupby("taxonID").domainID.unique()        
            domain_confusion = metrics.site_confusion(y_true = results.true_taxonID, y_pred = results.predicted_taxonID, site_lists=domain_lists)
            self.experiment.log_metric(name = "Within_domain confusion[training]", value = domain_confusion)
            
            #Genus of all the different taxonID variants should be the same, take the first
            scientific_dict = self.train_shp.groupby('taxonID')['scientific'].apply(lambda x: x.head(1).values.tolist()).to_dict()
            genus_confusion = metrics.genus_confusion(y_true = results.true_taxonID, y_pred = results.predicted_taxonID, scientific_dict=scientific_dict)
            self.experiment.log_metric(name = "Within Genus confusion", value = genus_confusion)
            
            #Most confused
            most_confused = results.groupby(["true_taxonID","predicted_taxonID"]).size().reset_index(name="count")
            most_confused = most_confused[~(most_confused.true_taxonID == most_confused.predicted_taxonID)].sort_values("count", ascending=False)
            self.experiment.log_table("most_confused.csv",most_confused.values)
            
    def on_epoch_end(self, epoch, logs={}):
        
        if not epoch % self.n == 0:
            return None
            
        majority = metrics.predict_crowns(self.model, self.eval_dataset_with_ids, self.submodel)
    
        #F1
        macro, micro = metrics.f1_scores(majority.true, majority.predicted)
        self.experiment.log_metric("MicroF1", micro)
        self.experiment.log_metric("MacroF1", macro)
        
        #Log number of predictions to make sure its constant
        self.experiment.log_metric("Prediction samples",majority.shape[0])
                               
class ConfusionMatrixCallback(Callback):

    def __init__(self, experiment, dataset, label_names, y_true, submodel):
        self.experiment = experiment
        self.dataset = dataset
        self.label_names = label_names
        self.submodel = submodel
        self.y_true = y_true
        
    def on_train_end(self, epoch, logs={}):
        
        metrics.predict_crowns(self.model, self.eval_dataset_with_ids)
                    
        if self.submodel is "metadata":
            name = "Metadata Confusion Matrix"        
        elif self.submodel in ["ensemble"]:
            name = "Ensemble Matrix"
        else:
            name = "Confusion Matrix"

        cm = self.experiment.log_confusion_matrix(
            self.y_true,
            y_pred,
            title=name,
            file_name= name,
            labels=self.label_names,
            max_categories=90,
            max_example_per_cell=1)
        
        
class ImageCallback(Callback):

    def __init__(self, experiment, dataset, label_names, submodel=False):
        self.experiment = experiment
        self.dataset = dataset
        self.label_names = label_names
        self.submodel = submodel

    def on_train_end(self, epoch, logs={}):
        """Plot sample images with labels annotated"""

        #fill until there is atleast 20 images
        images = []
        y_pred = []
        y_true = []
        box_index = []
        
        limit = 20
        num_images = 0
        for index, batch in self.dataset:
            data, label = batch
            if num_images < limit:
                pred = self.model.predict(data)                    
                images.append(data)
                
                if self.submodel:
                    y_pred.append(pred[0])
                    y_true.append(label[0])
                    box_index.append(index[0])
                else:
                    y_pred.append(pred)
                    y_true.append(label)    
                    box_index.append(index)

                num_images += label.shape[0]
            else:
                break
        
        images = np.vstack(images)
        y_true = np.concatenate(y_true)
        y_pred = np.concatenate(y_pred)
        box_index = np.concatenate(box_index)

        y_true = np.argmax(y_true, axis=1)
        y_pred = np.argmax(y_pred, axis=1)

        true_taxonID = [self.label_names[x] for x in y_true]
        pred_taxonID = [self.label_names[x] for x in y_pred]
        
        results = pd.DataFrame({"true": true_taxonID, "predicted": pred_taxonID,"box_index":box_index})
        grouped = results.groupby("box_index")
        
        for name, group in grouped:
            label = group.true.unique()[0]
            figure = visualize.plot_prediction(group)
            self.experiment.log_figure(figure_name="{}_{}".format(label,name))


def create(experiment, train_data, validation_data, train_shp, log_dir=None, label_names=None, submodel=False):
    """Create a set of callbacks
    Args:
        experiment: a comet experiment object
        train_data: a tf data object to generate data
        validation_data: a tf data object to generate data
        train_shp: the original shapefile for the train data to check site error
        """
    
    #turn off callbacks for metadata
    callback_list = []
    reduce_lr = ReduceLROnPlateau(monitor='val_loss',
                                  factor=0.5,
                                  patience=10,
                                  min_delta=0.1,
                                  min_lr=0.00001,
                                  verbose=1)
    callback_list.append(reduce_lr)

    #Get the true labels since they are not shuffled
    y_true = [ ]
    for data, label in validation_data:
        if submodel in ["spatial","spectral"]:
            label = label[0]
        y_true.append(label)
            
    y_true = np.concatenate(y_true)
    
    if not submodel in ["spatial","spectral"]:
        confusion_matrix = ConfusionMatrixCallback(experiment=experiment, y_true=y_true, dataset=validation_data, label_names=label_names, submodel=submodel)
        callback_list.append(confusion_matrix)

    f1 = F1Callback(experiment=experiment, y_true=y_true, eval_dataset=validation_data, label_names=label_names, submodel=submodel, train_shp=train_shp)
    callback_list.append(f1)
    
    #if submodel is None:
        #plot_images = ImageCallback(experiment, validation_data, label_names, submodel=submodel)
        #callback_list.append(plot_images)
        
    if log_dir is not None:
        print("saving tensorboard logs at {}".format(log_dir))
        tensorboard = TensorBoard(log_dir=log_dir, histogram_freq=0, profile_batch=30)
        callback_list.append(tensorboard)      

    return callback_list
