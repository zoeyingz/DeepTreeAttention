# Run Experiment
## Sleep for a moment to allow queries to build up in SLURM queue
import os
from random import randint
from time import sleep
from datetime import datetime

from comet_ml import Experiment
from DeepTreeAttention.trees import AttentionModel
from DeepTreeAttention.utils import metrics, resample, start_cluster
from DeepTreeAttention.models.layers import WeightedSum
from DeepTreeAttention.visualization import visualize

import tensorflow as tf
from tensorflow.keras import metrics as keras_metrics
from tensorflow.keras.models import load_model
from distributed import wait

import glob
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def find_shapefiles(dirname):
    files = glob.glob(os.path.join(dirname,"*.shp"))
    return files

def predict(dirname, savedir, generate=True, cpus=2, parallel=True, height=40, width=40, channels=3):
    """Create a wrapper dask cluster and run list of shapefiles in parallel (optional)
        Args:
            dirname: directory of DeepForest predicted shapefiles to run
            savedir: directory to write processed shapefiles
            generate: Do tfrecords need to be generated/overwritten or use existing records?
            cpus: Number of dask cpus to run
    """
    shapefiles = find_shapefiles(dirname=dirname)
    
    if parallel:
        client = start_cluster.start(cpus=cpus)
        futures = client.map(_predict_,shapefiles, create_records=generate, savedir=savedir, height=height, width=width, channels=channels)
        wait(futures)
        
        for future in futures:
            print(future.result())
    else:
        for shapefile in shapefiles:
            _predict_(shapefile, model_path, savedir=savedir, create_records=generate)
            
if __name__ == "__main__":
    sleep(randint(0,20))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = "{}/{}".format("/orange/idtrees-collab/DeepTreeAttention/snapshots/",timestamp)
    os.mkdir(save_dir)
    
    experiment = Experiment(project_name="neontrees", workspace="bw4sz")
    experiment.add_tag("Train")

    #Create output folder
    experiment.log_parameter("timestamp",timestamp)
    experiment.log_parameter("log_dir",save_dir)
    
    #Create a class and run
    model = AttentionModel(config="/home/b.weinstein/DeepTreeAttention/conf/tree_config.yml", log_dir=save_dir)
    model.create()
    
    if model.config["train"]["pretraining_dir"]:
        model.HSI_model.load_weights("{}/HSI_model.h5".format(model.config["train"]["pretraining_dir"]))
        
    #Log config
    experiment.log_parameters(model.config["train"])
    experiment.log_parameters(model.config["evaluation"])    
    experiment.log_parameters(model.config["predict"])
    experiment.log_parameters(model.config["train"]["ensemble"])
    
    ##Train
    #Train see config.yml for tfrecords path with weighted classes in cross entropy
    model.read_data(mode="HSI")
    
    #Log the size of the training data
    counter=0
    for data, label in model.train_split:
        counter += data.shape[0]
    experiment.log_parameter("Training Samples", counter)
        
    #Load from file and compile or train new models
    if model.config["train"]["checkpoint_dir"] is not None:
        dirname = model.config["train"]["checkpoint_dir"]        
        if model.config["train"]["gpus"] > 1:
            with model.strategy.scope():   
                print("Running in parallel on {} GPUs".format(model.strategy.num_replicas_in_sync))
                #model.RGB_model = load_model("{}/RGB_model.h5".format(dirname), custom_objects={"WeightedSum": WeightedSum}, compile=False)
                model.HSI_model = load_model("{}/HSI_model.h5".format(dirname), custom_objects={"WeightedSum": WeightedSum}, compile=False)  
                model.metadata_model = load_model("{}/metadata_model.h5".format(dirname), compile=False)  
        else:
            #model.RGB_model = load_model("{}/RGB_model.h5".format(dirname), custom_objects={"WeightedSum": WeightedSum})
            model.HSI_model = load_model("{}/HSI_model.h5".format(dirname), custom_objects={"WeightedSum": WeightedSum})     
            model.metadata_model = load_model("{}/metadata_model.h5".format(dirname), compile=False)  
                
    else:
        if model.config["train"]["pretrain"]:
            #metadata network
            with experiment.context_manager("metadata"):
                print("Train metadata")
                model.read_data(mode="metadata")
                print(model.metadata_model.summary())
                
                model.train(submodel="metadata", experiment=experiment)
                model.metadata_model.save("{}/metadata_model.h5".format(save_dir))
            
            with experiment.context_manager("HSI_spectral_subnetwork"):
                print("Train HSI spectral subnetwork")    
                model.train(submodel="spectral", sensor="hyperspectral", experiment=experiment)
                    
            #Train full model
            with experiment.context_manager("HSI_model"):
                experiment.log_parameter("Class Weighted", True)
                model.read_data(mode="HSI")
                model.train(sensor="hyperspectral", experiment=experiment)
                model.HSI_model.save("{}/HSI_model.h5".format(save_dir))
                
            
    ##Ensemble
    model.read_data(mode="ensemble")
    with experiment.context_manager("ensemble"):    
        print("Train Ensemble")
        model.ensemble(experiment=experiment)
    
    #Final score, be absolutely sure you get all the data, feed slowly in batches of 1
    final_score = model.ensemble_model.evaluate(model.val_split.unbatch().batch(1))    
    experiment.log_metric("Ensemble Accuracy", final_score[1])
    
    #Save model and figure
    #tf.keras.utils.plot_model(model.ensemble_model, to_file="{}/Ensemble.png".format(save_dir))
    #experiment.log_figure("{}/Ensemble.png".format(save_dir))
    model.ensemble_model.save("{}/Ensemble.h5".format(save_dir))
    
    #save predictions
    predicted_shp = model.predict(model = model.ensemble_model)
    predicted_shp.to_file("{}/prediction.shp".format(save_dir))
    experiment.log_asset("{}/prediction.shp".format(save_dir))
    experiment.log_asset("{}/prediction.dbf".format(save_dir))
    experiment.log_asset("{}/prediction.shx".format(save_dir))
    experiment.log_asset("{}/prediction.cpg".format(save_dir))
    
    #per species accurracy
    predicted_shp["match"] = predicted_shp.apply(lambda x: x.true_taxonID == x.predicted_taxonID, 1)
    per_species = predicted_shp.groupby("true_taxonID").apply(lambda x: x["match"].sum()/len(x))
    per_species.to_csv("{}/perspecies.csv".format(save_dir))
    experiment.log_asset("{}/perspecies.csv".format(save_dir))
    
    per_site = predicted_shp.groupby("siteID").apply(lambda x: x["match"].sum()/len(x))
    per_site.to_csv("{}/persite.csv".format(save_dir))
    experiment.log_asset("{}/persite.csv".format(save_dir))   
    
    #Plots - this function needs to be rewritten because the dataset is now nested: ids, (data, label). probably predict on batch.
    #ax = visualize.plot_crown_position(y_pred = predicted_shp.predicted_taxonID, y_true=predicted_shp.true_taxonID, box_index=predicted_shp.id, path = model.config["evaluation"]["ground_truth_path"])
    #experiment.log_figure(ax)    
