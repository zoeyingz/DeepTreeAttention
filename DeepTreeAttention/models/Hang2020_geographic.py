#Create spectral attention model
from .layers import *
from tensorflow.keras import metrics

def define_model(height=11, width=11, channels=48, classes=2):
    """
    Create model and return output layers to allow training at different levels
    """
    input_shape = (height, width, channels)
    sensor_inputs = layers.Input(shape=input_shape, name="data_input")
    
    #spectral network
    spectral_attention_outputs, spectral_attention_pool = spectral_network(sensor_inputs, classes=classes)
    sensor_output = spectral_attention_outputs[2]
        
    return sensor_inputs, sensor_output, spectral_attention_outputs

def create_models(height, width, channels, classes, learning_rate):
    #Define model structure
    sensor_inputs, sensor_outputs, spectral_attention_outputs = define_model(
        height = height,
        width = width,
        channels = channels,
        classes = classes)

    #Full model compile
    model = tf.keras.Model(inputs=sensor_inputs,
                                outputs=sensor_outputs,
                                name="DeepTreeAttention")

    #compile full model
    metric_list = [metrics.CategoricalAccuracy(name="acc")]    
    model.compile(loss="categorical_crossentropy",
                       optimizer=tf.keras.optimizers.Adam(
                           lr=float(learning_rate)),
                       metrics=metric_list)

    # Spectral Attention softmax model
    spectral_model = tf.keras.Model(inputs=sensor_inputs,
                                         outputs=spectral_attention_outputs,
                                         name="DeepTreeAttention")

    #compile loss dict
    loss_dict = {
        "spectral_attention_1": "categorical_crossentropy",
        "spectral_attention_2": "categorical_crossentropy",
        "spectral_attention_3": "categorical_crossentropy"
    }

    spectral_model.compile(
        loss=loss_dict,
        loss_weights=[0.01, 0.1, 1],
        optimizer=tf.keras.optimizers.Adam(
            lr=float(learning_rate)),
        metrics=metric_list)
    
    
    return model, spectral_model

def strip_sensor_softmax(model, classes, index, squeeze=False, squeeze_size=128):
    #prepare RGB model
    spectral_relu_layer = model.get_layer("spectral_pooling_filters_128").output    
    if squeeze:
        weighted_relu = layers.Dense(squeeze_size)(spectral_relu_layer)
        
    stripped_model = tf.keras.Model(inputs=model.inputs, outputs = weighted_relu)
    
    #for x in model.layers:
        #rename(model, x, x.name + str(index))        
    
    return stripped_model

def rename(model, layer, new_name):
    def _get_node_suffix(name):
        for old_name in old_nodes:
            if old_name.startswith(name):
                return old_name[len(name):]

    old_name = layer.name
    old_nodes = list(model._network_nodes)
    new_nodes = []

    for l in model.layers:
        if l.name == old_name:
            l._name = new_name
            # vars(l).__setitem__('_name', new)  # bypasses .__setattr__
            new_nodes.append(new_name + _get_node_suffix(old_name))
        else:
            new_nodes.append(l.name + _get_node_suffix(l.name))
    model._network_nodes = set(new_nodes)
    
def learned_ensemble(HSI_model, metadata_model, classes, freeze=True):
    stripped_HSI_model = strip_sensor_softmax(HSI_model, classes, index = "HSI", squeeze=True, squeeze_size=classes)      
    normalized_metadata = layers.BatchNormalization()(metadata_model.get_layer("last_relu").output)
    stripped_metadata = tf.keras.Model(inputs=metadata_model.inputs, outputs = normalized_metadata)
    
    #concat and learn ensemble weights
    merged_layers = layers.Concatenate(name="submodel_concat")([stripped_HSI_model.output, stripped_metadata.output])    
    merged_layers = layers.Dropout(0.7)(merged_layers)
    ensemble_learn = layers.Dense(classes,name="ensemble_learn")(merged_layers)
    ensemble_softmax = layers.Softmax()(ensemble_learn)

    #Take joint inputs    
    ensemble_model = tf.keras.Model(inputs=HSI_model.inputs+metadata_model.inputs,
                                    outputs=ensemble_softmax,
                           name="ensemble_model")    
    
    return ensemble_model