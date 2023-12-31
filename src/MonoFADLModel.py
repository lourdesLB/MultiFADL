import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers
from src.fadl_layer import FADLSelectionLayer, binary_sigmoid_unit

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from pandas import DataFrame

# -----------------------------------------------
# BinaryModel definition


class MonoFADLModel:

    def __init__(self, n_inputs, n_class) -> None:
        # Create model

        # Input layer
        input_layer = layers.Input(
            shape=(n_inputs,),
            name='input_layer')                
        
        # Feature selection layer
        selection_layer = FADLSelectionLayer(
            num_outputs=n_class, 
            name='selection_layer')(input_layer)
        
        # Base model layers        
        dense1 = layers.Dense(
            units=32, 
            activation="relu", 
            name="dense1_layer")(selection_layer)
        
        dense2 = layers.Dense(
            units=8, 
            activation="relu", 
            name="dense2_layer")(dense1)

        # Output layer 
        n_outputs = 1 if n_class == 2 else n_class
        activation = "sigmoid" if n_class == 2 else "softmax"
        output_layer = layers.Dense(
            n_outputs, 
            activation= activation,
            name="output_layer")(dense2)

        # Model definition
        model = keras.Model(
            inputs=input_layer, 
            outputs=output_layer, 
            name="model")
        
        # Model error function, optimizer and metrics
        loss = keras.losses.BinaryCrossentropy() if n_class == 2 else keras.losses.SparseCategoricalCrossentropy()
        metrics = [keras.metrics.BinaryAccuracy()] if n_class == 2 else [keras.metrics.SparseCategoricalAccuracy()]

        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.01),
            loss=loss,
            metrics=metrics
        )

        self.model = model
        self.history = None

        self.selected_features = np.array([])

        self.predictionsproba = np.array([])
        self.results = np.array([])


    def fit(self, 
            X_train : pd.DataFrame, y_train: pd.DataFrame, 
            X_val: pd.DataFrame, y_val: pd.DataFrame,
            epochs=100, batch_size=32, verbose=1) -> None:
        
        callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

        self.model.fit(
            X_train, 
            y_train, 
            validation_data=(X_val, y_val),
            epochs=epochs, 
            batch_size=batch_size, 
            verbose=verbose,
            callbacks=[callback])
        
        self.history = self.model.history.history
        
        fadl_layer = self.model.get_layer('selection_layer')
        mask = binary_sigmoid_unit(fadl_layer.get_mask()).numpy()
        features_names = X_train.columns.values
        self.selected_features = features_names[ mask.flatten().astype(bool).tolist() ]
     

    def predict(self, X_test: pd.DataFrame) -> np.array:
        self.predictionsproba = self.model.predict(X_test)
        return self.predictionsproba
    
    def evaluate(self, X_test: pd.DataFrame, y_test: pd.DataFrame) -> np.array:
        self.predictionsproba = self.predict(X_test)
        self.results = self.model.evaluate(X_test, y_test)

        y_pred = np.argmax(self.predictionsproba, axis=1)
        f1 = f1_score(y_test, y_pred, average='weighted')
        self.results = np.append(self.results, f1)

        self.results = {
            'loss': self.results[0],
            'accuracy': self.results[1],
            'f1': self.results[2]
        }
        return self.results
    
    def get_verbose(self) -> dict:

        return {
            'model': self.model,
            'history': self.history,

            'selected_features': self.selected_features,

            'predictionsproba': self.predictionsproba,
            'results': self.results
        }

    
