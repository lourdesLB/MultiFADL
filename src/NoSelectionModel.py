import tensorflow as tf, keras
from tensorflow.keras import layers

from functools import partial
import numpy as np
import pandas as pd

# -----------------------------------------------
# BinaryModel definition


class NoSelectionModel:

    def __init__(self, n_inputs, n_class) -> None:

        # Create model

        # Input layer
        input_layer = layers.Input(
            shape=(n_inputs,),
            name='input_layer')                
        
        # Base model layers
        dense1 = layers.Dense(
            units=32, 
            activation="relu", 
            name="dense1_layer")(input_layer)
        
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
        
        features_names = X_train.columns.values
        self.selected_features = features_names
     

    def predict(self, X_test: pd.DataFrame) -> np.array:
        self.predictionsproba = self.model.predict(X_test)
        return self.predictionsproba
    
    def evaluate(self, X_test: pd.DataFrame, y_test: pd.DataFrame) -> np.array:
        self.predictionsproba = self.predict(X_test)
        self.results = self.model.evaluate(X_test, y_test)
        return self.results
    
    def get_verbose(self) -> dict:

        return {
            'model': self.model,

            'selected_features': self.selected_features,

            'predictionsproba': self.predictionsproba,
            'results': self.results
        }
    
