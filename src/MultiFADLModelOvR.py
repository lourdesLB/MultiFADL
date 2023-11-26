import tensorflow as tf, keras
from tensorflow.keras import layers

from functools import partial
import numpy as np
import pandas as pd

from .MonoFADLModel import MonoFADLModel


# -----------------------------------------------
# BinaryModeldefinition


class MultiFADLModelOvR:

    """
    Given a dataset with a target column with n classes, construct N one-vs-rest models using the BinaryModelclass.
    """

        
    def __init__(self) -> None:
        
        self.models = {}

        self.selected_features_per_class = {}    
        
        self.predictionsproba_per_model = {}
        self.predictions_global = np.array([]) 

        self.lossAcc_per_model = {}
        self.acc_global = 0


    def fit(self, 
            X_train: pd.DataFrame, y_train: pd.DataFrame, 
            X_val: pd.DataFrame, y_val: pd.DataFrame,
            epochs=100, batch_size=32, verbose=1) -> None:
        
        classes = y_train.unique()
        features_names = X_train.columns.values
        
        # For each class, train a model
        for clas in classes:

            print(f"--> Training model class {clas} vs rest")

            y_train_clas = y_train.apply(lambda y: 1 if y == clas else 0)
            y_val_clas = y_val.apply(lambda y: 1 if y == clas else 0)

            model = MonoFADLModel(n_inputs=features_names.shape[0], n_class=2)
    
            model.fit(X_train=X_train, y_train=y_train_clas, 
                        X_val=X_val, y_val=y_val_clas,
                        epochs=epochs, batch_size=batch_size, verbose=verbose)
            
            self.models[clas] = model
            self.selected_features_per_class[clas] = model.selected_features        


    def predict(self, X_test: pd.DataFrame) -> (dict, np.array):
        
        # Predictions per model
        for clas, model in self.models.items():
            self.predictionsproba_per_model[clas] = model.predict(X_test)

        # Global predictions
        predictionsproba_global = np.array([])
        for key in sorted(self.predictionsproba_per_model.keys()):
            # insertar columna en predictions_global
            if predictionsproba_global.shape[0] == 0:
                predictionsproba_global = self.predictionsproba_per_model[key]
            else:
                predictionsproba_global = np.concatenate((predictionsproba_global, self.predictionsproba_per_model[key]), axis=1)

        self.predictions_global = np.argmax(predictionsproba_global, axis=1)

        return self.predictionsproba_per_model, self.predictions_global
        
    
    def evaluate(self, X_test: pd.DataFrame, y_test: pd.DataFrame) -> (dict, float):

        self.predictionsproba_per_model, self.predictions_global = self.predict(X_test)

        # Results per model
        for clas, model in self.models.items():
            y_test_clas = y_test.apply(lambda y: 1 if y == clas else 0)
            self.lossAcc_per_model[clas] = model.evaluate(X_test, y_test_clas)

        # Global results
        self.acc_global = np.mean(self.predictions_global == y_test)


        return self.lossAcc_per_model, self.acc_global
    
    
    def get_verbose(self) -> dict:
        return {            
            'models': self.models,

            'selected_features_per_class': self.selected_features_per_class,

            'predictionsproba_per_model': self.predictionsproba_per_model,
            'predictions_global': self.predictions_global,
            
            'lossAcc_per_model': self.lossAcc_per_model,
            'acc_global': self.acc_global
        }

        
    

