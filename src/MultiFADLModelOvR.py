import tensorflow as tf, keras
from tensorflow.keras import layers

from functools import partial
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score

from .MonoFADLModel import MonoFADLModel


# -----------------------------------------------
# BinaryModeldefinition


class MultiFADLModelOvR:

    """
    Given a dataset with a target column with n classes, construct N one-vs-rest models using the BinaryModelclass.
    """

        
    def __init__(self) -> None:
        
        self.models = {}
        self.histories = {}

        self.selected_features = {}    
        
        self.predictionsproba = {}

        self.results = {}


    def fit(self, 
            X_train: pd.DataFrame, y_train: pd.DataFrame, 
            X_val: pd.DataFrame, y_val: pd.DataFrame,
            epochs=100, batch_size=32, verbose=1) -> None:
        
        classes = y_train.unique()
        features_names = X_train.columns.values

        total_selected_feaures = set()
        
        # For each class, train a model
        for clas in classes:

            print(f"--> Training model class {clas} vs rest")

            y_train_clas = y_train.apply(lambda y: 1 if y == clas else 0)
            y_val_clas = y_val.apply(lambda y: 1 if y == clas else 0)

            model_class = MonoFADLModel(n_inputs=features_names.shape[0], 
                                        n_class=2)
    
            model_class.fit(X_train=X_train, y_train=y_train_clas, 
                        X_val=X_val, y_val=y_val_clas,
                        epochs=epochs, batch_size=batch_size, verbose=verbose)
            
            self.models[clas] = model_class
            self.histories[clas] = model_class.history
            self.selected_features[clas] = model_class.selected_features
            total_selected_feaures.update(model_class.selected_features)

        self.selected_features['global'] = np.array(list(total_selected_feaures))


    def predict(self, X_test: pd.DataFrame) -> (dict, np.array):
        self.predictionsproba = dict()

        # Predictions per model
        for clas, model in self.models.items():
            self.predictionsproba[clas] = model.predict(X_test)

        # Global predictions
        predictionsproba_global = None
        for clas in sorted(self.predictionsproba.keys()):
            if predictionsproba_global is None:
                predictionsproba_global = self.predictionsproba[clas]
            else:
                predictionsproba_global = np.concatenate((predictionsproba_global, self.predictionsproba[clas]), axis=1)

        self.predictionsproba['global'] = predictionsproba_global

        return self.predictionsproba

        
    
    def evaluate(self, X_test: pd.DataFrame, y_test: pd.DataFrame) -> (dict, float):

        self.predictionsproba = self.predict(X_test)

        avg_loss = 0
        # Results per model
        for clas, model_clas in self.models.items():

            print(f"--> Evaluating model class {clas} vs rest")

            y_test_clas = y_test.apply(lambda y: 1 if y == clas else 0)
            self.results[clas] = model_clas.evaluate(X_test, y_test_clas)
            print(self.results[clas])
            avg_loss += self.results[clas]['loss']
            

        # Global results
        ypred_global = np.argmax(self.predictionsproba['global'], axis=1)

        self.results['global'] = {
            'loss': avg_loss / len(self.models),
            'accuracy': accuracy_score(y_test, ypred_global),
            'f1': f1_score(y_test, ypred_global, average='weighted')
        }

        # Transform results data structure 
        self.results = {
            'loss': {clas: res['loss'] for clas, res in self.results.items()},
            'accuracy': {clas: res['accuracy'] for clas, res in self.results.items()},
            'f1': {clas: res['f1'] for clas, res in self.results.items()}
        }

        return self.results
    
    
    def get_verbose(self) -> dict:
        return {            
            'models': self.models,
            'histories': self.histories,

            'selected_features': self.selected_features,
            'predictionsproba': self.predictionsproba,
            'results': self.results
        }

        
    

