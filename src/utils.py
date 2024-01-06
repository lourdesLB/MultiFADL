import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler


def train_test_split_stratify(df, target, SEED=1223):
    X = df.drop([target], axis=1)
    y = df[target]

    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, 
        test_size=0.15, 
        stratify=y,
        random_state=SEED)

    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, 
        test_size=0.17, 
        stratify=y_trainval,
        random_state=SEED)
    
    print("Class distribution in train:")
    print(y_train.value_counts())
    print("Class distribution in val:")
    print(y_val.value_counts())
    print("Class distribution in test:")
    print(y_test.value_counts())
    
    return X_train, X_val, X_test, y_train, y_val, y_test
    
    
# def categorize_variables(df):

#     categorical = []
#     numerical = []

#     for column in df.columns:
#         unique_values = df[column].unique()
#         n_unique = len(unique_values)

#         if n_unique <= 10:
#             categorical.append((column, unique_values.tolist()))
#         else:
#             numerical.append(column)

#     return {
#         'categorical': categorical,
#         'numerical': numerical
#     }


def scale_numerical_variables(X_train, X_test, X_val, numerical_variables):
    # Inicializa el escalador
    scaler = MinMaxScaler()

    # Asegúrate de escalar solo las variables numéricas
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    X_val_scaled = X_val.copy()

    X_train_scaled[numerical_variables] = scaler.fit_transform(X_train[numerical_variables])
    X_test_scaled[numerical_variables] = scaler.transform(X_test[numerical_variables])
    X_val_scaled[numerical_variables] = scaler.transform(X_val[numerical_variables])

    return X_train_scaled, X_val_scaled, X_test_scaled
