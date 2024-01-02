import numpy as np
import pandas as pd


# preprocessing
from sklearn.preprocessing import OneHotEncoder, TargetEncoder, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# handling detection
from sklearn.ensemble import IsolationForest

# imbalanced data 
from imblearn.over_sampling import SMOTE, SMOTENC

# pipeline
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn.preprocessing import FunctionTransformer


# Function to preprocess the data
def pre_process1(data, inplace=False):
    if not inplace :
        data = data.copy()

    # data.drop(columns=["id"], inplace=True)
    # Convert number of days into years for more clarification
    data["N_Days"] = (data["N_Days"] / 365)
    data.rename(columns={"N_Days": "N_years"}, inplace=True)

    # Some of the users took actual medicine and others didn't
    data["took_drug"] = (data["Drug"] == "D-penicillamine")
    data.drop(columns=["Drug"], inplace=True)

    # Convert age from days into years for more clarification
    data["Age"] = (data["Age"] / 365)
    
    # Fixing boolean features
    data["Sex"] = data["Sex"] == "M"
    data.rename(columns={"Sex": "is_male"}, inplace=True)
    data["Ascites"] = data["Ascites"] == "Y"
    data["Hepatomegaly"] = data["Hepatomegaly"] == "Y"
    data["Spiders"] = data["Spiders"] == "Y"

    # fix True False Features
    data = data.astype(
        {
            "is_male": float,
            "Ascites": int,
            "Spiders": int,
            "took_drug": int,
            "Hepatomegaly": int,
        }
    )
    # data = data.convert_dtypes()
    
    # fixing Status dtype using TargetEncoding
    if "Status" in data.columns :
        encoder = LabelEncoder()
        data["Status"] = encoder.fit_transform(data["Status"])

    return data  # Return the modified DataFrame

def imputation(data, inplace=False):
    if not inplace :
        data = data.copy()
    # drop features that contain little Nan values
    not_nan_rows = data[["Prothrombin", "Stage", "Platelets"]].dropna().index.to_numpy()
    data = data.iloc[not_nan_rows, :]
    # a function that returns the count of Nan in each row
    def count_nan(data, index_range: int, threshhold:int=3) :
        '''checks the last X samples for their number of Nan values where X is `index_range`'''
        indexes = []
        n = data.shape[0]
        for row in data.iloc[-index_range: ,:].iterrows() :
            row_ind, row = row # it's a tuple
        # for row_ind in range(n-index_range, n) :
            row_ind = row.name
            row_nan_values = row[row.isna()]
            if len(row_nan_values) >= threshhold :
                indexes.append(row_ind)
                
        return indexes
    indexes = count_nan(data, 200)
    data = data.drop(indexes) 
    # impute by the mean 
    data.fillna(data.select_dtypes(["int", "float"]).mean(), inplace=True)
    data.reset_index(inplace=True)

    return data

def encode_data(data, inplace=False):
    if not inplace :
        data = data.copy()
        
    # encoded_frame = pd.DataFrame(
    #     OneHotEncoder().fit_transform(data[["Edema"]]).toarray(),
    #     columns=["no_Edema_no_diuretic", "Edema_no_diuretic", "Edema_diuretic"]
    # )
    # data = pd.concat(
    #     [data, encoded_frame],
    #     axis=1,
    # )
    # data.drop(columns=["Edema"], inplace=True)



    # # fixing Status dtype using TargetEncoding
    if "Status" in data.columns :
        encoder = LabelEncoder()
        data["Status"] = encoder.fit_transform(data["Status"])


    return data  # Return the modified DataFrame

def outlier_removal(data, inplace=False) : 
    if not inplace :
        data = data.copy()
    # IsolationForest
    def single_outlier_removal(column: pd.DataFrame, n_estimators=100, contamination="auto") :
        # single feature
        isolation_forest =IsolationForest(n_estimators=n_estimators, max_samples='auto', contamination=contamination, max_features=1)
        isolation_forest.fit(column)

        tmp_df = column.copy()
        tmp_df["anomaly"] = isolation_forest.predict(column)
        tmp_df = tmp_df.query("anomaly == 1")
        tmp_df.drop(columns="anomaly", inplace=True)
        return tmp_df

    outlier_list = ["Bilirubin", "Cholesterol", "Albumin", "Copper", "Alk_Phos", "SGOT", "Tryglicerides", "Platelets", "Prothrombin"]
    chosen_indexes = set()
    for feature in data[outlier_list] :
        new = single_outlier_removal(data[[feature]])
        indexes = new.index.tolist()
        chosen_indexes = set(indexes) if not chosen_indexes else chosen_indexes
        chosen_indexes = chosen_indexes.intersection(set(indexes))

    return data.iloc[list(chosen_indexes), :]

def scale(data, inplace=False) :
    # scale the data using z-score scaler
    scaler = StandardScaler()
    scaled = scaler.fit_transform(
        data[
            [
                "Bilirubin", "Cholesterol", "Albumin", "Copper", "Alk_Phos", "SGOT", "Tryglicerides", "Platelets", "Prothrombin",
                "Age", "N_years",
            ]
        ]
    )
    data[
        [
            "Bilirubin", "Cholesterol", "Albumin", "Copper", "Alk_Phos", "SGOT", "Tryglicerides", "Platelets", "Prothrombin",
            "Age", "N_years"
        ]
    ] = scaled

    return data

def beta_preprocess(data, inplace=False) :
    if not inplace :
        data = data.copy()
        
    # Isolation Forest
    def single_outlier_removal(column: pd.DataFrame, n_estimators=50, contamination=0.03) :
        # single feature
        isolation_forest =IsolationForest(n_estimators=n_estimators, max_samples='auto', contamination=contamination, max_features=1)
        isolation_forest.fit(column)

        tmp_df = column.copy()
        tmp_df["anomaly"] = isolation_forest.predict(column)
        tmp_df = tmp_df.query("anomaly == 1")
        tmp_df.drop(columns="anomaly", inplace=True)
        return tmp_df
    outlier_list = ["Bilirubin", "Cholesterol", "Albumin", "Copper", "Alk_Phos", "SGOT", "Tryglicerides", "Platelets", "Prothrombin"]
    chosen_indexes = set()
    for feature in data[outlier_list] :
        new = single_outlier_removal(data[[feature]])
        indexes = new.index.tolist()
        chosen_indexes = set(indexes) if not chosen_indexes else chosen_indexes
        chosen_indexes = chosen_indexes.intersection(set(indexes))
    data = data.iloc[list(chosen_indexes), :]

    # # fixing Status dtype using TargetEncoding
    if "Status" in data.columns :
        encoder = LabelEncoder()
        data["Status"] = encoder.fit_transform(data["Status"])

    return data  # Return the modified DataFrame

# Column Transformer to apply the pre_process function to the entire DataFrame

basic_pipeline = Pipeline(
    [
        ("EDA_preprocessing", FunctionTransformer(func=pre_process1, validate=False)),
    ]
)

pipeline = Pipeline(
    [
        ("imputation_preprocess", FunctionTransformer(func=imputation, validate=False)),
        ("outlier_removal_preprocess", FunctionTransformer(func=outlier_removal, validate=False)),

    ]
)

train_pipeline = Pipeline(
    # last steps to do before train the model
    [
        ("encode", FunctionTransformer(func=encode_data, validate=False)),

    ]
)

test_date_pipeline = Pipeline(
    [
        ("EDA_preprocessing", FunctionTransformer(func=pre_process1, validate=False)),
    ]
)

beta_pipeline = Pipeline(
    [
        ("EDA_preprocessing", FunctionTransformer(func=pre_process1, validate=False)),
        # ("model_preprocessing", FunctionTransformer(func=pre_process2, validate=False)),
        ("beta_preprocess", FunctionTransformer(func=beta_preprocess, validate=False)),
    ]
)