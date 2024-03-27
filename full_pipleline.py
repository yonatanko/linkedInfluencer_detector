import ast
import pickle 
import numpy as np
import pandas as pd 
from sentence_transformers import SentenceTransformer
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')


def parse_complex_column(text):
    try:
        return ast.literal_eval(text)
    except (SyntaxError, ValueError):
        return None
    

def get_sentence_embedding(text):
    embedding = embedding_model.encode(text, convert_to_tensor=False)
    return np.array(embedding)


def string_to_embedding_feature(df, col):
    """
    Convert string column to embedding vector 
    """
    df[col] = df[col].astype(str).fillna(" ") # fill missing values 
    df[f'{col}_embedding'] = df[col].apply(get_sentence_embedding)
    pickle_in = open(f"trained_models/pca_trained_{col}.pickle", "rb")
    pca_loaded = pickle.load(pickle_in)
    df_pca = pca_loaded.transform(np.stack(df[f'{col}_embedding']))
    df_pca = pd.DataFrame(df_pca, columns=[f'{col}_pca_{i+1}' for i in range(df_pca.shape[1])]).reset_index(drop=True)
    return df_pca


def pre_process(df, used_features):
    """
    Prepare df for modeling- split into X (features) and y (labels)
    """
    df['connections'] = df['connections'].astype(int)
    numeric_features = used_features['numeric']
    X = df[numeric_features].reset_index(drop=True)
    for text_col in used_features['textual']:
        df_pca = string_to_embedding_feature(df, text_col)
        X = pd.concat([X, df_pca], axis=1)
    return X


def prepare_data_for_modeling(df):
    """
    Features preprocessing
    """
    df['education'] = df['education'].apply(parse_complex_column)
    df['experience'] = df['experience'].apply(parse_complex_column)
    df['posts'] = df['posts'].apply(parse_complex_column)
    df['education_count'] = df['education'].apply(lambda x: len(x) if isinstance(x, list) else None)
    df['experience_count'] = df['experience'].apply(lambda x: len(x) if isinstance(x, list) else None)
    df['posts_count'] = df['posts'].apply(lambda x: len(x) if isinstance(x, list) else None)
    df['followers'] = df['followers'].astype(float)
    used_features = {'numeric': ['followers', 'posts_count', 'experience_count', 'education_count'], 'textual': ['about', 'position', 'recommendations']}
    X = pre_process(df, used_features)
    return X 


def run_and_predict_influencers(df):
    """
    Predict influencer label
    """
    X = prepare_data_for_modeling(df)
    pickle_model = open("trained_models/pca_trained_instance.pkl", "rb")
    model_loaded = pickle.load(pickle_model)
    predictions = model_loaded.predict(X)
    return predictions
