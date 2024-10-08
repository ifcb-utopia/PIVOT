"""
This module provides functions for user management, data retrieval, image handling, 
and connection status monitoring.

Functions:
    - create_user(user_info): Create a user from a dictionary of user_info and return the new user id.
    - get_user(email): Get the user id associated with the email if it exists, otherwise return None.
    - get_models(): Get all the unique models and their model ids.
    - get_dissimilarities(): Get all the unique dissimilarity metrics and their dissimilarity ids.
    - get_image(file_path): Get the image associated with the file_path.
    - await_connection(max_time=60, step=5): Wait for a connection status.
    - insert_label(df): Insert labels into a DataFrame.
"""
import time
from datetime import datetime
import cv2
from utils import data_utils
from utils import sql_utils
import numpy as np
import streamlit as st
import pandas as pd

def create_user(user_info):
    """
    Create a user from a dictionary of user_info and returns new user id.

    Args: 
        user_info (dict): Dict with keys associated to columns in user table.

    Returns:
        u_id (int): The u_id for the new user
    """

    u_id = data_utils.insert_data('users', user_info)
    return u_id

@st.cache_data(show_spinner="Retrieving User", ttl="1d")
def get_user(email):
    """
    Gets the u_id associated with the email if it exists. Otherwise returns None.

    Args: 
        email (str): String of a user's email.

    Returns:
        u_id (int/None): The u_id for the user with the email. Otherwise None.
    """
    user = data_utils.select('users', {'email':email}, ['u_id', 'name', 'experience', 'lab', 'email'])
    if user and len(user) > 0:
        return user[0]
    return None

@st.cache_data(show_spinner="Retrieving Models", ttl="1d")
def get_models():
    """
    Gets all the unique models and their m_id's.

    Returns:
        model_list (list<Dict>): A list of dictionaries with keys model_name and m_id.
    """
    return data_utils.select_distinct('models', ['model_name','m_id'])

@st.cache_data(show_spinner="Retrieving Dissimilarities", ttl="1d")
def get_dissimilarities():
    """
    Gets all the unique dissimilarity metrics and their d_id's.

    Returns:
        dissimilarity_list (list<Dict>): A list of dictionaries with keys name and d_id.
    """
    return data_utils.select_distinct('dissimilarity', ['name','d_id'])

@st.cache_data(show_spinner="Retrieving Image")
def get_image(file_path):
    """
    Gets the image associated with the file_path. Otherwise returns None.

    Args: 
        file_path (str): String of the blob filepath.

    Returns:
        image (np.array): The image array in grayscale
    """
    image_contents = data_utils.get_blob_bytes(file_path)
    if image_contents.startswith(b'\x89PNG\r\n\x1a\n'):
        image = np.frombuffer(image_contents, dtype=np.uint8)
        image = cv2.imdecode(image, cv2.IMREAD_UNCHANGED) # pylint: disable=no-member
        image = data_utils.preprocess_input(image)
    else:
        print("The blob does not appear to be a valid PNG image.")
        image = np.zeros((128,128))
        image = data_utils.preprocess_input(image)
    return image.reshape(image.shape[:2])

def await_connection(max_time=60, step=5):
    """
    Wait for a connection status.

    Parameters:
        max_time (int): Maximum time (in seconds) to wait for the connection status.
                        Default is 60 seconds.
        step (int): Time interval (in seconds) to check the connection status.
                    Default is 5 seconds.

    Returns:
        bool: True if connection status is obtained within the specified time, False otherwise.
    """
    if max_time <= 0:
        raise ValueError(f"max_time={max_time}, must be > 0.")
    if step <= 0:
        raise ValueError(f"step={step}, must be > 0.")
    if max_time < step:
        raise ValueError(f"max_time={max_time} and step={step}, step must be less than max_time.")
    if not isinstance(step, int):
        raise TypeError(f"step must be an int but got type {type(step)}")
    if not isinstance(max_time, int):
        raise TypeError(f"max_time must be an int but got type {type(max_time)}")

    for _ in range(int(max_time//step)):
        if data_utils.get_status():
            return True
        time.sleep(step)
    return False

def insert_label(df):
    """
    Insert labels into a DataFrame.

    Parameters:
        df (DataFrame): DataFrame containing labels to be inserted.
                        It should have columns representing the data to be inserted.

    Returns:
        None
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"df must be of type pd.DataFrame, got {type(df)}")
    df['date'] = datetime.now()
    labels = df.to_dict(orient='records')
    data_utils.insert_data('labels', labels)

    # Update metrics table
    label_weights = df['weight'].unique()
    for label_weight in label_weights:
        subset = df.query(f'weight == {label_weight}')
        i_id_list = list(subset['i_id'].values)
        sql_utils.update_scores(i_ids=i_id_list, label_weight=label_weight, mode='insert')
