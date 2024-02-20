""" This python script contains multiple helper functions that are used throughout the experiment.  """

import os
import re
import time
from functools import wraps

import matplotlib.pyplot as plt

import cloudpickle
import pandas as pd
import yaml
from pycaret.classification import setup
#from sdmetrics.utils import get_columns_from_metadata, get_type_from_column_meta


def getPicklesFromDir(path: str) -> list[dict]:
    """Returns all pickles in the provided path as a list.

    In:
        'path': the relative path to the destination directory

    Out:
        List of all pickles in the provided path

    """

    pickles = []

    for dirname, _, filenames in os.walk(path):
        for filename in filenames:
            folder = os.path.join(dirname, filename)
            pickle_obj = cloudpickle.load(open(folder, "rb"))
            pickles.append(pickle_obj)

    return pickles


def getExperimentConfig() -> dict:
    """Returns the YAML experiment configuration that contains all
    global experiment settings

    """

    with open("../experiment_config.yml", "r") as file:
        config = yaml.safe_load(file)

    return config


def run_pycaret_setup(data_path: str, setup_param: dict, meta: dict=None):
    """A wrapper function for the experiment to run pycaret setup()

    sends the correct params to the pycaret setup() function and
    returns its return value.
    Thus enabling iterative runs of the settings.
    """
    cols_dtype = None
    if meta != None and 'cols_dtype' in meta:
        cols_dtype = meta['cols_dtype']
    
    data=pd.read_csv(data_path, dtype=cols_dtype) 

    pycaret_setup = setup(data=data, **setup_param)

    return pycaret_setup

'''
def get_categorical_indices(data: pd.DataFrame, metadata: dict) -> list[int]:
    """Returns a list of indices of the categorical columns in the dataset"""

    indices = []

    columns = get_columns_from_metadata(metadata)
    for col in columns:
        col_type = get_type_from_column_meta(columns[col])
        if col_type == "categorical" or col_type == "boolean":
            col_index = data.columns.get_loc(col)
            indices.append(col_index)

    return indices
'''


def timefunction(func):
    """Used to time the Population fidelity measures."""

    @wraps(func)
    def timefunction_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        score = func(*args, **kwargs)
        end_time = time.perf_counter()

        total_time = end_time - start_time
        return {"score": score, "time": total_time}

    return timefunction_wrapper


def translate_model_name(model: str) -> str:
    model_names = {
        "lr": "Logistic Regression",
        "knn": "K-Nearest Neighbor",
        "nb": "Naive Bayes",
        "svm": "SVM",
        "rbfsvm": "SVM-RBF",
        "gpc": "Gaussian Process Classifier",
        "mlp": "Multilayer Perceptron",
        "ridge": "Ridge Classifier",
        "rf": "Random Forest",
        "dt": "Decision Tree Classifier",
        "et": "Extra Trees Classifier",
        "qda": "Quadratic Discriminant Analysis",
        "ada": "Ada Boost Clasifier",
        "gbc": "Gradient Boosting Classifier",
        "lda": "Linear Dicriminant Analysis",
        "xgboost": "Extreme Gradient Boosting",
        "lightgbm": "Light Gradient Boosting Machine",
    }

    if model in model_names:
        return model_names[model]
    else:
        return None


def unravel_metric_report(report_dict: dict):
    """
    Flattens the nested dictionary generated by the sklearn classification_report.

    Args:
        report_dict (dict): The dictionary output from sklearn.classification_report with output_dict=True.

    Returns:
        dict: Flattened dictionary with key-value pairs.
    """
    flat_dict = {}
    for key, value in report_dict.items():
        if isinstance(value, dict):
            for inner_key, inner_value in value.items():
                flat_key = f"{key}_{inner_key}"
                flat_dict[flat_key] = inner_value
        else:
            flat_dict[key] = value

    return flat_dict


def extract_loss_info_from_stdout(input_str: str) -> dict:
    """ Extract the loss information from the standard output from the CTGAN method,
            where the loss from each SDG is seperated by "#START#\n" and "#END#\n"
    """
    output_dict = {}

    # Define a regular expression pattern to match each block of text between #START# and #END#
    pattern = r"#START#\n(.+?)\n(.+?)#END#"

    # Use the regular expression to find all matches in the input string
    matches = re.findall(pattern, input_str, re.DOTALL)

    # Iterate over the matches and extract the dataset id and data string for each block
    for match in matches:
        dataset_id = match[0].strip()  # The key is the first line of the block, stripped of any whitespace
        data_str = match[1]     # The data string is the rest of the block

        # Extract the values for each epoch and the corresponding loss values for Loss G and Loss D
        epoch_list = [
            int(re.findall(r"Epoch (\d+)", line)[0])
            for line in data_str.split("\n")
            if line.startswith("Epoch")
        ]

        loss_g_list = [
            float(re.findall(r"Loss G: (.+?),", line)[0])
            for line in data_str.split("\n")
            if line.startswith("Epoch")
        ]
        loss_d_list = [
            float(re.findall(r"Loss D: (.+)", line)[0])
            for line in data_str.split("\n")
            if line.startswith("Epoch")
        ]

        # Use the extracted values to create a Pandas DataFrame
        df = pd.DataFrame(
            {"Epoch": epoch_list, "Loss_G": loss_g_list, "Loss_D": loss_d_list}
        )
        # Add the DataFrame to the dictionary with the key value being the string after #START#
        output_dict[dataset_id] = df
    # Return the dictionary containing all the extracted information
    return output_dict

def create_loss_plot(dataset_id:str, loss_df:pd.DataFrame):

    # Create the plot
    fig, ax = plt.subplots()
    ax.plot(loss_df['Epoch'], loss_df['Loss_G'], label='Loss Generator')
    ax.plot(loss_df['Epoch'], loss_df['Loss_D'], label='Loss Discriminator')
    ax.legend()
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss value')
    ax.set_title(f"{dataset_id} - Gen vs Dis loss")

    return fig

def extract_filenames(directory, extension='.csv', return_relative_path=False):
    """
    Extracts a list of filenames from a directory with a specified file extension.

    Args:
        directory (str): The path to the directory to search for files.
        extension (str): The file extension to search for. Defaults to '.txt'.
        return_relative_path (bool): Whether to return the filenames with the directory path
                              relative to the current working directory. Defaults to False.
                              False: only return the filenames (no path).

    Returns:
        list: A list of filenames in the directory with the specified extension.
    """
    # Get the list of files in the directory
    files = os.listdir(directory)

    # Filter the list by file extension
    files_filtered = [file for file in files if file.endswith(extension)]

    # Return the list of filenames
    if return_relative_path:
        return [os.path.join(directory, file) for file in files_filtered]
    else:
        return files_filtered

def get_synthetic_filepaths_from_original_data_id(original_data_id):
    """
    Returns a list of filepaths for the synthetic data that was generated 
    on the specified original data id.

    Args:
        original_data_id: (string) the id for the original data.

    Returns:
        list: A list of filenames for the synthetic data that was generated on 
        the specified original data
    """

    config=getExperimentConfig()
    # Get the list of synthetic data filenames
    synthetic_data_files = extract_filenames(config['folders']['sd_dir'])

    # rule to match for the dataset id
    regex_rule = r'SD({})Q\d+_\d+\.csv'.format(original_data_id[1:])
    # Filter the list by the dataset id
    files_filtered = [filename for filename in synthetic_data_files if re.match(regex_rule, filename)]

    return files_filtered

def convert_and_clean_dict(dictionary:dict) -> dict:
    # Helper function to convert string values to their appropriate data types
    def convert(value):
        # If the value is an integer, convert it to int
        if value.isdigit():
            return int(value)
        # if the value is a negative integer
        if isinstance(value, str) and value.lstrip('-').isdigit():
            return int(value)
        # If the value is a float, try converting it to float
        try:
            return float(value)
        except ValueError:
            pass
        # Convert string 'True' and 'False' to boolean True and False
        if value.lower() == "true":
            return True
        if value.lower() == "false":
            return False
        # Convert string 'None' to None
        if value.lower() == "none":
            return None
        # Return value unchanged if it cannot be converted to any other data type
        return value

    # Create an empty dictionary to store cleaned key-value pairs
    cleaned_dict = {}

    # Iterate over each key-value pair in the input dictionary
    for key, value in dictionary.items():
        # If the value is not empty, convert it and add the key-value pair to the cleaned_dict
        if value.strip() != "" and value.strip() != '{}':
            cleaned_dict[key] = convert(value)

    # Return the cleaned dictionary with values converted to their appropriate data types
    return cleaned_dict