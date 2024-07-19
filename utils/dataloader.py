import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def load_train_test_imdb_data(data_dir):
    """Loads the IMDB train/test datasets from a folder path.
    Input:
    data_dir: path to the "aclImdb" folder.
    
    Returns:
    train/test datasets as pandas dataframes.
    """

    data = {}
    for split in ["train", "test"]:
        data[split] = []
        for sentiment in ["neg", "pos"]:
            score = 1 if sentiment == "pos" else 0

            path = os.path.join(data_dir, split, sentiment)
            file_names = os.listdir(path)
            for f_name in file_names:
                with open(os.path.join(path, f_name), "r",encoding="utf8") as f:
                    review = f.read()
                    data[split].append([review, score])

    np.random.shuffle(data["train"])        
    data["train"] = pd.DataFrame(data["train"],
                                 columns=['text', 'label'])

    np.random.shuffle(data["test"])
    data["test"] = pd.DataFrame(data["test"],
                                columns=['text', 'label'])
    return data["train"], data["test"]

def load_mr_data(data_dir = "/home/ubuntu/Robustness_Gym/data/rt-polaritydata/rt-polarity.",):
    """Loads the IMDB train/test datasets from a folder path.
    Input:
    data_dir: path to the "aclImdb" folder.
    
    Returns:
    train/test datasets as pandas dataframes.
    """

    data = []
    for sentiment in ["neg", "pos"]:
        score = 1 if sentiment == "pos" else 0

        path = data_dir+sentiment
        with (open(path, "r",encoding="utf8",errors='ignore')) as f:
            review = f.readlines()
            for i in review:
                data.append([i.replace("\n",""), score])

    data_table = pd.DataFrame(data,
                                 columns=['text', 'label'])
    train, test = train_test_split(data_table, test_size=0.2)

    return train, test