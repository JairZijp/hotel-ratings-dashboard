import sys
import pandas as pd
import pymongo
import json
import os

def import_content(filepath):
    mng_client = pymongo.MongoClient('localhost', 27017)
    mng_db = mng_client['assignment2']
    collection_name = 'reviews'
    db_cm = mng_db[collection_name]
    cdir = os.path.dirname(__file__)
    file_res = os.path.join(cdir, filepath)

    data = pd.read_csv(file_res)
    data_json = json.loads(data.to_json(orient='records'))
    db_cm.remove()
    db_cm.insert(data_json)

def import_balanced_content(filepath):
    mng_client = pymongo.MongoClient('localhost', 27017)
    mng_db = mng_client['assignment2']
    collection_name = 'balanced_reviews'
    db_cm = mng_db[collection_name]
    cdir = os.path.dirname(__file__)
    file_res = os.path.join(cdir, filepath)

    data = pd.read_csv(file_res)

    negative_df = data[["Negative_Review"]]
    negative_df["positive"] = 0
    negative_df.rename(columns={
        "Negative_Review": "review"
    }, inplace=True)

    positive_df = data[["Positive_Review"]]
    positive_df["positive"] = 1
    positive_df.rename(columns={
        "Positive_Review": "review"
    }, inplace=True)

    pos_neg_df = pd.concat([negative_df, positive_df], axis=0)

    data_json = json.loads(pos_neg_df.to_json(orient='records'))
    db_cm.remove()
    db_cm.insert(data_json)

if __name__ == "__main__":
    script_dir = os.path.dirname(__file__) #<-- absolute dir the script is in
    rel_path = "Hotel_Reviews.csv"
    abs_file_path = os.path.join(script_dir, rel_path)
    import_balanced_content(abs_file_path)
