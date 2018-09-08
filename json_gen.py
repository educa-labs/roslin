#!/usr/bin/python3

import requests
import pandas as pd
from io import StringIO
import json


def download_csv(path):
    return requests.get(path).text

def fix_json(json_data):
    for key in json_data:
        if type(json_data[key]) == str:
            try:
                json_data[key] = json_data[key].replace("\\","")
                json_data[key] = json_data[key].strip('"')
                json_data[key] = json.loads(json_data[key])
            except json.decoder.JSONDecodeError:
                print("Couldn't parse", key,json_data[key])
                print("If its a regular string its ok")

def fix_numerical(json_data):
    numerical_data = json_data['numerical_tags']
    if numerical_data:
        for key,value in numerical_data.items():
            numerical_data[key] = int(value)


if __name__ == '__main__':

    PATH = "https://dataclips.heroku.com/ljcywcsanvtwmrpkiuliqddgdful.csv"
    OUTPUT= "data.json"
    text = download_csv(PATH)
    df = pd.read_csv(StringIO(text))
    # row_to_json(df.iloc[0])
    data_array  = json.loads(df.to_json(orient='records'))
    for instance in data_array:
        fix_json(instance)
        fix_numerical(instance)
    print("Parsed {} projects".format(len(data_array)))
    with open(OUTPUT,"w") as out:
        json.dump(data_array,out,indent=2,ensure_ascii=False)
