import decoupler as dc
import omnipath as op
import re
import requests
import owncloud as oc
import zipfile
import os
import pandas as pd

def get_available_datasets():
    public_link="https://oc.embl.de/index.php/s/6KsHfeoqJOKLF6B"
    password="networkcommons_datasaezlab"
    occontents = oc.Client.from_public_link(public_link,folder_password=password)
    response = occontents.list('/')
    file_paths = [item.path.strip('/') for item in response]

    return file_paths
    

def download_dataset(dataset):
    available_datasets = get_available_datasets()
    if dataset not in available_datasets:
        raise ValueError(f"Dataset {dataset} not available. Check available datasets with get_available_datasets()")

    download_url(f'https://oc.embl.de/index.php/s/6KsHfeoqJOKLF6B/download?path=%2F&files={dataset}', f'./data/{dataset}.zip')
    # unzip files
    with zipfile.ZipFile(f'./data/{dataset}.zip', 'r') as zip_ref:
        zip_ref.extractall(f'./data/')
    os.remove(f'./data/{dataset}.zip')
    
    # list contents of dir, read them and append to list
    files = os.listdir(f'./data/{dataset}')
    file_list = []
    for file in files:
        df = pd.read_csv(f'./data/{dataset}/{file}', sep='\t')
        file_list.append(df)
    
    return file_list


def download_url(url, save_path, chunk_size=128):
    r = requests.get(url, stream=True)
    # mkdir if not exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'wb') as fd:
        for chunk in r.iter_content(chunk_size=chunk_size):
            fd.write(chunk)





