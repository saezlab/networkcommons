import decoupler as dc
import omnipath as op
import re
import requests
import owncloud as oc
import zipfile
import os
import pandas as pd
import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri
from rpy2.robjects.packages import importr
from rpy2.robjects.conversion import localconverter
import decoupler as dc
from pydeseq2.dds import DeseqDataSet
from pydeseq2.default_inference import DefaultInference
from pydeseq2.ds import DeseqStats

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

    save_path = f'./data/{dataset}.zip'
    if not os.path.exists(save_path):
        download_url(f'https://oc.embl.de/index.php/s/6KsHfeoqJOKLF6B/download?path=%2F&files={dataset}', save_path)
    
    # unzip files
    with zipfile.ZipFile(save_path, 'r') as zip_ref:
        zip_ref.extractall(f'./data/')
    
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



def run_deseq2_analysis(counts, 
                      metadata, 
                      test_group, 
                      ref_group, 
                      covariates=[]):

    counts.set_index('gene_symbol', inplace=True)
    metadata.set_index('sample_ID', inplace=True)

    design_factors = ['group']

    if len(covariates) > 0:
        if isinstance(covariates, str):
            covariates = [covariates]
        design_factors += covariates
    
    inference = DefaultInference(n_cpus=8)
    dds = DeseqDataSet(
        counts=counts.T,
        metadata=metadata,
        design_factors=design_factors,
        refit_cooks=True,
        inference=inference
    )
    dds.deseq2()

    results = DeseqStats(dds, contrast=["group", test_group, ref_group], inference=inference)
    results.summary()
    return results.results_df.astype('float64')






    




