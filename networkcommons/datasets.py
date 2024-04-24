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


def deseq2_analysis(counts, 
                    metadata,
                    covariates="",
                    deseq2_test='Wald',
                    deseq2_fitType='parametric',
                    deseq2_betaprior=False,
                    deseq2_quiet=False,
                    deseq2_minReplicatesForReplace=7,
                    ):
    """
    Perform DESeq2 analysis using rpy2.

    Parameters:
        counts (DataFrame): A pandas DataFrame containing raw count data.
        metadata (DataFrame): A pandas DataFrame containing metadata.
        additional_args (dict): Additional arguments for DESeq2 analysis.

    Returns:
        DESeq2 results as a DataFrame.
    """
    # Importing required R packages
    DESeq2 = importr("DESeq2")
    base = importr("base")

    # Set genesymbol as rownames
    counts.set_index('gene_symbol', inplace=True)
    metadata.set_index('sample_ID', inplace=True)

    # Convert pandas DataFrames to R DataFrames
    pandas2ri.activate()
    gene_counts = pandas2ri.py2rpy(counts)
    metadata_r = pandas2ri.py2rpy(metadata)

    if covariates != "" and len(covariates)>=1:
        covariates = ["" + covariates]

    # Create design formula
    design_formula = robjects.Formula("~ 0 + group" + " + ".join(covariates))


    # Create DESeqDataSet object
    formatted_data = DESeq2.DESeqDataSetFromMatrix(countData=gene_counts,
                                                   colData=metadata_r,
                                                   design=design_formula)

    # Get study groups
    studygroups = list(set(metadata['group']))


    # Run DESeq2 analysis
    results = DESeq2.DESeq(formatted_data,
                            test=deseq2_test,
                            fitType=deseq2_fitType,
                            betaPrior=deseq2_betaprior,
                            quiet=deseq2_quiet,
                            minReplicatesForReplace=deseq2_minReplicatesForReplace)
    results = DESeq2.results(results, contrast=robjects.StrVector(['group', studygroups[0], studygroups[1]]))
    results = base.as_data_frame(results)

    # Convert DESeq2 results to pandas DataFrame
    with localconverter(robjects.default_converter + pandas2ri.converter):
        results_df = robjects.conversion.rpy2py(results)
        

    return results_df




    




