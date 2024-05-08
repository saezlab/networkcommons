import requests
import owncloud as oc
import zipfile
import os
import pandas as pd
from pydeseq2.dds import DeseqDataSet
from pydeseq2.default_inference import DefaultInference
from pydeseq2.ds import DeseqStats


def get_available_datasets():
    """
    Retrieves a list of available datasets from a specified public link.

    Returns:
        list: A list of file paths for the available datasets.
    """
    public_link = "https://oc.embl.de/index.php/s/6KsHfeoqJOKLF6B"
    password = "networkcommons_datasaezlab"
    occontents = oc.Client.from_public_link(public_link,
                                            folder_password=password)
    response = occontents.list('/')
    file_paths = [item.path.strip('/') for item in response]

    return file_paths


def download_dataset(dataset):
    """
    Downloads a dataset and returns a list of pandas DataFrames.

    Args:
        dataset (str): The name of the dataset to download.

    Returns:
        list: A list of pandas DataFrames, each representing a file in
            the downloaded dataset.

    Raises:
        ValueError: If the specified dataset is not available.

    """
    available_datasets = get_available_datasets()
    if dataset not in available_datasets:
        error_message = f"Dataset {dataset} not available. Check available datasets with get_available_datasets()"  # noqa: E501
        raise ValueError(error_message)

    save_path = f'./data/{dataset}.zip'
    if not os.path.exists(save_path):
        download_url(f'https://oc.embl.de/index.php/s/6KsHfeoqJOKLF6B/download?path=%2F&files={dataset}', save_path)  # noqa: E501

    # unzip files
    with zipfile.ZipFile(save_path, 'r') as zip_ref:
        zip_ref.extractall('./data/')

    # list contents of dir, read them and append to list
    files = os.listdir(f'./data/{dataset}')
    file_list = []
    for file in files:
        df = pd.read_csv(f'./data/{dataset}/{file}', sep='\t')
        file_list.append(df)

    return file_list


def download_url(url, save_path, chunk_size=128):
    """
    Downloads a file from the given URL and saves it to the specified path.

    Args:
        url (str): The URL of the file to download.
        save_path (str): The path where the downloaded file will be saved.
        chunk_size (int, optional): The size of each chunk to download.
            Defaults to 128.
    """
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
    """
    Runs DESeq2 analysis on the given counts and metadata.

    Args:
        counts (DataFrame): The counts data with gene symbols as index.
        metadata (DataFrame): The metadata with sample IDs as index.
        test_group (str): The name of the test group.
        ref_group (str): The name of the reference group.
        covariates (list, optional): List of covariates to include in the
            analysis. Defaults to an empty list.

    Returns:
        DataFrame: The results of the DESeq2 analysis as a DataFrame.
    """
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

    results = DeseqStats(
        dds,
        contrast=["group", test_group, ref_group],
        inference=inference
    )
    results.summary()
    return results.results_df.astype('float64')
