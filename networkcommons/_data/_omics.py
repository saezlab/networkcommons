#!/usr/bin/env python

#
# This file is part of the `networkcommons` Python module
#
# Copyright 2024
# Heidelberg University Hospital
#
# File author(s): Saez Lab (omnipathdb@gmail.com)
#
# Distributed under the GPLv3 license
# See the file `LICENSE` or read a copy at
# https://www.gnu.org/licenses/gpl-3.0.txt
#

import requests
import owncloud as oc
import zipfile
import os
import pandas as pd
from pydeseq2.dds import DeseqDataSet
from pydeseq2.default_inference import DefaultInference
from pydeseq2.ds import DeseqStats
from ftplib import FTP
import re
import glob
import shutil
import requests
import bs4

from ._builtin import _module_data
from networkcommons import _conf
from networkcommons._session import _log


def _datasets() -> dict[str, dict]:

    return _module_data('datasets').get('omics', {}).get('datasets', {})


def datasets() -> dict[str, str]:
    """
    Built-in omics datasets.

    Returns:
        A dict with dataset labels as keys and descriptions as values.
    """

    return {k: v['description'] for k, v in _datasets().items()}


def _download(url: str, path: str) -> str:

    timeouts = tuple(_conf.get(f'http_{k}_timout') for k in ('read', 'connect'))

    _log(f'Downloading `{url}` to `{path}`.')

    with requests.get(url, timeout = timeouts, stream = True) as r:

        r.raise_for_status()

        with open(path, 'wb') as f:

            for chunk in r.iter_content(chunk_size = 8192):

                f.write(chunk)

    _log(f'Finished downloading `{url}` to `{path}`.')


def download_dataset(dataset, **kwargs):
    """
    Downloads a dataset and returns a list of pandas DataFrames.

    Args:
        dataset (str): The name of the dataset to download.
        **kwargs: Additional keyword arguments to pass to the decryptm call:
            - experiment (str): The name of the experiment.
            - data_type (str): The type of data to download (Phosphoproteome,
            Full proteome, Acetylome...). Not all data types are available for
            all experiments.

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
    elif dataset == 'decryptm':
        file_list = decryptm_handler(**kwargs)
    else:
        save_path = f'./data/{dataset}.zip'
        if not os.path.exists(save_path):
            download_url(f'https://oc.embl.de/index.php/s/6KsHfeoqJOKLF6B/download?path=%2F&files={dataset}', save_path)  # noqa: E501

            # download?path=%2Fdecryptm%2F3_EGFR_Inhibitors%2FFullproteome&files=curves.txt

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


def get_decryptm():
    ftp = FTP('ftp.pride.ebi.ac.uk')
    ftp.login()

    files = ftp.nlst('pride/data/archive/2023/03/PXD037285/')

    curve_files = [
        "https://ftp.pride.ebi.ac.uk/" + file
        for file in files
        if re.search(r'Curves\.zip', file)
    ]

    for curve_file in curve_files:
        download_url(curve_file, f'./tmp/{os.path.basename(curve_file)}')

        with zipfile.ZipFile(f'./tmp/{os.path.basename(curve_file)}', 'r') as zip_ref:  # noqa: E501
            zip_ref.extractall('./data/')

        pdfs_toremove = glob.glob('./data/*/*/pdfs', recursive=True)
        for pdf in pdfs_toremove:
            shutil.rmtree(pdf)

        os.remove(f'./tmp/{os.path.basename(curve_file)}')


def decryptm_handler(experiment, data_type='Phosphoproteome'):
    save_path = f'./data/decryptm/{experiment}/{data_type}/'

    curve_files = list_directories(f'decryptm/{experiment}/{data_type}')[1]

    curve_files = [
        os.path.basename(file) for file in curve_files if 'curves' in file
    ]

    for curve_file in curve_files:
        if not os.path.exists(save_path + curve_file):
            download_url(
                f'https://oc.embl.de/index.php/s/6KsHfeoqJOKLF6B/download?path=%2Fdecryptm%2F{experiment}%2F{data_type}&files={curve_file}',  # noqa: E501
                save_path + curve_file
            )  # noqa: E501

    file_list = {}
    for curve_file in curve_files:
        df = pd.read_csv(
            f'./data/decryptm/{experiment}/{data_type}/{curve_file}',
            sep='\t'
        )
        file_list[curve_file] = df

    return file_list


def list_directories(path):
    public_link = (
        "https://oc.embl.de/index.php/s/6KsHfeoqJOKLF6B"
        "?path=%2Fdecryptm"
    )
    password = "networkcommons_datasaezlab"
    occontents = oc.Client.from_public_link(public_link,
                                            folder_password=password)
    response = occontents.list(path)
    file_paths = [item.path.strip('/') for item in response]
    return response, file_paths


def _ls(path: str) -> list[str]:
    """
    List files in a remote directory.

    Args:
        path:
            HTTP URL of a directory with standard nginx directory listing.
    """

    resp = requests.get(path)

    if resp.status_code == 200:
        soup = bs4.BeautifulSoup(resp.content, 'html.parser')
        return [
            href for a in soup.find_all('a')
            if (href := a['href'].strip('/')) != '..'
        ]

    else:
        raise FileNotFoundError(
            f'URL {path} returned status code {resp.status_code}'
        )
