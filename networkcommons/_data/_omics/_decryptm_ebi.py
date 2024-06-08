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

from __future__ import annotations

import os
import zipfile
import re
import shutil
import glob


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
