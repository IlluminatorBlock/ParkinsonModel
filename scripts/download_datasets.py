#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This script downloads and sets up the required datasets for the Parkinson's MRI Detection project.
It handles downloading, extraction, and organization of the data.
"""

import os
import sys
import argparse
import urllib.request
import zipfile
import tarfile
import shutil
import pandas as pd
from tqdm import tqdm
import json

# Define data directories
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DATA_DIR = os.path.join(ROOT_DIR, 'data', 'raw')
PROCESSED_DATA_DIR = os.path.join(ROOT_DIR, 'data', 'processed')
METADATA_DIR = os.path.join(ROOT_DIR, 'data', 'metadata')

# Ensure directories exist
os.makedirs(RAW_DATA_DIR, exist_ok=True)
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
os.makedirs(METADATA_DIR, exist_ok=True)

# Dataset configurations
DATASETS = {
    'ppmi': {
        'name': 'Parkinson\'s Progression Markers Initiative (PPMI)',
        'description': 'Clinical and imaging data for Parkinson\'s research',
        'note': 'You need to register at https://www.ppmi-info.org/access-data-specimens/download-data to access the data',
        'instructions': [
            '1. Register at https://www.ppmi-info.org/',
            '2. Apply for data access',
            '3. Once approved, download the T1 MRI data and clinical information'
        ],
        'citation': 'Marek, K. et al. (2011). The Parkinson Progression Marker Initiative (PPMI). Progress in Neurobiology, 95(4), 629-635.'
    },
    'ukbiobank': {
        'name': 'UK Biobank Neuroimaging',
        'description': 'Large-scale population neuroimaging',
        'note': 'Access requires an approved research application',
        'instructions': [
            '1. Apply for access at https://www.ukbiobank.ac.uk/',
            '2. Request MRI brain imaging data and PD clinical information',
            '3. Download approved datasets'
        ],
        'citation': 'Miller, K. L. et al. (2016). Multimodal population brain imaging in the UK Biobank prospective epidemiological study. Nature Neuroscience, 19(11), 1523-1536.'
    },
    'ixi': {
        'name': 'IXI Dataset',
        'description': 'Normal and healthy brain images from multiple modalities',
        'url': 'https://brain-development.org/ixi-dataset/',
        'files': {
            'T1': 'https://brain-development.org/download/?file_id=2',
            'T2': 'https://brain-development.org/download/?file_id=3',
            'MRA': 'https://brain-development.org/download/?file_id=4',
            'DTI': 'https://brain-development.org/download/?file_id=5',
        },
        'citation': 'IXI Dataset. Information eXtraction from Images (IXI). https://brain-development.org/ixi-dataset/'
    },
    'oasis': {
        'name': 'OASIS-3',
        'description': 'Longitudinal neuroimaging, clinical, and cognitive dataset for normal aging and Alzheimer Disease',
        'url': 'https://www.oasis-brains.org/',
        'note': 'Registration required for download',
        'citation': 'LaMontagne, P.J., et al. (2019). OASIS-3: Longitudinal Neuroimaging, Clinical, and Cognitive Dataset for Normal Aging and Alzheimer Disease. medRxiv.'
    }
}

# Progress bar for downloads
class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)

def download_url(url, output_path):
    """
    Download a file from a URL with a progress bar
    """
    with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=url.split('/')[-1]) as t:
        urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)

def simulate_dataset():
    """
    Create a simulated dataset for development purposes
    """
    print("Creating simulated dataset for development...")
    
    # Create directory
    sim_dir = os.path.join(RAW_DATA_DIR, 'simulated')
    os.makedirs(sim_dir, exist_ok=True)
    
    # Create dummy metadata
    metadata = {
        'subjects': []
    }
    
    # Simulate 50 subjects (25 PD, 25 control)
    for i in range(50):
        subject_id = f'SIM{i:03d}'
        group = 'PD' if i < 25 else 'Control'
        age = 55 + (i % 15)  # Ages 55-69
        sex = 'M' if i % 2 == 0 else 'F'
        
        subject_data = {
            'subject_id': subject_id,
            'group': group,
            'age': age,
            'sex': sex,
            'has_follow_up': i < 10 or (i >= 25 and i < 35),  # Some subjects have follow-up
        }
        
        if group == 'PD':
            subject_data['disease_duration'] = 1 + (i % 5)
            subject_data['updrs_score'] = 20 + (i * 2)
        
        metadata['subjects'].append(subject_data)
    
    # Save metadata
    with open(os.path.join(METADATA_DIR, 'simulated_metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Create CSV version for easier processing
    df = pd.DataFrame(metadata['subjects'])
    df.to_csv(os.path.join(METADATA_DIR, 'simulated_metadata.csv'), index=False)
    
    print(f"Created simulated metadata for {len(metadata['subjects'])} subjects")
    print(f"Metadata saved to {os.path.join(METADATA_DIR, 'simulated_metadata.json')}")
    print(f"CSV version saved to {os.path.join(METADATA_DIR, 'simulated_metadata.csv')}")
    
    print("\nNote: This simulated dataset does not contain actual MRI images.")
    print("To generate synthetic MRI data, run the synthetic_data_generation.py script.")

def main():
    """
    Main function to download and set up datasets
    """
    parser = argparse.ArgumentParser(description='Download and set up datasets for Parkinson\'s MRI Detection')
    parser.add_argument('--dataset', choices=['all'] + list(DATASETS.keys()) + ['simulated'], 
                        default='simulated', help='Dataset to download')
    args = parser.parse_args()
    
    print("Parkinson's MRI Detection - Dataset Setup")
    print("=========================================")
    
    if args.dataset == 'simulated':
        simulate_dataset()
        return
    
    if args.dataset == 'all':
        print("Note: Most datasets require manual download after registration.")
        print("This script will provide instructions for each dataset.")
        
        for dataset_key, dataset_info in DATASETS.items():
            print(f"\n{dataset_info['name']}:")
            print(f"Description: {dataset_info['description']}")
            
            if 'note' in dataset_info:
                print(f"Note: {dataset_info['note']}")
            
            if 'instructions' in dataset_info:
                print("Instructions:")
                for instruction in dataset_info['instructions']:
                    print(f"  {instruction}")
            
            if 'url' in dataset_info:
                print(f"URL: {dataset_info['url']}")
            
            print(f"Citation: {dataset_info['citation']}")
    else:
        # Handle specific dataset
        if args.dataset not in DATASETS:
            print(f"Error: Dataset '{args.dataset}' not recognized")
            return
        
        dataset_info = DATASETS[args.dataset]
        print(f"\n{dataset_info['name']}:")
        print(f"Description: {dataset_info['description']}")
        
        if 'note' in dataset_info:
            print(f"Note: {dataset_info['note']}")
        
        if 'instructions' in dataset_info:
            print("Instructions:")
            for instruction in dataset_info['instructions']:
                print(f"  {instruction}")
        
        if 'url' in dataset_info:
            print(f"URL: {dataset_info['url']}")
        
        # If the dataset has direct download URLs
        if 'files' in dataset_info:
            print("This dataset has files available for direct download.")
            download_dir = os.path.join(RAW_DATA_DIR, args.dataset)
            os.makedirs(download_dir, exist_ok=True)
            
            for file_type, url in dataset_info['files'].items():
                print(f"Download option available for {file_type} data.")
                print(f"URL: {url}")
                print("Note: Manual download required for most neuroimaging datasets.")
        
        print(f"Citation: {dataset_info['citation']}")
    
    print("\nSetup complete!")
    print("For actual patient data, you will need to register and download from the respective sources.")
    print("The simulated dataset option can be used for initial development and testing.")

if __name__ == "__main__":
    main() 