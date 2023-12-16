import onedrivedownloader
import pandas as pd
from io import StringIO
from scipy.io import arff
import yaml


DATASETS_URLS = 'data/datasets_urls.yml'


def download_dataset(yml_item):

    print(yml_item)

    dataset_name = yml_item[0]
    dataset_url = yml_item[1]['url']
    dataset_type = yml_item[1]['format']

    print(f'Downloading {dataset_name}...')
    onedrivedownloader.download(dataset_url, filename=f'data/{dataset_name}.{dataset_type}')
    

def download_datasets(datasets_yml_path):

    datasets_yml = yaml.safe_load(open(datasets_yml_path, 'r'))
    datasets_yml = datasets_yml['datasets']


    for item in datasets_yml.items():

        download_dataset(item)


if __name__ == '__main__':
    download_datasets(DATASETS_URLS)
