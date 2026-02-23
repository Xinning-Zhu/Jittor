from __future__ import print_function
import os
import tarfile
import requests
from warnings import warn
from zipfile import ZipFile
from bs4 import BeautifulSoup
from os.path import abspath, isdir, join, basename

# 用于自动下载 CycleGAN 或 pix2pix 模型所需的数据集
class GetData(object):
    """从伯克利大学的官方数据集仓库下载 CycleGAN 或 pix2pix 所需的预设数据集
       提供交互式选择界面，让用户可以从可用数据集中挑选需要的数据集
       自动处理下载、解压和清理过程，将数据集整理到指定目录
    """

    def __init__(self, technique='cyclegan', verbose=True):
        url_dict = {
            'pix2pix': 'http://efrosgans.eecs.berkeley.edu/pix2pix/datasets/',
            'cyclegan': 'https://people.eecs.berkeley.edu/~taesung_park/CycleGAN/datasets'
        }
        self.url = url_dict.get(technique.lower())
        self._verbose = verbose

    def _print(self, text):
        if self._verbose:
            print(text)

    @staticmethod
    def _get_options(r):
        soup = BeautifulSoup(r.text, 'lxml')
        options = [h.text for h in soup.find_all('a', href=True)
                   if h.text.endswith(('.zip', 'tar.gz'))]
        return options

    def _present_options(self):
        r = requests.get(self.url)
        options = self._get_options(r)
        print('Options:\n')
        for i, o in enumerate(options):
            print("{0}: {1}".format(i, o))
        choice = input("\nPlease enter the number of the "
                       "dataset above you wish to download:")
        return options[int(choice)]
    # 下载
    def _download_data(self, dataset_url, save_path):
        if not isdir(save_path):
            os.makedirs(save_path)

        base = basename(dataset_url)
        temp_save_path = join(save_path, base)

        with open(temp_save_path, "wb") as f:
            r = requests.get(dataset_url)
            f.write(r.content)

        if base.endswith('.tar.gz'):
            obj = tarfile.open(temp_save_path)
        elif base.endswith('.zip'):
            obj = ZipFile(temp_save_path, 'r')
        else:
            raise ValueError("Unknown File Type: {0}.".format(base))

        self._print("Unpacking Data...")
        obj.extractall(save_path)
        obj.close()
        os.remove(temp_save_path)
        
    # 对外提供的下载入口，整合上述所有步骤
    def get(self, save_path, dataset=None):
        """

        Download a dataset.

        Parameters:
            save_path (str) -- A directory to save the data to.
            dataset (str)   -- (optional). A specific dataset to download.
                            Note: this must include the file extension.
                            If None, options will be presented for you
                            to choose from.

        Returns:
            save_path_full (str) -- the absolute path to the downloaded data.

        """
        if dataset is None:
            selected_dataset = self._present_options()
        else:
            selected_dataset = dataset

        save_path_full = join(save_path, selected_dataset.split('.')[0])

        if isdir(save_path_full):
            warn("\n'{0}' already exists. Voiding Download.".format(
                save_path_full))
        else:
            self._print('Downloading Data...')
            url = "{0}/{1}".format(self.url, selected_dataset)
            self._download_data(url, save_path=save_path)

        return abspath(save_path_full)