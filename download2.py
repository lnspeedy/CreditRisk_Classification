import urllib
from urllib.request import urlretrieve as retrieve
import os 


url = "https://pkgstore.datahub.io/machine-learning/credit-g/credit-g_csv/data/ac05ce3bfd911258bd37fde1e8a3051f/credit-g_csv.csv"
destination_dir = str(os.getcwd())+"/"
file_name = 'Data_Credit_RISK'
extension = '.csv'

def download_csv(destination_dir, file_name, extension, url):
    """Method to download a file to a specific folder"""
    urllib.request.urlretrieve(url, destination_dir + file_name + extension)
    file_path = os.path.join(destination_dir, file_name)
    return file_path

download_csv(destination_dir, file_name, extension, url)
