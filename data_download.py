from lib import *

data_dir = "./data"

if not os.path.exists(data_dir):
    os.mkdir(data_dir)

url = "path_url"

target_path = os.path.join(data_dir, "path.zip")

if not os.path.exists(target_path):
    urllib.request.urlretrieve(url, target_path)

    #read file tar
    zip = zipfile.ZipFile(target_path)
    zip.extractall(target_path)
    zip.close

    os.remove(target_path)