from lib import *

batch_size = 8
num_epochs = 20

resize = 224
mean = (0.485, 0.456, 0.406)
std = (0.229, 0.244, 0.255)

train = 'train'
test = 'test'
val = 'val'

conv_arch = ((1, 64), (1, 128), (2, 256), (2, 512), (2, 512))

save_path = r'D:\Github\Classification\models\models_results\model_resnet.pth'
csv_path = r'D:\Github\Classification\models\csv_results\resnet50.csv'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")