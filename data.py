from lib import *
from config import *


def make_datapath_list(root_path):
    train_path = 'train'
    test_path = 'test'
    val_path = 'val'

    train_target_path =  osp.join(root_path, train_path, "**/*.png")
    test_target_path =  osp.join(root_path, test_path, "**/*.png")
    val_target_path =  osp.join(root_path, val_path, "**/*.png")

    train_list = []
    test_list = []
    val_list = [] 

    for path in glob.glob(train_target_path):
        train_list.append(path)

    for path in glob.glob(test_target_path):
        test_list.append(path)

    for path in glob.glob(val_target_path):
        val_list.append(path)


    return train_list, test_list, val_list


def make_datapath_list_super(root_path):
    train_path_normal = osp.join(root_path, r'normal\images_split\train\*.png')
    train_path_abnormal = osp.join(root_path, r'abnormal\images_split\train\*.png')

    val_path_normal = osp.join(root_path, r'normal\images_split\val\*.png')
    val_path_abnormal = osp.join(root_path, r'abnormal\images_split\val\*.png')

    train_list = []
    val_list = []

    for path in glob.glob(train_path_normal):
        train_list.append(path)

    for path in glob.glob(train_path_abnormal):
        train_list.append(path)

    for path in glob.glob(val_path_normal):
        val_list.append(path)

    for path in glob.glob(val_path_abnormal):
        val_list.append(path)


    return train_list
    #return train_list, val_list


class ImageTranform():
    def __init__(self, resize, mean, std):
        self.data_transform = {
            'train': transforms.Compose([
                transforms.RandomResizedCrop(resize, scale=(0.5, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean , std)
            ]),

            'test' : transforms.Compose([
                transforms.RandomResizedCrop(resize, scale=(0.5, 1.0)),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ]),

            'val' : transforms.Compose([
                transforms.RandomResizedCrop(resize, scale=(0.5, 1.0)),
                transforms.CenterCrop(resize),
                transforms.ToTensor(),
                transforms.Normalize(mean , std)
            ])
        }

    def __call__(self, img, phase='train'):
        return self.data_transform[phase](img)
    
class MyDataset(data.Dataset):
    def __init__(self, file_list, transform=None, phase='train'):
        self.file_list = file_list
        self.transform = transform
        self.phase = phase

    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, idx):
        img_path = self.file_list[idx]
        img = Image.open(img_path).convert("RGB")

        img_transformed = self.transform(img, self.phase)

        label_name = os.path.basename(os.path.dirname(img_path))

        if label_name == "arnormal":
            label = 0
        elif label_name == "normal":
            label = 1
        else:
            raise ValueError(f"Undefine label from path {img_path}")
        
        return img_transformed, label

class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor(Tensor): Tensor image of size(C, H, W) to be normalize
        Returns:
            Tensor: Normalize Image
        """

        for t, m, s  in zip(tensor, self.mean, self.std):
            t.mul_(s).add(m)
            #the normalize code -> t.sub(m).div_(s)
        return tensor
    
if __name__ == "__main__":
    root_path = r"root_path"
    train_list, test_list, val_list = make_datapath_list(root_path)
    print("Train list:", len(train_list))
    print(train_list[0])

    transforms = ImageTranform(resize=resize, mean=mean, std=std)

    test_dataset = MyDataset(test_list, transforms, test)
    train_dataset = MyDataset(train_list, transforms, train)
    val_dataset = MyDataset(val_list,  transforms, val) 

    index = 100
    unorm = UnNormalize(mean=mean, std=mean)

    # Read image
    img_path = test_list[index]
    img_original = Image.open(img_path)

    # Image to tensor, label
    img_transform, label = test_dataset.__getitem__(index)

    # Image transform
    img_transformed = transforms(img_original, phase=train)
    img_transformed = img_transformed.numpy().transpose(1, 2, 0)
    img_transformed = np.clip(img_transformed, 0, 1)

    print(f"Label: {label}")
    print(img_transform.shape)

    plt.subplot(1,3,1)
    plt.imshow(img_original)

    plt.subplot(1, 3, 2)
    plt.imshow(img_transformed)

    plt.subplot(1, 3, 3)
    plt.imshow(unorm(img_transform).permute(1, 2, 0))
    plt.show()    

