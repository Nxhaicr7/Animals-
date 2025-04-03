import os.path
from torch.utils.data import Dataset
import cv2
from torchvision.transforms import Compose, Resize, ToTensor
import numpy as np

class Animals(Dataset):
    def __init__(self, root_path, is_train = True, transform = None):
        self.transform = transform
        if is_train:
            data_paths = os.path.join(root_path, "train")
        else:
            data_paths = os.path.join(root_path, "test")
        self.labels = []
        self.images = []

        self.catelogies = ['butterfly', 'cat', 'chicken', 'cow', 'dog', 'elephant', 'horse', 'sheep', 'spider', 'squirrel']
        for ind, catelogy in enumerate(self.catelogies):
            catelogy_path = os.path.join(data_paths, catelogy)
            for file in os.listdir(catelogy_path):
                file_path = os.path.join(catelogy_path, file)
                self.images.append(file_path)
                self.labels.append(ind)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, item):
        image_path = self.images[item]
        image = cv2.imread(image_path)
        if self.transform:
            image = self.transform(image)
        label = self.labels[item]
        return image, label

if __name__ == "__main__":
    transform = Compose([
        ToTensor(),
        Resize((128, 128))
    ])
    data = Animals("/home/nxhai/Downloads/VietNguyenAI/ComputerVision_CoBan/ComputerVision_CoBan/animals", True, transform)
    test_image, test_label = data[9876]
    label = data.catelogies[test_label]
    test_image = test_image.permute(1, 2, 0).numpy()
    test_image = (test_image * 255).astype(np.uint8)
    cv2.imshow("{}".format(label), test_image)
    cv2.waitKey(0)
