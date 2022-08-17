from typing import final
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms, models, datasets
from PIL import Image
import torch
from tqdm import tqdm
import os
import csv

ImageNet_val_data_path = './data/ImageNet_data/ImageNet_val'
imagenet_classes_path = './data/imagenet_classes.txt' # 사실 이거 안쓰는거아니냐?
val_txt_path = './data/val.txt'

def find_classes(dir): # imagenet_classes.txt
    classes_dict = {}
    with open(dir, 'r') as f:
        for s in f.readlines():             
            class_id, class_name = s.split(', ')
            classes_dict[class_id] = class_name.rstrip('\n')
            
    print("here in find_classes")
    print(classes_dict)
    return classes_dict

def make_dataset(datadir): # class_dic # val.txt
    images, labels = [], []
    with open(datadir, 'r') as f:
        for s in f.readlines():
            image_path, label_num = s.split(' ')
            images.append(image_path)
            #labels.append(class_dic[label_num.rstrip('\n')]) # label_num -> label_name
            labels.append(int(label_num.rstrip('\n'))) 
    return images, labels

    # open the csv file -> sketch data 전용
    # f = open(datadir, 'r')
    # csv_reader = csv.reader(f)
    # for row in csv_reader:
    #     images.append(row[0])
    #     labels.append(row[1])
        
class ImageDataset(Dataset):
    def __init__(self, path, transform=None, train=False):
        self.path = path # path = val.txt
        # self.imgs = list(sorted(os.listdir(self.path)))
        self.transform = transform
        # self.class_dict = find_classes(dir=imagenet_classes_path)
        self.images, self.labels = make_dataset(self.path) #self.class_dict
        
        assert len(self.images) == len(self.labels)

    def __len__(self):
        return len(self.images) 
    
    def __getitem__(self, idx):
        image = Image.open('./data/ImageNet_data/ImageNet_val/' + self.images[idx]).convert('RGB')
        transform_image = self.transform(image)
        return transform_image, self.labels[idx]

class ImageDataLoader(DataLoader):
    def __init__(self):
        
        # transform using for EfficientNet_v2
        transform_val = transforms.Compose([transforms.ToTensor(), 
                        models.EfficientNet_V2_L_Weights.IMAGENET1K_V1.transforms()])
        
        val_dataset = ImageDataset(path=val_txt_path, transform=transform_val)
        self.val_loader = DataLoader(val_dataset, batch_size=10, shuffle=False, num_workers=0) #, num_workers=0
        
    def get_data_loader(self):
        return self.val_loader

def load_efficientnet_v2_model():
    # Load the pre-trained EfficientNetV2-L model
    # effNet_v2_large = models.efficientnet_v2_l(weights=models.EfficientNet_V2_L_Weights.DEFAULT)
    effNet_v2_large = models.efficientnet_v2_l()
    # load the pretrained weights
    effNet_v2_large.load_state_dict(torch.load('./model_checkpoints/efficientnet_v2_l-59c71312.pth'))
    effNet_v2_large.eval()
    return effNet_v2_large

def validate():
    if torch.cuda.is_available():
        device = torch.device("cuda:0") 
    else:
        device = torch.device("cpu")
    
    # transform using for EfficientNet_v2
    transform_val = transforms.Compose([transforms.ToTensor(), 
                    models.EfficientNet_V2_L_Weights.IMAGENET1K_V1.transforms()])
    
    val_dataset = ImageDataset(path=val_txt_path, transform=transform_val)
        
    # load the pre-trained EfficientNetV2-L model
    effNet_v2_large = load_efficientnet_v2_model().to(device)
    
    # load the dataloader
    data_loader = ImageDataLoader()
    val_loader = data_loader.get_data_loader()
    
    # validate top-1 & top-5 accuracy 
    correct_top1 = 0
    correct_top5 = 0
    total = 0
    idx = 0
    final_correct_top1 = []
    final_correct_top5 = []
    with torch.no_grad():
        for idx, data in tqdm(enumerate(val_loader)):
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            outputs = effNet_v2_large(images)
            
            # top-1 accuracy
            predicted = torch.argmax(outputs.data, 1)
            total += labels.size(0)
            correct_top1 += (predicted == labels).sum().item()
            
            # top-5 accuracy
            _, rank5 = outputs.topk(5, 1, largest=True, sorted=True) # [1]
            rank5 = rank5.t()
            correct5 = rank5.eq(labels.view(1, -1).expand_as(rank5))
            
            for k in range(6):
                correct_k = correct5[:k].reshape(-1).float().sum(0, keepdim=True)
            
            correct_top5 += correct_k.item()
            
            print("step : {} / {}".format(idx + 1, len(val_dataset)/int(labels.size(0))))
            print("top-1 percentage :  {0:0.2f}%".format(correct_top1 / total * 100))
            print("top-5 percentage :  {0:0.2f}%".format(correct_top5 / total * 100))
            
            final_correct_top1.append(correct_top1 / total * 100)
            final_correct_top5.append(correct_top5 / total * 100)
            
    print("top-1 percentage :  {0:0.2f}%".format(correct_top1 / total * 100))
    print("top-5 percentage :  {0:0.2f}%".format(correct_top5 / total * 100))
        
    print("######### final value ##########")
    print("final top-1 percentage :  {0:0.2f}%".format(sum(final_correct_top1) / len(final_correct_top1) * 100))
    print("final top-5 percentage : {0:0.2f}%".format(sum(final_correct_top5) / len(final_correct_top5) * 100))
        
if __name__ == '__main__':
    validate()