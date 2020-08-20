#!/usr/bin/env python
# coding: utf-8

# In[80]:


from PIL import Image
import os
import cv2
import numpy as np
import time
import torch
import torchvision
from torch.utils.data import Dataset
from torchvision import transforms
import albumentations
from albumentations.pytorch import ToTensorV2
from matplotlib import pyplot as plt

albumentations_transform = albumentations.Compose([
        albumentations.ToGray(p=0.35),
        albumentations.HorizontalFlip(p=0.5),
        albumentations.VerticalFlip(p=0.5),
        albumentations.OneOf([
            albumentations.HueSaturationValue(p=0.7),
            albumentations.ChannelShuffle(p=0.7),
            albumentations.RGBShift(r_shift_limit =20, g_shift_limit=20,b_shift_limit=20, p=0.4),
        ], p=0.8),
        albumentations.OneOf([
            albumentations.RandomResizedCrop(224, 224, scale=(0.25, 0.9), ratio=(0.8, 1.2), interpolation=0, p=0.7),
            albumentations.RandomResizedCrop(512, 512, scale=(0.25, 0.9), ratio=(0.8, 1.2), interpolation=0, p=0.7),
            albumentations.RandomResizedCrop(1024, 1024, scale=(0.25, 0.9), ratio=(0.8, 1.2), interpolation=0, p=0.7),
        ], p=1),
        albumentations.OneOf([
            albumentations.RandomBrightness(limit=0.4, p=0.7),
            albumentations.RandomBrightnessContrast(brightness_limit=0.4, contrast_limit=0.4, p=0.7),
        ], p=0.8),
        albumentations.pytorch.transforms.ToTensor()
    ], bbox_params=albumentations.BboxParams(format='yolo', min_area=4500, min_visibility=0.3))

albumentations_include_rotate = albumentations.Compose([
        albumentations.ToGray(p=0.35),
        albumentations.HorizontalFlip(p=0.5),
        albumentations.VerticalFlip(p=0.5),
        albumentations.OneOf([
            albumentations.HueSaturationValue(p=0.7),
            albumentations.ChannelShuffle(p=0.7),
            albumentations.RGBShift(r_shift_limit =20, g_shift_limit=20,b_shift_limit=20, p=0.4),
        ], p=0.8),
        albumentations.OneOf([
            albumentations.RandomResizedCrop(224, 224, scale=(0.25, 0.25), ratio=(0.8, 1.2), interpolation=0, p=0.7),
            albumentations.RandomResizedCrop(512, 512, scale=(0.25, 0.25), ratio=(0.8, 1.2), interpolation=0, p=0.7),
            albumentations.RandomResizedCrop(1024, 1024, scale=(0.25, 0.25), ratio=(0.8, 1.2), interpolation=0, p=0.7),
            albumentations.ShiftScaleRotate(border_mode=1, interpolation=3, value=10, p=0.5),
            albumentations.Rotate(border_mode=1,p=0.55),
        ], p=1),
        albumentations.OneOf([
            albumentations.RandomBrightness(limit=0.4, p=0.7),
            albumentations.RandomBrightnessContrast(brightness_limit=0.4, contrast_limit=0.4, p=0.7),
        ], p=0.8),
        albumentations.OneOf([ 
            albumentations.Resize(224, 224, p=0.7),
            albumentations.Resize(512, 512, p=0.7),
            albumentations.Resize(1024, 1024, p=0.7),
        ], p=1),
        albumentations.pytorch.transforms.ToTensor()
    ], bbox_params=albumentations.BboxParams(format='yolo', min_visibility=0.4))

def ref_bboxes(im,bboxes,num_lines):
    area = []
    visib = []
    width, height, channel=im.shape
    resolution=width*height
    for i in range(0,num_lines):
        bbox=list(bboxes[i])
        bbox_width_ratio=bbox[2]
        bbox_height_ratio=bbox[3]
        bbox_width=bbox_width_ratio*width
        bbox_height=bbox_height_ratio*height
        area.append(int(bbox_width)*int(bbox_height))
        visib.append(area[i]/resolution)
    
    return area, visib

def cancel(im,cpath,ccnt):
    f_name= "cancel_data_" + str(ccnt) + ".jpg"
    target = "/canceled_img/"
    path = cpath + target + f_name
    print(path)
    cv2.imwrite(path,im)

def write_train_txt(cpath):
    image_list=[]
    name = "train.txt"
    search = ".jpg"
    
    target = "\\train_set"
    path = cpath + target
    save_path = cpath +"\\"+ name
    
    data_list=os.listdir(path)
    
    for i in data_list:
        if search in i:
            image_list.append(i)
    
    with open(save_path,'w') as f:
        for i in range(0, len(image_list)):
            data = "data/obj/" +str(image_list[i])+"\n"
            f.write(data)
            
def mak_save_img(im,cpath,ite):
    f_name= "train_data_" + str(ite) + ".jpg"
    path = cpath +"/train_set/" + f_name
    cv2.imwrite(path,im)

def mak_plt(im): # plt 그리기
    fig = plt.figure(figsize=(10,10))
    plt.axis('off')
    plt.xticks([]), plt.yticks([])
    plt.tight_layout()
    plt.subplots_adjust(left = 0, bottom = 0, right = 1, top= 1, hspace= 0, wspace=0)
    plt.imshow(transforms.ToPILImage()(im))
    
def save_plt(ite,cpath): # plt 저장
    path = cpath + "/train_set/"
    f_name = "train_data_" + str(ite) + ".jpg"
    save_path_img = path + f_name
    plt.savefig(save_path_img, bbox_inces='tight',pad_inches=0)
    plt.close()

def save_bboxes_format(ite, transformed_bboxes, cpath, num_lines): # bbox format을 지정한 파일에 저장
    path = cpath + "/train_set/"
    f_name1 = "train_data_" + str(ite) + ".txt"
    save_path_txt = path + f_name1
    
    for i in range(0,num_lines):
        bboxes_txt=list(transformed_bboxes[i])
        temp=bboxes_txt.pop(4)
        bboxes_txt.insert(0,int(temp))
        
        with open(save_path_txt,'a') as f1:
            for s in bboxes_txt:
                f1.write(str(s)+ " ")
            f1.write("\n")
            
def load_bboxes_format(cpath): # bbox format 불러오기
    lines=[]
    yolo_bbox=[]
    yolo_bboxes= []
    with open(cpath,'rt',encoding='UTF8') as f:
        lines=f.readlines()
    for i in range(0,len(lines)):
        for s in lines[i].split(" "):
            yolo_bbox.append(float(s))
        temp=yolo_bbox.pop(0)
        temp=int(temp)
        yolo_bbox.insert(4,str(temp))
        yolo_bboxes.append(yolo_bbox)
        
        yolo_bbox=[]

    return yolo_bboxes, str(temp), len(lines)

def load_info(cnt,cpath): # 정보 불러오기
    image_list = []
    bbox_list = []
    
    target = "\\obj"
    path = cpath + target
    
    data_list=os.listdir(path)
    search = ".jpg"
    search2 = ".JPG"
    for i in data_list:
        if search in i:
            image_list.append(i)
        elif search2 in i:
            image_list.append(i)
        else:
            bbox_list.append(i)
    image_path = path +"\\"+ str(image_list[cnt])
    bbox_path = path +"\\"+ str(bbox_list[cnt])

    bboxes, labels, num_lines = load_bboxes_format(bbox_path)

    return image_path, bboxes, labels, image_list, num_lines

class AlbumentationsDataset(Dataset): # Dataset 지정
    def __init__(self, file_paths, bboxes, labels, transform=None):
        self.file_paths = file_paths
        self.labels = labels
        self.bboxes = bboxes
        self.transform = transform
    
    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        label = self.labels
        file_path = self.file_paths
        bboxes = self.bboxes
        image = cv2.imread(file_path)
        # By default OpenCV uses BGR color space for color images,
        # so we need to convert the image to RGB color space.
        #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        start_t = time.time()
        if self.transform:
            augmented = self.transform(image=image, bboxes=bboxes) 
            image = augmented['image']
            bboxes = augmented['bboxes']
            total_time = (time.time() - start_t)
        return image, bboxes, label, total_time
    
def main():
    area=[]
    visibility=[]
    check=0
    ccnt = 0 # 제외된 이미지 개수
    cnt = 0 # 원본 이미지 개수
    ite= 10960 # 복사된 이미지 개수
    cpath=os.getcwd()
    image_path, bbox, label, images, num_lines=load_info(cnt,cpath)
    

    for i in range(0, len(images)):
    #for i in range(0,20):
        # 이미지, 경계박스 정보 불러오기
        image_path, bbox, label, ims, num_lines =load_info(cnt,cpath)

        cnt+=1 #obj 폴더 이미지 카운터     
        iterations = 25 # 한 그림에 대한 반복횟수
        for i in range(0, iterations):
            #dataset 불러오기
            albumentations_dataset=AlbumentationsDataset(
                file_paths=image_path,
                bboxes=bbox,
                labels=[label],
                transform=albumentations_include_rotate,
            )
            transformed_image, transformed_bboxes, label, transform_time = albumentations_dataset[ite]
            im=transformed_image.numpy() # tensor -> numpy
            im=(im * 255).round().astype(np.uint8) # float32 -> uint8
            im=np.transpose(im,(1,2,0)) # shape 변형
            
            # plt 여백 및 축 제거, img 저장 % rotate포함한 composition set을 가지고 변형하면 안됨
            #mak_plt(transformed_image)# RGB COLOR
            #save_plt(ite,cpath)
            
            if len(transformed_bboxes) != 0:
                area, visibility=ref_bboxes(im,transformed_bboxes,num_lines)
            print("ite: "+str(ite))
            print("area: "+str(area))
            print("visibility: "+str(visibility))
            print("num_lines: "+str(num_lines))
            print("bboxes: "+str(transformed_bboxes))
            print("Num_bboxes: "+str(len(transformed_bboxes)))
            
            # 이미지, bbox 저장
            if int(len(transformed_bboxes)) == int(num_lines):
                #for i in range(0,num_lines):
                    #if visibility[i] > 0.5:
                        #check+=1
                #if check == num_lines:
                mak_save_img(im,cpath,ite)# BGR COLOR / cv2.imwrite 저장
                save_bboxes_format(ite, transformed_bboxes,cpath,num_lines)
                ite+=1 # 복제된 모든 이미지 카운터
                #check=0
            else:
                #cancel(im,cpath,ccnt)
                i-=1
                ccnt+=1
            area=[]
            visibility=[]
            
    write_train_txt(cpath) # train.txt 작성
    
main()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




