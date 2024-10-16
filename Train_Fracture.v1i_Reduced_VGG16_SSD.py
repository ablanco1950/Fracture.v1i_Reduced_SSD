
# https://medium.com/aimonks/ssd-neural-network-revolutionizing-object-detection-f655d8b4b521
# modified by Alfonso Blanco 

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image, ImageDraw
import torchvision.models as models

#from keras.applications.vgg16 import VGG16

# 147 registros / 21 = 7
# batch con 7 registros

#BATCH_SIZE=21
BATCH_SIZE=1

########################################################################
def loadlabels(dirnameLabels):
 #########################################################################
 # adapted from:
 #  https://www.aprendemachinelearning.com/clasificacion-de-imagenes-en-python/
 # by Alfonso Blanco García
 ########################################################################  
     imgpath = dirnameLabels + "\\"
     
     Labels = []
     TabFileLabelsName=[]
     Tabxyxy=[]
     ContLabels=0
     ContNoLabels=0
         
     print("Reading labels from ",imgpath)
        
     for root, dirnames, filenames in os.walk(imgpath):
         
         for filename in filenames:
                           
                 filepath = os.path.join(root, filename)
                
                 f=open(filepath,"r")

                 Label=""
                 xyxy=""
                 for linea in f:
                      
                      indexFracture=int(linea[0])
                      Label=class_list[indexFracture]
                      #xyxy=linea[2:]
                      xyxy=linea[1:]
                      #print(xyxy)
                      
                                            
                 Labels.append(Label)
                 
                 if Label=="":
                      ContLabels+=1
                 else:
                     ContNoLabels+=1 
                 
                 TabFileLabelsName.append(filename)
                 Tabxyxy.append(xyxy)
     return Labels, TabFileLabelsName, Tabxyxy, ContLabels, ContNoLabels

# Dataset

import numpy as np
import os

import torch

from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageFile

# allows PIL to load images even if they are truncated or incomplete
#ImageFile.LOAD_TRUNCATED_IMAGES = True


IMG_DIR = "trainFractureOJumbo1\\images"
LABEL_DIR = "trainFractureOJumbo1\\labels"
IMG_DIR_TEST = "validFractureOJumbo1\\images"
LABEL_DIR_TEST = "validFractureOJumbo1\\labels"

class_list=["Fracture"]
class YOLODataset(Dataset):
  def __init__(self,  img_dir, label_dir): # only 2 objects "Fracture" "no object"
    
    ClassName, self.annotations, Tabxyxy, ContLabels, ContNoLabels=loadlabels(label_dir)
       
    self.img_dir = img_dir
    self.label_dir = label_dir
    
   

  def __len__(self):
    return len(self.annotations)

  def __getitem__(self, index):
   
    NameImage=self.annotations[index]
    NameImage=NameImage[:len(NameImage)-4]
    ImageLabel=NameImage+".txt"
        
    
    label_path = os.path.join(self.label_dir, ImageLabel)

    
    bboxes = np.roll(np.loadtxt(fname=label_path, delimiter=" ", ndmin=2), 4, axis=1).tolist() # np.roll with shift 4
                                                                                               # on axis 1:
                                                                                               #[class, x, y, w, h] -->
                                                                                               #[x, y, w, h, class]

    Imagepath=NameImage+ ".jpg"
    img_path = os.path.join(self.img_dir, Imagepath)
    #image = Image.open(img_path)
    import cv2
    image=cv2.imread(img_path)
    #image=cv2.resize(image,(224,224))
     
    # labels from train dataset labeled
    # a image may have several labels
    for box in bboxes:   
   
      box=box[:5]            
      
      x, y, width, height, class_label = box
      #print(box) 
           
      break # only one label is considered  
   
    label=0.0
    image = np.array(image) / 255.0
    image = np.transpose(image, (2, 0, 1))

    return torch.FloatTensor(image), torch.tensor(label, dtype=torch.float), torch.FloatTensor([x, y, width, height])
 
# Simplified SSD Model Definition
class SimplifiedSSD(nn.Module):
    def __init__(self, num_classes=2):
        super(SimplifiedSSD, self).__init__()
        self.feature_extractor = models.vgg16(pretrained=True).features[:-1]  # Removing the last maxpool layer
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.Conv2d= nn.Conv2d(640, 160, 3, stride=2)
        self.classifier = nn.Sequential(
            #nn.Linear(640*7*7, 4096),
            #nn.Linear(100352, 4096),
            nn.Linear(25088, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 1024),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(1024, num_classes),
        )
        self.regressor = nn.Sequential(
                       
            #nn.Linear(640*7*7, 4096),
            #nn.Linear(160*7*7, 4096),
            #nn.Linear(100352, 4096),
            nn.Linear(25088, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 1024),
            nn.ReLU(True),
            nn.Dropout(),
            
            nn.Linear(1024, 4),  # 4 for bounding box [x1, y1, x2, y2]
        )
    
    def forward(self, x):
        x = self.feature_extractor(x)
        #print(x.shape)
        x = self.avgpool(x)
        #print(x.shape)
        #x = torch.flatten(x, 1)
        x = torch.flatten(x)
        #print(x.shape)
        class_preds = self.classifier(x)
        bbox_preds = self.regressor(x)
        return class_preds, bbox_preds

# Initialize Dataset, DataLoader, and Model

dataset=YOLODataset(        
        img_dir=IMG_DIR,
        label_dir=LABEL_DIR,      
    )
# Initialize Dataset, DataLoader, and Model

dataset=YOLODataset(        
        img_dir=IMG_DIR,
        label_dir=LABEL_DIR,      
    )
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
model = SimplifiedSSD()

# Training Setup
# https://medium.com/@giorgio.martinez1926/nesterov-momentum-explained-with-examples-in-tensorflow-and-pytorch-4673dbf21998
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, nesterov=True) 
classification_criterion = nn.CrossEntropyLoss()
bbox_criterion = nn.SmoothL1Loss()

# Training Loop
num_epochs = 50

for epoch in range(num_epochs):
 
    running_loss = 0.0
    
    
    for inputs, class_labels, bbox_labels in dataloader:
      # LOSS With each epoch
      SwIni=0
      optimizer.zero_grad() # https://stats.stackexchange.com/questions/153531/what-is-batch-size-in-neural-network
                            # answer 421
 
      batch_size, A, S, _ = inputs.shape   
      
      for i in range(batch_size):
          class_preds, bbox_preds = model(inputs[i])
                   
          bbox_loss = bbox_criterion(bbox_preds, bbox_labels[i])

          #loss = classification_loss + bbox_loss
          loss1 =  bbox_loss
          if SwIni==0:
           loss=loss1
           SwIni=1
          else:
           loss+= loss1
      
          running_loss += loss1.item()
      
      loss.backward()
      optimizer.step()    
    # Answer from  COPILOT to "Reducing the size of a .pth"
    if (epoch+1) % num_epochs ==0:
         torch.save(model.state_dict(), f'SSD_epoch{epoch+1}.pth', _use_new_zipfile_serialization=False)    
    print(f'Epoch {epoch+1}, Loss: {running_loss / len(dataloader)}')
    
