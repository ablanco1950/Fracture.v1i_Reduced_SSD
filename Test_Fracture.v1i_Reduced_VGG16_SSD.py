

model_path = "SSD_epoch50.pth" 

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image, ImageDraw
import torchvision.models as models
import matplotlib.pyplot as plt
import matplotlib.patches as patches

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PASCAL_CLASSES = [
    "Fracture"
]

import cv2
import re
import os
########################################################################
def loadimages(dirname):
 #########################################################################
 # adapted from:
 #  https://www.aprendemachinelearning.com/clasificacion-de-imagenes-en-python/
 # by Alfonso Blanco García
 ########################################################################  
     imgpath = dirname + "\\"
     
     images = []
     TabFileName=[]
   
    
     print("Reading images from ",imgpath)
     NumImage=-2
     
     Cont=0
     for root, dirnames, filenames in os.walk(imgpath):
        
         NumImage=NumImage+1
         
         for filename in filenames:
             
             if re.search("\.(jpg|jpeg|png|bmp|tiff)$", filename):
                 
                 
                 filepath = os.path.join(root, filename)
                
                 
                 image = cv2.imread(filepath)
                 #print(filepath)
                 #print(image.shape)                           
                 images.append(image)
                 TabFileName.append(filename)
                 
                 Cont+=1
     
     return images, TabFileName


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
                      xyxy=linea[2:]
                      
                                            
                 Labels.append(Label)
                 
                 if Label=="":
                      ContLabels+=1
                 else:
                     ContNoLabels+=1 
                 
                 TabFileLabelsName.append(filename)
                 Tabxyxy.append(xyxy)
     return Labels, TabFileLabelsName, Tabxyxy, ContLabels, ContNoLabels

def plot_image(image, box, boxesTrue, imageCV, score_boxes):

    image = np.transpose(image, (1, 2, 0))
    """Plots predicted bounding boxes on the image"""
    cmap = plt.get_cmap("tab20b")
    class_labels = PASCAL_CLASSES
    colors = [cmap(i) for i in np.linspace(0, 1, len(class_labels))]
    im = np.array(image)
    
    height, width, _ = im.shape
    #_, height, width, _ = im.shape

    # Create figure and axes
    fig, ax = plt.subplots(1)
    # Display the image
    ax.imshow(im)
    
    #print(box)
    upper_left_x = box[0] - box[2] /2
    upper_left_y = box[1] - box[3] / 2
    rect = patches.Rectangle(
            (upper_left_x * width, upper_left_y * height),
            box[2] * width,
            box[3] * height,
            linewidth=2,
            edgecolor="blue",
            facecolor="none",
        )
    
    # Add the patch to the Axes
    ax.add_patch(rect)
   
    upper_left_x_true = boxesTrue[0] - boxesTrue[2] / 2
    upper_left_y_true = boxesTrue[1] - boxesTrue[3] / 2
    rect1 = patches.Rectangle(
            (upper_left_x_true * width, upper_left_y_true * height),
            boxesTrue[2] * width,
            boxesTrue[3] * height,
            linewidth=2,
            edgecolor="green",
            facecolor="none",
        )
    # Add the patch to the Axes
    ax.add_patch(rect1)
    
    plt.show()

# Dataset

import numpy as np
import os
#import pandas as pd
import torch

from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageFile

# allows PIL to load images even if they are truncated or incomplete
ImageFile.LOAD_TRUNCATED_IMAGES = True

IMG_DIR_TEST = "testFractureOJumbo1\\images"
LABEL_DIR_TEST = "testFractureOJumbo1\\labels"

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
           
      break # only one label is considered  
        
    
    label=0.0
    image = np.array(image) / 255.0
    image = np.transpose(image, (2, 0, 1))

    #print("retornoa " + str(label))
    
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


dirnameCV="testFractureOJumbo1\\images"
dirnameLabels="testFractureOJumbo1\\labels"

imagesCV, TabFileName=loadimages(dirnameCV)

labelsTrue=loadlabels(dirnameLabels)

# Inference

dataset=YOLODataset(        
        img_dir=IMG_DIR_TEST,
        label_dir=LABEL_DIR_TEST,      
    )
dataloader = DataLoader(dataset, batch_size=1, shuffle=False)


# Load the model
model = SimplifiedSSD()

state_dict = torch.load(model_path)
model.load_state_dict(state_dict)
model = model.to(DEVICE)

model.eval()

for inputs, class_labels, bbox_labels in dataloader:
   batch_size, A, S, _ = inputs.shape
   class_preds, bbox_preds = model(inputs)
  
   for i in range(batch_size):
      #class_preds, bbox_preds = model(inputs[i])
      if len(bbox_preds) == 0:
            print("NON DETECTED FRACTURE")
            nms_boxes=[]
      else:
            print(bbox_preds)
            #https://stackoverflow.com/questions/49158935/extracting-a-sub-tensor-in-tensorflow
            #bbox_preds=bbox_preds[0, :]
            
            #bbox_preds=bbox_preds[0]
            #print("bbox_preds[0]")
            #print(bbox_preds)
            bbox_preds=bbox_preds.detach().numpy()
            #print(bbox_preds)
            score_boxes=class_preds[0]
            
      boxesTrue=bbox_labels[i]
      
      plot_image(inputs[i], bbox_preds, boxesTrue, imagesCV[i], score_boxes)
      
