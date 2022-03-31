# CS6910 Assignment 2
 CS6910 Fundamentals of Deep Learning.

Team members: Dip Narayan Gupta(CS21Z025),Monica (CS21Z023)

---
  1. Training CNN from Scratch
  2. Transfer Learning(pre-training)
  3. Application using YoloV3(object detection)
  
---

## Data set downloaded procedure iNaturalist

Next the google drive needs to be mounted and the iNaturalist file needs to be unzipped.

```python coding 
#Mount Google Drive
#Link to download dataset
#!wget https://storage.googleapis.com/wandb_datasets/nature_12K.zip
from google.colab import drive
drive.mount('/content/drive')  # location google drive 

#Load and unzip iNaturalist zip file onto google colob
zip_path = "drive/MyDrive/nature_12K.zip"
!cp "{zip_path}" .
!unzip -q nature_12K.zip

---
  
## Part A  Training CNN from Scratch 

There are functions defined to build a custom CNN using tensorflow and keras.To prepare the image data generators for training and testing which need to be compiled.

## data spliting 
validation_split = 0.1 # 10% splitting for validation dataset

def spliting_train_data(validation_split = 0.1) 

## Intialize the  data prepartion 
def data_preparation(data_dir , data_augmentation , batch_size)

data_dir = "inaturalist_12K"
data_agumention = True
batch_size = 250 

it return the return train_generator , val_generator, test_generator

## Defining the Convolution Netural Networking 

def CNN(filter,filter_size,dense_layer,dropout, normalisation) 

1.Filter :[16,32,64,128,256]   list of number in filter for all layers
2.Filter sizes : [ [(3, 3), (3, 3), (3, 3), (3, 3), (3, 3)] list of filter size for all layers
3.Denser Layer size : 256 size of dense layers
4.Drop out : [0.1,0.2,0.3,0.4] The values of dropout to used at the dense layers 
6.Batch Normalization : True or false Batch Normalization layer can be used several times in a CNN network 

it return the model 

## Intialize the train for sweeps configuration 

def train():
    # Default values for hyper-parameters
    config_defaults = {
        "data_augmentation": True,
        "batch_size": 250,
        "normalisation": True,
        "dropout": 0.1,
        "filter": [16, 32, 64, 128, 256],
        "dense_layer": 256,
        "learning_rate": 0.0001, 
        "filters_size": [(3, 3), (3, 3), (3, 3), (3, 3), (3, 3)],
        "epochs "  : 5
    }

## sweep configuration 

sweep_config = {
  "name": "CNN Part A",
  "metric": {
      "name":"val_accuracy",
      "goal": "maximize"
  },
  "method": "bayes",
  "parameters": {
        "data_augmentation": {
            "values": [True, False]
        },
        "batch_size": {
            "values": [128, 256]
        },
         "epochs": {
            "values": [10]
        },
        "learning_rate": {
            "values": [0.001, 0.0001]
        },
        "normalisation": {
            "values": [True , False]
        },
        "dropout": {
            "values": [0, 0.1, 0.2,0.4]
        },
        "filter": {
            "values": [[16, 32, 64, 128, 256], [32, 64, 128, 256, 512], [32, 32, 32, 32, 32],
                       [256, 128, 64, 32, 16]]
        },
        "dense_layer": {
            "values": [128, 256]
        },
        "filters_size": {
            "values": [[(3, 3), (3, 3), (3, 3), (3, 3), (3, 3)],
                       [(5, 5), (5, 5), (5, 5), (5, 5), (5, 5)],
                       [(7, 7), (5, 5), (5, 5), (3, 3), (3, 3)],
                       [(3, 3), (5, 5), (3, 3), (5, 5), (7, 7)]]
        }
    }
}
## Model Intialization 
## Creating the data generators
train_generator , val_generator , test_generator = data_preparation(data_dir , data_augmentation , batch_size)
##Defining model
model = CNN(filter,filters_size,dense_layer,dropout, normalisation)
## compiling  the model  
model.compile(optimizer=Adam(learning_rate=learning_rate), loss='categorical_crossentropy', metrics=['accuracy'])
 
## Early Stopping 
earlyStopping = EarlyStopping(monitor='val_loss', patience=5, verbose=0, mode='min')

## To save the model with best validation accuracy
checkpoint = ModelCheckpoint('bestmodel.h5', monitor='val_accuracy', mode='max', verbose=0, save_best_only=True)

##Training model and returning it
history = model.fit(train_generator,
                    steps_per_epoch = train_generator.n//train_generator.batch_size,
                    validation_data = val_generator,
                    validation_steps = val_generator.n//val_generator.batch_size,
                    epochs=epochs, verbose = 2
                    ,callbacks=[earlyStopping, checkpoint])

## running sweeps 
wandb.agent(id, train, entity="fdl-moni_dip", project="test_cnn_part_a" , count=50)

## Visualization Guided backprogation 

For the visualization of Guided Backpropgation we have made a function `guided_backprop`.
To run it for visualizing the guided backpropagation of 10 images

7.Guided Backprogation (You can simply run the cell)
```python

```
To run it for visualizing the guided backpropagation of 10 images

---
```python coding
## Part B  Transfer Learning(pre-training) 

## data spliting 
validation_split = 0.1 # 10% splitting for validation dataset

def spliting_train_data(validation_split = 0.1) 

## data prepartion code
def data_preparation(data_dir , data_augmentation , batch_size)

data_dir = "inaturalist_12K"
data_agumention = True    #Augmenting data 
batch_size = 250          #size used to train model

it returns train_generator , val_generator, test_generator

## Pre trained model function

def pretrain_model(pretrained_model_name, dropout, dense_layer, pre_layer_train=None)

pre_train_model: [ResNet50, Xception,InceptionV3, InceptionResNetV2]
Data augmentation (data_augmentation): [True ,False]
Batch size for training (batch_size): [128,256]
Number of neurons in the fully connected layer (dense layer): [256,512]
Learning Rate : [0.01,0.001]
Dropout (dropout) : [0.1,0.2,0.3]
pre_layer_train : [None,10,20]


It return the model 

## Intialize the train for sweeps configuration 

def train():
    # Default values for hyper-parameters
    config_defaults = {
        "data_augmentation": True,
        "batch_size": 250,
        "dropout": 0.1,
        "dense_layer": 256,
        "learning_rate": 0.0001,
        "epochs": 5,
        "pre_layer_train": None,
    }

## sweep configuration 

sweep_config = {
  "name": "DL_assi_2_part_b",
  "metric": {
      "name":"val_accuracy",
      "goal": "maximize"
  },
  "method": "bayes",
  "parameters": {
        "data_augmentation": {
            "values": [True]
        },
        "batch_size": {
            "values": [256]
        },
        "learning_rate": {
            "values": [0.001]
        },
        "epochs": {
            "values": [10]
        },
        "dropout": {
            "values": [0.4]
        },
        "dense_layer": {
            "values": [512]
        },
                "pre_layer_train": {
            "values": [20]
        }
           }
}



## Model Intialization 
## Creating the data generators
train_generator , val_generator , test_generator = data_preparation(data_dir , data_augmentation , batch_size)
##Defining model
model = pretrain_model(pretrained_model_name = pre_train_model, dropout = dropout, dense_layer = dense_layer, pre_layer_train=pre_layer_train)

## compiling the model  
 model.compile(optimizer=Adam(learning_rate = learning_rate), loss = 'categorical_crossentropy', metrics = ['accuracy'])

## Early Stopping callback
earlyStopping = EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='min')

## To save the model with best validation accuracy
checkpoint = ModelCheckpoint('best_model.h5', monitor='val_accuracy', mode='max', verbose=0, save_best_only=True)

##Training model and returning it
history = model.fit(train_generator,
                    steps_per_epoch = train_generator.n//train_generator.batch_size,
                    validation_data = val_generator,
                    validation_steps = val_generator.n//val_generator.batch_size,
                    epochs=epochs, verbose = 2
                    ,callbacks=[earlyStopping, checkpoint])

## running in sweeps 
wandb.agent(id, train, entity="moni6264", project="test_cnn_part_b" , count=40)

```python

# Part C - Application using YoloV3 (Object Detection )
We have done
1. Mask detection using Webcam
2. Social distancing violence
2. Multiple objects detection(eg. person, bag)




