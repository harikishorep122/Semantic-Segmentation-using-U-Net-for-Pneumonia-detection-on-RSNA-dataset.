# Semantic Segmentation using U-Net for Pneumonia detection on RSNA dataset.


## INTRODUCTION

Medical imaging and applications of Artificial intelligence for diagnosis has been one of the hot topics since the onset of deep learning. Quite a lot of datasets have come up in this field. 

This project is an attempt at using UNET architecture for diagnosing pneumonia in chest X-ray images. The [RSNA dataset](https://www.kaggle.com/c/rsna-pneumonia-detection-challenge/) from Kaggle has been used for training the model. This project taught me how to structure a deep learning project, follow the coding standards and more importantly gave insights into image segmentation.
		
Considering the fact that even though the dataset was a region localization dataset the results were promising. The dataset consisted of bounding boxes of pneumonia regions, so on generating masks from the coordinates of the bounding boxes there will be class overlap, even then the model did a decent job in predicting the class maps. On drawing bounding boxes for the predicted pneumonia region and then calculating the metrics, the model achieves good results. 


----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

## ABOUT THE DATASET


### 2.1 Dataset

The dataset was directly imported from Kaggle pneumonia detection challenge page onto a Google colab instance. It consisted of close to 30,000 images in .dcm format.  Train and validation sets were separately available with ~27,000 and ~4000 train and validation images. Images were of 1024x1024 resolution. A .csv file was also provided as train lables. The csv files consisted of bounding box coordinates of the pneumonia region in each image.

### 2.2 Data Generator class for efficient data flow.

The primary step of the data generator class was to convert the files from .dcm format to .jpg followed by resizing the image to 256 x 256 pixels. Along with the image a mask image was also prepared using the bounding box coordinates from the .csv file.
The mask image was of binary type with 1 as value inside the bounding box and 0 outside it. The mask rectangle was first drawn on 1024x1024x1 numpy array and then resized to 256x256. Finally using the to_categorical() it was converted to one hot encodings.  Binary value 1 represents pneumonia affected regions and 0 represents normal regions.

The data generator class passes a number of images at a time corresponding to the batch size specified. The returned arrays are of size 

```python
images : (self.batch_size, self.img_height, self.img_width, 1)
```
```python
masks : (self.batch_size, self.img_height, self.img_width, 2)
```
----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

##  METHOD

### 3.1 Model build

<img src="https://github.com/harikishorep122/UNET/blob/main/Final_results/unet.png" height="300" align = "right">

The model used is similar to a standard UNET architecture. The model involves an encoder path and a decoder path. The encoder consists of a stacks of convolutions and max pooling layers. This helps to capture the context of the image. 
The decoder path is symmetric to the encoder it consists of stacks of upsampling and convolution layers.
<br />

### Activation functions.

Relu activation was used in the convolution layers. In the output layer  softmax function was used, this gives us the class label of the predicted class. 

### Loss function.
Diceloss and MeanIoU were the loss functions used.

Diceloss is calculated by : 

<img src="https://github.com/harikishorep122/UNET/blob/main/Final_results/diceloss.png" align = "center">
 
MeanIoU is calculated by:
		
<img src="https://github.com/harikishorep122/UNET/blob/main/Final_results/Iou.jpg" align = "center" width="200">


Where TP is the true positives (ie correct class predictions) FP is the false positives (class 0 getting predicted as 1), and FN is the false negatives  (class 1 getting predicted as 0).


**Function for implementing dice loss :**

<img src="https://github.com/harikishorep122/UNET/blob/main/Final_results/Loss function.jpg" align = "center">


## Training the model.

The model was trained on google colab instance for 50 epochs, the callbacks used were checkpoint (to save the model at every epoch), earlystopping with min_delta = 0.0001 on validation accuracy.


----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

## RESULTS

The model achieved a dice coefficient of 0.65 and Mean IoU 0.5 on the validation set.
The above measures are about how good the model predicts the segmentation map. But however our intended purpose is to finally predict bounding boxes for pneumonia region. Since our dataset was one which had bounding boxes rather than a classification mask.
On drawing bounding boxes for the predicted regions with pneumonia class and comparing with the ground truth we get the below results.

<img src="https://github.com/harikishorep122/UNET/blob/main/Final_results/Validation score.jpg" width="900">

Dice coefficient of **0.97** and Mean IoU of **0.95**


### Learning curve :

<img src="https://github.com/harikishorep122/UNET/blob/main/Final_results/learning curve.jpg" width="400" height="400"> |<img src="https://github.com/harikishorep122/UNET/blob/main/Final_results/Dice curve.jpg" width="400" height="400">
------------ | -------------

-------------------------------
### Some predicted x-rays.

<img src="https://github.com/harikishorep122/UNET/blob/main/Final_results/R1.jpg" width="900">

<img src="https://github.com/harikishorep122/UNET/blob/main/Final_results/R2.jpg" width="900">

---------------------

### Results with bounding boxes:

<img src="https://github.com/harikishorep122/UNET/blob/main/Final_results/R3.jpg" width="800">

<img src="https://github.com/harikishorep122/UNET/blob/main/Final_results/R4.jpg" width="800">

----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


## CONCLUSION

This project was a great learning experience for me. I learned how to preprocess a dataset and also how to code a data generator function to feed data while training the model. 

By doing this project I could appreciate the working of U-Net architecture for image segmentation. I understood the encoder decoder batches in the network and how it works to first capture the context of the image and then predict the classes for each classes.

Finally it was a great experience to code the whole workflow. I learned the standard coding methods for data preprocessing, building the network architecture, training the model with proper checkpoints such as callbacks and finally testing the model.

----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

## BIBLIOGRAPHY


1. U-Net: Convolutional Networks for Biomedical Image Segmentation Olaf Ronneberger, Philipp Fischer, and Thomas Brox
2. https://towardsdatascience.com/understanding-semantic-segmentation-with-unet-6be4f42d4b47
3. https://www.kaggle.com/c/rsna-pneumonia-detection-challenge/
