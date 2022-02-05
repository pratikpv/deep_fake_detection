# Quantifying DeepFake Detection Accuracy for a Variety of Natural Settings.

This is the official code repository of the thesis https://scholarworks.sjsu.edu/etd_projects/962/. Visit the link for the full paper.


# Abstract

Deep fakes are videos generated from a starting video of a person where that person's face has been swapped for someone else's. In this report, we describe our work to develop general, deep learning-based models to classify Deep Fake content. Our first experiments involved simple Convolution Neural Network (CNN)-based models where we varied how individual frames from the source video were passed to the CNN. These simple models tended to give low accuracy scores for discriminating fake versus non-fake videos of less than 60%. We then developed three more sophisticated models: one based on choosing test frames, one based on video Optical Flow, and one that uses Generative Adversarial Networks (GANs) to determine structural differences in images. This last technique we call MRI-GAN and is new to the literature.  We tested our models using the Deep Fake Detection Challenge dataset and found our plain frames-based model achieves 90% test accuracy, our MRI model achieves 79% test accuracy, and Optical Flow-based model achieves 69% test accuracy.

# Data augmentation

Various types of noise are added to each frame of the videos. Types of noise implemented are Gaussian, Speckle, Salt-and-Pepper, Pepper, Salt, Poisson, and Localvar. We also applied other augmentation methods such as blur, rotation, horizontal flip, rescale, brightness, and contrast. We also implemented static, rolling, and spontaneous methods of data distractions. Each method includes text and geometric shapes as an object to generate distractions. When a text is used as a distraction, a random alpha-numeric text of length eight is generated and applied to the video.

<img src="https://github.com/pratikpv/deep_fake_detection/blob/master/images/Augmentation_sample.gif" width=110% height=150%>


# Model arch# 1. Plain-frames

## Pipeline for Plain-frames-based model 

Figure below shows the overall core design. We begin with a video sample in Step 1. Then we extract frames from the video in Step 2. We used pre-trained MTCNN in Step 3 to locate the faces of the humans in each frame. There could be more than one person in each frame. Detected faces are cropped from the frames and saved on disk. We repeat this process for each video of the dataset. At the end of Step 3, we have a set of frames for all videos. These frames contain only the faces of humans. The faces could be either fake or real. Then in Step 4, we use our model to predict if each of these frames is of a fake or real person. In the model, each frame is passed to the pre-trained Efficient-Net B0 model to extract the features. These features are passed to a custom classifier to predict if the given frame is fake or real.  In Step 5 we need to aggregate the results of each frame and at last in Step 6 we need to predict if the given video is fake or real.

<img src="https://github.com/pratikpv/deep_fake_detection/blob/master/images/Picture1.png" width=110% height=150%>


# Model arch# 2. Optical-flow

## Optical-flow generation process

We use two consecutive frames of a video to predict the optical flow of the subjects. Pre-trained Flow-net is used to predict the flow vectors. Flowiz library is used to color-code the vectors and generate an image for the optical-flow vectors.

<img src="https://github.com/pratikpv/deep_fake_detection/blob/master/images/Picture2.png" width=60% height=40%>

## Optical-flow to detect DeepFake videos

Fake videos have disturbed optical-flow vectors.

<img src="https://github.com/pratikpv/deep_fake_detection/blob/master/images/Picture3.png" width=30% height=20%>


## Pipeline for Optical-flow-based model
With the Optical Flow variant, we use the same pipeline as described in Model arch#1, but we pass images of Optical Flow at Step 3 instead of ‘plain frames’. Figure below shows the end-to-end pipeline for the Optical Flow-based model. In addition to architecture defined earlier, we have Step 3a and Step 3b depicted below. At Step 3a we use two consecutive frames and generate an Optical Flow image. At Step 3b we pass these images for dynamic augmentation and then to our model for classification.

<img src="https://github.com/pratikpv/deep_fake_detection/blob/master/images/Picture5.png" width=110% height=150%>

# Model arch# 3. MRI-GAN (a novel approach that uses perceptual similarity)

## MRI-GAN at high-level

MRI-GAN generates MRI of the input image. The MRI of DeepFake image contains artifacts that highlight regions of synthesized pixels. The MRI of non-DeepFake image is just black image. More details can be found in the paper and my another GitHub repo: https://github.com/pratikpv/mri_gan_deepfake

<img src="https://github.com/pratikpv/deep_fake_detection/blob/master/images/Picture4.png" width=50% height=50%>


## MRI-GAN sample output on validation set

<img src="https://github.com/pratikpv/deep_fake_detection/blob/master/images/Picture8.png" width=110% height=150%>

## Pipeline for MRI-GAN-based model

Using MRI-GAN, we have generated MRIs of all samples in the dataset. Just like ‘plain frames’ and Optical Flow methods, we have also passed MRIs as inputs to the model to classify if the video is fake or real. Figure below shows the pipeline for the model utilizing MRIs. In addition to architecture defined in earlier models, we have Step 3a and Step 3b. At Step 3a we pass the frame to MRI-GAN and generate an MRI of the face.  At Step 3b we pass these MRIs for dynamic augmentation and then to our model for classification.

<img src="https://github.com/pratikpv/deep_fake_detection/blob/master/images/Picture6.png" width=110% height=150%>


# High-level arch of the classifier used in all three models (Plain-frames, Optical-flow, MRI-gan-based)

Figure below shows the model architecture used for classification. In Step 1, we provide an input image to the model. This image can be a plain frame, Optical Flow image, or MRI of the face. The input image is expected to be 224 x 224 to match the native size of the Efficient-Net B0 model. The pre-trained Efficient-Net B0 is deployed at Step 2, in our model to extract the convolution features of the input image.  At Step 3 we apply adaptive average pooling. The output of Step 3 is flattened and sent to the classifier in Step 4. The classifier is two layers of fully connected MLP with dropout and Relu activation applied in between. Finally Step 6 is the final output of the model which is a single neuron to indicate if the input image is fake or real. We aggregate the results of the frames to quantify the entire video.

<img src="https://github.com/pratikpv/deep_fake_detection/blob/master/images/Picture7.png" width=80% height=70%>


# Confusion-matrix for all three methods (Plain-frames, Optical-flow, MRI-gan-based)


<img src="https://github.com/pratikpv/deep_fake_detection/blob/master/images/Picture9.png" width=110% height=150%>