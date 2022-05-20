[![Tensorflow - Version](https://img.shields.io/badge/Tensorflow%2FKeras-2.6+-blue&logo=tensorflow)](https://github.com/carlgogo/dl4ds) 
[![Python - Version](https://img.shields.io/badge/PYTHON-3.6+-red?style=flat&logo=python&logoColor=white)](https://github.com/carlgogo/dl4ds) 


# Deep Learning for empirical DownScaling

`DL4DS` (Deep Learning for empirical DownScaling) is a Python package that implements state-of-the-art and novel deep learning algorithms for empirical downscaling of gridded Earth science data. 

The general architecture of `DL4DS` is shown on the image below. A low-resolution gridded dataset can be downscaled, with the help of (an arbitrary number of) auxiliary predictor and static variables, and a high-resolution reference dataset. The mapping between the low- and high-resolution data is learned with either a supervised or a conditional generative adversarial DL model.

<img src="https://github.com/carlos-gg/dl4ds/raw/master/docs/img/fig_workflow.png" alt="drawing" width="800"/>

The training can be done from explicit pairs of high- and low-resolution samples (MOS-style, e.g., high-res observations and low-res numerical weather prediction model output) or only with a HR dataset (PerfectProg-style, e.g., high-res observations or high-res model output).

A wide variety of network architectures have been implemented in `DL4DS`. The main modelling approaches can be combined into many different architectures:

|Downscaling type               |Training (loss type)         |Sample type     |Backbone section              |Upsampling method   |
|---                            |---                          |---             |---                          |---|
|MOS (explicit pairs of HR and LR data)           |Supervised (non-adversarial) |Spatial         |Plain convolutional          |Pre-upsampling via interpolation |
|PerfectProg (implicit pairs, only HR data)   |Conditional Adversarial    |Spatio-temporal |Residual                     |Post-upsampling via sub-pixel convolution |
|                               |                             |                |Dense                        |Post-upsampling via resize convolution |
|                               |                             |                |Unet (PIN, Spatial samples)  |Post-upsampling via deconvolution   |
|                               |                             |                |Convnext (Spatial samples)   |                                      |

In `DL4DS`, we implement a channel attention mechanism to exploit inter-channel relationship of features by providing a weight for each channel in order to enhance those that contribute the most to the optimizaiton and learning process. Aditionally, a Localized Convolutional Block (LCB) is located in the output module of the networks in `DL4DS`. With the LCB we learn location-specific information via a locally connected layer with biases. 

`DL4DS` is built on top of Tensorflow/Keras and supports distributed GPU training (data parallelism) thanks to Horovod. 

# API documentation 

Check out the API documentation [here](https://carlos-gg.github.io/dl4ds/).

# Example notebooks

Colab notebooks are under construction. Stay tuned!




