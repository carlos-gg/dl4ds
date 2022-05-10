[![Tensorflow - Version](https://img.shields.io/badge/Tensorflow%2FKeras-2.6+-blue&logo=tensorflow)](https://github.com/carlgogo/dl4ds) 
[![Python - Version](https://img.shields.io/badge/PYTHON-3.6+-red?style=flat&logo=python&logoColor=white)](https://github.com/carlgogo/dl4ds) 


# DL4DS - Deep Learning for empirical DownScaling

Python package implementing state-of-the-art and novel deep learning algorithms for empirical downscaling of gridded data. DL4DS is built on top of Tensorflow/Keras and supports distributed GPU training (data parallelism) thanks to Horovod. The training can be done from explicit pairs of HR and LR samples (e.g., HR observations and LR numerical weather prediction model output) or only with a HR dataset (e.g., HR observations or HR model output). All the models are able to handle multiple predictors and an arbitrary number of static variables (e.g., elevation or land-ocean mask).

<img src="https://github.com/carlos-gg/dl4ds/raw/master/docs/img/fig_workflow.png" alt="drawing" width="800"/>

A wide variety of network architectures have been implemented. The main modelling approaches can be combined into many different architectures:

|Downscaling type               |Training (loss type)         |Sample type     |Backbone module              |Upsampling method   |
|---                            |---                          |---             |---                          |---|
|MOS (explicit pairs)           |Supervised (non-adversarial) |Spatial         |Plain convolutional          |Pre-upsampling: interpolation (PIN) |
|PerfectProg (implicit pairs)   |Adversarial (conditional)    |Spatio-temporal |Residual                     |Post-upsampling: sub-pixel convolution (SPC)|
|                               |                             |                |Dense                        |Post-upsampling: resize convolution (RC) |
|                               |                             |                |Unet (PIN, Spatial samples)  |Post-upsampling: deconvolution (DC)   |
|                               |                             |                |Convnext (Spatial samples)   |                                      |

## Extended documentation 

Check out the API documentation [here](https://carlos-gg.github.io/dl4ds/).

## Examples

Colab notebooks are WIP. Stay tuned!




