[![Tensorflow - Version](https://img.shields.io/badge/Tensorflow%2FKeras-2.6+-blue&logo=tensorflow)](https://github.com/carlgogo/dl4ds) 
[![Python - Version](https://img.shields.io/badge/PYTHON-3.6+-red?style=flat&logo=python&logoColor=white)](https://github.com/carlgogo/dl4ds) 


# DL4DS - Deep Learning for empirical DownScaling

Python package implementing state-of-the-art and novel deep learning algorithms for empirical downscaling of gridded data. DL4DS is built on top of Tensorflow/Keras and supports distributed GPU training (data parallelism) thanks to Horovod. The training can be done from explicit pairs of HR and LR samples (e.g., HR observations and LR numerical weather prediction model output) or only with a HR dataset (e.g., HR observations or HR model output). All the models are able to handle multiple predictors and an arbitrary number of static variables (e.g., elevation or land-ocean mask).

A wide variety of network architectures have been implemented. The main modelling approahces can be combined into many different architectures:

|Downscaling type      |Training (loss type)         |Sample type     |Backbone section     |Upsampling strategy   |
|---                   |---                          |---             |---                  |---|
|MOS (explicit pairs)  |Supervised (non-adversarial) |Spatial         |Plain convolutional  |Pre-upsampling: interpolation  |
|PP (implicit pairs)   |Adversarial (conditional)    |Spatio-temporal |Residual             |Post-upsampling: sub-pixel convolution (SPC)|
|                      |                             |                |Dense                |Post-upsampling: resize convolution (RC) |
|                      |                             |                |Unet (only pre-ups)  |Post-upsampling: deconvolution (DC)   |

## Extended documentation 

Work in progress. 

## Examples

Colab notebooks are being prepared. Stay tuned!

Examples: 
* choosing spatial samples (proving both HR and LR data), a residual backbone, pre-upsampling strategy and supervised training, we end up with the _resnet_pin_ model 
* choosing spatio-temporal samples (with only HR data), a dense backbone, post-upsampligng strategy (DC) and adversarial training, we end up with the _cgan_recdensenet_dc_ model. 



