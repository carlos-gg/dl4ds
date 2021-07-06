# DL4DS

Python package with state-of-the-art deep learning algorithms for spatial statistical downscaling. 

A wide variety of models have been implemented. The main modelling techniques can be combined into 48 different architectures:

|Training                       |Sample type        |Backbone block |Upsampling strategy   |
|---                            |---                |---            |---|
|Supervised (non-adversarial)   |Spatial            |Plain Conv     |Pre-upsampling: interpolation  |
|Adversarial (conditional)      |Spatio-temporal    |Residual       |Post-upsampling: sub-pixel convolution (SPC)   |
|                               |                   |Dense          |Post-upsampling: resize convolution (RC)    |
|                               |                   |               |Post-upsampling: deconvolution (DC)    |

Examples: 
* using spatial samples, a residual backbone, pre-upsampling strategy and supervised training, we end up with the _resnet_pin_ model 
* using spatio-temporal samples, a dense backbone, post-upsampligng strategy (DC) and adversarial training, we end up with the _cgan_recdensenet_dc_ model. 

For adversarial training, we traing a discriminator network simultaneosly as we train the main generator model. All the models handle 
multiple predictors and static information such as the elevation and the land-ocean mask.

DL4DS is built on top of Tensorflow/Keras and supports distributed GPU training (data parallelism) thanks to Horovod.
