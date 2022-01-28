# DL4DS

Python package with state-of-the-art and novel deep learning algorithms for spatial empirical downscaling of gridded data. DL4DS is built on top of Tensorflow/Keras and supports distributed GPU training (data parallelism) thanks to Horovod. The training can be done from explicit pairs of HR and LR samples (MOS-like) or only with a HR dataset (PerfecProg-like). 

A wide variety of network architectures have been implemented. The main modelling approahces can be combined into many different architectures:

|Downscaling type      |Training                     |Sample type     |Backbone block |Upsampling strategy   |
|---                   |---                          |---             |---            |---|
|MOS (explicit pairs)  |Supervised (non-adversarial) |Spatial         |Plain convolutional     |Pre-upsampling: interpolation  |
|PP (implicit pairs)   |Adversarial (conditional)    |Spatio-temporal |Residual       |Post-upsampling: sub-pixel convolution (SPC)|
|                      |                             |                |Dense          |Post-upsampling: resize convolution (RC) |
|                      |                             |                |               |Post-upsampling: deconvolution (DC)   |

Examples: 
* choosing spatial samples (proving both HR and LR data), a residual backbone, pre-upsampling strategy and supervised training, we end up with the _resnet_pin_ model 
* choosing spatio-temporal samples (with only HR data), a dense backbone, post-upsampligng strategy (DC) and adversarial training, we end up with the _cgan_recdensenet_dc_ model. 

For adversarial training, we traing a discriminator network simultaneosly as we train the main generator model. All the models handle 
multiple predictors and an arbitrary number of static variables such as the elevation or a land-ocean mask.


