# DL4DS

Python package with deep learning algorithms (on top of Tensorflow/Keras) for spatial statistical downscaling.

Currently the following models have been implemented:
    * 'resnet_spc': Resnet with pixel shuffle post-upscaling (based on EDSR)
    * 'resnet_int': Resnet with interpolation-based pre-upscaling
    * 'resnet_rec': Resnet with resize convolution post-upscaling (bilinear interpolation)
    * 'cgan_resnet_spc': Conditional generatie adversarial network with 'resnet_spc' as generator 
    * 'cgan_resnet_int': Conditional generatie adversarial network with 'resnet_int' as generator 
    * 'cgan_resnet_rec': Conditional generatie adversarial network with 'resnet_rec' as generator 