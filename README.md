# DL4DS

Python package with novel deep learning algorithms for spatial statistical downscaling. It's built on top of Tensorflow/Keras.

Currently the following models have been implemented:

    * 'resnet_spc': Resnet with pixel shuffle post-upscaling (based on EDSR)
    * 'resnet_bi': Resnet with interpolation-based pre-upscaling
    * 'resnet_rc': Resnet with resize convolution post-upscaling (bilinear interpolation)
    * 'cgan_resnet_spc': Conditional generative adversarial network with 'resnet_spc' as generator 
    * 'cgan_resnet_bi': Conditional generative adversarial network with 'resnet_bi' as generator 
    * 'cgan_resnet_rc': Conditional generative adversarial network with 'resnet_rc' as generator 