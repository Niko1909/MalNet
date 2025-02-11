# MalNet
ESC204: Team 104-C high fidelity prototype for malaria plasmodium parasitism classification in blood cells.

The current MalariaNet.py file generates a training model which can produce CNN models with roughly 94%-95% validation set test accuracy in 5-6 epochs.
Other notes for the current build:
  - Current batch size: 128
  - Current training split: 80% training, 10% validation, 10% test
  - Current layer architecture: 4 convolutional layers and 4 max pooling layers in alternating sequence, followed by a single linear layer (DNN)
  - Current activation function: selu
  - Current number of filters per convolutional layer: convolution_layer_nodes(128,64,32,16) -> linear layer
  - Current input shape: (128,128,128,3) -> (n,h,w,c)
  - Current output shape: (128,) -> (bool)
  - Current optimizer: Adam
  - Current learning rate: 0.001
  - Current loss function: SparseCategoricalCrossEntropy
  - Current maximum epochs: 20
  - Current kernel size: (2,2)
  - Current stride distance: (2,2)
  - Current batching method: resizing (h,w) to (128,128)

A checklist to complete before running the MalariaNet.py file

1. install python 3.9 to your device
      - See https://www.python.org/downloads/ for further details

2. install tensorflow (TF/tf), numpy (np), matplotlib, (pandas, pending.) libraries for python using pip
      - For help using pip see documentation: https://pip.pypa.io/en/stable/cli/pip_install/
      - General commands for windows appear as: C:\Users\Username>python -m pip install <library>
          - Note: the <> indicate values to be filled based on desired library
  
3. Optional*: For GPU support see https://www.tensorflow.org/install/gpu
      - Webpage provides support for Nvidia GPUs with CUDA core support
      - Be sure to follow specified instructions to a tee in terms of cuDNN package installation and version support for TF, 
        otherwise your GPU won't be able to detect the correct library in your environment path, resulting in an error.
      - WARNING: Computation times may be extremely long without proper GPU or TPU.

TODO:
  - Create a .dll file or makefile to do all the above for you upon installation. May not actually happen, only if I have extra time.
  - Work on hyperparameter tuning during test on validation set using random search optimizers
    - Research Bayesian hyperparameter algorithm as an alternative
  
