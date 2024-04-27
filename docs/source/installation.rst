Installation
============

The requirements are:

-  A compiler that fully supports C++17, such as clang or gcc (gcc 9.4.0
   or newer is required, but gcc 12.x is not supported yet)
-  Python 3.9 or later
-  PyTorch 1.8 or later
-  CUDA Toolkit 11.8 or 12.1 (optional, for CUDA accelerated layers)

For more detailed information, you can check the `requirements of
PyTorch <https://github.com/pytorch/pytorch?tab=readme-ov-file#prerequisites>`__.

Currently, the engine needs to be built from source. We provide
instructions how to install Python/PyTorch (and CUDA/MLX) for:

-  Conda + Linux (with CUDA)
-  Docker (with CUDA)
-  Conda + MacOS (with MLX)

We recommend managing your BITorch Engine installation in a conda
environment (otherwise you should adapt/remove certain variables,
e.g. ``CUDA_HOME``). You may want to keep everything (environment, code,
etc.) in one directory or use the default directory for conda
environments. You may wish to adapt the CUDA version to 12.1 where
applicable.

Conda on Linux (with CUDA)
--------------------------

1. Create Environment for Python 3.9 and activate it:

.. code:: bash

   conda create -y --name bitorch-engine python=3.9
   conda activate bitorch-engine

2. Install CUDA

.. code:: bash

   conda install -y -c "nvidia/label/cuda-11.8.0" cuda-toolkit

3. `Download customized Torch
   2.1.0 <https://drive.google.com/drive/folders/1T22b8JhN-E3xbn3h332rI1VjqXONZeB7?usp=sharing>`__
   (it allows gradients on INT tensors, built for Python 3.9 and CUDA
   11.8) and install it with pip:

.. code:: bash

   pip install torch-2.1.0-cp39-cp39-linux_x86_64.whl
   # optional: install corresponding torchvision (check https://github.com/pytorch/vision?tab=readme-ov-file#installation in the future)
   pip install "torchvision==0.16.0" --index-url https://download.pytorch.org/whl/cu118

Alternatively, you can also save the environment and clone the
repository within the same directory.

    
    
    
.. collapse:: Click to here to expand the instructions for this.
            
    
    
    0. Set workspace dir (use an absolute path!):
    
    .. code:: bash
    
       export BITORCH_WORKSPACE="${HOME}/bitorch-workspace"
       mkdir -p "${BITORCH_WORKSPACE}" && cd "${BITORCH_WORKSPACE}"
    
    1. Create Environment for Python 3.9 and activate it:
    
    .. code:: bash
    
       conda create -y --prefix ./conda-env python=3.9
       conda activate ./conda-env
    
    2. Install CUDA
    
    .. code:: bash
    
       conda install -y -c "nvidia/label/cuda-11.8.0" cuda-toolkit
    
    3. `Download customized Torch
       2.1.0 <https://drive.google.com/drive/folders/1T22b8JhN-E3xbn3h332rI1VjqXONZeB7?usp=sharing>`__,
       select the package fit for the cuda version you installed in the
       previous step (it allows gradients on INT tensors, built for Python
       3.9 and CUDA 11.8) and install it with pip:
    
    .. code:: bash
    
       pip install torch-2.1.0-cp39-cp39-linux_x86_64.whl
       # optional: install corresponding torchvision (check https://github.com/pytorch/vision?tab=readme-ov-file#installation in the future)
       pip install "torchvision==0.16.0" --index-url https://download.pytorch.org/whl/cu118
    
    

After setting up the environment, clone the code and build with pip (to
hide the build output remove ``-v``):

.. code:: bash

   git clone --recursive https://github.com/GreenBitAI/bitorch-engine
   cd bitorch-engine
   # only gcc versions 9.x, 10.x, 11.x are supported
   # to select the correct gcc, use:
   # export CC=gcc-11 CPP=g++-11 CXX=g++-11
   CUDA_HOME="${CONDA_PREFIX}" pip install -e . -v

Docker (with CUDA)
------------------

You can also use our prepared Dockerfile to build a docker image (which
includes building the engine under ``/bitorch-engine``):

.. code:: bash

   cd docker
   docker build -t bitorch/engine .
   docker run -it --rm --gpus all --volume "/path/to/your/project":"/workspace" bitorch/engine:latest

Check the `docker readme <https://github.com/GreenBitAI/bitorch-engine/blob/HEAD/docker/README.md>`__ for options and more
details.

Conda on MacOS (with MLX)
-------------------------

1. We recommend to create a virtual environment for and activate it. In
   the following example we use a conda environment for python 3.9, but
   virtualenv should work as well.

.. code:: bash

   conda create -y --name bitorch-engine python=3.9
   conda activate bitorch-engine

2. Download `customized Torch for
   arm <https://drive.google.com/drive/folders/1T22b8JhN-E3xbn3h332rI1VjqXONZeB7?usp=sharing>`__
   (it allows gradients on INT tensors, built for Python 3.9 and CUDA
   11.8) and install it with pip:

.. code:: bash

   pip install path/to/torch-2.2.1-cp39-none-macosx_11_0_arm64.whl
   # optional: install corresponding torchvision (check https://github.com/pytorch/vision?tab=readme-ov-file#installation in the future)
   pip install "torchvision==0.17.1"

3. For MacOS users and to use OpenMP acceleration, install OpenMP with
   Homebrew and configure the environment:

.. code:: bash

   brew install libomp
   # during libomp installation it should remind you, you need something like this:
   export LDFLAGS="-L$(brew --prefix)/opt/libomp/lib"
   export CPPFLAGS="-I$(brew --prefix)/opt/libomp/include"

4. To use the `mlx <https://github.com/ml-explore/mlx>`__ accelerated
   ``MPQLinearLayer``, you need to install the python library.

.. code:: bash

   # use one of the following, to either install with pip or conda:
   pip install mlx==0.4.0
   conda install conda-forge::mlx=0.4.0

Currently, we only tested version 0.4.0. However, newer versions might
also work. To train the ``MPQLinearLayer`` you need to install our
custom PyTorch version (see steps above). Without it, you need to
specify ``requires_grad=False`` when initializing ``MPQLinearLayer``. 5.
You should now be able to build with:

.. code:: bash

   git clone --recursive https://github.com/GreenBitAI/bitorch-engine
   cd bitorch-engine
   pip install -e . -v
