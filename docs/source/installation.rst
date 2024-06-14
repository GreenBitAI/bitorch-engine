Installation
============

The requirements are:

-  A compiler that fully supports C++17, such as clang or gcc (gcc 9.4.0
   or newer is required, but gcc 12.x is not supported yet)
-  Python 3.9 or later
-  PyTorch 1.8 or later

Please check your operating system’s options for the C++ compiler. For
more detailed information, you can check the `requirements to build
PyTorch from
source <https://github.com/pytorch/pytorch?tab=readme-ov-file#prerequisites>`__.
In addition, for layers to speed up on specific hardware (such as CUDA
devices, or MacOS M1/2/3 chips), we recommend installing:

-  CUDA Toolkit 11.8 or 12.1 for CUDA accelerated layers
-  `MLX <https://github.com/ml-explore/mlx>`__ for mlx-based layers on
   MacOS
-  `CUTLASS <https://github.com/NVIDIA/cutlass>`__ for cutlass-based
   layers

Binary Release
--------------

**A first experimental binary release for Linux with CUDA 12.1 is
ready.** It only supports GPUs with CUDA compute capability with 8.6 or
higher (`check here <https://developer.nvidia.com/cuda-gpus>`__). For
MacOS or lower compute capability, build the package from source
(additional binary release options are planned in the future). We
recommend to create a conda environment to manage the installed CUDA
version and other packages:

1. Create Environment for Python 3.10 and activate it:

.. code:: bash

   conda create -y --name bitorch-engine python=3.10
   conda activate bitorch-engine

As an alternative, you can also store the environment in a relative
path.

    
    
    
.. dropdown:: Click to here to expand the instructions for this.
            
    
    
    .. code:: bash
    
       export BITORCH_WORKSPACE="${HOME}/bitorch-workspace"
       mkdir -p "${BITORCH_WORKSPACE}" && cd "${BITORCH_WORKSPACE}"
       conda create -y --prefix ./conda-env python=3.10
       conda activate ./conda-env
    
    

2. Install CUDA (if it is not installed already on the system):

.. code:: bash

   conda install -y -c "nvidia/label/cuda-12.1.0" cuda-toolkit

3. Install our customized torch that allows gradients on INT tensors and
   install it with pip (this URL is for CUDA 12.1 and Python 3.10 - you
   can find other versions `here <https://packages.greenbit.ai/whl/>`__)
   together with bitorch engine:

.. code:: bash

   pip install \
     "https://packages.greenbit.ai/whl/cu121/torch/torch-2.3.0-cp310-cp310-linux_x86_64.whl" \
     "https://packages.greenbit.ai/whl/cu121/bitorch-engine/bitorch_engine-0.2.6-cp310-cp310-linux_x86_64.whl"

Build From Source
-----------------

We provide instructions for the following options:

-  `Conda + Linux <#conda-on-linux-with-cuda>`__ (with CUDA and cutlass)
-  `Docker <#docker-with-cuda>`__ (with CUDA and cutlass)
-  `Conda + MacOS <#conda-on-macos-with-mlx>`__ (with MLX)

We recommend managing your BITorch Engine installation in a conda
environment (otherwise you should adapt/remove certain variables,
e.g. ``CUDA_HOME``). You may want to keep everything (environment, code,
etc.) in one directory or use the default directory for conda
environments. You may wish to adapt the CUDA version to 12.1 where
applicable.

Conda on Linux (with CUDA)
~~~~~~~~~~~~~~~~~~~~~~~~~~

To use these instructions, you need to have
`conda <https://conda.io/projects/conda/en/latest/user-guide/getting-started.html>`__
and a suitable C++ compiler installed.

1. Create Environment for Python 3.9 and activate it:

.. code:: bash

   conda create -y --name bitorch-engine python=3.9
   conda activate bitorch-engine

2. Install CUDA

.. code:: bash

   conda install -y -c "nvidia/label/cuda-11.8.0" cuda-toolkit

3. Install our customized torch that allows gradients on INT tensors and
   install it with pip (this URL is for CUDA 11.8 and Python 3.9 - you
   can find other versions
   `here <https://packages.greenbit.ai/whl/>`__):

.. code:: bash

   pip install "https://packages.greenbit.ai/whl/cu118/torch/torch-2.1.0-cp39-cp39-linux_x86_64.whl"

4. To use cutlass layers, you should also install CUTLASS 2.8.0 (from
   source), adjust ``CUTLASS_HOME`` (this is where we clone and install
   cutlass) (if you have older or newer GPUs you may need to add your
   `CUDA compute capability <https://developer.nvidia.com/cuda-gpus>`__
   in ``CUTLASS_NVCC_ARCHS``):

.. code:: bash

   export CUTLASS_HOME="/some/path"
   mkdir -p "${CUTLASS_HOME}"
   git clone --depth 1 --branch "v2.8.0" "https://github.com/NVIDIA/cutlass.git" --recursive ${CUTLASS_HOME}/source
   mkdir -p "${CUTLASS_HOME}/build" && mkdir -p "${CUTLASS_HOME}/install"
   cd "${CUTLASS_HOME}/build"
   cmake ../source -DCMAKE_INSTALL_PREFIX="${CUTLASS_HOME}/install" -DCUTLASS_ENABLE_TESTS=OFF -DCUTLASS_ENABLE_EXAMPLES=OFF -DCUTLASS_NVCC_ARCHS='75;80;86'
   make -j 4
   cmake --install .

If you have difficulties installing cutlass, you can check the `official
documentation <https://github.com/NVIDIA/cutlass/tree/v2.8.0>`__, use
the other layers without installing it or try the docker installation.

As an alternative to the instructions above, you can also store the
environment and clone all repositories within one “root” directory.

    
    
    
.. dropdown:: Click to here to expand the instructions for this.
            
    
    
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
    
    3. Install our customized torch that allows gradients on INT tensors and
       install it with pip (this url is for CUDA 11.8 and Python 3.9 - you
       can find other versions
       `here <https://packages.greenbit.ai/whl/>`__):
    
    .. code:: bash
    
       pip install "https://packages.greenbit.ai/whl/cu118/torch/torch-2.1.0-cp39-cp39-linux_x86_64.whl"
    
    4. To use cutlass layers, you should also install CUTLASS 2.8.0 (if you
       have older or newer GPUs you may need to add your `CUDA compute
       capability <https://developer.nvidia.com/cuda-gpus>`__ in
       ``CUTLASS_NVCC_ARCHS``):
    
    .. code:: bash
    
       export CUTLASS_HOME="${BITORCH_WORKSPACE}/cutlass"
       mkdir -p "${CUTLASS_HOME}"
       git clone --depth 1 --branch "v2.8.0" "https://github.com/NVIDIA/cutlass.git" --recursive ${CUTLASS_HOME}/source
       mkdir -p "${CUTLASS_HOME}/build" && mkdir -p "${CUTLASS_HOME}/install"
       cd "${CUTLASS_HOME}/build"
       cmake ../source -DCMAKE_INSTALL_PREFIX="${CUTLASS_HOME}/install" -DCUTLASS_ENABLE_TESTS=OFF -DCUTLASS_ENABLE_EXAMPLES=OFF -DCUTLASS_NVCC_ARCHS='75;80;86'
       make -j 4
       cmake --install .
       cd "${BITORCH_WORKSPACE}"
    
    If you have difficulties installing cutlass, you can check the `official
    documentation <https://github.com/NVIDIA/cutlass/tree/v2.8.0>`__, use
    the other layers without installing it or try the docker installation.
    
    

After setting up the environment, clone the code and build with pip (to
hide the build output remove ``-v``):

.. code:: bash

   # make sure you are in a suitable directory, e.g. your bitorch workspace
   git clone --recursive https://github.com/GreenBitAI/bitorch-engine
   cd bitorch-engine
   # only gcc versions 9.x, 10.x, 11.x are supported
   # to select the correct gcc, use:
   # export CC=gcc-11 CPP=g++-11 CXX=g++-11
   CPATH="${CUTLASS_HOME}/install/include" CUDA_HOME="${CONDA_PREFIX}" pip install -e . -v

Docker (with CUDA)
~~~~~~~~~~~~~~~~~~

You can also use our prepared Dockerfile to build a docker image (which
includes building the engine under ``/bitorch-engine``):

.. code:: bash

   cd docker
   docker build -t bitorch/engine .
   docker run -it --rm --gpus all --volume "/path/to/your/project":"/workspace" bitorch/engine:latest

Check the `docker readme <https://github.com/GreenBitAI/bitorch-engine/blob/HEAD/docker/README.md>`__ for options and more
details.

Conda on MacOS (with MLX)
~~~~~~~~~~~~~~~~~~~~~~~~~

1. We recommend to create a virtual environment for and activate it. In
   the following example we use a conda environment for python 3.9, but
   virtualenv should work as well.

.. code:: bash

   conda create -y --name bitorch-engine python=3.9
   conda activate bitorch-engine

2. Install our customized torch that allows gradients on INT tensors and
   install it with pip (this URL is for macOS with Python 3.9 - you can
   find other versions `here <https://packages.greenbit.ai/whl/>`__):

.. code:: bash

   pip install "https://packages.greenbit.ai/whl/macosx/torch/torch-2.2.1-cp39-none-macosx_11_0_arm64.whl"

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

