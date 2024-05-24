Build options
=============

Building Specific Extensions
----------------------------

While developing, a specific cpp/cuda extension can be (re-)build, by
using the environment variable ``BIE_BUILD_ONLY``, like so:

.. code:: bash

   BIE_BUILD_ONLY="bitorch_engine/layers/qlinear/binary/cpp" pip install -e . -v

It needs to a relative path to one extension directory.

Building for a Specific CUDA Architecture
-----------------------------------------

To build for a different CUDA Arch, use the environment variable
``BIE_CUDA_ARCH`` (e.g. use ‘sm_75’, ‘sm_80’, ‘sm_86’):

.. code:: bash

   BIE_CUDA_ARCH="sm_86" pip install -e . -v

Force Building CUDA Modules
---------------------------

If you have CUDA development libraries installed, but
``torch.cuda.is_available()`` is False, e.g. in HPC or docker
environments, you can still build the extensions that depend on CUDA, by
setting ``BIE_FORCE_CUDA="true"``:

.. code:: bash

   BIE_FORCE_CUDA="true" pip install -e . -v

Skip Library File Building
--------------------------

If you just want to avoid rebuilding any files, you can set
``BIE_SKIP_BUILD``:

.. code:: bash

   BIE_SKIP_BUILD="true" python3 -m build --no-isolation --wheel

This would create a wheel and package ``.so`` files without trying to
rebuild them.

