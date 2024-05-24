import importlib
import os
import sys
from distutils.command.clean import clean
from os.path import isfile
from pathlib import Path
from typing import Union, List

import setuptools
from setuptools.extension import Extension
from setuptools.dist import Distribution


root_path = Path(__file__).resolve().parent


version = "unknown"
version_file = root_path / "version.txt"
if version_file.exists():
    with open(root_path / "version.txt") as handle:
        version_content = handle.read().strip()
        if version_content:
            version = version_content
print("Bitorch Engine: version", version)


ALL_EXTENSIONS = [
    "bitorch_engine/layers/qconv/binary/cpp",
    "bitorch_engine/layers/qconv/binary/cutlass",
    "bitorch_engine/layers/qconv/nbit/cutlass",
    "bitorch_engine/layers/qlinear/binary/cpp",
    "bitorch_engine/layers/qlinear/binary/cutlass",
    "bitorch_engine/layers/qlinear/binary/cuda",
    "bitorch_engine/layers/qlinear/nbit/cutlass",
    "bitorch_engine/layers/qlinear/nbit/cuda",
    "bitorch_engine/layers/qlinear/nbit/mps",
    "bitorch_engine/functions/cuda",
]


built_extensions = list(filter(lambda p: Path(p).is_file(), ALL_EXTENSIONS))
print("Extensions built:", built_extensions)


def get_requirements(file_path: Union[Path, str]):
    requires = []
    for requirement in (root_path / file_path).open().readlines():
        requires.append(requirement)
    if "BIE_TORCH_REQUIREMENT" in os.environ:
        requires.append(os.environ["BIE_TORCH_REQUIREMENT"])
    return requires


dependencies = get_requirements('requirements.txt')


class CustomClean(clean):
    def run(self):
        # clean all by default
        self.all = True
        super().run()
        import bitorch_engine.extensions as ext
        for file in Path(ext.__file__).parent.glob("*.pyd" if sys.platform == "win32" else "*.so"):
            if self.dry_run:
                print(f"would remove '{file}'")
            else:
                print(f"removing '{file}'")
                file.unlink()


def get_ext_modules() -> List[Extension]:
    dirs_to_process = ALL_EXTENSIONS
    if "BIE_BUILD_ONLY" in os.environ:
        dirs_to_process = os.environ.get("BIE_BUILD_ONLY").split(" ")
        print(f"Extensions to build from environment variable 'BIE_BUILD_ONLY': {dirs_to_process}")
        for extension_dir in dirs_to_process:
            if extension_dir not in ALL_EXTENSIONS:
                sep = "\n    - "
                print(f"The extension directory '{extension_dir}' is not valid.\n" +
                      "Must be one of:{}{}".format(sep, sep.join(ALL_EXTENSIONS)))
                sys.exit(1)
    extension_list = []
    cutlass_installed = is_cutlass_available()
    mlx_installed = is_mlx_available()
    force_cuda_build = os.environ.get("BIE_FORCE_CUDA", "false").lower() == "true"
    for extension_dir in dirs_to_process:
        # we can print debug information here, it is only shown in case of errors or with a verbose setting
        print(f"Loading extension for build from {extension_dir}")
        sys.path.insert(0, extension_dir)
        extension_path = extension_dir + "/extension"
        if not isfile("{}.py".format(extension_path)):
            print("Warning: no extension.py found in {}! Is something to build there?".format(extension_path))
            continue
        module = importlib.import_module(extension_path.replace("/", "."))
        if hasattr(module, "CUDA_REQUIRED") and module.CUDA_REQUIRED and not torch.cuda.is_available() and not force_cuda_build:
            print(f"Skipping {module}, since it seems as if CUDA is not available.")
            continue
        if hasattr(module, "CUTLASS_REQUIRED") and module.CUTLASS_REQUIRED and not cutlass_installed:
            print(f"Skipping {module}, since it seems as if CUTLASS is not available.")
            continue
        if hasattr(module, "NEON_REQUIRED") and module.NEON_REQUIRED and not linux_arch_ident.is_arm():
            print(f"Skipping {module}, since it seems as if ARM architecture is not available.")
            continue
        if hasattr(module, "MLX_REQUIRED") and module.MLX_REQUIRED and not (torch.backends.mps.is_available() and mlx_installed):
            print(f"Skipping {module}, since it seems as if Mlx is not available.")
            continue

        extension_list.append(module.get_ext(Path(extension_dir)))
        sys.path.pop(0)
    return extension_list


cmdclass = {'clean': CustomClean}
ext_modules = []

build_keywords = ["develop", "bdist_wheel"]
if not any(k in sys.argv for k in build_keywords) or os.environ.get("BIE_SKIP_BUILD", "false") == "true":
    print("Bitorch Engine: Not loading extension build commands.")
else:
    print("Bitorch Engine: Loading extension build commands.")
    # we have installed requirements already and (might) need to build extensions
    import torch
    from torch.utils.cpp_extension import BuildExtension, IS_LINUX
    from bitorch_engine.utils.cutlass_path import is_cutlass_available
    from bitorch_engine.utils.mlx_path import is_mlx_available
    from bitorch_engine.utils.cuda_extension import gcc_version
    from bitorch_engine.utils.arch_helper import linux_arch_ident

    class CustomBuildExtension(BuildExtension):
        def run(self):
            if IS_LINUX:
                major, minor, patch = gcc_version()
                print("GCC Version:", f"{major}.{minor}.{patch}")
            print("Python Path:", sys.path)
            print("Environment:\n", "\n    ".join(f"{k}='{v}'" for k, v in os.environ.items()))
            super().run()

    cmdclass['build_ext'] = CustomBuildExtension
    ext_modules = get_ext_modules()


with open("README.md", "r", encoding="utf-8") as handle:
    readme_content = handle.read()


# make a binary distribution to fix python version and platform
class BinaryDistribution(Distribution):
    def has_ext_modules(self):
        return True


setuptools.setup(
    name="bitorch_engine",
    url="https://github.com/hpi-xnor/bitorch-inference-engine",
    version=version,
    author="Hasso Plattner Institute",
    author_email="fb10-xnor@hpi.de",
    description="A package for building and training quantized and binary neural networks with Pytorch",
    long_description=readme_content,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(exclude=["tests*", "examples*"]),
    package_data={"bitorch_engine.extensions": ["*.so"]},
    install_requires=dependencies,
    extras_require={
        "dev": get_requirements("requirements-dev.txt"),
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires='>=3.8',
    data_files=[
        ('.', [
            'version.txt',
            'requirements.txt',
            'requirements-dev.txt',
        ]),
    ],
    ext_modules=ext_modules,
    cmdclass=cmdclass,  # type: ignore
    distclass=BinaryDistribution,
)
