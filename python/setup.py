import os

from setuptools import setup, find_packages, Extension


include_dirs = []
library_dirs = []

def _maybe_add_library_root(lib_name):
  if "%s_ROOT" % lib_name in os.environ:
    root = os.environ["%s_ROOT" % lib_name]
    include_dirs.append("%s/include" % root)
    for lib_dir in ("lib", "lib64"):
      path = "%s/%s" % (root, lib_dir)
      if os.path.exists(path):
        library_dirs.append(path)
        break

_maybe_add_library_root("BOOST")
_maybe_add_library_root("CTRANSLATE2")

ctranslate2_module = Extension(
    "ctranslate2.translator",
    sources=["translator.cc"],
    extra_compile_args=["-std=c++11"],
    include_dirs=include_dirs,
    library_dirs=library_dirs,
    libraries=[os.getenv("BOOST_PYTHON_LIBRARY", "boost_python"), "ctranslate2"])

setup(
    name="ctranslate2",
    version="1.0.1",
    packages=find_packages(exclude=["bin"]),
    ext_modules=[ctranslate2_module],
    install_requires=[
        "six",
    ],
    entry_points={
        "console_scripts": [
            "ct2-opennmt-py-converter=ctranslate2.bin.opennmt_py_converter:main",
            "ct2-opennmt-tf-converter=ctranslate2.bin.opennmt_tf_converter:main",
        ],
    }
)
