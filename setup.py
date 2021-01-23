import setuptools

with open("README.rst", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="modypy",
    version="1.0.1",
    author="Ralf Gerlich",
    author_email="ralf@ralfgerlich.biz",
    description="A framework for hierarchical modelling and simulation of dynamic systems",
    long_description=long_description,
    long_description_content_type="text/x-rst",
    url="https://github.com/ralfgerlich/modypy",
    license="BSD 2-clause \"Simplified\" License",
    packages=setuptools.find_packages(),
    project_urls={
        'documentation': 'https://modypy.readthedocs.io/',
    },
    classifiers=[
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Scientific/Engineering :: Mathematics",
    ],
    python_requires='>=3.6',
    install_requires=[
      "numpy>=1.18.1",
      "scipy>=1.5.4",
    ],
    setup_requires=[
      "pytest-runner",
    ],
    tests_require=[
      "pytest",
    ]
 )
