from setuptools import setup, find_packages

with open("README.md", mode="r", encoding="utf-8") as readme_file:
    readme = readme_file.read()



setup(
      name="ACTIVA",
      version="0.0.3",
      author="A. Ali Heydari",
      author_email="aliheydari@ucdavis.edu",
      description="Generating Realistic Synthetic Single Cells with Introspective Variational Autoencoders",
      long_description=readme,
      long_description_content_type="text/markdown",
      license="MIT",
      url="https://github.com/SindiLab/ACTIVA",
      download_url="https://github.com/dr-aheydari/ACTIVA",
      packages=find_packages(),
      dependency_links=['http://github.com/SindiLab/ACTINN-PyTorch/tarball/master#egg=ACTINN'],
      install_requires=[
                        'tqdm==4.47.0',
                        'numpy==1.18.5',
                        'pandas==1.2.0',
                        'torch==1.9.1',
                        'scanpy==1.7.0',
                        'tensorboardX==2.1',
                        ],
      classifiers=[
                   "Development Status :: 1 - Beta",
                   "Intended Audience :: Science/Research",
                   "License :: OSI Approved :: MIT Software License",
                   "Programming Language :: Python :: 3.6",
                   "Topic :: Scientific/Engineering :: Artificial Intelligence :: Bioinformatics :: Deep Learning"
                   ],
      keywords="Single Cell RNA-seq, Synthetic Data Generation, InroVAEs, Transfer Learning"
      )
