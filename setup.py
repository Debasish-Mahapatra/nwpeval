from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

#with open("CHANGELOG.md", "r") as fh:
 #   changelog = fh.read()

#long_description += "\n\n" + changelog

setup(
    name='nwpeval',
    version='1.5.0b2',
    description='A package for computing metrics for NWP model evaluation',
    long_description=long_description,
    long_description_content_type="text/markdown",
    author='Debasish Mahapatra',
    author_email='debasish.atmos@gmail.com | debasish.mahapatra@ugent.be',
    
    packages=find_packages(),
    install_requires=[
        'numpy>=1.20.0',
        'pygrib>=2.1.0',
        'xarray>=0.18.0',
        'scipy>=1.6.0',
        'pandas>=1.2.0',
        'matplotlib>=3.3.0',
    ],
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Operating System :: OS Independent',
    ],
)