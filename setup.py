from setuptools import setup, find_packages
from os import path

working_dir = path.abspath(path.dirname(__file__))

with open(path.join(working_dir, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='fed_rf_mk',
    version='0.0.4',
    author='Alexandre Cotorobai',
    url='https://github.com/ieeta-pt/fed_rf',
    description='A federated Random Forest implementation',
    long_description=long_description,
    long_description_content_type='text/markdown',
    packages=find_packages(),
    classifiers=[
        'Programming Language :: Python :: 3',
    ],
    python_requires = ">=3.10.12",
    install_requires=[
        # 'threadpoolctl>=3.5.0',
        # 'async-timeout>=5.0.1',
        'syft>=0.9.1,<0.9.2',
        'joblib>=1.4.2',
        'matplotlib>=3.9.0',
        # 'ipython>=8.26.0',
        # 'anyio>=4.7.0',
        # 'typing_extensions>=4.12.0',
        'scikit-learn>=1.6.0',
        'numpy>=1.24.4',
        'pandas>=2.2.2',
        # 'ctgan>=0.10.2',
        'cloudpickle>=3.1.1',
    ],
    entry_points={
        'console_scripts': [
            'fed_rf_mk = fed_rf_mk.client:hello_world',
        ],
    },
)