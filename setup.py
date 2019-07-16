from setuptools import setup
import fastentrypoints

setup(name='spark_vio_evaluation',
      version='0.2',
      description='Code for evaluating the performance of the SPARK VIO pipeline',
      url='https://github.com/ToniRV/spark_vio_evaluation',
      author='Antoni Rosinol',
      author_email='arosinol@mit.edu',
      license='MIT',
      packages=['evaluation', 'evaluation.tools'],
      install_requires=['numpy', 'pyyaml=3.12', 'evo-1'],
      zip_safe=False)
