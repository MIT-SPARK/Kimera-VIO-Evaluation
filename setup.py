from setuptools import setup
import fastentrypoints

setup(name='spark_vio_evaluation',
      version='0.2',
      description='Code for evaluating the performance of the SPARK VIO pipeline',
      url='https://github.com/ToniRV/spark_vio_evaluation',
      author='Antoni Rosinol',
      author_email='arosinol@mit.edu',
      license='MIT',
      packages=['evaluation', 'evaluation.tools', 'website'],
      install_requires=['numpy', 'glog', 'tqdm', 'ruamel.yaml', 'evo-1', 'open3d-python', 'plotly', 'chart_studio', 'pandas'],
      zip_safe=False)
