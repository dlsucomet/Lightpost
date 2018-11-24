from setuptools import setup

setup(name='lightpost',
      version='0.0.3',
      author='Jan Christian Blaise Cruz',
      author_email='jsnell@cs.toronto.edu',
      license='MIT',
      packages=['lightpost', 'lightpost.engine', 'lightpost.estimators', 'lightpost.datapipe', 'lightpost.utils'],
      install_requires=[
          'torch',
          'tqdm',
          'torchtext',
      ])
