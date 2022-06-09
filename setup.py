from distutils.core import setup
import setuptools

setup(name='Proml',
      version='0.0.2',
      description='ProML',
      author='Attayeb Mohsen',
      author_email='',
      package_dir = {"" : str("src")},
      url='',
      packages=setuptools.find_packages(where="src")
     )