import os
import setuptools
from numpy.distutils.core import setup


# project libraries
import lib

NAME = 'teslakit'


setup(
    name=NAME,
    version=wavespectra.__version__,
    description=wavespectra.__description__,
    long_description=read('README.rst'),
    keywords=wavespectra.__keywords__,
    author=wavespectra.__author__,
    author_email=wavespectra.__contact__,
    url=wavespectra.__url__,
    license='MIT',
    packages=setuptools.find_packages(exclude=['test*']),
    include_package_data=True,
    package_data={'attributes': ['wavespectra/core/attributes.yml']},
    platforms=['any'],
    install_requires=install_requires(),
    extras_require=extras_require(),
    setup_requires=['pytest-runner'],
    tests_require=reqs('test.txt'),
    python_requires=">=2.7, <3",
    classifiers=CLASSIFIERS,
    project_urls=PROJECT_URLS,
    **kwargs
)
