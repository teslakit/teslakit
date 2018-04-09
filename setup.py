import os
import setuptools
from numpy.distutils.core import setup


# project libraries
import teslakit

NAME = 'teslakit'


# TODO: igual no es necesaria tanta metafuncion
def _strip_comments(l):
    return l.split('#', 1)[0].strip()

def _pip_requirement(req):
    if req.startswith('-r '):
        _, path = req.split()
        return reqs(*path.split('/'))
    return [req]

def _reqs(*f):
    return [
        _pip_requirement(r) for r in (
            _strip_comments(l) for l in open(
                os.path.join(os.getcwd(), 'requirements', *f)).readlines()
        ) if r]

def reqs(*f):
    """
    Parse requirement file.
    Returns:
        List[str]: list of requirements specified in the file.
    Example:
        reqs('default.txt')          # requirements/default.txt
        reqs('extras', 'redis.txt')  # requirements/extras/redis.txt
    """
    return [req for subreq in _reqs(*f) for req in subreq]

def install_requires():
    return reqs('default.txt')

def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


setup(
    name=NAME,
    version=teslakit.__version__,
    description=teslakit.__description__,
    long_description=read('README.md'),
    keywords=teslakit.__keywords__,
    author=teslakit.__author__,
    author_email=teslakit.__contact__,
    url=teslakit.__url__,
    license='MIT',
    packages=setuptools.find_packages(exclude=['test*']),
    #include_package_data=True,
    #package_data={'attributes': ['wavespectra/core/attributes.yml']},
    platforms=['any'],
    install_requires=install_requires(),
    #extras_require=extras_require(),
    #setup_requires=['pytest-runner'],
    #tests_require=reqs('test.txt'),
    python_requires=">=2.7, <3",
    #classifiers=CLASSIFIERS,
    #project_urls=PROJECT_URLS,
    #**kwargs
)
