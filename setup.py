import os
from distutils.core import setup
#import setuptools
#from numpy.distutils.core import setup

import teslakit

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
                os.path.join(os.getcwd(), *f)).readlines()
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
    return reqs('requirements.txt')


setup(
    name             = 'teslakit',
    version          = teslakit.__version__,
    description      = teslakit.__description__,
    long_description = open('README.md').read(),
    keywords         = teslakit.__keywords__,
    author           = teslakit.__author__,
    author_email     = teslakit.__contact__,
    url              = teslakit.__url__,
    license          = 'LICENSE.txt',
    install_requires = install_requires(),
    python_requires  = ">=3.7",
    packages         = ['teslakit', 'teslakit.test', 'teslakit.plotting',
                        'teslakit.io', 'teslakit.util'],
    package_data     = {'teslakit' : ['resources/*']},
    include_package_data = True,
    #packages=setuptools.find_packages(exclude=['test*']),
    #include_package_data=True,
    #package_data={'attributes': ['teslakit/core/attributes.yml']},
    #platforms=['any'],
    #extras_require=extras_require(),
    #setup_requires=['pytest-runner'],
    #tests_require=reqs('test.txt'),
    #classifiers=CLASSIFIERS,
    #project_urls=PROJECT_URLS,
    #**kwargs
)

