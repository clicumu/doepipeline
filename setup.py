from setuptools import setup
import os
from doepipeline  import __version__

def readme():
    with open('README.md') as f:
        return f.read()

setup(
    name='doepipeline',
    version=__version__,
    description='Package for optimizing pipelines using DoE.',
    long_description=readme(),
    long_description_content_type="text/markdown",
    classifiers=[
        'Development Status :: 4 - Beta',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Topic :: Scientific/Engineering',
        'Operating System :: OS Independent'
    ],
    keywords='pipeline doe optimization',
    url='https://github.com/clicumu/doepipeline',
    author='Rickard Sjogren',
    author_email='rickard.sjogren@umu.se',
    license='MIT',
    packages=['doepipeline', 'doepipeline.executor'],
    install_requires=[
        'pyyaml',
        'pandas',
        'pyDOE2',
        'statsmodels'
    ],
    include_package_data=True,
    tests_require=[
        'mock',
        'nose'
    ],
    scripts=[
        os.path.join('bin', 'doepipeline')
    ],
    zip_safe=False
)
