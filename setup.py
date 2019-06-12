from setuptools import setup
import os

def readme():
    with open('README.md') as f:
        return f.read()

setup(
    name='doepipeline',
    version='1.1.0',
    description='Package for optimizing pipelines using DoE.',
    long_description=readme(),
    classifiers=[
        'Development Status :: 4 - Beta',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Topic :: Scientific/Engineering'
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
