from setuptools import setup

def readme():
    with open('README.rst') as f:
        return f.read()

setup(
    name='doepipeline',
    version='0.1',
    description='Package for optimizing pipelines using DoE.',
    long_description=readme(),
    classifiers=[
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 2.7',
        'Topic :: Scientific/Engineering'
    ],
    keywords='pipeline doe optimization',
    url='https://github.com/clicumu/doepipeline',
    author='Rickard Sjogren',
    author_email='rickard.sjogren@umu.se',
    license='MIT',
    packages=['doepipeline'],
    install_requires=[
        'pyyaml',
        'pandas',
        'paramiko'
    ],
    include_package_data=True,
    tests_require=[
        'mock',
    ],
    zip_safe=False
)