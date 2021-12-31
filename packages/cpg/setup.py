from setuptools import setup

REQUIREMENTS = ['click', 'gtools']

setup(
    name='cpg',
    version='0.1',
    description='CLI for paramfile generation.',
    url='https://github.com/cmhainje/gtools',
    author='Connor Hainje',
    author_email='cmhainje@gmail.com',
    packages=['cpg'],
    install_requires=REQUIREMENTS,
)