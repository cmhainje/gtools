from setuptools import setup

REQUIREMENTS = ['numpy', 'astropy', 'scipy']

setup(
    name='gtools',
    version='0.1',
    description='Tools for GIZMO analysis.',
    url='https://github.com/cmhainje/gtools',
    author='Connor Hainje',
    author_email='cmhainje@gmail.com',
    packages=['gtools'],
    # long_description=open('README.md').read(),
    install_requires=REQUIREMENTS,
)