from setuptools import setup

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name='Random NWNs',
    version='0.1.0',    
    description='Code for creating and analyzing random nanowire networks.',
    long_description=long_description,
    url='https://github.com/Marcus-Repository/Random-NWNs',
    author='Marcus Kasdorf',
    author_email='marcus.kasdorf@ucalgary.ca',
    license='MIT License',
    packages=['randomnwn']
)