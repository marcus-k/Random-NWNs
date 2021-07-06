from setuptools import setup

namespace = {}
with open("randomnwn/version.py", "r") as f:
    exec(f.read(), namespace)

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name='randomnwn',
    version=namespace["__version__"],    
    description='Code for creating and analyzing random nanowire networks.',
    long_description=long_description,
    url='https://github.com/marcus-k/Random-NWNs',
    author='Marcus Kasdorf',
    author_email='marcus.kasdorf@ucalgary.ca',
    license='MIT License',
    packages=['randomnwn'],
    python_requires='>=3.7'
)