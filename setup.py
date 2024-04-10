from setuptools import setup, find_packages

VERSION = '0.0.1' 
DESCRIPTION = 'Integrated framework for network inference and evaluation using prior knowledge'
with open("README.md", "r", encoding="utf-8") as fh:
    LONG_DESCRIPTION = fh.read()


# Setting up
setup(
       # the name must match the folder name 'verysimplemodule'
        name="networkcommons", 
        version=VERSION,
        author="Victor Paton",
        author_email="victor.paton_gonzalez@stud.uni-heidelberg.de",
        description=DESCRIPTION,
        long_description=LONG_DESCRIPTION,
        packages=find_packages(),
        install_requires=['networkx'],        
        keywords=['python', 'network inference'],
        classifiers= [
            "Development Status :: 3 - Alpha",
            "Programming Language :: Python :: 3",
            "Operating System :: OS Independent",
        ]
)