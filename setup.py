from setuptools import setup, find_packages
from typing import List

# Function to read requirements from requirements.txt file and return a list of requirements
def get_requirements() -> List[str]:
    """
    Read the requirements.txt file and return a list of requirements
    """
    try:
        with open("requirements.txt", "r") as f:
            requirements_list = [
                line.strip() for line in f.readlines() 
                if line.strip() and line.strip() != "-e ."
            ]
    except FileNotFoundError:
        print("requirements.txt file not found")
        requirements_list = []
    
    return requirements_list

setup(
    name="Network Security",
    version="0.0.1",
    author="Ricardo Rojas",
    author_email="ricardorojasm1991@gmail.com",
    packages=find_packages(),
    install_requires=get_requirements(),
)