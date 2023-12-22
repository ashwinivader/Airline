from setuptools import find_packages,setup
from typing import List

def get_requirements(file_path:str):
    '''
    this function will return the list of requirements
    '''
    requirments=[]
    with open(file_path) as file_obj:
        requirments=file_obj.readlines()
        requirments=[reg.replace("\n","") for reg in requirments]
        if "-e ." in requirments:
            requirments.remove('-e .')
    return requirments        

setup(
name='Airline',
version='0.0.1',
author= "Name",
author_email="abc.",
packages= find_packages(),
install_requires=get_requirements('requirments.txt')
)

