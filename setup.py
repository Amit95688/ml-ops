from setuptools import find_packages, setup

hypen='-e .'
def get_requirements(file_path):
    with open(file_path, 'r') as file_obj:
        requirements=[]
        requirements = file_obj.readlines()
        requirements=[req.replace("\n","") for req in requirements]
        if hypen in requirements:
            requirements.remove(hypen)
    return requirements

setup(        
    name='ml_project',
    version='0.1.0',
    author='Amit Dubey',
    author_email='amitdubey300705@gmail.com',
    packages=find_packages(),
    install_requires=[
        'numpy>=1.18.0',
        'pandas>=1.0.0',
        'scikit-learn>=0.22.0',
        'matplotlib>=3.1.0',
        'seaborn>=0.10.0',
        'joblib>=0.14.0'
    ],  
)