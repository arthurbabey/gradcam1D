from setuptools import setup, find_packages

# Read the contents of requirements.txt
with open('requirements.txt', 'r') as f:
    requirements = f.read().splitlines()

setup(
    name='gradcam1D',
    version='0.0.0',
    packages=find_packages(),
    install_requires=requirements,
    description='GradCAM implementation for 1D CNN',
    url='https://github.com/arthurbabey/gradcam1D'
)
