from setuptools import setup, find_packages

def parse_requirements(filename):
    with open(filename, "r") as f:
        return f.read().splitlines()

setup(
    name='pySAMetrics',
    version='0.2',  # Set an appropriate version number
    packages=find_packages(where='src'),  # Only include the 'pySAMetrics' package
    package_dir={"":"src"},
    install_requires=parse_requirements("requirements.txt")
)