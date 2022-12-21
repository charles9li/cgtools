from setuptools import setup


setup(
    name="cgtools",
    author="Charles Li",
    author_email="charlesli@ucsb.edu",
    description="Utilities for mapping to and modifying coarse-grained systems",
    keywords="molecular dynamics, coarse graining",
    packages=['cgtools'],
    python_requires=">=3.6",
    install_requires=["numpy", "networkx", "mdtraj"]
)
