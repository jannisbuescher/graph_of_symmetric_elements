from setuptools import setup, find_packages

setup(
    name="DSG",
    version="0.1.0",
    author="Jannis Buescher",
    author_email="definitelyjannisbuescher@gmail.com",
    description="Graphs of symmetric elements",
    packages=find_packages(),
    install_requires=[
        "torch",
        "torch-geometric",
        "numpy"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
)
