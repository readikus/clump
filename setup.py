import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="clump",
    version="0.0.3",
    author="Ian Read",
    author_email="ianharveyread@gmail.com",
    description="Easy-to-use package for grouping, tagging and finding similar documents, using advanced natural language processing techniques",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/readikus/clump",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6'
)
