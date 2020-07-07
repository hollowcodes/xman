
import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="xman",
    version="0.0.1",
    author="Theodor Peifer",
    author_email="teddypeifer@gmail.com",
    description="A simple libary to manage deep-learning experiments.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    py_modules=["xman"],
    package_dir={" ": "src"},
    url="https://github.com/hollowcodes/xman",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: Linux",
    ],
    python_requires='>=3.6',
)