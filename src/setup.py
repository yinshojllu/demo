#!/usr/bin/env python

from setuptools import setup, find_packages
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))


with open(path.join(here, "README.rst"), encoding='utf-8') as f:
    long_description = f.read()


setup(
    name="spider_ys",
    version='0.1',
    author="jllu",
    author_email="lujiale@yinsho.com",
    description="爬虫框架",
    long_description=long_description,
    classifiers=[
        'Development Status :: 3 - Alpha',

        'Intended Audience :: Developers',
        'Topic :: Software Development :: Web Spider',

        'Programming Language :: Python :: 3',
    ],
    packages=find_packages(exclude=['contrib', 'docs', 'tests']),
    install_requires=[
        'aliyun-mns',
    ],
    entry_points={
        'console_scripts': [
            # 'sample=sample:main',
        ],
    },
)
