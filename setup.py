#!/usr/bin/python3

import setuptools

with open('README.md', 'r', encoding='utf-8') as _:
    long_description = _.read()

with open('requirements.txt', 'r') as _:
    requires = [i.strip() for i in _.readlines()]

setuptools.setup(
    author='Ping Wu',
    author_email='wpwupingwp@outlook.com',
    description='Quantify GUS Stain Images',
    install_requires=requires,
    license='GNU AGPL v3',
    long_description=long_description,
    long_description_content_type='text/markdown',
    name='pyGUS',
    include_package_data=True,
    package_data={'pyGUS': ['1200dpi.png', ]},
    packages=setuptools.find_packages(),
    # f-string and Path-like
    python_requires='>=3.8',
    url='https://github.com/wpwupingwp/pyGUS',
    version='0.9',
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: GNU Affero General Public License v3',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3',
        'Topic :: Scientific/Engineering :: Bio-Informatics'
    ],
)
