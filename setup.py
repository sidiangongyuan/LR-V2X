# -*- coding: utf-8 -*-
# Author: Seth Z. Zhao <sethzhao506@g.ucla.edu>
# License: Academic Software License: © 2021 UCLA Mobility Lab (“Institution”).

from setuptools import setup, find_packages
from opencood.version import __version__


setup(
    name='lr-v2x',
    version=__version__,
    packages=find_packages(),
    license='Mixed; see THIRD_PARTY_NOTICES.md',
    author='Kang Yang, Tianci Bu, Peng Wang, Deying Li, Yongcai Wang',
    author_email='ycw@ruc.edu.cn',
    description='Loss-resilient collaborative LiDAR perception under low-bandwidth communication',
    long_description=open("README.md", encoding='utf-8').read(),
    long_description_content_type='text/markdown',
    install_requires=[],
)
