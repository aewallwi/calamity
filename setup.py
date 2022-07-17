"""A setuptools based setup module.
See:
https://packaging.python.org/guides/distributing-packages-using-setuptools/
https://github.com/pypa/sampleproject
"""

# Always prefer setuptools over distutils
from setuptools import setup, find_packages
import pathlib
import os
import sys
sys.path.append("calamity")
import version
import json

data = [version.git_origin, version.git_hash, version.git_description, version.git_branch]
with open(os.path.join('calamity', 'GIT_INFO'), 'w') as outfile:
    json.dump(data, outfile)


def package_files(package_dir, subdirectory):
    # walk the input package_dir/subdirectory
    # return a package_data list
    paths = []
    directory = os.path.join(package_dir, subdirectory)
    for (path, directories, filenames) in os.walk(directory):
        for filename in filenames:
            path = path.replace(package_dir + '/', '')
            paths.append(os.path.join(path, filename))
    return paths

here = pathlib.Path(__file__).parent.resolve()
long_description = (here / 'README.md').read_text(encoding='utf-8')
data_files = package_files('calamity', 'data')

setup(
    name='calamity',  # Required
    version=version.version,  # Required
    description='Mostly sky-independent calibration.',  # Optional
    long_description=long_description,  # Optional
    long_description_content_type='text/markdown',  # Optional (see note above)
    url='https://github.com/aewallwi/calamity',  # Optional
    author='A. Ewall-Wice',  # Optional
    author_email='aaronew@berkeley.edu',  # Optional
    classifiers=[
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3 :: Only',
    ],
    keywords='21cm, cosmology, foregrounds, radio astronomy, cosmic dawn',
    package_dir={'calamity': 'calamity'},
    packages=['calamity'],
    python_requires='>=3.6, <4',
    install_requires=[
                      'pyuvdata>=2.1.5',
                      'tensorflow>=2.4.0',
                      'scipy',
                      'tqdm',
                      'hera_filters @ git+http://github.com/HERA-Team/hera_filters',
                      'tensorflow-addons',
                      ],
    include_package_data=True,
    scripts = ['scripts/calibrate_and_model_dpss.py'],
    package_data={'calamity': data_files},
    exclude = ['tests'],
    zip_safe = False,
    )
