```python
from setuptools import setup, find_packages
import os

# Function to read the README.md file for the long description
def read_readme():
    try:
        with open(os.path.join(os.path.abspath(os.path.dirname(__file__)), 'README.md'), encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        return "A Pluggable Geometric Oracle Library / Geometric Reasoning Domain-Specific Language (DSL) prototype for SpatialEvo."

setup(
    name='geometric-reasoning-dsl-prototype',
    version='0.1.0',
    author='SpatialEvo Lead AI Developer',
    author_email='ai.dev@spatievo.com',
    description='A prototype for a Pluggable Geometric Oracle Library / Geometric Reasoning DSL to reduce coupling in SpatialEvo.',
    long_description=read_readme(),
    long_description_content_type='text/markdown',
    url='https://github.com/SpatialEvo/geometric-reasoning-dsl-prototype', # Placeholder URL
    project_urls={
        'Bug Tracker': 'https://github.com/SpatialEvo/geometric-reasoning-dsl-prototype/issues',
        'Source Code': 'https://github.com/SpatialEvo/geometric-reasoning-dsl-prototype',
    },
    packages=find_packages(where='src'),  # Automatically find packages in the 'src' directory
    package_dir={'': 'src'},  # Tell setuptools that packages are under 'src'
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',  # Assuming MIT License
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Operating System :: OS Independent',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'Topic :: Utilities',
    ],
    python_requires='>=3.8',
    install_requires=[
        'numpy>=1.20.0',  # Essential for numerical operations and geometric calculations
        'scipy>=1.7.0',   # Useful for advanced spatial algorithms and data structures
        # Add other core dependencies as the prototype evolves and integrates specific external libraries.
        # Examples might include:
        # 'trimesh>=3.9.0',   # If complex mesh processing is offloaded to a library
        # 'open3d>=0.14.0',   # For point cloud and general 3D data handling
        # 'pyvista>=0.32.0',  # For visualization or more advanced mesh operations
    ],
    extras_require={
        'dev': [
            'pytest>=7.0.0',
            'pytest-cov>=3.0.0',
            'flake8>=4.0.0',
            'black>=22.3.0',
            'isort>=5.10.0',
            'mypy>=0.950',
        ],
        'docs': [
            'sphinx>=5.0.0',
            'sphinx_rtd_theme>=1.0.0',
            'sphinx-autodoc-typehints>=1.18.0',
        ],
    },
    # If the project includes non-code files (e.g., example scene data, configs),
    # they can be specified here.
    # include_package_data=True,
    # package_data={
    #     'geometric_reasoning_dsl_prototype': ['data/*.obj', 'data/*.ply'],
    # },
)
```