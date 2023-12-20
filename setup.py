from setuptools import setup, find_packages


setup(
    name='proxskip',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'opt_methods @ git+https://github.com/konstmish/opt_methods.git',
        'jupyter',
        'matplotlib',
        'numpy',
        'ray',
        'seaborn',
        'sklearn'
    ]
)
