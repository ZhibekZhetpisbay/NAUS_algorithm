from setuptools import setup, find_packages
with open("README.md", "r") as fh:
    long_description = fh.read()
setup(
    name='NAUS_code',
    version='0.1',
    packages=find_packages(),
    author='Z. Buribayev, A. Yerkos, Z. Zhetpisbay',
    author_email='zhibekzhetpisbay@gmail.com',
    description='Package with NAUS algorithm and learning model',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ZhibekZhetpisbay/NAUS_algorithm",
    install_requires=[
            'scikit-learn', 'numpy','pandas','matplotlib',
            'scipy','umap-learn','xgboost','lightgbm',
            'torch','imbalanced-learn'],
)
