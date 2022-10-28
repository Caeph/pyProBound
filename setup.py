from setuptools import setup, find_packages

setup(
    name='pyProBound',
    version='0.0.1',
    description='API for manipulate and score ProBound model',
    install_requires=["numpy", "pandas", "jpype1"],
    packages=find_packages(),
    package_data={"pyProBound": ["ProBound-jar-with-dependencies.jar", "*.json"]},
    license='GPLv3',
)