from distutils.core import setup

setup(
    name='weat',
    version='1.0',
    packages=['weat',],
    license='Creative Commons Attribution-Noncommercial-Share Alike license',
    long_description=open('README.md').read(),
    install_requires=[
    	'pandas',
    	'scikit-learn',
    	'numpy'
    ]
)