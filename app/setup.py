from setuptools import setup, find_packages

with open('requirements.txt') as r:
    requirements = r.readlines()

setup(
    name='backend',
    version='0.1.0',
    packages=['backend'],
    install_requires=['requests',
        'importlib; python_version == "3.9"',
        ]
        +[l for l in requirements]
    )