from setuptools import setup

setup(
    name='boyardeep',
    version='0.1',
    description='Build canned lasagne models',
    author='Brian McFee',
    author_email='brian.mcfee@nyu.edu',
    url='http://github.com/bmcfee/augment_music',
    download_url='http://github.com/bmcfee/augment_music/releases',
    long_description="""\
    Build canned lasagne models
    """,
    packages=['boyardeep'],
    classifiers=[
        "License :: OSI Approved :: ISC License (ISCL)",
        "Programming Language :: Python",
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Multimedia :: Text Processing :: Markup",
    ],
    keywords='web template',
    license='ISC',
    install_requires=[
        'lasagne',
        'six',
        'jsonpickle',
        'pandas',
        'theano',
        'scikit-learn'
    ],
)
