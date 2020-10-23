from distutils.core import setup

reqs = [
    "astropy",
    "numpy",
    "scipy",
    "matplotlib",
    "hdf5",
]

setup(
    name='mwa_pb_lookup',
    version='1.0',
    author=["John Morgan", "Tim Galvin"],
    description="This package provides an alternative, far quicker way of generating MWA primary beams compared with the standard mwa_pb code.",
    url="https://github.com/johnsmorgan/mwa_pb_lookup",
    long_description=open('README.md').read(),
    packages=['mwa_pb_lookup',],
    license='GNU General Public License v3.0',
    requires=reqs,
    scripts=["mwa_pb_lookup/lookup_beam.py",
             "mwa_pb_lookup/lookup_jones.py"],
)