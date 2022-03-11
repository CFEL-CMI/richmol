from pkg_resources import get_distribution, DistributionNotFound

try:
    distr = get_distribution(__name__)
    __descr__ = {key : "".join(getattr(distr, key)) for key in (
        "version", "location", "project_name", "extras", "py_version", "platform"
        )}
    __version__ = __descr__["version"]
except DistributionNotFound:
    pass

#"""
#Concrete query strategy classes.
#"""
#from __future__ import absolute_import
##from .poten_from_tf import Potential

#__all__ = [
#    'Potential'
#]
