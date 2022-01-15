import os, sys
from os.path import join, dirname, abspath

def set_env():
    """This function does two things:
    1. It allows the code in cgn_framework to be imported from the experiments folder.
    2. It changes the current working directory to cgn_framework/. This is necessary,
    because the code present in this directory expects to be called from here."""
    sys.path.insert(0, dirname(dirname(abspath(__file__))))
    os.chdir(join(dirname(dirname(abspath(__file__))), "cgn_framework"))

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__
