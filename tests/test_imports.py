import importlib
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

def test_imports():
    modules = [
        'api.main',
        'api.ml.model',
        'api.ml.preprocess',
    ]
    for m in modules:
        importlib.import_module(m)

