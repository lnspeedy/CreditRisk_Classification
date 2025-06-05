import os
import sys
import types
import pytest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

mock_modules = {
    'pandas': types.ModuleType('pandas'),
    'sklearn': types.ModuleType('sklearn'),
    'sklearn.metrics': types.ModuleType('sklearn.metrics'),
    'api.ml': types.ModuleType('api.ml'),
    'api.ml.model': types.ModuleType('api.ml.model'),
    'api.ml.preprocess': types.ModuleType('api.ml.preprocess'),
    'api.config': types.ModuleType('api.config'),
    'uvicorn': types.ModuleType('uvicorn'),
}
for name, module in mock_modules.items():
    sys.modules.setdefault(name, module)

# Provide minimal attributes used during import
sys.modules['sklearn.metrics'].accuracy_score = lambda *a, **k: None
sys.modules['sklearn.metrics'].precision_score = lambda *a, **k: None
sys.modules['sklearn.metrics'].recall_score = lambda *a, **k: None
sys.modules['api.config'].Settings = object
sys.modules['api.ml.model'].CreditRisk_Classifier = object
sys.modules['api.ml.preprocess'].CreditRisk_Preprocess_Predict = object
sys.modules['api.ml.preprocess'].get_input_column_names = lambda: []

from api.main import ClientProfile


def test_invalid_model_version_raises_value_error():
    with pytest.raises(ValueError):
        ClientProfile(model_version="invalid")
