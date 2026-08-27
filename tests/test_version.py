import neurotabular
from neurotabular import NeuroTabularClassifier


def test_public_import_and_version():
    assert neurotabular.__version__ == "0.1.0"
    assert NeuroTabularClassifier.__name__ == "NeuroTabularClassifier"
