import neurotabular
from neurotabular import NeuroTabularClassifier


def test_public_import_and_version():
    assert neurotabular.__version__ == "0.2.0"
    assert NeuroTabularClassifier.__name__ == "NeuroTabularClassifier"
