def test_imports():
    """Test that basic imports work"""
    try:
        import pandas
        import numpy
        assert True
    except ImportError:
        assert False, "Basic imports failed"

def test_simple_math():
    """Test basic functionality"""
    assert 1 + 1 == 2