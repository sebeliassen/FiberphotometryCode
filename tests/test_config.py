import pytest

from fiberphotometry import config


def test_data_patterns_structure():
    assert isinstance(config.DATA_PATTERNS, dict)
    # should have at least 'cpt' and 'oft'
    for key in ('cpt', 'oft'):
        assert key in config.DATA_PATTERNS
        patterns = config.DATA_PATTERNS[key]
        assert isinstance(patterns, dict)
        for name, spec in patterns.items():
            assert 'glob' in spec and isinstance(spec['glob'], str)
            assert 'kwargs' in spec and isinstance(spec['kwargs'], dict)


def test_letter_to_freqs():
    assert isinstance(config.LETTER_TO_FREQS, dict)
    for k, v in config.LETTER_TO_FREQS.items():
        assert isinstance(k, str)
        assert isinstance(v, str)


def test_filter_columns_pattern():
    assert hasattr(config, 'FILTER_COLUMNS_PATTERN')
    assert isinstance(config.FILTER_COLUMNS_PATTERN, str)