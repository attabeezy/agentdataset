import pytest
from unittest.mock import MagicMock, patch
from agentdataset.core.extractor import Extractor

def test_extractor_init():
    ext = Extractor(model="test-model")
    assert ext.model == "test-model"

@patch('agentdataset.core.extractor.fitz')
def test_pdf_to_markdown(mock_fitz):
    mock_doc = MagicMock()
    mock_page = MagicMock()
    mock_page.get_text.return_value = "page text"
    mock_doc.__iter__.return_value = [mock_page]
    mock_fitz.open.return_value = mock_doc
    
    ext = Extractor()
    md = ext.pdf_to_markdown("test.pdf")
    assert md == "page text"

def test_extract_parameters():
    ext = Extractor()
    text = "The mean is 10.5 and the standard deviation is 2.1."
    params = ext.extract_parameters(text, "test_source")
    
    assert len(params.variables) == 1
    var = params.variables["var_1"]
    assert var.mean == 10.5
    assert var.std == 2.1
    assert params.meta.source == "test_source"

def test_check_statistical_density():
    ext = Extractor()
    text = "Word word 123 456 word"
    density = ext.check_statistical_density(text)
    # 2 numbers / 5 items = 0.4
    # Regex findall r'\w+': ['Word', 'word', '123', '456', 'word'] -> 5
    # Regex findall r'\d+': ['123', '456'] -> 2
    assert density == 0.4
