import unittest
import sys
import os
from unittest.mock import Mock, patch, MagicMock

# Add src to path to import modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from detection import load_model, infer


class TestDetection(unittest.TestCase):
    
    @patch('os.path.join')
    @patch('ultralytics.YOLO')
    def test_load_model(self, mock_yolo, mock_path_join):
        """Test model loading function"""
        # Mock path joining
        mock_path_join.return_value = "weights/PPE.pt"
        
        # Mock YOLO model
        mock_model = Mock()
        mock_model.names = {0: "person", 1: "mask", 2: "glove"}
        mock_yolo.return_value = mock_model
        
        model, class_names = load_model()
        
        # Check YOLO was called with correct path
        mock_yolo.assert_called_once_with("weights/PPE.pt")
        
        # Check model is returned
        self.assertEqual(model, mock_model)
        
        # Check class names are processed correctly
        expected_names = {0: "person", 1: "mask", 2: "glove"}
        self.assertEqual(class_names, expected_names)
    
    @patch('os.path.join')
    @patch('ultralytics.YOLO')
    def test_load_model_with_whitespace(self, mock_yolo, mock_path_join):
        """Test model loading with whitespace in class names"""
        # Mock path joining
        mock_path_join.return_value = "weights/PPE.pt"
        
        # Mock YOLO model with whitespace in names
        mock_model = Mock()
        mock_model.names = {0: " person ", 1: " mask ", 2: " glove "}
        mock_yolo.return_value = mock_model
        
        model, class_names = load_model()
        
        # Check class names are stripped
        expected_names = {0: "person", 1: "mask", 2: "glove"}
        self.assertEqual(class_names, expected_names)
    
    def test_infer(self):
        """Test inference function"""
        # Mock model
        mock_model = Mock()
        
        # Mock frame
        mock_frame = Mock()
        
        # Mock YOLO result
        mock_result = Mock()
        mock_model.return_value = [mock_result]
        
        result = infer(mock_model, mock_frame)
        
        # Check model was called with correct parameters
        mock_model.assert_called_once_with(mock_frame, conf=0.7)
        
        # Check result is returned
        self.assertEqual(result, mock_result)
    
    @patch('detection.MODEL_CONF')
    def test_infer_uses_config_threshold(self, mock_model_conf):
        """Test that inference uses the configured confidence threshold"""
        mock_model_conf.value = 0.8
        
        # Mock model
        mock_model = Mock()
        mock_frame = Mock()
        mock_result = Mock()
        mock_model.return_value = [mock_result]
        
        result = infer(mock_model, mock_frame)
        
        # Check model was called with config threshold
        mock_model.assert_called_once_with(mock_frame, conf=0.8)
    
    def test_infer_returns_first_result(self):
        """Test that inference returns the first result from the list"""
        # Mock model
        mock_model = Mock()
        mock_frame = Mock()
        
        # Mock multiple results
        mock_result1 = Mock()
        mock_result2 = Mock()
        mock_model.return_value = [mock_result1, mock_result2]
        
        result = infer(mock_model, mock_frame)
        
        # Should return first result
        self.assertEqual(result, mock_result1)


if __name__ == '__main__':
    unittest.main()
