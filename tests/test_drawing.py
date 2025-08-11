import unittest
import sys
import os
from unittest.mock import Mock, patch

# Add src to path to import modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from drawing import draw_ppes, draw_person_status


class TestDrawing(unittest.TestCase):
    
    @patch('cv2.rectangle')
    @patch('cv2.putText')
    def test_draw_ppes(self, mock_put_text, mock_rectangle):
        """Test drawing PPE bounding boxes and labels"""
        frame = Mock()
        ppes = [
            {"cls": "mask", "bbox": [10, 20, 30, 40]},
            {"cls": "glove", "bbox": [50, 60, 70, 80]},
        ]
        
        draw_ppes(frame, ppes)
        
        # Check rectangle calls
        expected_rectangle_calls = [
            ((10, 20), (30, 40), (255, 255, 0), 1),
            ((50, 60), (70, 80), (255, 255, 0), 1),
        ]
        self.assertEqual(mock_rectangle.call_count, 2)
        mock_rectangle.assert_has_calls(expected_rectangle_calls)
        
        # Check text calls
        expected_text_calls = [
            (frame, "mask", (10, 15), 0, 0.5, (255, 255, 0), 1),
            (frame, "glove", (50, 55), 0, 0.5, (255, 255, 0), 1),
        ]
        self.assertEqual(mock_put_text.call_count, 2)
        mock_put_text.assert_has_calls(expected_text_calls)
    
    @patch('cv2.rectangle')
    @patch('cv2.putText')
    def test_draw_ppes_empty(self, mock_put_text, mock_rectangle):
        """Test drawing PPE when no PPE is detected"""
        frame = Mock()
        ppes = []
        
        draw_ppes(frame, ppes)
        
        # Should not call any drawing functions
        mock_rectangle.assert_not_called()
        mock_put_text.assert_not_called()
    
    @patch('cv2.rectangle')
    @patch('cv2.putText')
    def test_draw_person_status_ok(self, mock_put_text, mock_rectangle):
        """Test drawing person status when all PPE is present"""
        frame = Mock()
        person_results = [
            {
                "person": {"bbox": [0, 0, 20, 20]},
                "missing": [],  # No missing PPE
                "found": {"mask"}
            }
        ]
        
        draw_person_status(frame, person_results)
        
        # Check rectangle call (green for OK)
        mock_rectangle.assert_called_once_with((0, 0), (20, 20), (0, 255, 0), 2)
        
        # Check text call
        mock_put_text.assert_called_once_with(
            frame, "Person 1: OK", (0, -10), 0, 0.7, (0, 255, 0), 2
        )
    
    @patch('cv2.rectangle')
    @patch('cv2.putText')
    def test_draw_person_status_missing(self, mock_put_text, mock_rectangle):
        """Test drawing person status when PPE is missing"""
        frame = Mock()
        person_results = [
            {
                "person": {"bbox": [0, 0, 20, 20]},
                "missing": ["mask", "glove"],  # Missing PPE
                "found": set()
            }
        ]
        
        draw_person_status(frame, person_results)
        
        # Check rectangle call (red for missing PPE)
        mock_rectangle.assert_called_once_with((0, 0), (20, 20), (0, 0, 255), 2)
        
        # Check text call
        mock_put_text.assert_called_once_with(
            frame, "Person 1: Missing: mask, glove", (0, -10), 0, 0.7, (0, 0, 255), 2
        )
    
    @patch('cv2.rectangle')
    @patch('cv2.putText')
    def test_draw_person_status_multiple(self, mock_put_text, mock_rectangle):
        """Test drawing multiple person statuses"""
        frame = Mock()
        person_results = [
            {
                "person": {"bbox": [0, 0, 20, 20]},
                "missing": [],
                "found": {"mask"}
            },
            {
                "person": {"bbox": [30, 30, 50, 50]},
                "missing": ["glove"],
                "found": {"mask"}
            }
        ]
        
        draw_person_status(frame, person_results)
        
        # Check rectangle calls
        expected_rectangle_calls = [
            ((0, 0), (20, 20), (0, 255, 0), 2),  # Green for OK
            ((30, 30), (50, 50), (0, 0, 255), 2),  # Red for missing
        ]
        self.assertEqual(mock_rectangle.call_count, 2)
        mock_rectangle.assert_has_calls(expected_rectangle_calls)
        
        # Check text calls
        expected_text_calls = [
            (frame, "Person 1: OK", (0, -10), 0, 0.7, (0, 255, 0), 2),
            (frame, "Person 2: Missing: glove", (30, 20), 0, 0.7, (0, 0, 255), 2),
        ]
        self.assertEqual(mock_put_text.call_count, 2)
        mock_put_text.assert_has_calls(expected_text_calls)
    
    @patch('cv2.rectangle')
    @patch('cv2.putText')
    def test_draw_person_status_empty(self, mock_put_text, mock_rectangle):
        """Test drawing person status when no persons are detected"""
        frame = Mock()
        person_results = []
        
        draw_person_status(frame, person_results)
        
        # Should not call any drawing functions
        mock_rectangle.assert_not_called()
        mock_put_text.assert_not_called()


if __name__ == '__main__':
    unittest.main()
