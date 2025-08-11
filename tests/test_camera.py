import unittest
import sys
import os
from unittest.mock import Mock, patch

# Add src to path to import modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from camera import VideoSource, should_infer


class TestCamera(unittest.TestCase):
    
    def test_should_infer(self):
        """Test should_infer function logic"""
        # Test inference every frame
        self.assertTrue(should_infer(0, 1))
        self.assertTrue(should_infer(1, 1))
        self.assertTrue(should_infer(2, 1))
        
        # Test inference every 2 frames
        self.assertTrue(should_infer(0, 2))
        self.assertFalse(should_infer(1, 2))
        self.assertTrue(should_infer(2, 2))
        self.assertFalse(should_infer(3, 2))
        
        # Test inference every 5 frames
        self.assertTrue(should_infer(0, 5))
        self.assertFalse(should_infer(1, 5))
        self.assertFalse(should_infer(4, 5))
        self.assertTrue(should_infer(5, 5))
        
        # Test default parameter
        self.assertTrue(should_infer(0))
        self.assertTrue(should_infer(1))
    
    @patch('cv2.VideoCapture')
    def test_video_source_initialization(self, mock_cv2):
        """Test VideoSource initialization"""
        mock_cap = Mock()
        mock_cv2.return_value = mock_cap
        
        # Test default initialization
        vs = VideoSource()
        mock_cv2.assert_called_once_with(0)
        
        # Test custom source
        vs = VideoSource(1)
        mock_cv2.assert_called_with(1)
        
        # Test with width and height
        vs = VideoSource(0, 640, 480)
        mock_cap.set.assert_any_call(mock_cv2.return_value.CAP_PROP_FRAME_WIDTH, 640)
        mock_cap.set.assert_any_call(mock_cv2.return_value.CAP_PROP_FRAME_HEIGHT, 480)
    
    @patch('cv2.VideoCapture')
    def test_video_source_read(self, mock_cv2):
        """Test VideoSource read method"""
        mock_cap = Mock()
        mock_cv2.return_value = mock_cap
        
        vs = VideoSource()
        
        # Mock return values
        mock_cap.read.return_value = (True, "frame_data")
        
        ok, frame = vs.read()
        self.assertTrue(ok)
        self.assertEqual(frame, "frame_data")
        mock_cap.read.assert_called_once()
    
    @patch('cv2.VideoCapture')
    def test_video_source_release(self, mock_cv2):
        """Test VideoSource release method"""
        mock_cap = Mock()
        mock_cv2.return_value = mock_cap
        
        vs = VideoSource()
        vs.release()
        
        mock_cap.release.assert_called_once()
    
    def test_video_source_context_manager(self):
        """Test VideoSource can be used as context manager"""
        with patch('cv2.VideoCapture') as mock_cv2:
            mock_cap = Mock()
            mock_cv2.return_value = mock_cap
            
            with VideoSource() as vs:
                self.assertIsInstance(vs, VideoSource)
            
            mock_cap.release.assert_called_once()


if __name__ == '__main__':
    unittest.main()
