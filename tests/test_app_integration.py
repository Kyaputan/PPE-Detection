import unittest
import sys
import os
from unittest.mock import Mock, patch, MagicMock

# Add src to path to import modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from app import main


class TestAppIntegration(unittest.TestCase):
    
    @patch('cv2.imshow')
    @patch('cv2.waitKey')
    @patch('cv2.destroyAllWindows')
    @patch('app.draw_person_status')
    @patch('app.draw_ppes')
    @patch('app.assign_ppes_to_persons')
    @patch('app.split_person_ppe')
    @patch('app.parse_detections')
    @patch('app.infer')
    @patch('app.load_model')
    @patch('app.VideoSource')
    def test_main_workflow(self, mock_video_source, mock_load_model, mock_infer, 
                          mock_parse_detections, mock_split_person_ppe, 
                          mock_assign_ppes, mock_draw_ppes, mock_draw_person_status,
                          mock_destroy, mock_wait_key, mock_imshow):
        """Test the complete main workflow"""
        # Mock video source
        mock_cam = Mock()
        mock_cam.read.side_effect = [
            (True, "frame1"),  # First frame
            (True, "frame2"),  # Second frame
            (False, None),     # End of video
        ]
        mock_cam.release = Mock()
        mock_video_source.return_value = mock_cam
        
        # Mock model loading
        mock_model = Mock()
        mock_class_names = {0: "person", 1: "mask"}
        mock_load_model.return_value = (mock_model, mock_class_names)
        
        # Mock inference
        mock_yolo_result = Mock()
        mock_infer.return_value = mock_yolo_result
        
        # Mock detection parsing
        mock_detections = [
            {"cls": "person", "cls_l": "person", "bbox": [0, 0, 20, 20], "conf": 0.9},
            {"cls": "mask", "cls_l": "mask", "bbox": [5, 5, 15, 15], "conf": 0.8},
        ]
        mock_parse_detections.return_value = mock_detections
        
        # Mock person/PPE splitting
        mock_persons = [mock_detections[0]]
        mock_ppes = [mock_detections[1]]
        mock_split_person_ppe.return_value = (mock_persons, mock_ppes)
        
        # Mock PPE assignment
        mock_person_results = [
            {
                "person": mock_persons[0],
                "found": {"mask"},
                "missing": []
            }
        ]
        mock_assign_ppes.return_value = mock_person_results
        
        # Mock wait key to exit after first frame
        mock_wait_key.return_value = ord('q')
        
        # Run main function
        main()
        
        # Verify model was loaded
        mock_load_model.assert_called_once()
        
        # Verify video source was created
        mock_video_source.assert_called_once_with(0)
        
        # Verify inference was called for first frame (frame_idx = 0, infer_every_n = 5)
        mock_infer.assert_called_once_with(mock_model, "frame1")
        
        # Verify detections were parsed
        mock_parse_detections.assert_called_once_with(mock_yolo_result, mock_class_names)
        
        # Verify person/PPE splitting
        mock_split_person_ppe.assert_called_once_with(mock_detections)
        
        # Verify PPE assignment
        mock_assign_ppes.assert_called_once_with(mock_persons, mock_ppes)
        
        # Verify drawing functions were called
        mock_draw_ppes.assert_called_once_with("frame1", mock_ppes)
        mock_draw_person_status.assert_called_once_with("frame1", mock_person_results)
        
        # Verify display
        mock_imshow.assert_called_once_with("PPE Detection (Per Person)", "frame1")
        
        # Verify cleanup
        mock_cam.release.assert_called_once()
        mock_destroy.assert_called_once()
    
    @patch('cv2.imshow')
    @patch('cv2.waitKey')
    @patch('cv2.destroyAllWindows')
    @patch('app.draw_person_status')
    @patch('app.draw_ppes')
    @patch('app.assign_ppes_to_persons')
    @patch('app.split_person_ppe')
    @patch('app.parse_detections')
    @patch('app.infer')
    @patch('app.load_model')
    @patch('app.VideoSource')
    def test_main_skip_inference_frames(self, mock_video_source, mock_load_model, mock_infer, 
                                       mock_parse_detections, mock_split_person_ppe, 
                                       mock_assign_ppes, mock_draw_ppes, mock_draw_person_status,
                                       mock_destroy, mock_wait_key, mock_imshow):
        """Test that inference is skipped for non-inference frames"""
        # Mock video source with more frames
        mock_cam = Mock()
        mock_cam.read.side_effect = [
            (True, "frame1"),  # Frame 0 - should infer
            (True, "frame2"),  # Frame 1 - should skip
            (True, "frame3"),  # Frame 2 - should skip
            (True, "frame4"),  # Frame 3 - should skip
            (True, "frame5"),  # Frame 4 - should skip
            (True, "frame6"),  # Frame 5 - should infer
            (False, None),     # End of video
        ]
        mock_cam.release = Mock()
        mock_video_source.return_value = mock_cam
        
        # Mock model loading
        mock_model = Mock()
        mock_class_names = {0: "person", 1: "mask"}
        mock_load_model.return_value = (mock_model, mock_class_names)
        
        # Mock inference results
        mock_yolo_result = Mock()
        mock_infer.return_value = mock_yolo_result
        
        # Mock detection parsing
        mock_detections = [
            {"cls": "person", "cls_l": "person", "bbox": [0, 0, 20, 20], "conf": 0.9},
        ]
        mock_parse_detections.return_value = mock_detections
        
        # Mock person/PPE splitting
        mock_persons = [mock_detections[0]]
        mock_ppes = []
        mock_split_person_ppe.return_value = (mock_persons, mock_ppes)
        
        # Mock PPE assignment
        mock_person_results = [
            {
                "person": mock_persons[0],
                "found": set(),
                "missing": ["mask"]
            }
        ]
        mock_assign_ppes.return_value = mock_person_results
        
        # Mock wait key to exit after processing all frames
        mock_wait_key.return_value = ord('q')
        
        # Run main function
        main()
        
        # Verify inference was called only for frames 0 and 5 (infer_every_n = 5)
        expected_infer_calls = [
            (mock_model, "frame1"),  # Frame 0
            (mock_model, "frame6"),  # Frame 5
        ]
        mock_infer.assert_has_calls(expected_infer_calls)
        self.assertEqual(mock_infer.call_count, 2)
        
        # Verify other functions were called for each frame
        self.assertEqual(mock_draw_ppes.call_count, 6)
        self.assertEqual(mock_draw_person_status.call_count, 6)
        self.assertEqual(mock_imshow.call_count, 6)
    
    @patch('cv2.imshow')
    @patch('cv2.waitKey')
    @patch('cv2.destroyAllWindows')
    @patch('app.draw_person_status')
    @patch('app.draw_ppes')
    @patch('app.assign_ppes_to_persons')
    @patch('app.split_person_ppe')
    @patch('app.parse_detections')
    @patch('app.infer')
    @patch('app.load_model')
    @patch('app.VideoSource')
    def test_main_video_failure(self, mock_video_source, mock_load_model, mock_infer, 
                               mock_parse_detections, mock_split_person_ppe, 
                               mock_assign_ppes, mock_draw_ppes, mock_draw_person_status,
                               mock_destroy, mock_wait_key, mock_imshow):
        """Test main function handles video failure gracefully"""
        # Mock video source that fails immediately
        mock_cam = Mock()
        mock_cam.read.return_value = (False, None)
        mock_cam.release = Mock()
        mock_video_source.return_value = mock_cam
        
        # Mock model loading
        mock_model = Mock()
        mock_class_names = {0: "person", 1: "mask"}
        mock_load_model.return_value = (mock_model, mock_class_names)
        
        # Run main function
        main()
        
        # Verify cleanup still happens
        mock_cam.release.assert_called_once()
        mock_destroy.assert_called_once()
        
        # Verify no processing occurred
        mock_infer.assert_not_called()
        mock_draw_ppes.assert_not_called()
        mock_draw_person_status.assert_not_called()


if __name__ == '__main__':
    unittest.main()
