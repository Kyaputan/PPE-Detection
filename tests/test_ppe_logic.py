import unittest
import sys
import os
from unittest.mock import Mock

# Add src to path to import modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from ppe_logic import (
    normalize_name, parse_detections, split_person_ppe, assign_ppes_to_persons
)


class TestPPELogic(unittest.TestCase):
    
    def test_normalize_name(self):
        """Test name normalization function"""
        # Test basic normalization
        self.assertEqual(normalize_name("MASK"), "mask")
        self.assertEqual(normalize_name("  Glove  "), "glove")
        self.assertEqual(normalize_name("PPE_OVERALL"), "ppe_coverall")
        
        # Test synonym mapping
        self.assertEqual(normalize_name("ppe_overall"), "ppe_coverall")
        
        # Test unknown names
        self.assertEqual(normalize_name("unknown"), "unknown")
        self.assertEqual(normalize_name(""), "")
    
    def test_parse_detections(self):
        """Test YOLO detection parsing"""
        # Mock YOLO result
        mock_result = Mock()
        mock_result.boxes.data.tolist.return_value = [
            [10, 20, 30, 40, 0.8, 0],  # mask with high confidence
            [50, 60, 70, 80, 0.3, 1],  # person with low confidence (should be filtered)
            [90, 100, 110, 120, 0.9, 2],  # glove with high confidence
        ]
        
        class_names = {0: "mask", 1: "person", 2: "glove"}
        
        dets = parse_detections(mock_result, class_names)
        
        # Should filter out low confidence detection
        self.assertEqual(len(dets), 2)
        
        # Check first detection (mask)
        self.assertEqual(dets[0]["cls"], "mask")
        self.assertEqual(dets[0]["cls_l"], "mask")
        self.assertEqual(dets[0]["bbox"], [10, 20, 30, 40])
        self.assertEqual(dets[0]["conf"], 0.8)
        
        # Check second detection (glove)
        self.assertEqual(dets[1]["cls"], "glove")
        self.assertEqual(dets[1]["cls_l"], "glove")
        self.assertEqual(dets[1]["bbox"], [90, 100, 110, 120])
        self.assertEqual(dets[1]["conf"], 0.9)
    
    def test_split_person_ppe(self):
        """Test splitting detections into persons and PPE"""
        dets = [
            {"cls": "person", "cls_l": "person", "bbox": [0, 0, 10, 10], "conf": 0.9},
            {"cls": "mask", "cls_l": "mask", "bbox": [5, 5, 15, 15], "conf": 0.8},
            {"cls": "glove", "cls_l": "glove", "bbox": [20, 20, 30, 30], "conf": 0.7},
            {"cls": "human", "cls_l": "human", "bbox": [40, 40, 50, 50], "conf": 0.9},
        ]
        
        persons, ppes = split_person_ppe(dets)
        
        # Check persons
        self.assertEqual(len(persons), 2)
        person_classes = {p["cls_l"] for p in persons}
        self.assertEqual(person_classes, {"person", "human"})
        
        # Check PPE
        self.assertEqual(len(ppes), 2)
        ppe_classes = {p["cls_l"] for p in ppes}
        self.assertEqual(ppe_classes, {"mask", "glove"})
    
    def test_assign_ppes_to_persons(self):
        """Test assigning PPE to persons based on containment"""
        persons = [
            {"cls": "person", "cls_l": "person", "bbox": [0, 0, 20, 20], "conf": 0.9},
            {"cls": "human", "cls_l": "human", "bbox": [50, 50, 70, 70], "conf": 0.9},
        ]
        
        ppes = [
            {"cls": "mask", "cls_l": "mask", "bbox": [5, 5, 15, 15], "conf": 0.8},  # inside person 1
            {"cls": "glove", "cls_l": "glove", "bbox": [25, 25, 35, 35], "conf": 0.7},  # outside both
            {"cls": "mask", "cls_l": "mask", "bbox": [55, 55, 65, 65], "conf": 0.8},  # inside person 2
        ]
        
        results = assign_ppes_to_persons(persons, ppes)
        
        self.assertEqual(len(results), 2)
        
        # Check first person
        person1 = results[0]
        self.assertEqual(person1["person"]["cls_l"], "person")
        self.assertEqual(person1["found"], {"mask"})
        self.assertEqual(set(person1["missing"]), {"mask"})  # Only mask is required in config
        
        # Check second person
        person2 = results[1]
        self.assertEqual(person2["person"]["cls_l"], "human")
        self.assertEqual(person2["found"], {"mask"})
        self.assertEqual(set(person2["missing"]), set())  # No missing PPE
    
    def test_assign_ppes_to_persons_no_ppe(self):
        """Test assigning PPE when no PPE is detected"""
        persons = [
            {"cls": "person", "cls_l": "person", "bbox": [0, 0, 20, 20], "conf": 0.9},
        ]
        
        ppes = []  # No PPE detected
        
        results = assign_ppes_to_persons(persons, ppes)
        
        self.assertEqual(len(results), 1)
        person = results[0]
        self.assertEqual(person["found"], set())
        self.assertEqual(set(person["missing"]), {"mask"})  # All required PPE missing
    
    def test_assign_ppes_to_persons_no_persons(self):
        """Test assigning PPE when no persons are detected"""
        persons = []
        ppes = [
            {"cls": "mask", "cls_l": "mask", "bbox": [5, 5, 15, 15], "conf": 0.8},
        ]
        
        results = assign_ppes_to_persons(persons, ppes)
        
        self.assertEqual(len(results), 0)


if __name__ == '__main__':
    unittest.main()
