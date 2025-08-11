import unittest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from config import (
    WEIGHTS_DIR, MODEL_NAME, CONF_THRESH, MODEL_CONF, 
    CONTAINMENT_RATIO, PERSON_PAD_PX, PERSON_ALIASES, 
    REQUIRED_CLASSES, CLASS_SYNONYMS
)


class TestConfig(unittest.TestCase):
    
    def test_weights_config(self):
        """Test weights directory and model name configuration"""
        self.assertIsInstance(WEIGHTS_DIR, str)
        self.assertIsInstance(MODEL_NAME, str)
        self.assertTrue(len(WEIGHTS_DIR) > 0)
        self.assertTrue(len(MODEL_NAME) > 0)
        self.assertTrue(MODEL_NAME.endswith('.pt'))
    
    def test_threshold_config(self):
        """Test confidence threshold configuration"""
        self.assertIsInstance(CONF_THRESH, (int, float))
        self.assertGreaterEqual(CONF_THRESH, 0.0)
        self.assertLessEqual(CONF_THRESH, 1.0)
        
        self.assertIsInstance(MODEL_CONF, (int, float))
        self.assertGreaterEqual(MODEL_CONF, 0.0)
        self.assertLessEqual(MODEL_CONF, 1.0)
    
    def test_geometry_config(self):
        """Test geometry-related configuration"""
        self.assertIsInstance(CONTAINMENT_RATIO, (int, float))
        self.assertGreater(CONTAINMENT_RATIO, 0.0)
        self.assertLessEqual(CONTAINMENT_RATIO, 1.0)
        
        self.assertIsInstance(PERSON_PAD_PX, int)
        self.assertGreaterEqual(PERSON_PAD_PX, 0)
    
    def test_person_aliases(self):
        """Test person alias configuration"""
        self.assertIsInstance(PERSON_ALIASES, set)
        self.assertTrue(len(PERSON_ALIASES) > 0)
        
        # Check all aliases are strings
        for alias in PERSON_ALIASES:
            self.assertIsInstance(alias, str)
            self.assertTrue(len(alias) > 0)
    
    def test_required_classes(self):
        """Test required PPE classes configuration"""
        self.assertIsInstance(REQUIRED_CLASSES, set)
        self.assertTrue(len(REQUIRED_CLASSES) > 0)
        
        # Check all classes are strings
        for cls in REQUIRED_CLASSES:
            self.assertIsInstance(cls, str)
            self.assertTrue(len(cls) > 0)
    
    def test_class_synonyms(self):
        """Test class synonym configuration"""
        self.assertIsInstance(CLASS_SYNONYMS, dict)
        
        # Check all keys and values are strings
        for key, value in CLASS_SYNONYMS.items():
            self.assertIsInstance(key, str)
            self.assertIsInstance(value, str)
            self.assertTrue(len(key) > 0)
            self.assertTrue(len(value) > 0)


if __name__ == '__main__':
    unittest.main()
