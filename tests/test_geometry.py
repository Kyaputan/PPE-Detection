import unittest
import sys
import os

# Add src to path to import modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from geometry import area, intersection, containment_ratio, pad_box


class TestGeometry(unittest.TestCase):
    
    def test_area(self):
        """Test area calculation for bounding boxes"""
        # Test normal bounding box
        self.assertEqual(area([0, 0, 10, 10]), 100)
        
        # Test zero area
        self.assertEqual(area([0, 0, 0, 10]), 0)
        self.assertEqual(area([0, 0, 10, 0]), 0)
        
        # Test negative coordinates (should return 0)
        self.assertEqual(area([10, 10, 0, 0]), 0)
        
        # Test single point
        self.assertEqual(area([5, 5, 5, 5]), 0)
    
    def test_intersection(self):
        """Test intersection calculation between bounding boxes"""
        # Test overlapping boxes
        self.assertEqual(intersection([0, 0, 10, 10], [5, 5, 15, 15]), 25)
        
        # Test no intersection
        self.assertEqual(intersection([0, 0, 5, 5], [10, 10, 15, 15]), 0)
        self.assertEqual(intersection([0, 0, 5, 5], [5, 5, 10, 10]), 0)
        
        # Test contained box
        self.assertEqual(intersection([0, 0, 10, 10], [2, 2, 8, 8]), 36)
        
        # Test edge touching
        self.assertEqual(intersection([0, 0, 5, 5], [5, 0, 10, 5]), 0)
    
    def test_containment_ratio(self):
        """Test containment ratio calculation"""
        # Test fully contained
        self.assertAlmostEqual(containment_ratio([2, 2, 8, 8], [0, 0, 10, 10]), 1.0)
        
        # Test partially contained
        self.assertAlmostEqual(containment_ratio([0, 0, 5, 5], [0, 0, 10, 10]), 0.25)
        
        # Test no containment
        self.assertEqual(containment_ratio([10, 10, 15, 15], [0, 0, 5, 5]), 0.0)
        
        # Test zero area inner box
        self.assertEqual(containment_ratio([5, 5, 5, 5], [0, 0, 10, 10]), 0.0)
    
    def test_pad_box(self):
        """Test bounding box padding"""
        # Test normal padding
        self.assertEqual(pad_box([0, 0, 10, 10], 5), [-5, -5, 15, 15])
        
        # Test default padding
        self.assertEqual(pad_box([0, 0, 10, 10]), [-10, -10, 20, 20])
        
        # Test zero padding
        self.assertEqual(pad_box([0, 0, 10, 10], 0), [0, 0, 10, 10])
        
        # Test negative padding (shrinking)
        self.assertEqual(pad_box([0, 0, 10, 10], -2), [2, 2, 8, 8])


if __name__ == '__main__':
    unittest.main()
