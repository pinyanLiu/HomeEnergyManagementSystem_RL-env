from . import analysis
import unittest

class Test(unittest.TestCase):
    def test_calScheduleSimilarity1(self):
        a = [1,1,1,1,1,0,0,0,0,0]
        b = [1,0,0,1,1,1,1,0,0,0]
        self.assertEqual(analysis.calScheduleSimilarity(a,b),0.6)
    def test_calScheduleSimilarity2(self):
        a = [1,1,1,1,1,0,0,0,0,0]
        b = [1,1,1,0,1,1,1,0,0,0]
        self.assertEqual(analysis.calScheduleSimilarity(a,b),0.7)
    def test_calScheduleSimilarity3(self):
        a = [1,0,0,0,0,0,0,0,0,0]
        b = [1,0,0,1,1,1,1,0,0,0]
        self.assertEqual(analysis.calScheduleSimilarity(a,b),0.6)




if __name__ == '__main__':
    unittest.main()