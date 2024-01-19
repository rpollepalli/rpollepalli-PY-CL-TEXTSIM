import unittest
from src.main.lab import cosSimExercise, jaccardExercise
class TestTextSimilarityExercises(unittest.TestCase):
    def test_cosine_sim(self):
        list1 = ['Natural language processing is fascinating', 'I\'m intrigued by the wonders of natural language processing']
        expected1 = 0.4743416490252569
        actual1 = cosSimExercise(list1)

        self.assertAlmostEqual(actual1, expected1)

        list2 = ['cat loves meat', 'cat hates freezer burn']
        expected2 = 0.2886751345948129
        actual2 = cosSimExercise(list2)
        
        self.assertAlmostEqual(actual2, expected2)

    def test_jaccard_index(self):
        self.assertGreaterEqual(jaccardExercise(), 0.3)

if __name__ == '__main__':
    unittest.main()