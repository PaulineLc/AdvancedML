import unittest
import pandas as pd

from Assignment.classifier import TextClassifier
from Assignment.document import Document
from Assignment.dataset import Dataset


class ClassifierClassify(unittest.TestCase):

    def test_classification(self):
        # creating mock database
        Dataset.article_data = pd.DataFrame({'doc_id': [1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3],
                                             'term_id': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 6, 5, 11, 12],
                                             'nb_occurrences': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]})

        Dataset.article_labels = pd.DataFrame({'doc_id': [1, 2, 3], 'class': ['car', 'car', 'animal']})

        # Set training set
        TextClassifier.training_set = Dataset.article_labels['doc_id']

        # Create new classifier for document 1
        clf = TextClassifier(2)

        # classify example
        predicted_class = clf.classify(2)

        self.assertEqual(clf.sorted_similarities[0][0], 3)  # test that the strongest similarity is with document 3
        self.assertEqual(clf.sorted_similarities[1][0], 1)  # test that the lowest similarity is with document 1
        self.assertAlmostEqual(clf.sorted_similarities[0][1], 0.4, places=1)  # check similarity value is correct
        self.assertAlmostEqual(clf.sorted_similarities[1][1], 0.0, places=1)  # check similarity value is correct
        self.assertEqual(predicted_class, 'animal')  # check it predicted the expected class

if __name__ == '__main__':
    unittest.main()
