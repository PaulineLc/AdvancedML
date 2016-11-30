import unittest
import pandas as pd

from Assignment.classifier import TextClassifier
from Assignment.document import Document
from Assignment.dataset import Dataset


class ClassifierUpdateSimilarities(unittest.TestCase):

    def test_empty_similarities(self):
        # creating mock database
        Dataset.article_data = pd.DataFrame({'doc_id': [1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3],
                                             'term_id': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 6, 5, 11, 12],
                                             'nb_occurrence': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]})

        Dataset.article_labels = pd.DataFrame({'doc_id': [1, 2, 3], 'class': ['car', 'car', 'animal']})

        # Create new classifier for document 1
        clf = TextClassifier(1)

        self.assertEqual(clf.document, Document(1))
        self.assertEqual(clf.similarities_dict, {})
        self.assertEqual(clf.sorted_similarities, None)

    def test_populate_dictionary(self):
        # creating mock database
        Dataset.article_data = pd.DataFrame({'doc_id': [1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3],
                                             'term_id': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 6, 5, 11, 12],
                                             'nb_occurrences': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]})

        Dataset.article_labels = pd.DataFrame({'doc_id': [1, 2, 3], 'class': ['car', 'car', 'animal']})

        # Create new classifier for document 1
        clf = TextClassifier(2)

        # update similarities
        clf.update_similarities_dict([1, 3])

        # assert equality -- Can't compare the dictionaries directly because of the float they have aas value.
        # Instead, I check that they have the same keys, and that the keys contain a similar float value.
        self.assertEqual(set(clf.similarities_dict.keys()), {1, 3})
        self.assertAlmostEqual(clf.similarities_dict[1], 0.0, places=1)
        self.assertAlmostEqual(clf.similarities_dict[3], 0.4, places=1)

if __name__ == '__main__':
    unittest.main()
