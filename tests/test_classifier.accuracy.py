import unittest
import pandas as pd

from Assignment.classifier import TextClassifier
from Assignment.dataset import Dataset


class ClassifierClassify(unittest.TestCase):

    def test_classification_training_set(self):
        # creating mock database
        Dataset.article_data = pd.DataFrame({'doc_id': [1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3],
                                             'term_id': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 6, 5, 11, 12],
                                             'nb_occurrences': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]})

        Dataset.article_labels = pd.DataFrame({'doc_id': [1, 2, 3], 'class': ['car', 'car', 'animal']})

        accuracy = TextClassifier.calculate_accuracy(0, 0.67)

        if 2 in list(TextClassifier.test_set) or 3 in list(TextClassifier.test_set):
            self.assertEqual(accuracy, (0, 0))
        else:
            self.assertEqual(accuracy, (1, 1))

    def test_classification_cross_validation(self):
        # creating mock database
        Dataset.article_data = pd.DataFrame({'doc_id': [1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3],
                                             'term_id': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 6, 5, 11, 12],
                                             'nb_occurrences': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]})

        Dataset.article_labels = pd.DataFrame({'doc_id': [1, 2, 3], 'class': ['car', 'car', 'animal']})

        # compute accuracy
        accuracy = TextClassifier.calculate_accuracy(1, 3)

        self.assertAlmostEqual(accuracy[0], 0.33, places=1)
        self.assertAlmostEqual(accuracy[1], 0.33, places=1)

    def test_classification_default_input(self):
        # creating mock database
        Dataset.article_data = pd.DataFrame({'doc_id': [1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3],
                                             'term_id': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 6, 5, 11, 12],
                                             'nb_occurrences': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]})

        Dataset.article_labels = pd.DataFrame({'doc_id': [1, 2, 3], 'class': ['car', 'car', 'animal']})

        # compute accuracy
        accuracy = TextClassifier.calculate_accuracy()

        self.assertAlmostEqual(accuracy[0], 0.33, places=1)
        self.assertAlmostEqual(accuracy[1], 0.33, places=1)

    def test_classification_default_with_traiing_set(self):
        # creating mock database
        Dataset.article_data = pd.DataFrame({'doc_id': [1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3],
                                             'term_id': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 6, 5, 11, 12],
                                             'nb_occurrences': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]})

        Dataset.article_labels = pd.DataFrame({'doc_id': [1, 2, 3], 'class': ['car', 'car', 'animal']})

        # compute accuracy
        accuracy = TextClassifier.calculate_accuracy(0)

        if 2 in list(TextClassifier.test_set) or 3 in list(TextClassifier.test_set):
            self.assertEqual(accuracy, (0, 0))
        else:
            self.assertEqual(accuracy, (1, 1))


if __name__ == '__main__':
    unittest.main()
