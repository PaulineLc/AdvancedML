import unittest
import pandas as pd

from Assignment.classifier import TextClassifier
from Assignment.document import Document
from Assignment.dataset import Dataset


class ClassifierUpdateSimilarities(unittest.TestCase):

    def test_empty_similarities(self):
        # creating mock database
        Dataset.article_data = pd.DataFrame({'doc_id': [99997,
                                                        99997,
                                                        99997,
                                                        99997,
                                                        99996,
                                                        99996,
                                                        99996,
                                                        99996,
                                                        99996,
                                                        99995,
                                                        99995,
                                                        99995,
                                                        99995,
                                                        99995],
                                             'term_id': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 6, 5, 11, 12],
                                             'nb_occurrences': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]})

        Dataset.article_labels = pd.DataFrame({'doc_id': [99997, 99996, 99995], 'class': ['car', 'car', 'animal']})

        # Create new classifier for document 1
        clf = TextClassifier(99997)

        self.assertEqual(clf.document, Document(99997))
        self.assertEqual(clf.similarities_dict, {})
        self.assertEqual(clf.sorted_similarities, None)

    def test_populate_dictionary(self):
        # creating mock database
        Dataset.article_data = pd.DataFrame({'doc_id': [99997,
                                                        99997,
                                                        99997,
                                                        99997,
                                                        99996,
                                                        99996,
                                                        99996,
                                                        99996,
                                                        99996,
                                                        99995,
                                                        99995,
                                                        99995,
                                                        99995,
                                                        99995],
                                             'term_id': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 6, 5, 11, 12],
                                             'nb_occurrences': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]})

        Dataset.article_labels = pd.DataFrame({'doc_id': [99997, 99996, 99995], 'class': ['car', 'car', 'animal']})

        Dataset.word_bank_size = 12

        TextClassifier.training_set = [99997, 99995]

        # Create new classifier for document 1
        clf = TextClassifier(99996)

        # update similarities
        clf.update_similarities_dict()

        # assert equality -- Can't compare the dictionaries directly because of the float they have aas value.
        # Instead, I check that they have the same keys, and that the keys contain a similar float value.
        self.assertEqual(set(clf.similarities_dict.keys()), {99997, 99995})
        self.assertAlmostEqual(clf.similarities_dict[99997], 0.0, places=1)
        self.assertAlmostEqual(clf.similarities_dict[99995], 0.4, places=1)

if __name__ == '__main__':
    unittest.main()
