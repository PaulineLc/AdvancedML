import unittest

from Assignment.document import Document
from Assignment.dataset import Dataset
import pandas as pd

class TestDocumentBagOfWords(unittest.TestCase):
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

    def test_create_bag_of_words(self):
        d1 = Document(99997)
        expected_dic = [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
        self.assertEqual(list(d1._create_bag_of_words()), expected_dic)


if __name__ == '__main__':
    unittest.main()