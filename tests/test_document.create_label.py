import unittest

from Assignment.document import Document
from Assignment.dataset import Dataset


class TestDocumentLabel(unittest.TestCase):

    def test_get_category(self):
        Dataset.define_article_data('../Assignment/data/news_articles.mtx')
        Dataset.define_article_labels('../Assignment/data/news_articles.labels')

        d1 = Document(1)
        self.assertEqual(d1._create_label(), 'business')

        d2 = Document(511)
        self.assertEqual(d2._create_label(), 'politics')

        d3 = Document(928)
        self.assertEqual(d3._create_label(), 'sport')

        d4 = Document(1439)
        self.assertEqual(d4._create_label(), 'technology')


if __name__ == '__main__':
    unittest.main()