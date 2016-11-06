import unittest

from Assignment.text_data import Document
from Assignment.dataset import TextData

class TestDocumentMethods((unittest.TestCase)):

    TextData.define_article_data('../Assignment/data/news_articles.mtx')
    TextData.define_article_labels('../Assignment/data/news_articles.labels')

    def test_get_category(self):
        d1 = Document(1)
        self.assertEqual(d1.get_category(), 'business')

        d2 = Document(511)
        self.assertEqual(d2.get_category(), 'politics')

        d3 = Document(928)
        self.assertEqual(d3.get_category(), 'sport')

        d4 = Document(1439)
        self.assertEqual(d4.get_category(), 'technology')


if __name__ == '__main__':
    unittest.main()