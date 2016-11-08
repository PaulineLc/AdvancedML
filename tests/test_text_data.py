import unittest

from Assignment.text_data import Document
from Assignment.dataset import TextData

class TestDocumentMethods(unittest.TestCase):

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

    def test_create_bag_of_words(self):
        d1 = Document(666)
        expected_dic = {4864: 1, 3073: 1, 4098: 1, 4099: 1, 2276: 1, 1030: 1, 2567: 1, 268: 3, 3853: 3, 3598: 1, 387: 1, 3604: 1, 2070: 1, 345: 1, 792: 1, 537: 1, 3359: 1, 3617: 1, 2342: 1, 3111: 1, 1322: 1, 1817: 1, 557: 1, 558: 1, 816: 1, 2865: 8, 2537: 2, 3125: 2, 4150: 1, 3897: 1, 2911: 3, 3902: 2, 96: 1, 3139: 2, 2627: 1, 48: 1, 1607: 1, 1609: 2, 842: 3, 3659: 1, 844: 2, 4429: 1, 3919: 1, 2129: 1, 2943: 1, 4691: 3, 2133: 3, 2134: 1, 1081: 1, 2393: 4, 346: 1, 3907: 1, 869: 4, 4192: 1, 609: 4, 4283: 2, 3684: 1, 2917: 1, 2967: 2, 3693: 1, 2670: 3, 3517: 1, 3511: 1, 2561: 1, 375: 1, 122: 2, 2428: 1, 1149: 1, 1685: 1, 131: 1, 298: 2, 2698: 1, 3735: 1, 3987: 1, 3989: 5, 3222: 1, 4247: 1, 2200: 1, 1693: 1, 1950: 1, 1695: 1, 4513: 1, 1187: 1, 2758: 2, 1625: 2, 1191: 1, 4778: 1, 199: 1, 2973: 2, 3782: 3, 1975: 1, 3145: 1, 1720: 1, 4025: 5, 955: 1, 4541: 1, 959: 1, 3520: 1, 1217: 1, 196: 1, 2757: 1, 1478: 1, 3015: 1, 3528: 1, 1399: 1, 3023: 1, 1234: 1, 1747: 1, 2262: 1, 727: 1, 3801: 1, 4826: 2, 4827: 1, 3749: 1, 1248: 1, 2786: 1, 995: 1, 3556: 1, 1251: 1, 2534: 1, 488: 1, 3817: 1, 2062: 2, 2599: 1, 2540: 1, 1261: 1, 2941: 1, 1521: 1, 757: 4, 1273: 3, 1530: 1, 119: 1, 252: 2, 3626: 1, 4863: 2}
        self.assertEqual(d1.create_bag_of_words(), expected_dic)


if __name__ == '__main__':
    unittest.main()