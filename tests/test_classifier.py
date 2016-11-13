import unittest

from Assignment.classifier import TextClassifier
from Assignment.text_data import Document


class TestCosineCalculation(unittest.TestCase):

    def test_cosine1(self):
        # Create a document for the sentence "world cup winners 2014'
        # term id 1 = 'world', 2 = 'cup', 3 = 'winners', 4 = '2014', 5 = '2016'
        d1 = Document(1)
        d1.bag_of_words = {1: 1, 2: 1, 3: 1, 4: 1}

        # Create a document for the sentence "cup winners 2016"
        d2 = Document(2)
        d2.bag_of_words = {2: 1, 3: 1, 5: 1}

        # creating mock objects
        clf = TextClassifier(d1)
        clf.all_bags_of_words[2] = d2.bag_of_words
        cosine = clf.calculate_cosine(2)

        self.assertAlmostEqual(cosine, 0.577, places=3)

    def test_cosine2(self):
        # same experimental setup as above
        # term id: 1: 'jaguars', 2: 'are', 3: 'expensive', 4: 'cars', 5: 'a', 6: 'jaguar', 7: 'is',
        # 8: 'costly', 9: 'vehicle', 10: 'the', 11: 'feline', 12: 'animal'

        # d1 sentence: "jaguars are expensive cars"
        d1 = Document(1)
        d1.bag_of_words = {1:1, 2:1, 3:1, 4: 1}

        # d2 sentence: "a jaguar is a costly vehicle"
        d2 = Document(2)
        d2.bag_of_words = {5:1, 6:1, 7:1, 8:1, 9:1}

        # d2 sentence: "the jaguar, a feline animal"
        d3 = Document(3)
        d3.bag_of_words = {10:1, 6:1, 5:1, 11:1, 12:1}

        # creating mock objects
        clf = TextClassifier(d2)
        clf.all_bags_of_words[1] = d1.bag_of_words
        clf.all_bags_of_words[3] = d3.bag_of_words

        # calculating cosine
        cos_d1 = clf.calculate_cosine(1)
        cos_d3 = clf.calculate_cosine(3)

        self.assertAlmostEqual(cos_d1, 0.0, places=1)
        self.assertAlmostEqual(cos_d3, 0.4, places=1)

    def test_cosine3(self):
        # same experimental setup as above
        # term id 1:zoe, 2:is, 3:the, 4:prettiest, 5:cat, 6:that, 7:has, 8:ever, 9:been, 10:or, 11:will, 12:be, 13:my
        # 14:and, 15:she, 16:pretty, 17:very

        # d1 sentence: zoe is the prettiest cat that has ever been or will ever be
        d1 = Document(1)
        d1.bag_of_words = {1:1, 2:1, 3:1, 4:1, 5:1, 6:1, 7:1, 8:1, 9:1, 10:1, 11:1, 12:1}

        # d2 sentence: zoe is my cat and she is very pretty
        d2 = Document(2)
        d2.bag_of_words = {1:1, 2:1, 13:1, 5:1, 14:1, 15:1, 17:1, 16:1}

        # creating mock objects
        clf = TextClassifier(d1)
        clf.all_bags_of_words[2] = d2.bag_of_words

        # calculating cosine
        cosine = clf.calculate_cosine(2)

        self.assertAlmostEqual(cosine, 0.3062, places=4)

    def test_cosine4(self):
        # last example taken from:
        # https://stackoverflow.com/questions/1746501/can-someone-give-an-example-of-cosine-similarity-in-a-very-simple-graphical-wa
        d1 = Document(1)
        d1.bag_of_words = {'Julie': 1, 'Linda': 1, 'loves': 2, 'me': 2, 'more': 1, 'than': 1}

        d2 = Document(2)
        d2.bag_of_words = {'Jane': 1, 'Julie': 1, 'likes': 1, 'loves': 1, 'me': 2, 'more': 1, 'than': 1}

        # creating mock classifier
        clf = TextClassifier(d1)
        clf.all_bags_of_words[2] = d2.bag_of_words

        self.assertAlmostEqual(clf.calculate_cosine(2), 0.822, places=3)