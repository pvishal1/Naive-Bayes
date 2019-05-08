from __future__ import division
import nltk
import os
import sys
from nltk.tokenize import RegexpTokenizer
from nltk.stem.snowball import SnowballStemmer
from nltk.stem import LancasterStemmer
import glob
import math

def nb():
    ham_files = len(glob.glob("train/ham/*.txt"))
    spam_files = len(glob.glob("train/spam/*.txt"))
    total_no_of_files = ham_files + spam_files
    prob_ham = math.log((ham_files / total_no_of_files), 10)
    prob_spam = math.log10(spam_files / total_no_of_files)

    all_words = {}
    no_of_all_words = 0
    no_of_all_words = stem("train/*/*", all_words, no_of_all_words)
    no_of_unique_training_words = len(all_words)

    ham_words = {}
    no_of_ham_words = 0
    no_of_ham_words = stem("train/ham/*", ham_words, no_of_ham_words)

    spam_words = {}
    no_of_spam_words = 0
    no_of_spam_words = stem("train/spam/*", spam_words, no_of_spam_words)

    accuracy = 0
    accuracy = test(no_of_unique_training_words, prob_ham, prob_spam, ham_words, spam_words, no_of_ham_words, no_of_spam_words, accuracy)
    print "accuracy with stop word: " ,accuracy

def test(no_of_unique_training_words, prob_ham, prob_spam, ham_words, spam_words, no_of_ham_words, no_of_spam_words, accuracy):
    filepath = glob.glob("test-2/ham/*.txt")
    test_file_count = len(filepath)
    wrong_decision = 0

    for file in filepath:
        p_ham = 0
        p_spam = 0
        if not os.path.isfile(file):
            print("File path {} does not exist. Exiting...".format(file))
            sys.exit()

        with open(file,'r') as fp:
            p_ham, p_spam = get_probability(no_of_unique_training_words, prob_ham, prob_spam, ham_words, spam_words, no_of_ham_words, no_of_spam_words, p_ham, p_spam, fp)
        if (p_ham < p_spam) :
            wrong_decision += 1

    filepath = glob.glob("test-2/spam/*.txt")
    test_file_count += len(filepath)
    for file in filepath:
        p_ham = 1
        p_spam = 1
        if not os.path.isfile(file):
            print("File path {} does not exist. Exiting...".format(file))
            sys.exit()

        with open(file,'r') as fp:
            get_probability(no_of_unique_training_words, prob_ham, prob_spam, ham_words, spam_words, no_of_ham_words, no_of_spam_words, p_ham, p_spam, fp)
        if (p_ham > p_spam) :
            wrong_decision += 1

    accuracy = (test_file_count - wrong_decision) / test_file_count
    return accuracy*100

def get_probability(no_of_unique_training_words, prob_ham, prob_spam, ham_words, spam_words, no_of_ham_words, no_of_spam_words, p_ham, p_spam, fp):
    tokenizer = RegexpTokenizer("[a-zA-Z]+");
    stemmer = SnowballStemmer("english")

    for line in fp:
        tokens = tokenizer.tokenize(line)
        stemmed = [stemmer.stem(str(t)) for t in tokens]
        for word in stemmed:
            if word in ham_words:
                p_ham += math.log10((ham_words[word] + 1) / (no_of_ham_words + no_of_unique_training_words))
            else:
                p_ham += math.log10((1) / (no_of_ham_words + no_of_unique_training_words))
            if word in spam_words:
                p_spam += math.log10((spam_words[word] + 1) / (no_of_spam_words + no_of_unique_training_words))
            else:
                p_spam += math.log10((1) / (no_of_spam_words + no_of_unique_training_words))
    p_ham = prob_ham + p_ham
    p_spam = prob_spam + p_spam
    return p_ham, p_spam

def stem(path, bag_of_words, no_of_words):
    filepath = glob.glob(path + ".txt")
    tokenizer = RegexpTokenizer("[a-zA-Z]+");
    stemmer = LancasterStemmer()#("english")

    for file in filepath:
        if not os.path.isfile(file):
            print("File path {} does not exist. Exiting...".format(file))
            sys.exit()

        with open(file,'r') as fp:
            for line in fp:
                tokens = tokenizer.tokenize(line)
                no_of_words += len(tokens)
                stemmed = [stemmer.stem(t) for t in tokens]
                record_word_cnt(stemmed, bag_of_words)
    return no_of_words

def record_word_cnt(words, bag_of_words):
    for word in words:
        if word != '':
            if word.lower() in bag_of_words:
                bag_of_words[word.lower()] += 1
            else:
                bag_of_words[word.lower()] = 1
