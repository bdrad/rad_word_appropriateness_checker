import random
import nltk
from nltk.corpus import words
from nltk.corpus import wordnet as wn
from collections import defaultdict
import pandas as pd
import collections
import json
from time import *
from pyphonetics import Metaphone

"""
nltk.download('words')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('word_tokenize')
nltk.download('universal_tagset')
nltk.download('punkt')
"""

# ## Error_generator

# 1. random replacement
# 2. same part of speech https://www.nltk.org/book/ch05.html
# 3. phonetically similar replacement

metaphone = Metaphone()

# My edit distance
# Add pruning when calculating edit distance to improve efficiency
def _edit_dist_init(len1, len2):
    lev = []
    for i in range(len1):
        lev.append([0] * len2)  # initialize 2D array to zero
    for i in range(len1):
        lev[i][0] = i           # column 0: 0,1,2,3,4,...
    for j in range(len2):
        lev[0][j] = j           # row 0: 0,1,2,3,4,...
    return lev

def _edit_dist_step(lev, i, j, s1, s2, threshold):
  """
  Dp update
  return True if current element <= 1
  """
  c1 = s1[i - 1]
  c2 = s2[j - 1]

  # skipping a character in s1
  a = lev[i - 1][j] + 1
  # skipping a character in s2
  b = lev[i][j - 1] + 1
  # substitution
  c = lev[i - 1][j - 1] + (c1 != c2)

  # pick the cheapest
  lev[i][j] = min(a, b, c)
  if lev[i][j] <= threshold:
    return True
  return False

def my_edit_distance_ori(s1, s2, threshold = 1):
  """
  Calculate the Levenshtein edit-distance between two strings.
  Improved from nltk library, with pruning.
  """
  # set up a 2-D array
  len1 = len(s1)
  len2 = len(s2)
  lev = _edit_dist_init(len1 + 1, len2 + 1)

  # iterate over the array
  prev_flag = True
  cur_flag = False
  #if both prev and cur are False, then stop
  for i in range(len1):
    cur_flag = False
    for j in range(len2):
      _edit_dist_step(lev, i + 1, j + 1, s1, s2, threshold)

  for i in range(len1 + 1):
    for j in range(len2 + 1):
      print(lev[i][j],end="")
    print('\n')
  return lev[len1][len2]

def my_edit_distance(s1, s2, threshold = 1):
  """
  Calculate the Levenshtein edit-distance between two strings.
  Improved from nltk library, with pruning.
  """
  # set up a 2-D array
  len1 = len(s1)
  len2 = len(s2)
  lev = _edit_dist_init(len1 + 1, len2 + 1)

  # iterate over the array
  prev_flag = True
  cur_flag = False
  #if both prev and cur are False, then stop
  for i in range(len1):
    cur_flag = False
    for j in range(len2):
      if _edit_dist_step(lev, i + 1, j + 1, s1, s2, threshold) and not cur_flag:
        # when cur row doesn't contain 0/1 and cur postion is 0/1
        cur_flag = True 
    if not prev_flag and not cur_flag:
      return 5 # anything bigger than threshold
    prev_flag = cur_flag
  return lev[len1][len2]


def get_random_word(words):
  """
  Input: List of words
  Output: A random whole word
  """
  word = random.choice(words)
  while not isLetter(word[0]):
    word = random.choice(words)
  return word

def isLetter(x):
  if (x >= 'A' and x <= 'Z') or (x >= 'a' and x <= 'z'):
    return True
  return False


class Error_generator():
  def __init__(self, vocab, prob = 0.2):
    """
    vocab: list
    """
    self.prob = prob
    self.vocab = vocab
    self.meta_lst = [metaphone.phonetics(word) for word in self.vocab]
    '''
    self.vocab_tagged = nltk.pos_tag(self.vocab, tagset = 'universal')
    wordsets = defaultdict(list)
    #words = nltk.corpus.words.words('en')
    for word, tag in self.vocab_tagged:
      if isLetter(word[0]):
        wordsets[tag].append(word)
    self.pos_sets = wordsets
    '''

  def load_vocab(self, vocab_list):
    self.vocab = vocab_list

  def metaphone_sim(self, target, threshold = 1):
    '''
    Returns a list of 
    First generate Metaphone representation for the word
    Then, calculate the Levenshtein distance of the Mataphone representations
    '''
    ret = []
    target_meta = metaphone.phonetics(target)
    target_len = len(target)
    for i in range(len(self.vocab)): 
      word = self.vocab[i]
      if target_len * 2 < len(word) or target_len / 2 > len(word):
        continue
      if isLetter(word[0]) and word != target:
        word_meta = self.meta_lst[i]
        # Levenshtein distance
        if abs(len(target_meta) - len(word_meta)) > threshold:
          continue
        score = my_edit_distance(target_meta, word_meta)
        if score < threshold:
          ret.append(word)
    if len(ret) == 0 and threshold < 2:
      return self.metaphone_sim(target, threshold + 1)
    elif len(ret) == 0:
      return get_random_word(self.vocab)
    return ret

  def spelling_sim(self, target, lst):
    """
    Returns the most similar word with target in spelling from lst
    """
    if len(lst) == 0:
      return ""
    min_dis = float('INF')
    most_sim = lst[0]
    for word in lst:
      tmpscore = my_edit_distance(target, word)
      if tmpscore < min_dis and len(word) > 1:
        most_sim = word
        min_dis = tmpscore
    return most_sim


  def random_replace(self, text, prob = 0.2, verbose = False):
    """
    Randomly replace each word with probability = prob.
    """
    self.prob = prob
    words = nltk.word_tokenize(text.lower())
    for i in range(len(words)):
      if not isLetter(words[i][0]):
        continue
      if random.random() < self.prob:
        random_word = get_random_word(self.vocab)
        if verbose:
          print(i, words[i], "->", random_word)
        words[i] = random_word
    return ' '.join(words).capitalize()

  def pos_replace(self, text, prob = 0.2, verbose = False):
    """
    Replace each word with same POS word with probability = prob.
    """
    self.prob = prob
    words = nltk.word_tokenize(text.lower())
    tagged = nltk.pos_tag(words, tagset = 'universal')
    for i in range(len(words)):
      if not isLetter(words[i][0]):
        continue
      if random.random() < self.prob:
        random_word = get_random_word(self.pos_sets[tagged[i][1]])
        if verbose:
          print(i, words[i], "->", random_word)
        words[i] = random_word
    return ' '.join(words).capitalize()

  def phonetic_replace(self, text, prob = 0.2, verbose = False, random_choice = True):
    self.prob = prob
    words = nltk.word_tokenize(text.lower())
    for i in range(len(words)):
      if not isLetter(words[i][0]):
        continue
      if random.random() < self.prob:
        word_list = self.metaphone_sim(words[i], 1)
        if random_choice:
          chosen_word = random.choice(word_list)
        else:
          chosen_word = self.spelling_sim(words[i], word_list)
        if verbose:
          print(i, words[i], "->", chosen_word)
        words[i] = chosen_word
    return ' '.join(words).capitalize()

  def phonetic_replace_three(self, text, prob = 0.2, verbose = False):
    """
    Only change words with length longer than 3
    """
    self.prob = prob
    words = nltk.word_tokenize(text.lower())
    for i in range(len(words)):
      if not isLetter(words[i][0]) or len(words[i]) < 3:
        continue
      if random.random() < self.prob:
        word_list = self.metaphone_sim(words[i], 1)
        chosen_word = random.choice(word_list)
        if verbose:
          print(i, words[i], "->", chosen_word)
        words[i] = chosen_word
    return ' '.join(words).capitalize()

  def phonetic_replace_num(self, text, num = 1, verbose = False, random_choice = True):
    """
    Randomly replace #num word(s) in the given text
    Usually less than 3
    """
    words = nltk.word_tokenize(text.lower())
    words_len = len(words)
    if words_len == 0:
      return 

    num_cnt = 0
    changed = []
    while num_cnt < num:
      chosen_idx = random.choice(range(words_len))
      cnt = 0
      while len(words[chosen_idx]) < 3 and chosen_idx not in changed:
        chosen_idx = random.choice(range(words_len))
        cnt += 1
        if cnt > 5:
          chosen_idx = 0
          break

      changed.append(chosen_idx)
      word_list = self.metaphone_sim(words[chosen_idx], 1)
      if random_choice:
        chosen_word = random.choice(word_list)
      else:
        chosen_word = self.spelling_sim(words[i], word_list)
      if verbose:
        print(chosen_idx, words[chosen_idx], "->", chosen_word)
      words[chosen_idx] = chosen_word
      num_cnt += 1

    return ' '.join(words).capitalize()

  def phonetic_replace_num_json(self, text, num = 1, verbose = False):
    """
    Randomly replace #num word(s) in the given text
    Usually less than 3
    Return a dict as json form
    """
    result = dict()
    res_idx = []
    res_truth = []
    res_error = []
    words = nltk.word_tokenize(text.lower())
    words_len = len(words)
    if words_len == 0:
      return 

    num_cnt = 0
    while num_cnt < num:
      chosen_idx = random.choice(range(words_len))
      cnt = 0
      while len(words[chosen_idx]) < 3 and chosen_idx not in res_idx:
        chosen_idx = random.choice(range(words_len))
        cnt += 1
        if cnt > 5:
          chosen_idx = 0
          break

      res_idx.append(chosen_idx)
      res_truth.append(words[chosen_idx])
      word_list = self.metaphone_sim(words[chosen_idx], 1)
      chosen_word = random.choice(word_list)
      res_error.append(chosen_word)

      if verbose:
        print(chosen_idx, words[chosen_idx], "->", chosen_word)
      words[chosen_idx] = chosen_word
      num_cnt += 1

    res_text = ' '.join(words).capitalize()
    result['text'] = res_text
    result['index'] = res_idx
    result['truth'] = res_truth
    result['error'] = res_error
    return result