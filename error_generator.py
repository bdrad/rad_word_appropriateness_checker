import random
import nltk
from nltk.corpus import words
from nltk.corpus import wordnet as wn
from pyphonetics import Metaphone
from collections import defaultdict
import pandas as pd
import collections
import json
from time import *
from pyphonetics import Metaphone

vocab = None 
meta_lst = None
freq_vocab = None
metaphone = Metaphone()

def _set_vocab(in_vocab_f):
    """
    Set your own vocabulary here. It is suggested that the vocab file include a column of word frequency count.
    """
    global meta_lst 
    global vocab
    global freq_vocab

    if in_vocab_f.endswith(".csv"):
        rad_vocab_df = pd.read_csv(in_vocab_f, encoding='utf-8')
        vocab = [word for word in rad_vocab_df[rad_vocab_df['count'] >3]['word'].tolist() if isinstance(word,str) and len(word)>1]
        
        freq_vocab = [word for word in rad_vocab_df[rad_vocab_df['count'] >20]['word'].tolist() if isinstance(word,str) and len(word)>1
                     and len(word)<=5]
        
    elif in_vocab_f.endswith(".txt"):
        with open(in_vocab_f, 'r') as f:
            vocab = [line.strip() for line in f if len(line.strip()) > 1]
            freq_vocab = [word for word in vocab if len(word) <= 5]
            
    # stores all the metaphone phonetic representations while initialising vocab
    meta_lst = [metaphone.phonetics(word) for word in vocab]
    
    print('Vocab Len:', len(vocab))
    print('Freq vocab Len:', len(freq_vocab))
    return vocab

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
  """
  for i in range(len1 + 1):
    for j in range(len2 + 1):
      print(lev[i][j],end="")
    print('\n')
  """
  return lev[len1][len2]

def get_random_word(vocabs):
  """
  Input: List of words
  Output: A random whole word
  """
  word = random.choice(vocabs)
  while not isLetter(word[0]):
    word = random.choice(vocabs)
  return word

def isLetter(x):
  if (x >= 'A' and x <= 'Z') or (x >= 'a' and x <= 'z'):
    return True
  return False

def metaphone_sim_original(target, threshold = 1):
  '''
  First generate Metaphone representation for the word
  Then, calculate the Levenshtein distance of the Mataphone representations
  '''
  ret = []
  target_meta = metaphone.phonetics(target)
  for i in range(len(vocab)): 
    word = vocab[i]
    if isLetter(word[0]) and word != target:
      word_meta = meta_lst[i]
      # Levenshtein distance
      score = my_edit_distance(target_meta, word_meta)
      if score < threshold:
        ret.append(word)
  if len(ret) == 0 and threshold < 2:
    return metaphone_sim(target, threshold + 1)
  elif len(ret) == 0:
    return get_random_word(vocab)
  return ret

import Levenshtein
def metaphone_sim(target, threshold = 1):
  '''
  First generate Metaphone representation for the word
  Then, calculate the Levenshtein distance of the Mataphone representations
  '''
  ret = []
  target_meta = metaphone.phonetics(target)
  target_len = len(target)
  for i in range(len(vocab)): 
    word = vocab[i]
    if target_len * 2 < len(word) or target_len / 2 > len(word):
      continue
    if word[0].isalpha() and word != target:
      word_meta = meta_lst[i]
      # Levenshtein distance
      if abs(len(target_meta) - len(word_meta)) > threshold:
        continue
      score = Levenshtein.distance(target_meta, word_meta) #my_edit_distance(target_meta, word_meta)
      if score < threshold:
        ret.append(word)
  if len(ret) == 0 and threshold < 2:
    return metaphone_sim(target, threshold + 1)
  elif len(ret) == 0:
    return get_random_word(vocab)
  return ret

def spelling_sim(target, lst):
  '''
  Generate the most similar word in spelling
  '''
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
    

class Error_generator():
  def __init__(self, vocab_f, prob = 0.2):
    self.prob = prob
    
    self.vocab = _set_vocab(vocab_f)
    self.vocab_tagged = nltk.pos_tag(self.vocab, tagset = 'universal')
    wordsets = defaultdict(list)
    #words = nltk.corpus.words.words('en')
    for word, tag in self.vocab_tagged:
      if isLetter(word[0]):
        wordsets[tag].append(word)
    self.pos_sets = wordsets

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

  def phonetical_replace(self, text, prob = 0.2, verbose = False):
    self.prob = prob
    words = nltk.word_tokenize(text.lower())
    for i in range(len(words)):
      if not isLetter(words[i][0]):
        continue
      if random.random() < self.prob:
        word_list = metaphone_sim(words[i], 1)
        chosen_word = random.choice(word_list)
        if verbose:
          print(i, words[i], "->", chosen_word)
        words[i] = chosen_word
    return ' '.join(words).capitalize()

  def phonetical_replace_advanced(self, text, prob = 0.2, verbose = False):
    self.prob = prob
    words = nltk.word_tokenize(text.lower())
    for i in range(len(words)):
      if not isLetter(words[i][0]):
        continue
      if random.random() < self.prob:
        word_list = metaphone_sim(words[i], 1)
        chosen_word = spelling_sim(words[i], word_list)
        if verbose:
          print(i, words[i], "->", chosen_word)
        words[i] = chosen_word
    return ' '.join(words).capitalize()

  def phonetical_replace_three(self, text, prob = 0.2, verbose = False):
    """
    Only change words with length longer than 3
    """
    self.prob = prob
    words = nltk.word_tokenize(text.lower())
    for i in range(len(words)):
      if not isLetter(words[i][0]) or len(words[i]) < 3:
        continue
      if random.random() < self.prob:
        word_list = metaphone_sim(words[i], 1)
        chosen_word = random.choice(word_list)
        if verbose:
          print(i, words[i], "->", chosen_word)
        words[i] = chosen_word
    return ' '.join(words).capitalize()

  def phonetical_replace_num(self, text, num = 1, verbose = False):
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
      word_list = metaphone_sim(words[chosen_idx].lower(), 1)
      chosen_word = random.choice(word_list)
      if verbose:
        print(chosen_idx, words[chosen_idx], "->", chosen_word)
      words[chosen_idx] = chosen_word
      num_cnt += 1

    return ' '.join(words).capitalize()

  def phonetical_replace_num_json(self, text, num = 1, verbose = False):
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
      while len(words[chosen_idx]) < 3 or chosen_idx in res_idx:
        chosen_idx = random.choice(range(words_len))
        cnt += 1
        if cnt > 5:
          chosen_idx = 0
          break

      res_idx.append(chosen_idx)
      res_truth.append(words[chosen_idx])
     
      word_list = metaphone_sim(words[chosen_idx].lower(), 1)

      chosen_word = random.choice(word_list)
      res_error.append(chosen_word)

      if verbose:
        print(chosen_idx, words[chosen_idx], "->", chosen_word)
      words[chosen_idx] = chosen_word
      num_cnt += 1
    
    result['text'] = words 
    result['index'] = res_idx
    result['truth'] = res_truth
    result['error'] = res_error
    return result

  def substitute_word(self, result, num = 1, verbose = False, hs_code=3):
    """
    Randomly replace #num word(s) in the given text ARRAY, preserving input text capitalization
    Return a dict as json form
    """
#     result = dict()
    res_idx = result['index']
    res_truth = result['truth']
    res_error = result['error']
    words_raw = result['text'] #.split()
    hotshot =   result['hotshot']
    words = [nltk.word_tokenize(x)[0] for x in words_raw]
    words_len = len(words)
    
    assert words_len >= num, 'too many words requested to replace'
    
    if words_len == 0:
      return 

    num_cnt = 0
    while num_cnt < num:
      chosen_idx = random.choice(range(words_len))
      
      ct=0
      while len(words[chosen_idx]) < 3 or chosen_idx in res_idx:
        chosen_idx = random.choice(list(range(words_len)))
        ct+=1
        if ct==10: 
          chosen_idx=0
          break
        
      res_idx.append(chosen_idx)
      res_truth.append(words_raw[chosen_idx])
        
      word_list = metaphone_sim(words[chosen_idx].lower(), 1)

      chosen_word = random.choice(word_list)
      
      if words[chosen_idx][0].isupper():
          chosen_word = chosen_word.capitalize()
      
      res_error.append(chosen_word)

      if verbose:
        print(chosen_idx, words[chosen_idx], "->", chosen_word)
      
      words_raw[chosen_idx] = chosen_word
      hotshot[chosen_idx] = hs_code
      num_cnt += 1
    
#     res_text = ' '.join(words).capitalize()
    result['text'] = words_raw #res_text
    result['index'] = res_idx
    result['truth'] = res_truth
    result['error'] = res_error
    result['hotshot'] = hotshot
    
    return result


  def insert_word(self, result, num = 1, verbose=False, hs_code=1):
    """
    Randomly insert frequent, short #num word(s) in the given text ARRAY
    Return a dict as json form
    """
    ins_idx = result['index']
    ins_truth = result['truth']
    ins_errors = result['error']
    text = result['text']
    hotshot = result['hotshot']

    for _ in range(num):
      insert_ind = random.choice(list(range(len(text)+1)))
      insert_word = random.choice(freq_vocab)
      
      ins_idx = [x + int(x>=insert_ind) for x in ins_idx]
        
      ins_idx.append(insert_ind)
      ins_errors.append(insert_word)
      text.insert(insert_ind, insert_word)
      hotshot.insert(insert_ind, hs_code)
      ins_truth.append(None)
    
    result['text'] = text
    result['index'] = ins_idx
    result['truth'] = ins_truth
    result['error'] = ins_errors
    result['hotshot'] = hotshot
    
    return result

  def delete_word(self, result, num = 1, verbose=False, hs_code=2):
    """
    Randomly delete #num smaller word(s) in the given text ARRAY
    Return a dict as json form
    """
    del_idx = result['index']
    del_truth = result['truth']
    del_errors = result['error']
    text = result['text']
    hotshot = result['hotshot']

    for _ in range(num):
      del_ind = random.choice(list(range(len(text)-1)))
      ct=0
      while del_ind in del_idx:
        del_ind = random.choice(list(range(len(text)-1)))
        ct+=1
        if ct==10: 
          del_ind=0
          break
      
      del_idx = [x - int(x>=del_ind) for x in del_idx]
      
      if del_ind in del_idx: #should not happen unless above while loop defaults
        rep_ind = del_idx.index(del_ind)
        del_idx.pop(rep_ind)
        del_truth.pop(rep_ind)
        del_errors.pop(rep_ind)
        
      del_idx.append(del_ind)
      del_errors.append(None)
      true_word = text.pop(del_ind)
      hotshot.pop(del_ind)
      hotshot[del_ind] = hs_code
      del_truth.append(true_word)
    
    result['text'] = text
    result['index'] = del_idx
    result['truth'] = del_truth
    result['error'] = del_errors
    result['hotshot'] = hotshot
    
    return result