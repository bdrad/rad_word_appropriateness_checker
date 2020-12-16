import pandas as pd
from error_generator import *
from tqdm import tqdm
import random
import argparse
import os

def generate_errors(input_text, generator, max_subs=15, error_prob=0.074, type_probs=[0.347,0.270,0.383], verbose=False):
    type_choices = ['insert', 'delete', 'substitute']

    input_split = input_text.split()
    if input_split[-1] not in ['!', '.', '?']:
        input_split.append('.')
    num_input_words = len(input_split)

    num_errors = sum([random.random()<error_prob for _ in range(num_input_words)])
    
    if verbose:
        print('Generating %d errors' % num_errors)
    
    result = {'text': input_split, 'index': [], 'truth': [], 'error': [], 'hotshot': [0 for _ in range(num_input_words)]}
    for x in range(num_errors):
        error_type = random.choices(type_choices, weights=type_probs)[0]

        if error_type=='insert':
            result = generator.insert_word(result)
        elif error_type=='delete':
            result = generator.delete_word(result)
        else: # error_type=='substitution'
            result = generator.substitute_word(result)
    
        if verbose:
            print(x)
            print(result)
    return result

if __name__=='__main__':

    parser = argparse.ArgumentParser(description = 'Parameters to generate corrupted corpus')
    parser.add_argument("-v", "--vocab_file", help = "The vocabulary from which the word is selected", required=True)
    parser.add_argument("-c", "--corpus_file", help = "File that needs to be changed", required=True)
    parser.add_argument("-s", "--save_dir", help = "Directory to save corrupted corpus", required=True)
  
    args = parser.parse_args()

    vocab_file = args.vocab_file
    corpus_file = args.corpus_file
    save_dir = args.save_dir

    type_choices = ['insert', 'delete', 'substitute']
    type_probs  = [0.347   ,  0.270  ,  0.383      ]
    error_prob  = 0.074  #Probability of one word having an error
    max_subs=15 #Max number of errors per impression

    vocab_f = vocab_file
    generator = Error_generator(vocab_f, 0.2)
        
    with open(corpus_file, 'r') as f:
        imp_lines_raw = f.readlines()
      
    print('Num Input Lines:',len(imp_lines_raw))
    imp_lines_proc = [x.replace('\n',' ').replace('.', ' .').strip() for x in imp_lines_raw if len(x)>15]
    
    print('Num Processed Input Lines:',len(imp_lines_proc))
    
    imp_lines_errors = pd.DataFrame(columns=['text_orig','text_changed','index_changed','word_true','word_error','oneshot'])
    imp_lines_errors_dict = dict()
    for iii in range(len(imp_lines_proc)):
        if iii%100 == 0:
            print(iii)
        
        if len(imp_lines_proc[iii].split())<5: # min 5 words
            continue
            
        result = generate_errors(imp_lines_proc[iii], generator, max_subs=max_subs, error_prob=error_prob, type_probs=type_probs)
        
        imp_lines_errors_dict[iii] = {'text_orig': imp_lines_proc[iii].split(),
                                               'text_changed': result['text'],
                                               'index_changed': result['index'],
                                               'word_true': result['truth'],
                                               'word_error': result['error'],
                                               'oneshot':  result['hotshot']}

    imp_lines_errors = pd.DataFrame.from_dict(imp_lines_errors_dict,orient='index')
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    imp_lines_errors.to_pickle(os.path.join(save_dir, "error_corpus.pickle"))
    imp_lines_errors.to_csv(os.path.join(save_dir, "error_corpus.csv"))
    print('Successfully wrote %d lines.' % (len(imp_lines_errors)))