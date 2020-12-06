import pandas as pd
import numpy as np
from tqdm import tqdm, trange
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertTokenizer, BertConfig
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import transformers
from transformers import BertForTokenClassification, AdamW
from sklearn.metrics import classification_report, confusion_matrix, multilabel_confusion_matrix, f1_score, accuracy_score, matthews_corrcoef, roc_auc_score
import os, argparse
import tensorflow as tf
from torch.nn.functional import softmax
import pickle
import keras

from sklearn.metrics import roc_curve, auc
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

def to_cuda_manual(Tokenizer_output):
    tokens_tensor = Tokenizer_output['input_ids'].to('cuda:0')
    token_type_ids = Tokenizer_output['token_type_ids'].to('cuda:0')
    attention_mask = Tokenizer_output['attention_mask'].to('cuda:0')

    output = {'input_ids' : tokens_tensor, 
              'token_type_ids' : token_type_ids, 
              'attention_mask' : attention_mask}

    return output

def main(parser):
        
    tf.logging.set_verbosity(tf.logging.INFO)
    
    in_data, sv_dir, model_dir, batch_size = parser.data_dir, parser.save_dir, parser.model_dir, int(parser.batch_size)
    tf.logging.info(in_data)
    tf.logging.info(sv_dir)
    tf.logging.info(model_dir)
    tf.logging.info(batch_size)

    df = pd.read_pickle(in_data)
    
    raw_labels = df.oneshot.values
    comments = df.text_changed.values
    
    tag_values = ['normal', 'insert', 'delete', 'sub', 'PAD']
    tag2idx = {t: i for i, t in enumerate(tag_values)}
    
    tf.logging.info('Example pairs')
    tf.logging.info('%s : %s' % (raw_labels[200], comments[200]))
    tf.logging.info('%s : %s' % (raw_labels[20], comments[20]))
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
    tf.logging.info(str(torch.cuda.get_device_name(0)))
    
    #Load in tokenizer
    max_length = 512
    tokenizer = BertTokenizer.from_pretrained(model_dir, do_lower_case=False) # tokenizer
    
    def tokenize_and_preserve_labels(sentence, text_labels):
        tokenized_sentence = []
        labels = []

        for word, label in zip(sentence, text_labels):

            # Tokenize the word and count # of subwords the word is broken into
            tokenized_word = tokenizer.tokenize(word)
            n_subwords = len(tokenized_word)

            # Add the tokenized word to the final tokenized word list
            tokenized_sentence.extend(tokenized_word)

            # Add the same label to the new list of labels `n_subwords` times
            labels.extend([label] * n_subwords)

        return tokenized_sentence, labels
    
    
    tokenized_texts_and_labels = [tokenize_and_preserve_labels(sent, labs) for sent, labs in zip(comments, raw_labels)]
    tokenized_texts = [token_label_pair[0] for token_label_pair in tokenized_texts_and_labels]
    labels = [token_label_pair[1] for token_label_pair in tokenized_texts_and_labels]
    
    input_ids = pad_sequences([tokenizer.convert_tokens_to_ids(txt) for txt in tokenized_texts],
                          maxlen=max_length, dtype="long", value=0.0,
                          truncating="post", padding="post")

    tags = pad_sequences([[l for l in lab] for lab in labels],
                     maxlen=max_length, value=tag2idx["PAD"], padding="post",
                     dtype="long", truncating="post")

    attention_masks = [[float(i != 0.0) for i in ii] for ii in input_ids]
    
    test_inputs = input_ids
    test_labels = tags
    test_masks = attention_masks
    
    test_inputs = torch.tensor(test_inputs)
    test_labels = torch.tensor(test_labels)
    test_masks = torch.tensor(test_masks)

    # Create an iterator of our data with torch DataLoader. This helps save on memory during training because, unlike a for loop, 
    # with an iterator the entire dataset does not need to be loaded into memory
    
    test_data = TensorDataset(test_inputs, test_masks, test_labels)#, test_token_types)
    test_sampler = SequentialSampler(test_data)
    test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=batch_size)
    tf.logging.info('test: %s' % (str(test_inputs.shape)))
    
    model = BertForTokenClassification.from_pretrained(model_dir, num_labels=5, output_attentions = False,
        output_hidden_states = False)

    model = model.to('cuda:0')

    # ========================================
    #               Testing
    # ========================================  

    # Put the model into evaluation mode
    model.eval()
    # Reset the validation loss for this epoch.
    eval_loss, eval_accuracy = 0, 0
    nb_eval_steps, nb_eval_examples = 0, 0
    predictions , true_labels, raw_predictions = [], [], []
    for batch in test_dataloader:
            
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask, b_labels = batch

        # Telling the model not to compute or store gradients,
        # saving memory and speeding up validation
        with torch.no_grad():
            # Forward pass, calculate logit predictions.
            # This will return the logits rather than the loss because we have not provided labels.
            outputs = model(b_input_ids, token_type_ids=None,
                            attention_mask=b_input_mask, labels=b_labels)
        # Move logits and labels to CPU
        logits = outputs[1].detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()

        # Calculate the accuracy for this batch of test sentences.
        eval_loss += outputs[0].mean().item()
        raw_predictions.extend(logits)
        predictions.extend([list(p) for p in np.argmax(logits, axis=2)])
        true_labels.extend(label_ids)

    eval_loss = eval_loss / len(test_dataloader)
    tf.logging.info("loss: {}".format(eval_loss))
    pred_tags = [tag_values[p_i] for p, l in zip(predictions, true_labels)
                                 for p_i, l_i in zip(p, l) if tag_values[l_i] != "PAD"]
    test_tags = [tag_values[l_i] for l in true_labels
                                  for l_i in l if tag_values[l_i] != "PAD"]
    
    raw_preds = np.array([p_i for p, l in zip(raw_predictions, true_labels)
                                 for p_i, l_i in zip(p, l) if tag_values[l_i] != "PAD"])
    raw_tags_sing = [l_i for p, l in zip(raw_predictions, true_labels)
                                 for p_i, l_i in zip(p, l) if tag_values[l_i] != "PAD"]
    raw_tags = np.array([np.eye(len(tag2idx))[xx] for xx in raw_tags_sing])
    tf.logging.info(str(raw_preds.shape))
    tf.logging.info(str(raw_tags.shape))
    
    tf.logging.info('F1 Accuracy: %s' % (str( f1_score(test_tags, pred_tags,average=None) )))
    tf.logging.info('Flat Accuracy: %s' % (str( accuracy_score(test_tags, pred_tags) )))
   
    clf_report = classification_report(test_tags,pred_tags)
    tf.logging.info(str( clf_report ))
    
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    plt.figure()
    
    roc_auc = dict()
    lw = 2
    for i in range(len(tag2idx)):
        fpr[i], tpr[i], _ = roc_curve(raw_tags[:, i], raw_preds[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(raw_tags.ravel(), raw_preds.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    
    tf.logging.info(str(roc_auc))
    
    for i, color in enumerate(['red','orange','yellow','green']):
        plt.plot(fpr[i], tpr[i], color=color,
                 lw=lw, label='ROC curve class %s (area = %0.2f)' % (tag_values[i], roc_auc[i]))
        
    plt.plot(fpr["micro"], tpr["micro"], color='purple',
                 lw=lw, label='ROC curve micro (area = %0.2f)' % roc_auc["micro"])
    
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    
    plt.savefig(os.path.join(sv_dir, in_data.split('/')[-1] + '_results.png'))
    
        
if __name__=='__main__': 
    parser = argparse.ArgumentParser(description = 'Test pretrained model on corpus of errors')
    parser.add_argument("-d", "--data_dir", help = "Errors corpus file pickle", required=True)
    parser.add_argument("-m", "--model_dir", help = "Directory of transformers-compatible BERT model", required=True)
    parser.add_argument("-s", "--save_dir", help = "Directory to save outputs", required=True)
    parser.add_argument("-b", "--batch_size", help = "Batch size for fine-tuning model", default=32)
  
    args = parser.parse_args()
        
    main(args)