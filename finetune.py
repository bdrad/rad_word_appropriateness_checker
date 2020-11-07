# import pandas as pd
# import numpy as np
# import tensorflow as tf
# import torch
# from torch.nn import BCEWithLogitsLoss, BCELoss
# from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
# from keras.preprocessing.sequence import pad_sequences
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import MultiLabelBinarizer
# import pickle
# # from transformers import *
# # from pytorch_pretrained_bert import BertTokenizer, BertConfig
# # from pytorch_pretrained_bert import BertAdam, BertForSequenceClassification
# from tqdm import tqdm, trange
# from transformers import *
# # from ast import literal_eval
# import os, argparse
# import pickle


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
import pickle
import pdb

    
def main(parser):
    
    in_data, sv_dir, model_dir, epochs, learn_rate, batch_size = parser.data_dir, parser.save_dir, parser.model_dir, int(parser.epochs), float(parser.learn_rate), int(parser.batch_size)
    
    tf.logging.set_verbosity(tf.logging.INFO)
    
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
    max_length = 250
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
    
#     encodings = tokenizer.batch_encode_plus(comments,max_length=max_length,pad_to_max_length=True) # tokenizer's encoding method
#     input_ids = encodings['input_ids'] # tokenized and encoded sentences
#     token_type_ids = encodings['token_type_ids'] # token type ids
#     attention_masks = encodings['attention_mask'] # attention masks
    
#     # Identifying indices of 'one_hot_labels' entries that only occur once - this will allow us to stratify split our training data later
#     label_counts = df.one_hot_labels.astype(str).value_counts()
#     one_freq = label_counts[label_counts==1].keys()
#     one_freq_idxs = sorted(list(df[df.one_hot_labels.astype(str).isin(one_freq)].index), reverse=True)
#     tf.logging.info('df label indices with only one instance:  %s' % (str(one_freq_idxs)))
    
#     # Gathering single instance inputs to force into the training set after stratified split
#     one_freq_input_ids = [input_ids.pop(i) for i in one_freq_idxs]
#     one_freq_token_types = [token_type_ids.pop(i) for i in one_freq_idxs]
#     one_freq_attention_masks = [attention_masks.pop(i) for i in one_freq_idxs]
#     one_freq_labels = [labels.pop(i) for i in one_freq_idxs]
    
    # Use train_test_split to split our data into train and validation sets
    
    input_ids = pad_sequences([tokenizer.convert_tokens_to_ids(txt) for txt in tokenized_texts],
                          maxlen=max_length, dtype="long", value=0.0,
                          truncating="post", padding="post")
#     pdb.set_trace()
    
#     tags = pad_sequences([[tag2idx.get(l) for l in lab] for lab in labels],
#                      maxlen=max_length, value=tag2idx["PAD"], padding="post",
#                      dtype="long", truncating="post")
    tags = pad_sequences([[l for l in lab] for lab in labels],
                     maxlen=max_length, value=tag2idx["PAD"], padding="post",
                     dtype="long", truncating="post")

    attention_masks = [[float(i != 0.0) for i in ii] for ii in input_ids]
    
    train_inputs, not_train_inputs, train_labels, not_train_labels, train_masks, not_train_masks = train_test_split(input_ids, tags, attention_masks, random_state=4, test_size=0.30)#, stratify = labels)
    
    validation_inputs, test_inputs, validation_labels, test_labels, validation_masks, test_masks = train_test_split(not_train_inputs, not_train_labels, not_train_masks, random_state=4, test_size=0.50)#, stratify = not_train_labels)

#     # Add one frequency data to train data
#     train_inputs.extend(one_freq_input_ids)
#     train_labels.extend(one_freq_labels)
#     train_masks.extend(one_freq_attention_masks)
#     train_token_types.extend(one_freq_token_types)

    # Convert all of our data into torch tensors, the required datatype for our model
    train_inputs = torch.tensor(train_inputs)
    train_labels = torch.tensor(train_labels)
    train_masks = torch.tensor(train_masks)
#     train_token_types = torch.tensor(train_token_types)

    validation_inputs = torch.tensor(validation_inputs)
    validation_labels = torch.tensor(validation_labels)
    validation_masks = torch.tensor(validation_masks)
#     validation_token_types = torch.tensor(validation_token_types)
    
    test_inputs = torch.tensor(test_inputs)
    test_labels = torch.tensor(test_labels)
    test_masks = torch.tensor(test_masks)
#     test_token_types = torch.tensor(test_token_types)

    # Create an iterator of our data with torch DataLoader. This helps save on memory during training because, unlike a for loop, 
    # with an iterator the entire dataset does not need to be loaded into memory

    train_data = TensorDataset(train_inputs, train_masks, train_labels)#, train_token_types)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

    validation_data = TensorDataset(validation_inputs, validation_masks, validation_labels)#, validation_token_types)
    validation_sampler = SequentialSampler(validation_data)
    validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size=batch_size)
    
    test_data = TensorDataset(test_inputs, test_masks, test_labels)#, test_token_types)
    test_sampler = SequentialSampler(test_data)
    test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=batch_size)
    tf.logging.info('train: %s ; val: %s ; test: %s' % (str(train_inputs.shape), str(validation_inputs.shape), str(test_inputs.shape)))
    
    tf.logging.info('Saving data loaders..')
    torch.save(validation_dataloader, os.path.join(sv_dir,'validation_data_loader'))
    torch.save(train_dataloader, os.path.join(sv_dir,'train_data_loader'))
    torch.save(test_dataloader, os.path.join(sv_dir,'test_data_loader'))
    
    
    # Load model, the pretrained model will include a single linear classification layer on top for classification. 
#     model = BertForSequenceClassification.from_pretrained(model_dir, num_labels=num_labels)
    model = BertForTokenClassification.from_pretrained(model_dir, num_labels=len(tag2idx), output_attentions = False,
        output_hidden_states = False)
    model.cuda()
    
    
    FULL_FINETUNING = True
    if FULL_FINETUNING:
        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'gamma', 'beta']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
             'weight_decay_rate': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
             'weight_decay_rate': 0.0}
        ]
    else:
        param_optimizer = list(model.classifier.named_parameters())
        optimizer_grouped_parameters = [{"params": [p for n, p in param_optimizer]}]

    optimizer = AdamW(
        optimizer_grouped_parameters,
        lr=learn_rate,
        eps=1e-8
    )
    
    from transformers import get_linear_schedule_with_warmup

    max_grad_norm = 1.0

    # Total number of training steps is number of batches * number of epochs.
    total_steps = len(train_dataloader) * epochs

    # Create the learning rate scheduler.
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps
    )
    
    ## Store the average loss after each epoch so we can plot them.
    loss_values, validation_loss_values = [], []

    for _ in trange(epochs, desc="Epoch"):
        # ========================================
        #               Training
        # ========================================
        # Perform one full pass over the training set.

        # Put the model into training mode.
        model.train()
        # Reset the total loss for this epoch.
        total_loss = 0

        # Training loop
        for step, batch in enumerate(train_dataloader):
            # add batch to gpu
            batch = tuple(t.to(device) for t in batch)
            b_input_ids, b_input_mask, b_labels = batch
            # Always clear any previously calculated gradients before performing a backward pass.
            model.zero_grad()
            # forward pass
            # This will return the loss (rather than the model output)
            # because we have provided the `labels`.
            outputs = model(b_input_ids, token_type_ids=None,
                            attention_mask=b_input_mask, labels=b_labels)
            # get the loss
            loss = outputs[0]
            # Perform a backward pass to calculate the gradients.
            loss.backward()
            # track train loss
            total_loss += loss.item()
            # Clip the norm of the gradient
            # This is to help prevent the "exploding gradients" problem.
            torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=max_grad_norm)
            # update parameters
            optimizer.step()
            # Update the learning rate.
            scheduler.step()

        # Calculate the average loss over the training data.
        avg_train_loss = total_loss / len(train_dataloader)
        tf.logging.info("Average train loss: {}".format(avg_train_loss))

        # Store the loss value for plotting the learning curve.
        loss_values.append(avg_train_loss)


        # ========================================
        #               Validation
        # ========================================
        # After the completion of each training epoch, measure our performance on
        # our validation set.

        # Put the model into evaluation mode
        model.eval()
        # Reset the validation loss for this epoch.
        eval_loss, eval_accuracy = 0, 0
        nb_eval_steps, nb_eval_examples = 0, 0
        predictions , true_labels = [], []
        for batch in validation_dataloader:
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
            predictions.extend([list(p) for p in np.argmax(logits, axis=2)])
            true_labels.extend(label_ids)

        eval_loss = eval_loss / len(validation_dataloader)
        validation_loss_values.append(eval_loss)
        tf.logging.info("Validation loss: {}".format(eval_loss))
        pred_tags = [tag_values[p_i] for p, l in zip(predictions, true_labels)
                                     for p_i, l_i in zip(p, l) if tag_values[l_i] != "PAD"]
        valid_tags = [tag_values[l_i] for l in true_labels
                                      for l_i in l if tag_values[l_i] != "PAD"]
        tf.logging.info('Validation F1 Accuracy: %s' % (str( f1_score(valid_tags, pred_tags,average=None) )))
        tf.logging.info('Validation Flat Accuracy: %s' % (str( accuracy_score(valid_tags, pred_tags) )))
        
    
    # Saving trained model    
    torch.save(model.state_dict(), os.path.join(sv_dir,'finetuned_model_v1') )
    
    
    # ========================================
    #               Testing
    # ========================================  
    
    # Put the model into evaluation mode
    model.eval()
    # Reset the validation loss for this epoch.
    eval_loss, eval_accuracy = 0, 0
    nb_eval_steps, nb_eval_examples = 0, 0
    predictions , true_labels = [], []
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
        predictions.extend([list(p) for p in np.argmax(logits, axis=2)])
        true_labels.extend(label_ids)
    
    
    
    eval_loss = eval_loss / len(test_dataloader)
    validation_loss_values.append(eval_loss)
    tf.logging.info("Test loss: {}".format(eval_loss))
    pred_tags = [tag_values[p_i] for p, l in zip(predictions, true_labels)
                                 for p_i, l_i in zip(p, l) if tag_values[l_i] != "PAD"]
    test_tags = [tag_values[l_i] for l in true_labels
                                  for l_i in l if tag_values[l_i] != "PAD"]
    tf.logging.info('Test F1 Accuracy: %s' % (str( f1_score(test_tags, pred_tags,average=None) )))
    tf.logging.info('Test Flat Accuracy: %s' % (str( accuracy_score(test_tags, pred_tags) )))
    
    clf_report = classification_report(test_tags,pred_tags)
    tf.logging.info(str( clf_report ))
    pickle.dump(clf_report, open(os.path.join(sv_dir, 'classification_report.txt'),'wb')) #save report
    
    
    
#     # setting custom optimization parameters. You may implement a scheduler here as well.
#     param_optimizer = list(model.named_parameters())
#     no_decay = ['bias', 'gamma', 'beta']
#     optimizer_grouped_parameters = [
#         {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
#          'weight_decay_rate': 0.01},
#         {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
#          'weight_decay_rate': 0.0}
#     ]
    
#     optimizer = AdamW(optimizer_grouped_parameters,lr=learn_rate,correct_bias=True)
    
#     # Store our loss and accuracy for plotting
#     train_loss_set = []

#     # trange is a tqdm wrapper around the normal python range
#     for _ in trange(epochs, desc="Epoch"):

#       # Training

#       # Set our model to training mode (as opposed to evaluation mode)
#       model.train()

#       # Tracking variables
#       tr_loss = 0 #running loss
#       nb_tr_examples, nb_tr_steps = 0, 0

#       # Train the data for one epoch
#       for step, batch in enumerate(train_dataloader):
#         # Add batch to GPU
#         batch = tuple(t.to(device) for t in batch)
#         # Unpack the inputs from our dataloader
#         b_input_ids, b_input_mask, b_labels, b_token_types = batch
#         # Clear out the gradients (by default they accumulate)
#         optimizer.zero_grad()

#         # # Forward pass for multiclass classification
#         # outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)
#         # loss = outputs[0]
#         # logits = outputs[1]

#         # Forward pass for multilabel classification
#         outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)
#         logits = outputs[0]
#         loss_func = BCEWithLogitsLoss() 
#         loss = loss_func(logits.view(-1,num_labels),b_labels.type_as(logits).view(-1,num_labels)) #convert labels to float for calculation
#         # loss_func = BCELoss() 
#         # loss = loss_func(torch.sigmoid(logits.view(-1,num_labels)),b_labels.type_as(logits).view(-1,num_labels)) #convert labels to float for calculation
#         train_loss_set.append(loss.item())    

#         # Backward pass
#         loss.backward()
#         # Update parameters and take a step using the computed gradient
#         optimizer.step()
#         # scheduler.step()
#         # Update tracking variables
#         tr_loss += loss.item()
#         nb_tr_examples += b_input_ids.size(0)
#         nb_tr_steps += 1

#       tf.logging.info("Train loss: {}".format(tr_loss/nb_tr_steps))

#       ###############################################################################

#       # Validation

#       # Put model in evaluation mode to evaluate loss on the validation set
#       model.eval()

#       # Variables to gather full output
#       logit_preds,true_labels,pred_labels,tokenized_texts = [],[],[],[]

#       # Predict
#       for i, batch in enumerate(validation_dataloader):
#         batch = tuple(t.to(device) for t in batch)
#         # Unpack the inputs from our dataloader
#         b_input_ids, b_input_mask, b_labels, b_token_types = batch
#         with torch.no_grad():
#           # Forward pass
#           outs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)
#           b_logit_pred = outs[0]
#           pred_label = torch.sigmoid(b_logit_pred)

#           b_logit_pred = b_logit_pred.detach().cpu().numpy()
#           pred_label = pred_label.to('cpu').numpy()
#           b_labels = b_labels.to('cpu').numpy()

#         tokenized_texts.append(b_input_ids)
#         logit_preds.append(b_logit_pred)
#         true_labels.append(b_labels)
#         pred_labels.append(pred_label)

#       # Flatten outputs
#       pred_labels = [item for sublist in pred_labels for item in sublist]
#       true_labels = [item for sublist in true_labels for item in sublist]

#       # Calculate Accuracy
#       threshold = 0.50
#       pred_bools = [pl>threshold for pl in pred_labels]
#       true_bools = [tl==1 for tl in true_labels]
#       val_f1_accuracy = f1_score(true_bools,pred_bools,average='binary')*100
#       val_flat_accuracy = accuracy_score(true_bools, pred_bools)*100
        
#       tf.logging.info('F1 Validation Accuracy: %s' % (str(val_f1_accuracy)))
#       tf.logging.info('Flat Validation Accuracy: %s' % (str(val_flat_accuracy)))
    
#     torch.save(model.state_dict(), os.path.join(sv_dir,'radioped_model_v1') )
    
#     # Test

#     # Put model in evaluation mode to evaluate loss on the validation set
#     model.eval()

#     #track variables
#     logit_preds,true_labels,pred_labels,tokenized_texts = [],[],[],[]

#     # Predict
#     for i, batch in enumerate(test_dataloader):
#       batch = tuple(t.to(device) for t in batch)
#       # Unpack the inputs from our dataloader
#       b_input_ids, b_input_mask, b_labels, b_token_types = batch
#       with torch.no_grad():
#         # Forward pass
#         outs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)
#         b_logit_pred = outs[0]
#         pred_label = torch.sigmoid(b_logit_pred)

#         b_logit_pred = b_logit_pred.detach().cpu().numpy()
#         pred_label = pred_label.to('cpu').numpy()
#         b_labels = b_labels.to('cpu').numpy()

#       tokenized_texts.append(b_input_ids)
#       logit_preds.append(b_logit_pred)
#       true_labels.append(b_labels)
#       pred_labels.append(pred_label)

#     # Flatten outputs
#     tokenized_texts = [item for sublist in tokenized_texts for item in sublist]
#     pred_labels = [item for sublist in pred_labels for item in sublist]
#     true_labels = [item for sublist in true_labels for item in sublist]
#     # Converting flattened binary values to boolean values
#     true_bools = [tl==1 for tl in true_labels]
    
#     pred_bools = [pl>0.50 for pl in pred_labels] #boolean output after thresholding

#     # Print and save classification report
#     tf.logging.info('Test F1 Accuracy: %s' % (str( f1_score(true_bools, pred_bools,average='binary') )))
#     tf.logging.info('Test Flat Accuracy: %s' % (str( accuracy_score(true_bools, pred_bools) )))
    
#     clf_report = classification_report(true_bools,pred_bools)
#     pickle.dump(clf_report, open(os.path.join(sv_dir, 'classification_report.txt'),'wb')) #save report
#     tf.logging.info(str( clf_report ))   
    
#     tf.logging.info('Test matthews corr. coef: {0:0.4%}'.format(matthews_corrcoef(true_bools, pred_bools)))
#     tf.logging.info('Test roc auc: {0:0.4%}'.format(roc_auc_score(true_labels, pred_labels)))
    
#     model = BertForSequenceClassification.from_pretrained(model_dir, num_labels=num_labels)
#     model.cuda()
    
#     # BERT fine-tuning parameters
#     param_optimizer = list(model.named_parameters())
#     no_decay = ['bias', 'gamma', 'beta']
#     optimizer_grouped_parameters = [
#         {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
#          'weight_decay_rate': 0.01},
#         {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
#          'weight_decay_rate': 0.0}
#     ]

#     optimizer = BertAdam(optimizer_grouped_parameters,
#                          lr=learn_rate,
#                          warmup=.1)

#     # Function to calculate the accuracy of our predictions vs labels
#     def flat_accuracy(preds, labels):
#         pred_flat = np.argmax(preds, axis=1).flatten()
#         labels_flat = labels.flatten()
#         return np.sum(pred_flat == labels_flat) / len(labels_flat)

#     # Store our loss and accuracy for plotting
#     train_loss_set = []

#     # BERT training loop
#     for _ in trange(epochs, desc="Epoch"):  

#       ## TRAINING

#       # Set our model to training mode
#       model.train()  
#       # Tracking variables
#       tr_loss = 0
#       nb_tr_examples, nb_tr_steps = 0, 0
#       # Train the data for one epoch
#       for step, batch in enumerate(train_dataloader):
#         # Add batch to GPU
#         batch = tuple(t.to(device) for t in batch)
#         # Unpack the inputs from our dataloader
#         b_input_ids, b_input_mask, b_labels = batch
#         # Clear out the gradients (by default they accumulate)
#         optimizer.zero_grad()
#         # Forward pass
#         loss = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)
#         train_loss_set.append(loss.item())    
#         # Backward pass
#         loss.backward()
#         # Update parameters and take a step using the computed gradient
#         optimizer.step()
#         # Update tracking variables
#         tr_loss += loss.item()
#         nb_tr_examples += b_input_ids.size(0)
#         nb_tr_steps += 1
#       tf.logging.info("Train loss: {}".format(tr_loss/nb_tr_steps))

#       ## VALIDATION

#       # Put model in evaluation mode
#       model.eval()
#       # Tracking variables 
#       eval_loss, eval_accuracy = 0, 0
#       nb_eval_steps, nb_eval_examples = 0, 0
#       # Evaluate data for one epoch
#       for batch in validation_dataloader:
#         # Add batch to GPU
#         batch = tuple(t.to(device) for t in batch)
#         # Unpack the inputs from our dataloader
#         b_input_ids, b_input_mask, b_labels = batch
#         # Telling the model not to compute or store gradients, saving memory and speeding up validation
#         with torch.no_grad():
#           # Forward pass, calculate logit predictions
#           logits = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)    
#         # Move logits and labels to CPU
#         logits = logits.detach().cpu().numpy()
#         label_ids = b_labels.to('cpu').numpy()
#         tmp_eval_accuracy = flat_accuracy(logits, label_ids)    
#         eval_accuracy += tmp_eval_accuracy
#         nb_eval_steps += 1
#       tf.logging.info("Validation Accuracy: {}".format(eval_accuracy/nb_eval_steps))
    
    
#     ## Prediction on test set
#     # Put model in evaluation mode
#     model.eval()
#     # Tracking variables 
#     predictions , true_labels = [], []
#     # Predict 
#     for batch in prediction_dataloader:
#       # Add batch to GPU
#       batch = tuple(t.to(device) for t in batch)
#       # Unpack the inputs from our dataloader
#       b_input_ids, b_input_mask, b_labels = batch
#       # Telling the model not to compute or store gradients, saving memory and speeding up prediction
#       with torch.no_grad():
#         # Forward pass, calculate logit predictions
#         logits = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)
#       # Move logits and labels to CPU
#       logits = logits.detach().cpu().numpy()
#       label_ids = b_labels.to('cpu').numpy()  
#       # Store predictions and true labels
#       predictions.append(logits)
#       true_labels.append(label_ids)

#     # Import and evaluate each test batch using Matthew's correlation coefficient
#     from sklearn.metrics import matthews_corrcoef
#     matthews_set = []
#     for i in range(len(true_labels)):
#       matthews = matthews_corrcoef(true_labels[i],
#                      np.argmax(predictions[i], axis=1).flatten())
#       matthews_set.append(matthews)

#     # Flatten the predictions and true values for aggregate Matthew's evaluation on the whole dataset
#     flat_predictions = [item for sublist in predictions for item in sublist]
#     flat_predictions = np.argmax(flat_predictions, axis=1).flatten()
#     flat_true_labels = [item for sublist in true_labels for item in sublist]

#     # Converting flattened binary values to boolean values
#     true_bools = [tl==1 for tl in flat_true_labels]    
#     pred_bools = [pl>0.50 for pl in flat_predictions] #boolean output after thresholding

#     tf.logging.info('Classification accuracy using BERT Fine Tuning: {0:0.2%}'.format(matthews_corrcoef(flat_true_labels, flat_predictions)))
    
#     # Print and save classification report
#     tf.logging.info('Test F1 Accuracy: %s' % (str( f1_score(true_bools, pred_bools,average='micro') )))
#     tf.logging.info('Test Flat Accuracy: %s' % (str( accuracy_score(true_bools, pred_bools) )))
    
#     clf_report = classification_report(true_bools,pred_bools)
#     pickle.dump(clf_report, open(os.path.join(sv_dir, 'classification_report.txt'),'wb')) #save report
#     tf.logging.info(str( clf_report ))
                   
if __name__=='__main__': 
    parser = argparse.ArgumentParser(description = 'Fine Tune and Test on corpus of errors')
    parser.add_argument("-d", "--data_dir", help = "Errors corpus file", required=True)
    parser.add_argument("-m", "--model_dir", help = "Directory of transformers-compatible BERT model", required=True)
    parser.add_argument("-s", "--save_dir", help = "Directory to save dataloaders, fine-tuned model, and outputs", required=True)
    parser.add_argument("-e", "--epochs", help = "Numbers of epochs to run fine-tuning", default=3)
    parser.add_argument("-l", "--learn_rate", help = "Learning rate for fine-tuning model", default=3e-5)
    parser.add_argument("-b", "--batch_size", help = "Batch size for fine-tuning model", default=32)
  
    args = parser.parse_args()
        
    main(args)
    
