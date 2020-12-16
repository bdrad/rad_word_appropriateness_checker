# Word Appropriateness Detector

## Relevant Files

### error_generator.py

A tool to generate error according to the methodology in our manuscript. To properly generate phonetically similar errors, you need to prepare your own vocabulary file. You can edit the method `_set_vocab()`to fit your vocab file into the algorithm.

### generate_error_corpus.py

Run this script to generate corrupted corpus using our `error_generator.py` tool. It will create a pickle file where each row contains the original and corrupted sentence, the changed index and words, and the oneshot vector. In the oneshot vector, `True` means the word at this position is changed while `False`  means it remains the same. The default output file name is `error_corpus.pickle`.

The input VOCAB_FILE should be either a csv or a txt file, being consistent with the vocab file in `error_generator.py` . The input CORPUS_FILE should be a plain text file, with one sentence per line. You can change the proportion of the error types by modifying the script.  To run the script:

```python
python generate_error_corpus.py \
  --vocab_file=$VOCAB_FILE \
  --corpus_file=$CORPUS_FILE \
  --save_dir=$SAVE_DIR
```

### finetune.py

Run this python script to fine tune your BERT model. The model we adapt in finetuning is BertForTokenClassification from huggingface [transformers](https://github.com/huggingface/transformers) library. On top of the hidden states output from the BERT model, it applies a linear layer for token classification. 

Arguments：

* data_dir (required): The errored `.pickle` file.
* model_dir (required): Directory of transformers-compatible BERT model or a model identifier from transformers, e.g. 'bert-base-uncased'. The parameters will be initialised from this model.
* save_dir (required): Directory to save dataloaders, fine-tuned model, and outputs.
* epochs: Numbers of epochs to run fine-tuning. Default: 3.
* learn_rate: Learning rate for fine-tuning model. Default: 3e-5.
* batch_size: Batch size for fine-tuning model. Default: 32.

To start fine-tuning, just run:

```python
EPOCHS=3 
LEARN_RATE=3e-5
BATCH_SIZE=32

python finetune.py \
    --data_dir=$IN_DATA \
    --model_dir=$MODEL_DIR \
    --save_dir=$SAVE_DIR \
    --epochs=$EPOCHS \
    --learn_rate=$LEARN_RATE \
    --batch_size=$BATCH_SIZE 
```

### test.py

Run this script to test your finetuned model. It will compute micro-average ROC curve and ROC area and create a plot to illustrate.

Arguments:

* data_dir (required): The errored `.pickle` file.
* model_dir (required): Directory of transformers-compatible BERT model (or `SAVE_DIR` of finetune.py).
* save_dir (required): Directory to save dataloaders, fine-tuned model, and outputs.
* batch_size: Batch size for running the test. Default: 32.

To start testing, run:

```python
BATCH_SIZE=32

python test.py \
    --data_dir=$IN_DATA \
    --model_dir=$MODEL_DIR \
    --save_dir=$SAVE_DIR \
    --batch_size=$BATCH_SIZE 
```

### sample_data

In this folder, there're two sample files which serve as a good start for you to quickly run the code. In sample_corpus.txt, we scraped the findings section of ten radiology reports from [NATIONALRad](https://nationalrad.com/radiology/reports/). We performed sentence segmentation with [spaCy](https://spacy.io/). 

## Instructions for Running Locally

1. Clone this repository to your local PC.
2. Install Python 3.6 or above.
3. Install everything in the requirements.txt file.
4. Run generate_error_corpus.py script to generate data for fine-tuning. We've provided you with a sample corpus file and a vocab file. Feel free to generate your own errors with them.
5. Run finetuning.py to fine tune the BERT model for dictation error detection. You'll use the `error_corpus.pickle` that you created in step 4.
6. Run test.py to evaluate your finetuned model.  You can create another corrupted corpus (step 4) to test it.