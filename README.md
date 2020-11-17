# Word Appropriateness Detector

## Relevant Files

### error_generator.py

A tool to generate error according to the methodology in our manuscript. To properly generate phonetically similar errors, you need to prepare your own vocabulary.

### generate_error_corpus.py

Run this script to generate corrupted corpus using our error_generator tool. It will create a pickle file where each row contains the original and corrupted sentence, the changed index and words, and the oneshot vector. In the oneshot vector, `True` means the word at this position is changed while `False`  means it remains the same.

The proportion of each error type is set according to the paper (Jama paper). You can change it by modifying the script. To run the script:

```python
python generate_error_corpus.py \
  --vocab_file=$VOCAB_FILE \
  --corpus_file=$CORPUS_FILE \
  --save_dir=$SAVE_DIR
```



### finetune.py

Run this python script to fine tune your BERT model. The model we adapt in finetuning is BertForTokenClassification from huggingface [transformers](https://github.com/huggingface/transformers) library. On top of the hidden states output from the BERT model, it applies a linear layer for token classification. 

Arguments：

* data_dir (required): The errored .pickle file.
* model_dir (required): Directory of transformers-compatible BERT model.
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
    --save_dir=$SV_DIR \
    --epochs=$EPOCHS \
    --learn_rate=$LEARN_RATE \
    --batch_size=$BATCH_SIZE 
```



## Instructions for Running Locally

1. Clone this repository to your local PC.
2. Install Python 3.6 or above.
3. Install everything in the requirements.txt file.
4. Run generate_error_corpus.py script to generate data for fine-tuning.
5. Run finetuning.py to fine tune the BERT model for dictation error detection.