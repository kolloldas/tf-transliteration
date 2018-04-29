# tf-transliteration
TensorFlow implementation of the Google Transliteration model in the paper [Sequence-to-sequence neural network models for transliteration](https://arxiv.org/abs/1610.09565). Specifically it implements the BiLSTM-CTC model using Epsilon Insertion.

### Usage
For the code to run you need Python 3.5+ and Tensorflow 1.5+

**Training:**
The dataset must be provided as tab separated files. You can get an English-to-Hindi transliteration dataset [here](http://cse.iitkgp.ac.in/resgrp/cnerg/qa/fire13translit/Hindi%20-%20Word%20Transliteration%20Pairs%201.txt)
Train the model for 10,000 steps, evaluating every 1000 steps:
```
python transliterate.py --data_file=<filename> --train_steps=10000 --eval_steps=100 --min_eval_frequency=1000
```
During evaluation the CER will be displayed.

**Predicting:**
For predictions the inputs should be provided as a text file with one example per line.
```
python transliterate.py --decode_input_file=<filename>
```
The predictions will be written into a corresponding `filename.out.ext`
