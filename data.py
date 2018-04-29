from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import os, sys, io, re, random
from functools import reduce

#Special Tokens
PAD_TOKEN = '<PAD>'
START_TOKEN = '<GO>'
END_TOKEN = '<EOS>'
INS_TOKEN = '_'
UNKNOWN_TOKEN = ' '

SPECIALS = [PAD_TOKEN, START_TOKEN, END_TOKEN, INS_TOKEN, UNKNOWN_TOKEN]

def split_text_file(data_file, model_dir, eval_fraction):
    """
    Split a Text dataset into train and evaluation
    """
    with io.open(data_file, 'r', encoding='utf-8') as fp:
        data = fp.readlines()

    random.shuffle(data)

    root, ext = os.path.splitext(data_file)
    train_file = os.path.join(model_dir, "{}-train{}".format(root, ext))
    eval_file = os.path.join(model_dir,"{}-eval{}".format(root, ext))
    train_offset = int(len(data)*(1-eval_fraction))

    if not os.path.exists(train_file) or not os.path.exists(eval_file):
        tf.logging.info('Splitting into train and test datasets..')
        with io.open(train_file, 'w', encoding='utf-8') as tfp,\
            io.open(eval_file, 'w', encoding='utf-8') as efp:

            for i, line in enumerate(data):
                if i < train_offset:
                    tfp.write(line)
                else:
                    efp.write(line)

    return train_file, eval_file

def create_vocab(data_files, vocab_fname):
    """
    Creates the character vocabulary file from a
    text dataset. Adds special tokens
    """
    chars = set()
    for data_fname in data_files:
        with io.open(data_fname, 'r', encoding='utf8') as fp:
            raw = fp.read().lower()
            chars.update(raw)

    vocab = list(chars - set(['\t', '\n'])) + SPECIALS
    tf.logging.info('Creating vocab file..')
    with io.open(vocab_fname, 'w', encoding='utf8') as fp:
        fp.write('\n'.join(vocab))

def load_vocab(vocab_fname):
    with io.open(vocab_fname, 'r', encoding='utf-8') as f:
        characters = f.read().splitlines()
        char_vocab = {c:i for i, c in enumerate(characters)}

    return char_vocab, characters

def create_dataset(data_file, char_vocab, num_ep=3, force_create=False, maxlen=500):
    """
    Parses a TSV file with src target columns or text file with single source column.
    Numericalizes using provided vocab, pads and converts to TFRecords
    """
    ep = [INS_TOKEN]
    pad_id = char_vocab[PAD_TOKEN]
    start_id = char_vocab[START_TOKEN]
    end_id = char_vocab[END_TOKEN]
    unk_id = char_vocab[UNKNOWN_TOKEN]

    tfr_file = re.sub(r'\.([^\.]+$)',  '.tfrecord', data_file)
    
    if force_create or not os.path.exists(tfr_file):
        with io.open(data_file, 'r', encoding='utf-8') as fp:
            src, target = [], []
            src_lengths, target_lengths = [], []
            maxlen_src = 0
            # maxlen_target = 0
            tf.logging.info('Processing input file..')
            
            for i, line in enumerate(fp):
                if i % 10000 == 0:
                    tf.logging.info('Read %d lines', i)
                if '\t' in line:
                    s, t = line.strip().lower().split('\t')
                else:
                    s = line.strip().lower()
                    t = ''

                len_s = len(s)
                
                # Insert epsilons, basically spaces
                s_ex = list(reduce(lambda x,y: x + y, zip(list(s), *[ep*len_s for i in range(num_ep)])))
                
                if len(s_ex) + 2 < maxlen:
                    maxlen_src = max(maxlen_src, len(s_ex) + 2)
                    
                    src.append([start_id] + [char_vocab.get(c, unk_id) for c in s_ex] + [end_id])
                    target.append([start_id] + [char_vocab.get(c, unk_id) for c in t] + [end_id])
                    
                    src_lengths.append(len(src[-1]))
                    target_lengths.append(len(target[-1]))

            
            tf.logging.info('Total items %d', len(src))
            tf.logging.info('Max source length is %d', maxlen_src)

            src = [s + [pad_id]*(maxlen_src - len(s)) for s in src]
            
            tf.logging.info('Creating TFRecord file %s..', tfr_file)
            writer = tf.python_io.TFRecordWriter(tfr_file)
            
            for i, (s, t, l_s, l_t) in enumerate(zip(src, target, src_lengths, target_lengths)):

                features = tf.train.Features(feature={
                    'input': tf.train.Feature(int64_list=tf.train.Int64List(value=s)),
                    'input_length': tf.train.Feature(int64_list=tf.train.Int64List(value=[l_s])),
                    'target': tf.train.Feature(int64_list=tf.train.Int64List(value=t)),
                    'target_length': tf.train.Feature(int64_list=tf.train.Int64List(value=[l_t]))
                })
                example = tf.train.Example(features=features)
                writer.write(example.SerializeToString())
                if i % 10000 == 0:
                    tf.logging.info('Wrote %d lines', i)
                    sys.stdout.flush()
            
            writer.close()
            

    return tfr_file

def make_data_iter_fn(filename, batch_size, is_train):
    """
    Provides an input function for estimator that uses tf.data pipeline
    to load dataset from a TFRecord file
    """
    def parse_fn(example_proto):
        features = {
            'input': tf.FixedLenSequenceFeature([], tf.int64, allow_missing=True),
            'target': tf.VarLenFeature(tf.int64),
            'input_length': tf.FixedLenFeature([1], tf.int64),
            'target_length': tf.FixedLenFeature([1], tf.int64)
        }
        parsed_features = tf.parse_single_example(example_proto, features=features)

        parsed_features['input'] = tf.cast(parsed_features['input'], tf.int32)
        t = parsed_features['target']
        parsed_features['target'] = tf.SparseTensor(indices=t.indices, 
                                                    values=tf.cast(t.values, tf.int32),
                                                    dense_shape=t.dense_shape)
        
        parsed_features['input_length'] = tf.cast(parsed_features['input_length'], tf.int32)
        parsed_features['target_length'] = tf.cast(parsed_features['target_length'], tf.int32)

        return parsed_features

    def input_fn():
        dataset = tf.data.TFRecordDataset(filename)
        dataset = dataset.map(parse_fn)
                
        if is_train:
            dataset = dataset.shuffle(500).repeat()
    
        dataset = dataset.batch(batch_size).prefetch(2)
        features = dataset.make_one_shot_iterator().get_next()
        return features, features['target']

    return input_fn