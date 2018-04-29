from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np

import os, sys, io, re
import six

from data import create_vocab, load_vocab
from data import split_text_file, SPECIALS
from data import create_dataset, make_data_iter_fn


flags = tf.flags
FLAGS = flags.FLAGS


flags.DEFINE_integer("train_steps", 0,
                     "The number of steps to run training for.")
flags.DEFINE_integer("eval_steps", 100, "Number of steps in evaluation.")
flags.DEFINE_integer("min_eval_frequency", 101, "Minimum steps between evals")

flags.DEFINE_string("hparams", "", "Comma separated list of hyperparameters")
flags.DEFINE_string("model_name", "ei", "Name of model")                
flags.DEFINE_string("data_file", None, "TSV Data filename")    
flags.DEFINE_float("eval_fraction", 0.05, "Fraction dataset used for evaluation")
flags.DEFINE_string("decode_input_file", None, "File to decode")  

flags.DEFINE_string("vocab_file", "chars.vocab", "Character vocabulary file")


tf.logging.set_verbosity(tf.logging.INFO)

def decode_hparams(vocab_size, overrides=""):

    hp = tf.contrib.training.HParams(
        batch_size=32,
        embedding_size=64,
        char_vocab_size=vocab_size + 1, #Blank label for CTC loss
        hidden_size=128,
        learn_rate=0.0008
    )
    return hp.parse(overrides)

def get_model_dir(model_name):
    model_dir = os.path.join(os.getcwd(), model_name)
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
    return model_dir

def cer(labels, predictions):
    dist = tf.edit_distance(predictions, labels)
    return tf.metrics.mean(dist)

def create_model():
    """
    Actual model function. 
    Refer https://arxiv.org/abs/1610.09565
    """
    def model_fn(features, labels, mode, params):
        hparams = params
        inputs = features['input']
        input_lengths = features['input_length']
        targets = labels
        target_lengths = features['target_length']

        # Flatten input lengths
        input_lengths = tf.reshape(input_lengths, [-1])
        
        with tf.device('/cpu:0'):
            embeddings = tf.Variable(
                    tf.truncated_normal(
                        [hparams.char_vocab_size, hparams.embedding_size],
                        stddev=(1/np.sqrt(hparams.embedding_size))),
                    name='embeddings')

            input_emb = tf.nn.embedding_lookup(embeddings, inputs)

        cell_fw = tf.nn.rnn_cell.BasicLSTMCell(hparams.hidden_size//2)
        cell_bw = tf.nn.rnn_cell.BasicLSTMCell(hparams.hidden_size//2)


        with tf.variable_scope('encoder'):
            # BiLSTM
            enc_outputs, _ = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, input_emb, 
                                input_lengths, dtype=tf.float32)

            enc_outputs = tf.concat(enc_outputs, axis=-1)

        with tf.variable_scope('decoder'):
            # Project to vocab size
            logits = tf.layers.dense(enc_outputs, hparams.char_vocab_size)
            # CTC loss and decoder requires Time major
            logits = tf.transpose(logits, perm=[1, 0, 2])

        loss = None
        eval_metric_ops = None
        train_op = None
        predictions = None

        if mode == tf.estimator.ModeKeys.TRAIN:
            loss = tf.nn.ctc_loss(labels, logits, input_lengths)
            loss = tf.reduce_mean(loss)
            optimizer = tf.contrib.opt.LazyAdamOptimizer(learning_rate=hparams.learn_rate)
            train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())

        elif mode == tf.estimator.ModeKeys.EVAL:
            loss = tf.nn.ctc_loss(labels, logits, input_lengths,
                                 ignore_longer_outputs_than_inputs=True)
            loss = tf.reduce_mean(loss)
            eval_predictions, _ = tf.nn.ctc_greedy_decoder(logits, input_lengths)
            eval_metric_ops = {
                'CER': cer(labels, tf.cast(eval_predictions[0], tf.int32))
            }

        elif mode == tf.estimator.ModeKeys.PREDICT:
            predictions, _ = tf.nn.ctc_greedy_decoder(logits, input_lengths)
            predictions = tf.sparse_tensor_to_dense(tf.cast(predictions[0], tf.int32))
            predictions = {'decoded': predictions}

        return tf.estimator.EstimatorSpec(
                    mode,
                    predictions=predictions,
                    loss=loss,
                    train_op=train_op,
                    eval_metric_ops=eval_metric_ops
                )

    return model_fn

def train():
    """
    Train the model:
    1. Create vocab file from dataset if not created
    2. Split dataset into test/eval if not available
    3. Create TFRecord files if not available
    4. Load TFRecord files using tf.data pipeline
    5. Train model using tf.Estimator
    """
    model_dir = get_model_dir(FLAGS.model_name)
    vocab_file = os.path.join(model_dir, FLAGS.vocab_file)

    if not os.path.exists(vocab_file):
        create_vocab([FLAGS.data_file], vocab_file)
    
    vocab, characters = load_vocab(vocab_file)
    
    train_file, eval_file = split_text_file(FLAGS.data_file, model_dir, FLAGS.eval_fraction)

    train_tfr = create_dataset(train_file, vocab)
    eval_tfr = create_dataset(eval_file, vocab)
    
    hparams = decode_hparams(len(vocab), FLAGS.hparams)
    tf.logging.info('params: %s', str(hparams))

    train_input_fn = make_data_iter_fn(train_tfr, hparams.batch_size, True)
    eval_input_fn = make_data_iter_fn(eval_tfr, hparams.batch_size, False)

    estimator = tf.estimator.Estimator(
            model_fn=create_model(),
            model_dir=model_dir,
            params=hparams,
            config=tf.contrib.learn.RunConfig()
    )

    experiment = tf.contrib.learn.Experiment(
        estimator=estimator,
        train_input_fn=train_input_fn,
        eval_input_fn=eval_input_fn,
        train_steps=FLAGS.train_steps,
        eval_steps=FLAGS.eval_steps,
        min_eval_frequency=FLAGS.min_eval_frequency
    )

    experiment.train_and_evaluate()

def predict():
    """
    Perform transliteration using trained model. Input must be a text 
    file. Converts to a TFRecord first.
    """
    model_dir = get_model_dir(FLAGS.model_name)
    vocab_file = os.path.join(model_dir, FLAGS.vocab_file)
    
    if not os.path.exists(vocab_file):
        raise IOError("Could not find vocabulary file")
    
    vocab, rev_vocab = load_vocab(vocab_file)
    hparams = decode_hparams(len(vocab), FLAGS.hparams)
    tf.logging.info('params: %s', str(hparams))

    if FLAGS.decode_input_file is None:
        raise ValueError("Must provide input field to decode")

    tfr_file = create_dataset(FLAGS.decode_input_file, vocab)
    infer_input_fn = make_data_iter_fn(tfr_file, hparams.batch_size, False)
    
    estimator = tf.estimator.Estimator(
            model_fn=create_model(),
            model_dir=model_dir,
            params=hparams,
            config=tf.contrib.learn.RunConfig()
    )

    y = estimator.predict(input_fn=infer_input_fn, predict_keys=['decoded'])

    ignore_ids = set([vocab[c] for c in SPECIALS] + [0])
    
    decode_output_file = re.sub(r'\..+', '.out.txt', FLAGS.decode_input_file)

    count = 0
    with io.open(decode_output_file, 'w', encoding='utf-8') as fp:
        for pred in y:
            decoded = pred['decoded']
            if len(decoded.shape) == 1:
                decoded = decoded.reshape(1, -1)

            for r in range(decoded.shape[0]):
                fp.write(''.join([rev_vocab[i] for i in decoded[r, :] if i not in ignore_ids]) + '\n')
                count += 1
                if count % 10000 == 0:
                    tf.logging.info('Decoded %d lines', count)

def main(unused_argv):
    if FLAGS.decode_input_file:
        predict()
    elif FLAGS.train_steps > 0:
        train()

tf.app.run()