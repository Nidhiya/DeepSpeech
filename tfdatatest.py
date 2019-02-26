#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function

import functools
import multiprocessing
import numpy as np
import os
import pandas
import tensorflow as tf

from tensorflow.contrib.framework.python.ops import audio_ops as contrib_audio
from util.audio import audiofile_to_input_vector
from util.config import Config, initialize_globals
from util.text import text_to_char_array
from util.flags import create_flags, FLAGS
from timeit import default_timer as timer


tf.enable_eager_execution()


def read_csvs(csv_files):
    source_data = None
    for csv in csv_files:
        file = pandas.read_csv(csv, encoding='utf-8', na_filter=False)
        #FIXME: not cross-platform
        csv_dir = os.path.dirname(os.path.abspath(csv))
        file['wav_filename'] = file['wav_filename'].str.replace(r'(^[^/])', lambda m: os.path.join(csv_dir, m.group(1)))
        if source_data is None:
            source_data = file
        else:
            source_data = source_data.append(file)
    return source_data


def samples_to_mfccs(samples, sample_rate):
    spectrogram = contrib_audio.audio_spectrogram(samples, window_size=512, stride=320, magnitude_squared=True)
    mfccs = contrib_audio.mfcc(spectrogram, sample_rate, dct_coefficient_count=Config.n_input)

    tf.print('mfcc shape:', tf.shape(mfccs))

    # # Add empty initial and final contexts
    # empty_context = tf.fill([1, Config.n_context, Config.n_input], 0.0)
    # mfccs = tf.concat([empty_context, mfccs, empty_context], 1)

    return mfccs


def samples_to_features(samples, sample_rate):
    mfccs = samples_to_mfccs(samples, sample_rate)
    mfccs = tf.squeeze(mfccs, [0])
    # tf.print('after ctx:', tf.shape(mfccs))

    # window_width = 2*Config.n_context + 1
    # num_channels = Config.n_input

    # tf.print('shape before conv:', tf.shape(mfccs))

    # # Create a constant convolution filter using an identity matrix, so that the
    # # convolution returns patches of the input tensor as is, and we can create
    # # overlapping windows over the MFCCs.
    # eye_filter = tf.constant(np.eye(window_width * num_channels)
    #                            .reshape(window_width, num_channels, window_width * num_channels), tf.float32)

    # # Create overlapping windows
    # mfccs = tf.nn.conv1d(mfccs, eye_filter, stride=1, padding='SAME')

    # tf.print('shape after conv:', tf.shape(mfccs))

    # # Remove dummy depth dimension and reshape into n_windows, window_width, window_height
    # mfccs = tf.reshape(mfccs, [-1, window_width, num_channels])

    # # tf.print('after windows:', tf.shape(mfccs))

    return mfccs, tf.shape(mfccs)[0]


def file_to_features(wav_filename, transcript):
    samples = tf.read_file(wav_filename)
    decoded = contrib_audio.decode_wav(samples, desired_channels=1)
    features, features_len = samples_to_features(decoded.audio, decoded.sample_rate)

    return features, features_len, transcript


def file_to_mfccs(wav_filename, transcript):
    samples = tf.read_file(wav_filename)
    decoded = contrib_audio.decode_wav(samples, desired_channels=1)
    return samples_to_mfccs(decoded.audio, decoded.sample_rate)


def main(_):
    initialize_globals()

    print('Reading input files and processing transcripts...')
    df = read_csvs(FLAGS.train_files.split(','))
    df.sort_values(by='wav_filesize', inplace=True)
    df['transcript'] = df['transcript'].apply(functools.partial(text_to_char_array, alphabet=Config.alphabet))

    def generate_values():
        for _, row in df.iterrows():
            yield tf.cast(row.wav_filename, tf.string), tf.cast(row.transcript, tf.int32)

    num_gpus = len(Config.available_devices)

    print('Creating input pipeline...')
    dataset = (tf.data.Dataset.from_generator(generate_values,
                                              output_types=(tf.string, tf.int32),
                                              output_shapes=([], [None]))
                              .map(file_to_features, num_parallel_calls=multiprocessing.cpu_count())
                              .prefetch(FLAGS.train_batch_size * num_gpus * 8)
                              .cache()
                              .padded_batch(FLAGS.train_batch_size,
                                            padded_shapes=([None, Config.n_input], [], [None]),
                                            drop_remainder=True)
                              .repeat(FLAGS.epoch)
              )

    batch_count = 0
    batch_size = None
    batch_time = 0

    start_time = timer()
    for batch_x, batch_x_len, batch_y in dataset:
        tf.print('batch x shape from iter: ', tf.shape(batch_x))
        batch_count += 1
        batch_size = batch_x.shape[0]
        print('.', end='')
    total_time = timer() - start_time
    print()
    print('Iterating through dataset took {:.3f}s, {} batches, {} epochs, batch size from dataset = {}, average batch time = {:.3f}'.format(total_time, batch_count, FLAGS.epoch, batch_size, batch_time/batch_count))


if __name__ == '__main__' :
    create_flags()
    tf.app.run(main)
