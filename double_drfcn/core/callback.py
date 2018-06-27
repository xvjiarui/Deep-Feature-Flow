# --------------------------------------------------------
# Deep Feature Flow
# Copyright (c) 2017 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Modified by Yuwen Xiong
# --------------------------------------------------------
# Based on:
# MX-RCNN
# Copyright (c) 2016 by Contributors
# Licence under The Apache 2.0 License
# https://github.com/ijkguo/mx-rcnn/
# --------------------------------------------------------

import time
import logging
import mxnet as mx


class Speedometer(object):
    def __init__(self, batch_size, frequent=50):
        self.batch_size = batch_size
        self.frequent = frequent
        self.init = False
        self.tic = 0
        self.last_count = 0

    def __call__(self, param):
        """Callback to Show speed."""
        count = param.nbatch
        if self.last_count > count:
            self.init = False
        self.last_count = count

        if self.init:
            if count % self.frequent == 0:
                speed = self.frequent * self.batch_size / (time.time() - self.tic)
                s = ''
                if param.eval_metric is not None:
                    name, value = param.eval_metric.get()
                    s = "Epoch[%d] Batch [%d]\tSpeed: %.2f samples/sec\tTrain-" % (param.epoch, count, speed)
                    for n, v in zip(name, value):
                        s += "%s=%f,\t" % (n, v)
                else:
                    s = "Iter[%d] Batch [%d]\tSpeed: %.2f samples/sec" % (param.epoch, count, speed)

                logging.info(s)
                print(s)
                self.tic = time.time()
        else:
            self.init = True
            self.tic = time.time()

class PhillyProgressCallback(object):
    def __init__(self, total_iter, frequent=50):
        self.total_iter = total_iter
        self.frequent = frequent
        self.cur_iter = 0
    
    def __call__(self, param):
        if self.cur_iter % self.frequent == 0:
            print('\nPROGRESS: {:.2f}%\n'.format(100.0 * self.cur_iter / self.total_iter))
        self.cur_iter += 1

def do_checkpoint(prefix, means, stds, period=1):
    def _callback(iter_no, sym, arg, aux):
        if (iter_no+1) % period == 0:
            arg['bbox_pred_weight_test'] = (arg['bbox_pred_weight'].T * mx.nd.array(stds)).T
            arg['bbox_pred_bias_test'] = arg['bbox_pred_bias'] * mx.nd.array(stds) + mx.nd.array(means)
            mx.model.save_checkpoint(prefix, iter_no + 1, sym, arg, aux)
            arg.pop('bbox_pred_weight_test')
            arg.pop('bbox_pred_bias_test')
    return _callback

class LogMetricsCallback(object):
    """Log metrics periodically in TensorBoard.
    This callback works almost same as `callback.Speedometer`, but write TensorBoard event file
    for visualization. For more usage, please refer https://github.com/dmlc/tensorboard
    Parameters
    ----------
    logging_dir : str
        TensorBoard event file directory.
        After that, use `tensorboard --logdir=path/to/logs` to launch TensorBoard visualization.
    prefix : str
        Prefix for a metric name of `scalar` value.
        You might want to use this param to leverage TensorBoard plot feature,
        where TensorBoard plots different curves in one graph when they have same `name`.
        The follow example shows the usage(how to compare a train and eval metric in a same graph).
    Examples
    --------
    >>> # log train and eval metrics under different directories.
    >>> training_log = 'logs/train'
    >>> evaluation_log = 'logs/eval'
    >>> # in this case, each training and evaluation metric pairs has same name,
    >>> # you can add a prefix to make it separate.
    >>> batch_end_callbacks = [mx.contrib.tensorboard.LogMetricsCallback(training_log)]
    >>> eval_end_callbacks = [mx.contrib.tensorboard.LogMetricsCallback(evaluation_log)]
    >>> # run
    >>> model.fit(train,
    >>>     ...
    >>>     batch_end_callback = batch_end_callbacks,
    >>>     eval_end_callback  = eval_end_callbacks)
    >>> # Then use `tensorboard --logdir=logs/` to launch TensorBoard visualization.
    """
    def __init__(self, logging_dir, prefix=None):
        self.prefix = prefix
        try:
            from mxboard import SummaryWriter
            self.summary_writer = SummaryWriter(logging_dir)
        except ImportError:
            logging.error('You can install tensorboard via `pip install tensorboard`.')

    def __call__(self, param):
        """Callback to log training speed and metrics in TensorBoard."""
        if param.eval_metric is None:
            return
        name_value = param.eval_metric.get_name_value()
        for name, value in name_value:
            if self.prefix is not None:
                name = '%s-%s' % (self.prefix, name)
            self.summary_writer.add_scalar(name, value, global_step=param.nbatch)
