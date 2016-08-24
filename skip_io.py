"""
RandomSkipResizeIter is written by @taoari, from MXNet issue: https://github.com/dmlc/mxnet/issues/2968
"""
import mxnet as mx
import numpy as np

class RandomSkipResizeIter(mx.io.DataIter):
    """Resize a DataIter to given number of batches per epoch.
    May produce incomplete batch in the middle of an epoch due
    to padding from internal iterator.

    Parameters
    ----------
    data_iter : DataIter
        Internal data iterator.
    max_random_skip : maximum random skip number
        If max_random_skip is 1, no random skip.
    size : number of batches per epoch to resize to.
    reset_internal : whether to reset internal iterator on ResizeIter.reset
    """

    def __init__(self, data_iter, size, skip_ratio=0.5, reset_internal=False):
        super(RandomSkipResizeIter, self).__init__()
        self.data_iter = data_iter
        self.size = size
        self.reset_internal = reset_internal
        self.cur = 0
        self.current_batch = None
        self.prev_batch = None
        self.skip_ratio = skip_ratio

        self.provide_data = data_iter.provide_data
        self.provide_label = data_iter.provide_label
        self.batch_size = data_iter.batch_size

    def reset(self):
        self.cur = 0
        if self.reset_internal:
            self.data_iter.reset()

    def __get_next(self):
        try:
            return self.data_iter.next()
        except StopIteration:
            self.data_iter.reset()
            return self.data_iter.next()

    def iter_next(self):
        if self.cur == self.size:
            return False

        data, label = [], []
        if self.current_batch is None:
            # very first
            batch = self.__get_next()
            self.current_batch = mx.io.DataBatch(data=[mx.nd.empty(batch.data[0].shape)], label=[mx.nd.empty(batch.label[0].shape)])
            keep = np.random.rand(self.batch_size) > self.skip_ratio
            batch_data = batch.data[0].asnumpy()
            batch_label = batch.label[0].asnumpy()
            data.extend(batch_data[keep])
            label.extend(batch_label[keep])
        elif self.prev_batch is not None:
            # prev_batch
            batch_data, batch_label = self.prev_batch
            data.extend(batch_data)
            label.extend(batch_label)

        while len(data) < self.batch_size:
            batch = self.__get_next()
            keep = np.random.rand(self.batch_size) > self.skip_ratio
            batch_data = batch.data[0].asnumpy()
            batch_label = batch.label[0].asnumpy()
            data.extend(batch_data[keep])
            label.extend(batch_label[keep])

        if len(data) > self.batch_size:
            self.prev_batch = data[self.batch_size:], label[self.batch_size:]
        else:
            self.prev_batch = None
        self.current_batch.data[0][:] = np.asarray(data[:self.batch_size])
        self.current_batch.label[0][:] = np.asarray(label[:self.batch_size])

        self.cur += 1
        return True

    def getdata(self):
        return self.current_batch.data

    def getlabel(self):
        return self.current_batch.label

    def getindex(self):
        return self.current_batch.index

    def getpad(self):
        return self.current_batch.pad