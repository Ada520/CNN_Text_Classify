import tensorflow as tf
import numpy as np

class TextCNN(object):
    """
    A CNN for text classification。
    Use an embedding layer, followed by a convolutional, max-pooling and softmax
    """
