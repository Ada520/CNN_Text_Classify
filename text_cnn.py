import tensorflow as tf
import numpy as np

class TextCNN(object):
    """
    A CNN for text classification。
    Use an embedding layer, followed by a convolutional, max-pooling and softmax
    """
    def __init__(self,sequence_length,num_classes,vocab_size,
                 embedding_size,filter_size,num_filters):
        """
        :param sequence_length: the length of our sentences,Remeber that we padded all our
        sentence to have the same length (59 for our data set)
        :param num_classes: Number of classes in the output layer, two in our case(positive nad negative)
        :param vocab_size: the size of our vocabulary, this is needed to define the size of our embedding
        layer, which will have shape[vocabulary_size,embedding_size]
        :param embedding_size: The dimensionality of our embedding
        :param filter_size: The number of words we want our convolutinal filters to cover.we will
        have num_filters for each size specified here, for example.[3,4,5]means that we will have
        filters thar slide over 3,4 and 5 words respectively, for a total of 3 * num_fileter filters
        :param num_filters: the number of filters per filter size
        :return:
        """
        #Placeholder for input,output and dropout
        self.input_x=tf.placeholder(tf.int32,[None,sequence_length],name='input_x')
        self.input_y=tf.placeholder(tf.float32,[None,num_classes],name='input_y')
        self.dropout_keep_prob=tf.placeholder(tf.float32,name='dropout_keep_prob')

        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0)

        #Embedding layer
        with tf.device('cpu:0'),tf.name_scope("embedding"):
            W=tf.Variable(tf.random_uniform([vocab_size,embedding_size],-1.0,1.0),name="W")
            self.embedded_chars=tf.nn.embedding_lookup(W,self.input_x)
            self.embedded_chars_expanded=tf.expand_dims(self.embedded_chars,-1)

        # Create a convolution + maxpool layer for each filter size
            pooled_outputs=[]
            for i,filter_size in enumerate(filter_size):
                with tf.name_scope("conv-maxpool-%s"%filter_size):
                    #Convolution Layer
                    filter_shape=[filter_size,embedding_size,1,num_filters]
                    W=tf.Variable(tf.truncated_normal(filter_shape,stddev=0.1),name='W')
                    b=tf.Variable(tf.constant(0.1,shape=[num_filters]),name='b')
                    conv=tf.nn.conv2d(self.embedded_chars_expanded,
                                      W,
                                      strides=[1,1,1,1],
                                      padding='VALID',
                                      name='conv')
                    #Apply nonlinearity
                    h=tf.nn.relu(tf.nn.bias_add(conv,b),name='relu')
                    #Max_pooling over the output
                    pooled=tf.nn.max_pool(h,
                                          ksize=[1,sequence_length-filter_size+1,1,1],
                                          strides=[1,1,1,1],
                                          padding="VALID",
                                          name='pool')
                    pooled_outputs.append(pooled)

                    #Combined all the pooled features
                    num_filters_total=num_filters*len(filter_size)
                    self.h_pool=tf.concat(pooled_outputs,3)
                    self.h_pool_flat=tf.reshape(self.h_pool,[-1,num_filters_total])

































































