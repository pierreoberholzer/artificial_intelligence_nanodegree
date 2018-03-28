from keras import backend as K
from keras.models import Model
from keras.layers import (BatchNormalization, Conv1D, Dense, Input, 
    TimeDistributed, Activation, Bidirectional, SimpleRNN, GRU, LSTM, Dropout,
    MaxPooling1D)

def simple_rnn_model(input_dim, output_dim=29):
    """ Build a recurrent network for speech 
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # Add recurrent layer
    simp_rnn = GRU(output_dim, return_sequences=True, 
                 implementation=2, name='rnn')(input_data)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(simp_rnn)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: x
    print(model.summary())
    return model

def rnn_model(input_dim, units, activation, output_dim=29):
    """ Build a recurrent network for speech 
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # Add recurrent layer
    simp_rnn = GRU(units, activation=activation,
        return_sequences=True, implementation=2, name='rnn')(input_data)
    # TODO: Add batch normalization 
    bn_rnn = BatchNormalization(name='bn_rnn')(simp_rnn)
    # TODO: Add a TimeDistributed(Dense(output_dim)) layer
    time_dense = TimeDistributed(Dense(output_dim))(bn_rnn)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: x
    print(model.summary())
    return model


def cnn_rnn_model(input_dim, filters, kernel_size, conv_stride,
    conv_border_mode, units, output_dim=29):
    """ Build a recurrent + convolutional network for speech 
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # Add convolutional layer
    conv_1d = Conv1D(filters, kernel_size, 
                     strides=conv_stride, 
                     padding=conv_border_mode,
                     activation='relu',
                     name='conv1d')(input_data)
    # Add batch normalization
    bn_cnn = BatchNormalization(name='bn_conv_1d')(conv_1d)
    # Add a recurrent layer
    simp_rnn = SimpleRNN(units, activation='relu',
        return_sequences=True, implementation=2, name='rnn')(bn_cnn)
    # TODO: Add batch normalization
    bn_rnn = BatchNormalization(name='bn_simp_rnn')(simp_rnn)
    # TODO: Add a TimeDistributed(Dense(output_dim)) layer
    time_dense = TimeDistributed(Dense(output_dim))(bn_rnn)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: cnn_output_length(
        x, kernel_size, conv_border_mode, conv_stride)
    print(model.summary())
    return model

def cnn_output_length(input_length, filter_size, border_mode, stride,
                       dilation=1):
    """ Compute the length of the output sequence after 1D convolution along
        time. Note that this function is in line with the function used in
        Convolution1D class from Keras.
    Params:
        input_length (int): Length of the input sequence.
        filter_size (int): Width of the convolution kernel.
        border_mode (str): Only support `same` or `valid`.
        stride (int): Stride size used in 1D convolution.
        dilation (int)
    """
    if input_length is None:
        return None
    assert border_mode in {'same', 'valid'}
    dilated_filter_size = filter_size + (filter_size - 1) * (dilation - 1)
    if border_mode == 'same':
        output_length = input_length
    elif border_mode == 'valid':
        output_length = input_length - dilated_filter_size + 1
    return (output_length + stride - 1) // stride

def deep_rnn_model(input_dim, units, recur_layers, output_dim=29):
    """ Build a deep recurrent network for speech 
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # TODO: Add recurrent layers, each with batch normalization
    
    bn_rnn = None
    
    for i in range(recur_layers):
        layer_name_rn = 'deep_rnn_'+str(i)
        layer_name_bn = 'deep_bn_'+str(i)
        
        if i == 0:
            previous_layer = input_data
        else:
            previous_layer = bn_rnn
            
        simp_rnn = GRU(units, activation='relu', 
                             return_sequences=True, implementation=2, name=layer_name_rn)(previous_layer)        
        bn_rnn = BatchNormalization(name=layer_name_bn)(simp_rnn)
        
    # TODO: Add a TimeDistributed(Dense(output_dim)) layer
    time_dense = TimeDistributed(Dense(output_dim))(bn_rnn)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: x
    print(model.summary())
    return model

def bidirectional_rnn_model(input_dim, units, output_dim=29):
    """ Build a bidirectional recurrent network for speech
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # TODO: Add bidirectional recurrent layer
        #simp_rnn = GRU(units, activation=activation,
                        #return_sequences=True, implementation=2, name='rnn')(input_data)
    bidir_rnn = Bidirectional(LSTM(units, return_sequences=True))(input_data)
    # TODO: Add a TimeDistributed(Dense(output_dim)) layer
    time_dense = TimeDistributed(Dense(output_dim))(bidir_rnn)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: x
    print(model.summary())
    return model

def deep_bidirectional_rnn_model(input_dim, units, recur_layers, output_dim=29):
    """ Build a deep recurrent network for speech 
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # TODO: Add recurrent layers, each with batch normalization
    
    bn_rnn = None
    
    for i in range(recur_layers):
        layer_name_rn = 'bidir_deep_rnn_'+str(i)
        layer_name_bn = 'bidir_deep_bn_'+str(i)
        
        if i == 0:
            previous_layer = input_data
        else:
            previous_layer = bn_rnn
            
        bidir_rnn = Bidirectional(LSTM(units, return_sequences=True),name=layer_name_rn)(previous_layer)        
        bn_rnn = BatchNormalization(name=layer_name_bn)(bidir_rnn)
        
    # TODO: Add a TimeDistributed(Dense(output_dim)) layer
    time_dense = TimeDistributed(Dense(output_dim))(bn_rnn)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: x
    print(model.summary())
    return model

def final_model(input_dim, filters, kernel_size, conv_stride,
    conv_border_mode, units, recur_layers, pool=True, drop=True, RNN='GRU',output_dim=29):
    """ Build a deep network for speech 
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # TODO: Specify the layers in your network
    # Add convolutional layer
    conv_1d = Conv1D(filters, kernel_size, 
                     strides=conv_stride, 
                     padding=conv_border_mode,
                     activation='relu',
                     name='conv1d')(input_data)
    
    # Add max pooling
    
    if pool:
        pool_size=2
        pool_stride = conv_stride
        pool_border_mode = conv_border_mode
        pool_cnn = MaxPooling1D(pool_size=pool_size, strides=pool_stride, padding='valid',name='pool_cnn')(conv_1d)
    else:
        pool_cnn = conv_1d
    
    # Add batch normalization
    bn_cnn = BatchNormalization(name='bn_conv_1d')(pool_cnn)
      
    # Stacked RNNS
    
    drop_rnn= None
    
    for i in range(recur_layers):
        layer_name_rn = 'deep_rnn_'+str(i)
        layer_name_bn = 'deep_bn_'+str(i)
        layer_name_drop = 'deep_drop'+str(i)
        
        if i == 0:
            previous_layer = bn_cnn
        else:
            previous_layer = drop_rnn
          
        if RNN == 'GRU':
            simp_rnn = GRU(units, activation='relu', 
                                 return_sequences=True, implementation=2, name=layer_name_rn)(previous_layer)
        elif RNN == 'LSTM':
            simp_rnn = LSTM(units, activation='tanh', 
                                 return_sequences=True, implementation=2, name=layer_name_rn)(previous_layer)
        elif RNN == 'bidir':
            simp_rnn = Bidirectional(LSTM(units, return_sequences=True),name=layer_name_rn)(previous_layer) 
        
        bn_rnn = BatchNormalization(name=layer_name_bn)(simp_rnn)
        
        if drop:
            drop_rnn = Dropout(0.2,name=layer_name_drop)(bn_rnn)
        else:
            drop_rnn = bn_rnn
        
    # TODO: Add a TimeDistributed(Dense(output_dim)) layer
    time_dense = TimeDistributed(Dense(output_dim))(drop_rnn)
    
    # TODO: Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    # TODO: Specify model.output_length
    
    if pool:
        model.output_length = lambda x: cnn_output_length(
                                            cnn_output_length(
                                                x, kernel_size, conv_border_mode, conv_stride), pool_size, pool_border_mode, pool_stride)
    else:
        model.output_length = lambda x: cnn_output_length(
            x, kernel_size, conv_border_mode, conv_stride)

    print(model.summary())
    return model