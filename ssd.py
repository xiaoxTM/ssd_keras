"""Keras implementation of SSD."""

import keras.backend as K
from keras.layers import Activation
from keras.layers import Conv2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import GlobalAveragePooling2D
from keras.layers import Input
from keras.layers import MaxPooling2D
from keras.layers import concatenate
from keras.layers import Reshape
from keras.layers import ZeroPadding2D
from keras.models import Model

from layers import Normalize
from layers import PriorBox


def SSD300(input_shape, num_classes=21):
    """SSD300 architecture.

    # Arguments
        input_shape: Shape of the input image,
            expected to be either (300, 300, 3) or (3, 300, 300)(not tested).
        num_classes: Number of classes including background.

    # References
        https://arxiv.org/abs/1512.02325
    """
    print('begin building networks')
    kernel_size=(3, 3)

    net = {}
    # Block 1
    input_tensor = Input(shape=input_shape)
    img_size = (input_shape[1], input_shape[0])
    net['input'] = input_tensor
    net['conv1_1'] = Conv2D(64, kernel_size,
                                   activation='relu',
                                   padding='same',
                                   name='conv1_1')(net['input'])
    net['conv1_2'] = Conv2D(64, kernel_size,
                                   activation='relu',
                                   padding='same',
                                   name='conv1_2')(net['conv1_1'])
    net['pool1'] = MaxPooling2D((2, 2), strides=(2, 2), padding='same',
                                name='pool1')(net['conv1_2'])
    # Block 2
    net['conv2_1'] = Conv2D(128, kernel_size,
                                   activation='relu',
                                   padding='same',
                                   name='conv2_1')(net['pool1'])
    net['conv2_2'] = Conv2D(128, kernel_size,
                                   activation='relu',
                                   padding='same',
                                   name='conv2_2')(net['conv2_1'])
    net['pool2'] = MaxPooling2D((2, 2), strides=(2, 2), padding='same',
                                name='pool2')(net['conv2_2'])
    # Block 3
    net['conv3_1'] = Conv2D(256, kernel_size,
                                   activation='relu',
                                   padding='same',
                                   name='conv3_1')(net['pool2'])
    net['conv3_2'] = Conv2D(256, kernel_size,
                                   activation='relu',
                                   padding='same',
                                   name='conv3_2')(net['conv3_1'])
    net['conv3_3'] = Conv2D(256, kernel_size,
                                   activation='relu',
                                   padding='same',
                                   name='conv3_3')(net['conv3_2'])
    net['pool3'] = MaxPooling2D((2, 2), strides=(2, 2), padding='same',
                                name='pool3')(net['conv3_3'])
    # Block 4
    net['conv4_1'] = Conv2D(512, kernel_size,
                                   activation='relu',
                                   padding='same',
                                   name='conv4_1')(net['pool3'])
    net['conv4_2'] = Conv2D(512, kernel_size,
                                   activation='relu',
                                   padding='same',
                                   name='conv4_2')(net['conv4_1'])
    net['conv4_3'] = Conv2D(512, kernel_size,
                                   activation='relu',
                                   padding='same',
                                   name='conv4_3')(net['conv4_2'])
    net['pool4'] = MaxPooling2D((2, 2), strides=(2, 2), padding='same',
                                name='pool4')(net['conv4_3'])
    # Block 5
    net['conv5_1'] = Conv2D(512, kernel_size,
                                   activation='relu',
                                   padding='same',
                                   name='conv5_1')(net['pool4'])
    net['conv5_2'] = Conv2D(512, kernel_size,
                                   activation='relu',
                                   padding='same',
                                   name='conv5_2')(net['conv5_1'])
    net['conv5_3'] = Conv2D(512, kernel_size,
                                   activation='relu',
                                   padding='same',
                                   name='conv5_3')(net['conv5_2'])
    net['pool5'] = MaxPooling2D((3, 3), strides=(1, 1), padding='same',
                                name='pool5')(net['conv5_3'])
    # FC6
    net['fc6'] = Conv2D(1024, kernel_size, dilation_rate=(6, 6),
                                     activation='relu', padding='same',
                                     name='fc6')(net['pool5'])
    # x = Dropout(0.5, name='drop6')(x)
    # FC7
    net['fc7'] = Conv2D(1024, (1, 1), activation='relu',
                               padding='same', name='fc7')(net['fc6'])
    # x = Dropout(0.5, name='drop7')(x)
    # Block 6
    net['conv6_1'] = Conv2D(256, (1, 1), activation='relu',
                                   padding='same',
                                   name='conv6_1')(net['fc7'])
    net['conv6_2'] = Conv2D(512, kernel_size, strides=(2, 2),
                                   activation='relu', padding='same',
                                   name='conv6_2')(net['conv6_1'])
    # Block 7
    net['conv7_1'] = Conv2D(128, (1, 1), activation='relu',
                                   padding='same',
                                   name='conv7_1')(net['conv6_2'])
    net['conv7_2'] = ZeroPadding2D()(net['conv7_1'])
    net['conv7_2'] = Conv2D(256, kernel_size, strides=(2, 2),
                                   activation='relu', padding='valid',
                                   name='conv7_2')(net['conv7_2'])
    # Block 8
    net['conv8_1'] = Conv2D(128, (1, 1), activation='relu',
                                   padding='same',
                                   name='conv8_1')(net['conv7_2'])
    net['conv8_2'] = Conv2D(256, kernel_size, strides=(2, 2),
                                   activation='relu', padding='same',
                                   name='conv8_2')(net['conv8_1'])
    # Last Pool
    net['pool6'] = GlobalAveragePooling2D(name='pool6')(net['conv8_2'])
    print('base network built')

    # Prediction from conv4_3
    net['conv4_3_norm'] = Normalize(20, name='conv4_3_norm')(net['conv4_3'])
    num_priors = 3
    x = Conv2D(num_priors * 4, kernel_size, padding='same',
                      name='conv4_3_norm_mbox_loc')(net['conv4_3_norm'])
    net['conv4_3_norm_mbox_loc'] = x
    flatten = Flatten(name='conv4_3_norm_mbox_loc_flat')
    net['conv4_3_norm_mbox_loc_flat'] = flatten(net['conv4_3_norm_mbox_loc'])
    name = 'conv4_3_norm_mbox_conf'
    if num_classes != 21:
        name += '_{}'.format(num_classes)
    x = Conv2D(num_priors * num_classes, kernel_size, padding='same',
                      name=name)(net['conv4_3_norm'])
    net['conv4_3_norm_mbox_conf'] = x
    flatten = Flatten(name='conv4_3_norm_mbox_conf_flat')
    net['conv4_3_norm_mbox_conf_flat'] = flatten(net['conv4_3_norm_mbox_conf'])
    priorbox = PriorBox(img_size, 30.0, aspect_ratios=[2],
                        variances=[0.1, 0.1, 0.2, 0.2],
                        name='conv4_3_norm_mbox_priorbox')
    net['conv4_3_norm_mbox_priorbox'] = priorbox(net['conv4_3_norm'])
    print('conv4_3_norm_mbox_priorbox built')
    # Prediction from fc7
    num_priors = 6
    net['fc7_mbox_loc'] = Conv2D(num_priors * 4, kernel_size,
                                        padding='same',
                                        name='fc7_mbox_loc')(net['fc7'])
    flatten = Flatten(name='fc7_mbox_loc_flat')
    net['fc7_mbox_loc_flat'] = flatten(net['fc7_mbox_loc'])
    name = 'fc7_mbox_conf'
    if num_classes != 21:
        name += '_{}'.format(num_classes)
    net['fc7_mbox_conf'] = Conv2D(num_priors * num_classes, (3, 3),
                                         padding='same',
                                         name=name)(net['fc7'])
    flatten = Flatten(name='fc7_mbox_conf_flat')
    net['fc7_mbox_conf_flat'] = flatten(net['fc7_mbox_conf'])
    priorbox = PriorBox(img_size, 60.0, max_size=114.0, aspect_ratios=[2, 3],
                        variances=[0.1, 0.1, 0.2, 0.2],
                        name='fc7_mbox_priorbox')
    net['fc7_mbox_priorbox'] = priorbox(net['fc7'])
    print('fc7_mbox_priorbox built')
    # Prediction from conv6_2
    num_priors = 6
    x = Conv2D(num_priors * 4, kernel_size, padding='same',
                      name='conv6_2_mbox_loc')(net['conv6_2'])
    net['conv6_2_mbox_loc'] = x
    flatten = Flatten(name='conv6_2_mbox_loc_flat')
    net['conv6_2_mbox_loc_flat'] = flatten(net['conv6_2_mbox_loc'])
    name = 'conv6_2_mbox_conf'
    if num_classes != 21:
        name += '_{}'.format(num_classes)
    x = Conv2D(num_priors * num_classes, kernel_size, padding='same',
                      name=name)(net['conv6_2'])
    net['conv6_2_mbox_conf'] = x
    flatten = Flatten(name='conv6_2_mbox_conf_flat')
    net['conv6_2_mbox_conf_flat'] = flatten(net['conv6_2_mbox_conf'])
    priorbox = PriorBox(img_size, 114.0, max_size=168.0, aspect_ratios=[2, 3],
                        variances=[0.1, 0.1, 0.2, 0.2],
                        name='conv6_2_mbox_priorbox')
    net['conv6_2_mbox_priorbox'] = priorbox(net['conv6_2'])
    print('conv6_2_mbox_priorbox built')
    # Prediction from conv7_2
    num_priors = 6
    x = Conv2D(num_priors * 4, kernel_size, padding='same',
                      name='conv7_2_mbox_loc')(net['conv7_2'])
    net['conv7_2_mbox_loc'] = x
    flatten = Flatten(name='conv7_2_mbox_loc_flat')
    net['conv7_2_mbox_loc_flat'] = flatten(net['conv7_2_mbox_loc'])
    name = 'conv7_2_mbox_conf'
    if num_classes != 21:
        name += '_{}'.format(num_classes)
    x = Conv2D(num_priors * num_classes, kernel_size, padding='same',
                      name=name)(net['conv7_2'])
    net['conv7_2_mbox_conf'] = x
    flatten = Flatten(name='conv7_2_mbox_conf_flat')
    net['conv7_2_mbox_conf_flat'] = flatten(net['conv7_2_mbox_conf'])
    priorbox = PriorBox(img_size, 168.0, max_size=222.0, aspect_ratios=[2, 3],
                        variances=[0.1, 0.1, 0.2, 0.2],
                        name='conv7_2_mbox_priorbox')
    net['conv7_2_mbox_priorbox'] = priorbox(net['conv7_2'])
    print('conv7_2_mbox_priorbox built')
    # Prediction from conv8_2
    num_priors = 6
    x = Conv2D(num_priors * 4, kernel_size, padding='same',
                      name='conv8_2_mbox_loc')(net['conv8_2'])
    net['conv8_2_mbox_loc'] = x
    flatten = Flatten(name='conv8_2_mbox_loc_flat')
    net['conv8_2_mbox_loc_flat'] = flatten(net['conv8_2_mbox_loc'])
    name = 'conv8_2_mbox_conf'
    if num_classes != 21:
        name += '_{}'.format(num_classes)
    x = Conv2D(num_priors * num_classes, kernel_size, padding='same',
                      name=name)(net['conv8_2'])
    net['conv8_2_mbox_conf'] = x
    flatten = Flatten(name='conv8_2_mbox_conf_flat')
    net['conv8_2_mbox_conf_flat'] = flatten(net['conv8_2_mbox_conf'])
    priorbox = PriorBox(img_size, 222.0, max_size=276.0, aspect_ratios=[2, 3],
                        variances=[0.1, 0.1, 0.2, 0.2],
                        name='conv8_2_mbox_priorbox')
    net['conv8_2_mbox_priorbox'] = priorbox(net['conv8_2'])
    print('conv8_2_mbox_priorbox built')
    # Prediction from pool6
    num_priors = 6
    x = Dense(num_priors * 4, name='pool6_mbox_loc_flat')(net['pool6'])
    net['pool6_mbox_loc_flat'] = x
    name = 'pool6_mbox_conf_flat'
    if num_classes != 21:
        name += '_{}'.format(num_classes)
    x = Dense(num_priors * num_classes, name=name)(net['pool6'])
    net['pool6_mbox_conf_flat'] = x
    priorbox = PriorBox(img_size, 276.0, max_size=330.0, aspect_ratios=[2, 3],
                        variances=[0.1, 0.1, 0.2, 0.2],
                        name='pool6_mbox_priorbox')
    if K.image_dim_ordering() == 'tf':
        target_shape = (1, 1, 256)
    else:
        target_shape = (256, 1, 1)
    net['pool6_reshaped'] = Reshape(target_shape,
                                    name='pool6_reshaped')(net['pool6'])
    net['pool6_mbox_priorbox'] = priorbox(net['pool6_reshaped'])
    print('pool6_mbox_priorbox built')
    # Gather all predictions
    net['mbox_loc'] = concatenate([net['conv4_3_norm_mbox_loc_flat'],
                             net['fc7_mbox_loc_flat'],
                             net['conv6_2_mbox_loc_flat'],
                             net['conv7_2_mbox_loc_flat'],
                             net['conv8_2_mbox_loc_flat'],
                             net['pool6_mbox_loc_flat']],
                            axis=1, name='mbox_loc')
    net['mbox_conf'] = concatenate([net['conv4_3_norm_mbox_conf_flat'],
                              net['fc7_mbox_conf_flat'],
                              net['conv6_2_mbox_conf_flat'],
                              net['conv7_2_mbox_conf_flat'],
                              net['conv8_2_mbox_conf_flat'],
                              net['pool6_mbox_conf_flat']],
                             axis=1, name='mbox_conf')
    net['mbox_priorbox'] = concatenate([net['conv4_3_norm_mbox_priorbox'],
                                  net['fc7_mbox_priorbox'],
                                  net['conv6_2_mbox_priorbox'],
                                  net['conv7_2_mbox_priorbox'],
                                  net['conv8_2_mbox_priorbox'],
                                  net['pool6_mbox_priorbox']],
                                 axis=1, name='mbox_priorbox')
    print('gathering all prediction layers built')
    if hasattr(net['mbox_loc'], '_keras_shape'):
        # divide 4 for [xmin, ymin, xmax, ymax]
        num_boxes = net['mbox_loc']._keras_shape[-1] // 4
    elif hasattr(net['mbox_loc'], 'int_shape'):
        num_boxes = K.int_shape(net['mbox_loc'])[-1] // 4
    net['mbox_loc'] = Reshape((num_boxes, 4),
                              name='mbox_loc_final')(net['mbox_loc'])
    net['mbox_conf'] = Reshape((num_boxes, num_classes),
                               name='mbox_conf_logits')(net['mbox_conf'])
    net['mbox_conf'] = Activation('softmax',
                                  name='mbox_conf_final')(net['mbox_conf'])
    net['predictions'] = concatenate([net['mbox_loc'],
                               net['mbox_conf'],
                               net['mbox_priorbox']],
                               axis=2, name='predictions')
    print('prediction layers built')
    model = Model(net['input'], net['predictions'])
    return model

if __name__=='__main__':
    net = SSD300((300, 300, 3))
    import netstruct as ns
    ns.model2graph(net, to_file='model.svg', show_shapes=True, show_params=True, attrs=['clip', 'flip', 'variances', 'max_size', 'min_size', 'img_size'])
