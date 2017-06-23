"""Utilities related to model visualization."""
import os

layer_core = ['Dense', 'Activation', 'Dropout', 'Flatten', 'Permute', 'RepeatVector', 'Lambda', 'ActivityRegularization', 'Masking']
layer_conv = ['Conv1D', 'Conv2D', 'SeparableConv2D', 'Conv2DTranspose', 'Conv3D', 'Cropping1D', 'Cropping2D', 'Cropping3D', 'UpSampling1D', 'UpSampling2D', 'UpSampling3D', 'ZeroPadding1D', 'ZeroPadding2D', 'ZeroPadding3D']
layer_pool = ['MaxPooling1D', 'MaxPooling2D', 'MaxPooling3D', 'AveragePooling1D', 'AveragePooling2D', 'AveragePooling2D', 'AveragePooling3D', 'GlobalMaxPooling1D', 'GlobalMaxPooling2D', 'GlobalAveragePooling1D', 'GlobalAveragePooling2D']
layer_locn = ['LocallyConnected1D', 'LocallyConnected2D']
layer_recn = ['Recurrent', 'SimpleRNN', 'GRU', 'LSTM']
layer_embd = ['Embedding']
layer_mege = ['Add', 'Multiply', 'Average', 'Maximum', 'Concatenate', 'Dot', 'add', 'multiply', 'average', 'maximum', 'concatenate', 'dot']
layer_adac = ['LeakyReLU', 'PReLU', 'ELU', 'ThresholdedReLU']
layer_norm = ['BatchNormalization']
layer_nois = ['GaussianNoise', 'GaussianDropout']
layer_wrap = ['TimeDistributed', 'Bidirectional']

def layer2color(layername):
    if layername in layer_core:
        return '#4A88B3'
    elif layername in layer_conv:
        return '#98C1DE'
    elif layername in layer_pool:
        return '#6CA2C8'
    elif layername in layer_locn:
        return '#3173A2'
    elif layername in layer_recn:
        return '#17649B'
    elif layername in layer_embd:
        return '#FFBB60'
    elif layername in layer_adac:
        return '#FFDAA9'
    elif layername in layer_norm:
        return '#FFC981'
    elif layername in layer_nois:
        return '#FCAC41'
    elif layername in layer_wrap:
        return '#F29416'
    else:
        return '#C54AAA'

try:
    # pydot-ng is a fork of pydot that is better maintained.
    import pydot_ng as pydot
except ImportError:
    # Fall back on pydot if necessary.
    try:
        import pydot
    except ImportError:
        pydot = None


def _check_pydot():
    try:
        # Attempt to create an image of a blank graph
        # to check the pydot/graphviz installation.
        pydot.Dot.create(pydot.Dot())
    except Exception:
        # pydot raises a generic Exception here,
        # so no specific class can be caught.
        raise ImportError('Failed to import pydot. You must install pydot'
                          ' and graphviz for `pydotprint` to work.')


def model_to_dot(model,
                 show_shapes=False,
                 show_layer_names=True,
                 show_params=False,
                 rankdir='TB', **kwargs):
    """Convert a Keras model to dot format.
    # Arguments
        model: A Keras model instance.
        show_shapes: whether to display shape information.
        show_layer_names: whether to display layer names.
        show_params: show details of params
        rankdir: `rankdir` argument passed to PyDot,
            a string specifying the format of the plot:
            'TB' creates a vertical plot;
            'LR' creates a horizontal plot.
    # Returns
        A `pydot.Dot` instance representing the Keras model.
    """
    from keras.layers.wrappers import Wrapper
    from keras.models import Sequential

    _check_pydot()
    dot = pydot.Dot()
    dot.set('rankdir', rankdir)
    dot.set('concentrate', True)
    dot.set_node_defaults(shape='record')

    if isinstance(model, Sequential):
        if not model.built:
            model.build()
        model = model.model
    layers = model.layers

    attrs = ['filters', 'padding', 'use_bias', 'kernel_size', 'strides', 'pool_size', 'size', 'rate', 'dims', 'n', 'units', 'l1', 'l2', 'supporting_masking', 'epsilon', 'scale', 'momentum', 'dilation_rate']
    if 'attrs' in kwargs.keys():
        extra_attrs = kwargs['attrs']
        if isinstance(extra_attrs, list):
            attrs += extra_attrs
        else:
            raise TypeError('extra attributes can only be list, given {}'.format(type(extra_attrs)))
    # Create graph nodes.
    for layer in layers:
        layer_id = str(id(layer))

        # Append a wrapped layer's label to node's label, if it exists.
        layer_name = layer.name
        class_name = layer.__class__.__name__
        color = layer2color(class_name)
        if isinstance(layer, Wrapper):
            layer_name = '{}({})'.format(layer_name, layer.layer.name)
            child_class_name = layer.layer.__class__.__name__
            class_name = '{}({})'.format(class_name, child_class_name)

        # Create node's label.
        if show_layer_names:
            label = '{}: {}'.format(layer_name, class_name)
        else:
            label = class_name

        if show_params:
            attr_label = None
            for attr in attrs:
                if hasattr(layer, attr):
                    if attr_label is not None:
                        attr_label = '%s\n%s: %s' % (attr_label, attr, getattr(layer, attr))
                    else:
                        attr_label = '%s: %s' % (attr, getattr(layer, attr))
            if attr_label is not None:
                label = '%s\n%s' % (label, attr_label)

        # Rebuild the label as a table including input/output shapes.
        if show_shapes:
            try:
                outputlabels = str(layer.output_shape)
            except AttributeError:
                outputlabels = 'multiple'
            if hasattr(layer, 'input_shape'):
                inputlabels = str(layer.input_shape)
            elif hasattr(layer, 'input_shapes'):
                inputlabels = ', '.join(
                    [str(ishape) for ishape in layer.input_shapes])
            else:
                inputlabels = 'multiple'
            label = '%s\n|{input:|output:}|{{%s}|{%s}}' % (label,
                                                           inputlabels,
                                                           outputlabels)
        node = pydot.Node(layer_id, label=label, fillcolor=color, style='filled')
        dot.add_node(node)

    # Connect nodes with edges.
    for layer in layers:
        layer_id = str(id(layer))
        for i, node in enumerate(layer.inbound_nodes):
            node_key = layer.name + '_ib-' + str(i)
            if node_key in model.container_nodes:
                for inbound_layer in node.inbound_layers:
                    inbound_layer_id = str(id(inbound_layer))
                    layer_id = str(id(layer))
                    dot.add_edge(pydot.Edge(inbound_layer_id, layer_id))
    return dot


def model2graph(model,
               to_file='model.png',
               show_shapes=False,
               show_layer_names=True,
               show_params = False,
               rankdir='TB', **kwargs):
    """Converts a Keras model to dot format and save to a file.
    # Arguments
        model: A Keras model instance
        to_file: File name of the plot image.
        show_shapes: whether to display shape information.
        show_layer_names: whether to display layer names.
        rankdir: `rankdir` argument passed to PyDot,
            a string specifying the format of the plot:
            'TB' creates a vertical plot;
            'LR' creates a horizontal plot.
    """
    dot = model_to_dot(model, show_shapes, show_layer_names, show_params, rankdir, **kwargs)
    _, extension = os.path.splitext(to_file)
    if not extension:
        extension = 'png'
    else:
        extension = extension[1:]
    dot.write(to_file, format=extension)
