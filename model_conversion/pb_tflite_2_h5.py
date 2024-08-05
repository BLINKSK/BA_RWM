import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, Dense, Flatten
import argparse


def load_graph_def(pb_file):
    with tf.io.gfile.GFile(pb_file, "rb") as f:
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(f.read())
    return graph_def


def get_layers_from_graph(graph_def):
    layers_dict = {}
    for node in graph_def.node:
        if node.op == 'Placeholder':
            layers_dict[node.name] = {
                'type': 'InputLayer',
                'shape': [dim.size for dim in node.attr['shape'].shape.dim]
            }
        elif node.op == 'Conv2D':
            layers_dict[node.name] = {
                'type': 'Conv2D',
                'filters': node.attr['T'].type,
                'kernel_size': node.attr['ksize'].list.i,
                'strides': node.attr['strides'].list.i,
                'padding': node.attr['padding'].s.decode('utf-8')
            }
        elif node.op == 'MatMul':
            layers_dict[node.name] = {
                'type': 'Dense',
                'units': node.attr['T'].type
            }
        elif node.op == 'Relu':
            layers_dict[node.name] = {
                'type': 'ReLU'
            }
        elif node.op == 'MaxPool':
            layers_dict[node.name] = {
                'type': 'MaxPooling2D',
                'pool_size': node.attr['ksize'].list.i,
                'strides': node.attr['strides'].list.i,
                'padding': node.attr['padding'].s.decode('utf-8')
            }
        elif node.op == 'Reshape':
            layers_dict[node.name] = {
                'type': 'Reshape',
                'shape': node.attr['shape'].shape
            }
        elif node.op == 'Softmax':
            layers_dict[node.name] = {
                'type': 'Softmax'
            }
        elif node.op == 'Flatten':
            layers_dict[node.name] = {
                'type': 'Flatten'
            }

    return layers_dict


def load_weights_from_graph(graph_def):
    weights_dict = {}
    for node in graph_def.node:
        if 'Variable' in node.op or 'Const' in node.op:
            tensor = tf.make_ndarray(node.attr['value'].tensor)
            weights_dict[node.name] = tensor
    return weights_dict


def build_keras_model(layers_dict, weights_dict):
    model = models.Sequential()
    for name, layer in layers_dict.items():
        if layer['type'] == 'InputLayer':
            model.add(layers.InputLayer(input_shape=layer['shape']))
        elif layer['type'] == 'Conv2D':
            conv_layer = layers.Conv2D(
                filters=layer['filters'],
                kernel_size=layer['kernel_size'],
                strides=layer['strides'],
                padding=layer['padding'],
                activation='relu'
            )
            model.add(conv_layer)
            weights = weights_dict.get(name + '/kernel')
            biases = weights_dict.get(name + '/bias')
            if weights is not None and biases is not None:
                conv_layer.set_weights([weights, biases])
        elif layer['type'] == 'Dense':
            dense_layer = layers.Dense(units=layer['units'], activation='relu')
            model.add(dense_layer)
            weights = weights_dict.get(name + '/kernel')
            biases = weights_dict.get(name + '/bias')
            if weights is not None and biases is not None:
                dense_layer.set_weights([weights, biases])
        elif layer['type'] == 'ReLU':
            model.add(layers.ReLU())
        elif layer['type'] == 'MaxPooling2D':
            model.add(layers.MaxPooling2D(
                pool_size=layer['pool_size'],
                strides=layer['strides'],
                padding=layer['padding']
            ))
        elif layer['type'] == 'Reshape':
            model.add(layers.Reshape(target_shape=layer['shape']))
        elif layer['type'] == 'Softmax':
            model.add(layers.Softmax())
        elif layer['type'] == 'Flatten':
            model.add(layers.Flatten())

    return model


def pb_to_keras(pb_file, h5_file):
    graph_def = load_graph_def(pb_file)
    layers_dict = get_layers_from_graph(graph_def)
    weights_dict = load_weights_from_graph(graph_def)
    keras_model = build_keras_model(layers_dict, weights_dict)
    keras_model.save(h5_file)


def load_tflite_model(tflite_file):
    with open(tflite_file, 'rb') as f:
        tflite_model = f.read()
    interpreter = tf.lite.Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()
    return interpreter


def get_tflite_layers(interpreter):
    layers_dict = {}
    tensor_details = interpreter.get_tensor_details()
    for detail in tensor_details:
        if 'Identity' in detail['name']:
            continue  # Skip identity nodes
        index = detail['index']
        shape = interpreter.get_tensor(index).shape
        dtype = interpreter.get_tensor(index).dtype
        layers_dict[detail['name']] = {'shape': shape, 'dtype': dtype, 'detail': detail}
    return layers_dict


def build_keras_model_from_tflite_layers(layers_dict):
    model = tf.keras.Sequential()
    input_shape = list(layers_dict.values())[0]['shape'][1:]  # Assuming the first layer's shape is the input shape
    model.add(layers.InputLayer(input_shape=input_shape))
    
    for name, layer in layers_dict.items():
        if 'conv' in name.lower():
            filters = layer['shape'][-1]
            kernel_size = (layer['shape'][1], layer['shape'][2])
            model.add(layers.Conv2D(filters=filters, kernel_size=kernel_size, activation='relu'))
        elif 'dense' in name.lower():
            units = layer['shape'][-1]
            model.add(layers.Dense(units=units, activation='relu'))
        elif 'pool' in name.lower():
            pool_size = (layer['shape'][1], layer['shape'][2])
            model.add(layers.MaxPooling2D(pool_size=pool_size))
        elif 'flatten' in name.lower():
            model.add(layers.Flatten())
        elif 'softmax' in name.lower():
            model.add(layers.Softmax())
        elif 'reshape' in name.lower():
            target_shape = layer['shape'][1:]
            model.add(layers.Reshape(target_shape=target_shape))
        # Add more layers as needed

    return model


def set_keras_weights_from_tflite(interpreter, keras_model):
    tensor_details = interpreter.get_tensor_details()
    layer_index = 0
    for detail in tensor_details:
        if 'Identity' in detail['name']:
            continue  # Skip identity nodes
        tensor = interpreter.tensor(detail['index'])()
        layer_name = detail['name']
        
        if 'conv' in layer_name.lower():
            layer = keras_model.layers[layer_index]
            if isinstance(layer, tf.keras.layers.Conv2D):
                kernel, bias = tensor, None
                if len(layer.weights) == 2:
                    bias = interpreter.tensor(tensor_details[detail['index'] + 1]['index'])()
                    layer_index += 1  # Skip the bias tensor in the next iteration
                layer.set_weights([kernel, bias] if bias is not None else [kernel])
        
        elif 'dense' in layer_name.lower():
            layer = keras_model.layers[layer_index]
            if isinstance(layer, tf.keras.layers.Dense):
                kernel, bias = tensor, None
                if len(layer.weights) == 2:
                    bias = interpreter.tensor(tensor_details[detail['index'] + 1]['index'])()
                    layer_index += 1  # Skip the bias tensor in the next iteration
                layer.set_weights([kernel, bias] if bias is not None else [kernel])
        
        layer_index += 1
    return keras_model  # Return the model after setting weights


def tflite_to_keras(tflite_file, h5_file):
    interpreter = load_tflite_model(tflite_file)
    layers_dict = get_tflite_layers(interpreter)
    keras_model = build_keras_model_from_tflite_layers(layers_dict)
    keras_model = set_keras_weights_from_tflite(interpreter, keras_model)
    keras_model.save(h5_file)


parser = argparse.ArgumentParser(description='Pb / TFLite Models to H5 Models')
parser.add_argument("--pb_path", help="pb model path")
parser.add_argument("--tflite_path", help="tflite model path")
parser.add_argument("--save_path", help="save_model_path")
parser.add_argument("--pb", action="store_true", help="Run or not.")
args = parser.parse_args()


if args.pb:
    pb_model_path = args.pb_path
    
    pb_to_keras(pb_model_path, args.save_path)

else:
    tflite_model_path = args.tflite_path

    tflite_to_keras(tflite_model_path, args.save_path)
