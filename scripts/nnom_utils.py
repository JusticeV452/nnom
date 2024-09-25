'''
    Copyright (c) 2018-2020
    Jianjia Ma
    majianjia@live.com

    SPDX-License-Identifier: Apache-2.0

    Change Logs:
    Date           Author       Notes
    2019-02-05     Jianjia Ma   The first version


    This file provides:
    -> fake_quantisation layers which simulate the output quantisation on fixed-point NN models.
    -> weights/bias quantisation of Convolution and Dense Layer. "weight.h" file generations
    -> export "testing set" binary data file.
    -> print output ranges of each layers.

    Currently, this script does not support RNN (type) layers.
'''

import os
import io
import time
import warnings
import scipy.stats

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

import tensorflow.keras as keras
from tensorflow.keras import layers as kl
from tensorflow.keras.models import Model
from sklearn import metrics

from .fully_connected_opt_weight_generation import *


def is_input_layer(layer: kl.Layer):
    """
    Check if layer is an input layer

    Parameters
    ----------
    layer : kl.Layer

    Returns
    -------
    bool
        True if layer is an input layer, False otherwise.

    """
    return "input" in layer.name


def get_int_bits(min_value: float, max_value: float):
    """
    Determine the number of bits needed to represent a set of values

    Parameters
    ----------
    min_value : float
        The smallest value in a set of values.
    max_value : float
        The largest value in a set of values.

    Returns
    -------
    int
        The number of bits needed to represent a set of values with a minimum
        of `min_value` and maxmum of `max_value`.

    """
    return int(np.ceil(np.log2(max([abs(min_value), abs(max_value), 1e-10]))))


def pad_filter_sizes(*filter_sizes, pad_val=1, shape=2):
    padded_sizes = []
    for f_size in filter_sizes:
        if type(f_size) is int:
            f_size = [f_size]
        padded_sizes.append(
            # Extend shape with pad_val if len(f_size) < shape
            (*(pad_val,) * (shape - len(f_size)), *f_size) if len(f_size) < shape else tuple(f_size)
        )
    return padded_sizes


def flatten(L):
    if not isinstance(L, list | tuple):
        return [L]
    return sum([flatten(el) for el in L], start=[])


def to_transposed_x4_q7_weights(weights: np.array):
    transposed_wts = np.transpose(weights)
    return convert_to_x4_q7_weights(np.reshape(
        transposed_wts,
        (transposed_wts.shape[0], transposed_wts.shape[1], 1, 1)
    ))


def dec_bits_by_kld(layer: kl.Layer, features, dec_bits: int, verbose: bool=False):
    max_val = features.max()
    min_val = features.min()
    abs_max = max(abs(max_val), abs(min_val))
    small_var = 1e-5
    bins = np.arange(-abs_max, abs_max, abs_max / 2048 * 2)
    q_bins = np.arange(-abs_max, abs_max, abs_max / 256 * 2)
    flat_hist = np.histogram(features.flatten(), bins=bins)[0]
    kl_loss = []
    kl_shifts = []
    for shift in range(4):
        t = 2 ** (dec_bits + shift)     # 2-based threshold
        act = np.round(features.flatten() * t)
        act = act / t
        act = np.clip(act, -128 / t, 127 / t)
        act = np.histogram(act, bins=q_bins)[0]
        act_hist = np.zeros(2047)
        chunk = int(2048 / 256)
        for i in range(int(255)):
            none_zero = np.count_nonzero(flat_hist[i * chunk:(i + 1) * chunk])
            if none_zero == 0:
                continue
            for j in range(chunk):
                act_hist[i * chunk + j] = (
                    act[i] / none_zero
                    if flat_hist[i * chunk + j] != 0
                    else 0
                )
        flat_hist[flat_hist == 0] = small_var
        act_hist[act_hist == 0] = small_var
        kl = scipy.stats.entropy(flat_hist, act_hist)
        kl_loss.append(kl)
        kl_shifts.append(dec_bits + shift)

    # set the dec_bit to the KLD results
    new_dec = kl_shifts[np.argmin(kl_loss)]

    if verbose:
        print("KLD loss:", kl_loss)
        print("KLD shift:", kl_shifts)
    if verbose and dec_bits != new_dec:
        print(layer.name, "is using KLD method, original shift:", dec_bits, "KLD results:", new_dec)

    dec_bits = new_dec
    return dec_bits


def make_initial_shift_list(
        model: keras.Model, x_test: np.array | List[np.array],
        quantize_method: str="max_min", verbose: bool=False):
    shift_list = {}
    last_layer = None
    
    def get_features(model, inp):
        if verbose:
            return model.predict(inp)
        return model(inp).numpy()

    model_layers = model.layers
    if not is_input_layer(model.layers[0]):
        model_layers = [model.input] + model_layers

    inp_idx = 0
    for layer in model_layers: # layer loop
        if is_input_layer(layer):
            features = x_test[inp_idx] if isinstance(x_test, list) else x_test
            inp_idx += 1
        # batch_normalization will need to be handled differently, since we are fusing the weight to its predecessor.
        # sigmoid and tanh are different, their shift is fixed to 7
        elif is_shift_layer(layer) or "batch_normalization" in layer.name:
            layer_model = Model(inputs=model.input, outputs=layer.output)
            features = get_features(layer_model, x_test)
        # Otherwise leave the features not changed, so this layer shift will be the same
        # as its inputs
        
        #  calculate no saturation shift
        max_val = features.max()
        min_val = features.min()
        int_bits = get_int_bits(min_val, max_val)
        dec_bits = 7 - int_bits

        # saturation shift, using KLD method
        # Ref: http://on-demand.gputechconf.com/gtc/2017/presentation/s7310-8-bit-inference-with-tensorrt.pdf
        if (
            "kld" in quantize_method
            and not is_shift_fixed(layer)
            and not is_input_layer(layer)
            and "dense" not in layer.name
        ): # test, also do not use kld in input layer
            dec_bits = dec_bits_by_kld(layer, features, dec_bits, verbose=verbose)

        if verbose:
            print(layer.name, "max value:", max_val, "min value:", min_val, "dec bit:", dec_bits)
        
        # record the shift
        shift_name = layer.name
        if isinstance(model.input, tf.Tensor) and not is_input_layer(model.layers[0]):
            shift_name = shift_name.split(':')[0]
        shift_list[shift_name] = dec_bits

        if "batch_normalization" in layer.name:
            # use the bn layer shift to update the last layer.
            shift_list[last_layer.name] = dec_bits
        last_layer = layer
    return shift_list


""" 
this is the generate the test set data to a bin file
bin file can be used to validate the implementation in MCU

"""
def generate_test_bin(x, y, name='test_data_with_label.bin'):
    '''
    this method generate the
    :param x:  input x data size
    :param y:  input label (one hot label)
    :return:
    '''
    # quantize input x
    min_value = np.min(x)
    max_value = np.max(x)

    int_bits = int(np.ceil(np.log2(max(abs(min_value), abs(max_value)))))
    dec_bits = 7 - int_bits
    x = np.round(x*2**dec_bits).astype(np.int8)
    # get label
    if(len(y.shape) >1):
        test_label = np.argwhere(y == 1).astype(np.int8)  # test data
        test_label = test_label[:, 1]
    else:
        test_label = y

    # get data
    dat = x.astype(dtype="byte")  # test data
    batch_size = dat.shape[0]     # total pices of data	
    dat = dat.flatten()           # flatten to get the total size.
    block_size = int(dat.size / batch_size) # this must be integer but... just to confirm

    # write (label x 128) (data_block x 128)
    label_batch = 128       # the Y-modem example uses 128 batch
    with open(name, 'wb') as f:
        start = 0
        while start <= (test_label.size - label_batch):
            test_label[start: start + label_batch].tofile(f)
            dat[block_size * start: block_size * (start + label_batch)].tofile(f)
            start += label_batch

        # the rest data
        if (start < test_label.size):
            rest_len = test_label.size - start
            new_labls = test_label[start:]
            new_labls = np.pad(new_labls, (0, label_batch - rest_len), mode='constant')
            new_labls.tofile(f)
            dat[block_size * start:].tofile(f)

    print("binary test file generated:", name)
    print("test data length:", test_label.size)
    return


def is_shift_layer(layer):
    ''' layer which can change the output encoding'''
    #FIXME: add more which will change the output shift
    if('input' in layer.name or
       'conv2d' in layer.name or
       'conv1d' in layer.name or
       'dense' in layer.name or
       'softmax' in layer.name or
        'sigmoid' in layer.name or
        'tanh' in layer.name or
        ('add' in layer.name and 'zero' not in layer.name) or # the name, zero_padding contains 'add'
        'subtract' in layer.name or
        'multiply' in layer.name or
       ('activation' in layer.name and layer.get_config()['activation'] == 'softmax')or
       ('activation' in layer.name and layer.get_config()['activation'] == 'sigmoid') or
       ('activation' in layer.name and layer.get_config()['activation'] == 'tanh')
    ):
        return True
    return False


def is_shift_fixed(layer):
    ''' layer which shift to a fixed value'''
    #FIXME: add more which will change the output shift
    if('softmax' in layer.name or
        'sigmoid' in layer.name or
        'tanh' in layer.name or
        ('activation' in layer.name and layer.get_config()['activation'] == 'softmax') or
        ('activation' in layer.name and layer.get_config()['activation'] == 'sigmoid') or
        ('activation' in layer.name and layer.get_config()['activation'] == 'tanh')
    ):
        return True
    return False


def fuse_bn_to_conv(layer):
    # try to fuse BN layer to convolutional
    if ('conv' in layer.name) and \
            ('batch_normalization' in layer._outbound_nodes[0].outbound_layer.name):

        print("fusing batch normalization to", layer.name)
        bn_layer = layer._outbound_nodes[0].outbound_layer
        c_w = layer.get_weights()[0]
        c_b = layer.get_weights()[1]
        print('original weight max', c_w.max(), 'min', c_w.min())
        print('original bias max', c_b.max(), 'min', c_b.min())
        bn_gamma = bn_layer.get_weights()[0]
        bn_beta = bn_layer.get_weights()[1]
        bn_mean = bn_layer.get_weights()[2]
        bn_variance = bn_layer.get_weights()[3]

        if ('conv2d' in layer.name):
            epsilon = 1e-3  # default epsilon for tf.slim.batch_norm
            for l in range(c_w.shape[3]):
                for k in range(c_w.shape[2]):
                    for j in range(c_w.shape[1]):
                        for i in range(c_w.shape[0]):
                            if "depthwise" in layer.name:  # depthwise batchnorm params are ordered differently
                                c_w[i][j][k][l] *= bn_gamma[k] / np.sqrt(bn_variance[k] + epsilon)
                            else:
                                c_w[i][j][k][l] *= bn_gamma[l] / np.sqrt(bn_variance[l] + epsilon)

            if "depthwise" in layer.name:
                depth_dim = c_w.shape[2]
            else:
                depth_dim = c_w.shape[3]
            for l in range(depth_dim):
                c_b[l] = (bn_gamma[l] * (c_b[l] - bn_mean[l]) / np.sqrt(bn_variance[l] + epsilon)) + bn_beta[l]
        # conv1d
        else:
            epsilon = 1e-3  # default epsilon for tf.slim.batch_norm
            for k in range(c_w.shape[2]):
                for j in range(c_w.shape[1]):
                    for i in range(c_w.shape[0]):
                        if "depthwise" in layer.name:  # depthwise batchnorm params are ordered differently
                            c_w[i][j][k] *= bn_gamma[j] / np.sqrt(bn_variance[j] + epsilon)
                        else:
                            c_w[i][j][k] *= bn_gamma[k] / np.sqrt(bn_variance[k] + epsilon)

            if "depthwise" in layer.name:
                depth_dim = c_w.shape[1]
            else:
                depth_dim = c_w.shape[2]
            for l in range(depth_dim):
                c_b[l] = (bn_gamma[l] * (c_b[l] - bn_mean[l]) / np.sqrt(bn_variance[l] + epsilon)) + bn_beta[l]

        print('fused weight max', c_w.max(), 'min', c_w.min())
        print('fused bias max', c_b.max(), 'min', c_b.min())
        # write the weights back to the layer
        # after that, the model will be destroyed.. need a better way to pass the new weight
        layer.set_weights([c_w, c_b])


def generate_weights(
        model: keras.Model, x_test: np.array=None, quantize_method="max_min",
        max_calibrate_size=1000, fmt="hwc", verbose=False):
    # Quantize weights to 8-bits using (min,max) and write to file
    f = io.StringIO()
    f.write('#include "nnom.h"\n\n')

    if isinstance(x_test, type(None)):
        shift_list = None
    else:
        shift_list = layers_output_ranges(
            model, x_test, quantize_method=quantize_method,
            max_calibrate_size=max_calibrate_size, verbose=verbose
        )

    layer_quantize_info = {}
    layer_weights = {}
    for layer in model.layers:
        if not layer.weights:
            continue

        # before merging bn layer, check if the bn is "legally" after Conv
        if (
            "batch_normalization" in layer.name
            and "conv" not in layer.inbound_nodes[0].inbound_layers.name
        ):
            raise Exception(
                "Currently only support batch_normalization after conv",
                layer.name, layer._inbound_nodes[0].inbound_layers[0].name
            )

        # try to fuse BN layer to convolutional
        if (
            "conv" in layer.name
            and layer.outbound_nodes
            and "batch_normalization" in layer.outbound_nodes[0].outbound_layer.name
        ):
            fuse_bn_to_conv(layer)

        # generate weights and bias now
        weight_dec_shift = 0
        if verbose:
            print('weights for layer', layer.name)

        layer_quantize_info[layer.name] = {}
        layer_weights[layer.name] = {}
        for var in layer.weights:
            var_name = str(var.name)
            is_kernel = "kernel" in var_name
            if not is_kernel and "bias" not in var_name:
                continue

            var_values = var.numpy()
            min_value = np.min(var_values)
            max_value = np.max(var_values)

            int_bits = get_int_bits(min_value, max_value)
            dec_bits = 7 - int_bits

            if verbose:
                print(f" {'weight' if is_kernel else 'bias'}:", var_name)
                print("  original shape: ", var_values.shape)
                print("  dec bit", dec_bits)

            bSameAsKernel = False
            if is_shift_layer(layer):
                assert shift_list, f"Layer {layer.name} is classified as a shift layer so shift_list is required."
                inp = layer.input.name.replace(':', '/').split('/')[0]
                input_encoding = shift_list[inp]
                if is_kernel:
                    weight_dec_shift = dec_bits
                else:
                    shift = input_encoding + weight_dec_shift - dec_bits
                    if shift < 0:
                        bSameAsKernel = True

            if shift_list is None or bSameAsKernel:
                # check if bias shift > weight shift, then reduce bias shift to weight shift
                if is_kernel:
                    weight_dec_shift = dec_bits
                elif dec_bits > weight_dec_shift:
                    dec_bits = weight_dec_shift
                if verbose:
                    print("  new dec bit", dec_bits)

            layer_quantize_info[layer.name][var_name] = {
                "min": min_value,
                "max": max_value,
                "data_width": dec_bits
            }

            # convert to [-128,128) or int8
            var_values = np.round(var_values * 2 ** dec_bits)
            layer_weights[layer.name][int(not is_kernel)] = var_values
            var_name = var_name.replace('/', '_').replace(':', '_')
            f.write("#define " + var_name.upper() + " {")

            # CHW format
            if "chw" in fmt:
                if is_kernel and "dense" in var_name:
                    transposed_wts = to_transposed_x4_q7_weights(var_values)
                # all other kernels, bias stay the same
                else:
                    transposed_wts = var_values
            # HWC format
            else:
                if len(var_values.shape) == 3:  # 1D convolution layer weights
                    transposed_wts = np.transpose(var_values, (2, 0, 1))
                elif len(var_values.shape) == 4:  # 2D convolution layer weights
                    transposed_wts = np.transpose(var_values, (3, 0, 1, 2))
                elif is_kernel and "dense" in var_name:
                    # fully connected layer weights or biases of any layer
                    # test, use opt weight reorder
                    transposed_wts = to_transposed_x4_q7_weights(var_values)
                else:
                    transposed_wts = np.transpose(var_values)
            if verbose:
                print("  reshape to:", transposed_wts.shape)

            f.write(np.array2string(
                transposed_wts.flatten(),
                separator=", ",
                threshold=transposed_wts.size,
                formatter={"all": lambda x: str(int(x))}
            ).strip("[]").replace('\n', ''))
            # transposed_wts.tofile(f, sep=", ", format="%d")
            f.write("}\n\n")
            f.write(f"#define {var_name.upper()}_SHIFT ({dec_bits})\n\n")
            if not is_kernel:
                f.write("\n")
    return f, layer_weights, layer_quantize_info, shift_list


def layers_output_ranges(model, x_test, quantize_method="max_min", max_calibrate_size=1000, verbose=False):
    def clamp_input(input_arr):
        # limit the test data size
        np.random.shuffle(input_arr)
        if input_arr.shape[0] > max_calibrate_size:
            input_arr = input_arr[:max_calibrate_size]
        return input_arr

    if isinstance(x_test, list):
        x_test = [clamp_input(inp) for inp in x_test]
    else:
        x_test = clamp_input(x_test)
    shift_list = make_initial_shift_list(
        model, x_test,
        quantize_method=quantize_method, verbose=verbose
    )

    layer_dict = {}
    for layer in model.layers:
        layer_dict[layer.name] = layer

    def get_iname(layer):
        return layer.name.split('/')[0]

    def update_previous_layer_shift(init_layer, Qmin, skip_input=True):
        layers = init_layer.input
        if not isinstance(layers, list):
            layers = [layers]

        for layer in layers:
            if skip_input and is_input_layer(layer):
                continue
            iname = get_iname(layer)
            shift_list[iname] = Qmin
            if not is_shift_layer(layer_dict[iname]):
                update_previous_layer_shift(layer_dict[iname], Qmin)

    for layer in reversed(model.layers[1:]):
        if not isinstance(layer.input, list):
            continue

        # detemine Qmin
        Qmin = shift_list[get_iname(layer.input[0])]
        for inp in layer.input:
            Qmin = min(Qmin, shift_list[get_iname(inp)])

        update_previous_layer_shift(layer, Qmin, skip_input=False)

        if verbose:
            print(
                f"Set shift {Qmin} for the input of {layer.name}:",
                f"{[inp.name.split('/')[0] for inp in layer.input]}"
            )
        # update current layer's shift only when we cannot change the shift
        if not is_shift_layer(layer) or Qmin < shift_list[layer.name]:
            shift_list[layer.name] = Qmin

    if verbose:
        print("shift list:", shift_list)
    return shift_list


def generate_model(
        model, x_test, name='weights.h', fmt='hwc', quantize_method='max_min',
        max_calibrate_size=1000, verbose=False):
    f, *_, shift_list = generate_weights(
        model, x_test=x_test, fmt=fmt, quantize_method=quantize_method,
        max_calibrate_size=max_calibrate_size, verbose=verbose
    )

    model_layers = model.layers
    if not is_input_layer(model.layers[0]):
        model_layers = [model.input] + model_layers

    def get_iname(layer):
        return layer.name.replace(':', '/').split('/')[0]

    def to_cpp_var_name(layer_name):
        return layer_name.upper().replace('/', '_').replace(':', '_')

    def is_skipable_layer(layer):
        # FIXME: add more that could be skiped
        # flatten layer can be skipped in HWC but have to present in CHW
        return (
            "lambda" in layer.name
            or "dropout" in layer.name
            or "batch_normalization" in layer.name
            or ("flatten" in layer.name and "chw" not in fmt)
        )
    
    def add_activation(layer, inp, layer_id, cfg):
        activ_name = cfg.get("activation")
        if activ_name in ["tanh", "sigmoid"]:
            f.write(f"\tlayer[{layer_id}] = model.active(act_{activ_name}({inp.upper()}_OUTPUT_SHIFT), layer[{LI[inp][0]}]);\n")
        elif "re_lu" in layer.name or activ_name in ["softmax", "relu"]:
            func_name = "Softmax" if activ_name == "softmax" else "act_relu"
            func_type = "hook" if activ_name == "softmax" else "active"
            f.write(f"\tlayer[{layer_id}] = model.{func_type}({func_name}(), layer[{LI[inp][0]}]);\n")
        elif activ_name != "linear":
            raise Exception(f"{activ_name} activation is unsupported.")

    f.write('\n/* output encoding for each layer */\n')
    for layer in model_layers:
        iname = get_iname(layer)
        f.write(f"#define {iname.upper()}_OUTPUT_SHIFT {shift_list[iname]}\n")

    f.write('\n/* bias shift and output shift for each layer */\n')
    for layer in model_layers:
        if not is_shift_layer(layer):
            continue
        iname = layer.name.upper()
        if (
                len(layer.weights) == 2
                and "kernel" in layer.weights[0].name
                and "bias" in layer.weights[1].name
            ):
            kernel, bias = layer.weights
            kname = to_cpp_var_name(kernel.name)
            bname = to_cpp_var_name(bias.name)
            inp = get_iname(layer.input).upper()
            f.write(f"#define {iname}_OUTPUT_RSHIFT ({inp}_OUTPUT_SHIFT+{kname}_SHIFT-{iname}_OUTPUT_SHIFT)\n")
            f.write(f"#define {iname}_BIAS_LSHIFT   ({inp}_OUTPUT_SHIFT+{kname}_SHIFT-{bname}_SHIFT)\n")
            f.write(f"#if {iname}_OUTPUT_RSHIFT < 0\n#error {iname}_OUTPUT_RSHIFT must be bigger than 0\n#endif\n")
            f.write(f"#if {iname}_BIAS_LSHIFT < 0\n#error {iname}_BIAS_RSHIFT must be bigger than 0\n#endif\n")
        # add, sub
        elif "add" in layer.name or "subtract" in layer.name:
            # only consider the first, they have been set to same in out_put_range()
            inp = get_iname(layer.input[0]).upper()
            f.write(f"#define {iname}_OUTPUT_RSHIFT ({inp}_OUTPUT_SHIFT-{iname}_OUTPUT_SHIFT)\n")
            f.write(f"#if {iname}_OUTPUT_RSHIFT < 0\n#error {iname}_OUTPUT_RSHIFT must be bigger than 0\n#endif\n")
        # mult is different, Q3.4 * Q3.4 = Q6.8. if mult out is Q4.3, then shift (Q.4+q.4)-Q.3=5. Am I right?
        elif "multiply" in layer.name:
            inp = get_iname(layer.input[0]).upper()
            f.write(f"#define {iname}_OUTPUT_RSHIFT ({inp}_OUTPUT_SHIFT*2-{iname}_OUTPUT_SHIFT)\n")
            f.write(f"#if {iname}_OUTPUT_RSHIFT < 0\n#error {iname}_OUTPUT_RSHIFT must be bigger than 0\n#endif\n")

    ID = 0
    LI = {}
    f.write('\n/* weights for each layer */\n')
    for layer_id, layer in enumerate(model_layers):
        if is_skipable_layer(layer):
            inp = get_iname(layer.input)
            LI[layer.name] = (LI[inp][0], layer)
        else:
            layer_name = layer.name
            if isinstance(model.input, tf.Tensor) and not is_input_layer(model.layers[0]):
                layer_name = layer.name.split(':')[0]
            LI[layer_name] = (ID, layer)
            ID += 1

        if is_input_layer(layer) or not layer.weights:
            continue
        for var in layer.weights:
            var_name = to_cpp_var_name(var.name)
            if "KERNEL" in var_name:
                f.write(f"static const int8_t {layer.name}_weights[] = {var_name};\n")
                f.write('static const nnom_weight_t %s_w = { (const void*)%s_weights, %s_OUTPUT_RSHIFT};\n' % (layer.name, layer.name, layer.name.upper()))
            elif "BIAS" in var_name:
                f.write(f"static const int8_t {layer.name}_bias[] = {var_name};\n")
                f.write('static const nnom_bias_t %s_b = { (const void*)%s_bias, %s_BIAS_LSHIFT};\n' % (layer.name, layer.name, layer.name.upper()))

    f.write("\n/* nnom model */\n")
    # FIXME: now only support one output
    inp_sizes = []
    max_idx = 0
    inp_list = get_input_list(model)
    for i, inp in enumerate(inp_list):
        sz = 1
        for d in inp.shape[1:]:
            sz *= d
        inp_sizes.append(sz)
        if inp_sizes[i] > inp_sizes[max_idx]:
            max_idx = i
    f.write(f"const int8_t NUM_INPUTS = {len(inp_sizes)};\n")
    f.write(f"const int{'8_t' if sz < 128 else ''} INPUT_LENGTHS[] = ")
    f.write('{' + str(inp_sizes)[1:-1] + "};\n")
    f.write(f"const int8_t IN_DATA_WIDTH = {inp_sizes[max_idx]};\n"
    f.write(f"static int8_t nnom_input_data[NUM_INPUTS][IN_DATA_WIDTH];\n")
    sz = 1
    for d in model.output.shape[1:]:
        sz *= d
    f.write(f"const int{'8_t' if sz < 128 else ''} OUTPUT_LENGTH = {sz};\n")
    f.write("static int8_t nnom_output_data[OUTPUT_LENGTH];\n")
    f.write("static nnom_model_t* nnom_model_create(void)\n{\n")
    f.write("\tstatic nnom_model_t model;\n")

    if ID > 32:
        f.write(f"\tnnom_layer_t ** layer = malloc(sizeof(nnom_layer_t *)*{ID + 1});\n")
        f.write("\tif(NULL == layer) return NULL;\n")
    else:
        f.write(f"\tnnom_layer_t* layer[{ID + 1}];\n")

    f.write("\n\tnew_model(&model);\n\n")
    inp_idx = 0
    for layer in model_layers:
        if is_skipable_layer(layer):
            continue
        #FIXME: need a better solution to seperate the input 'tensor' from other layers
        if isinstance(model.input, tf.Tensor) and not is_input_layer(model.layers[0]):
            layer_id, _ = LI[layer.name.split(':')[0]]
        else:
            layer_id, _ = LI[layer.name]
        try:
            inp = get_iname(getattr(layer, "input", None))
        except AttributeError:
            inp = ""
        cfg = getattr(layer, "get_config", lambda: None)()

        if "input" in layer.name:
            try:
                inshape = layer.input_shape[0][1:] # new changes in tf2?
            except:
                inshape = layer.shape[1:]
            if len(inshape) == 1:  # 1-D input
                f.write(f"\tlayer[{layer_id}] = Input(shape({inshape[0]}, 1, 1), nnom_input_data[{inp_idx}]);\n")
            elif len(inshape) == 2:  # 1-D input
                f.write(f"\tlayer[{layer_id}] = Input(shape(1, {inshape[0]}, {inshape[1]}), nnom_input_data[{inp_idx}]);\n")
            else:
                f.write(f"\tlayer[{layer_id}] = Input(shape{inshape}, nnom_input_data[{inp_idx}]);\n")
            inp_idx += 1

        # convolutional
        elif "conv" in layer.name:
            is_depthwise = "depthwise" in layer.name
            num_filters = 1 if is_depthwise else cfg["filters"]
            conv_type = "Conv2D"
            if is_depthwise:
                conv_type = "DW_" + conv_type

            # Expand kernel, stride, and dilation for 1D conv
            kernel, stride, dilation = pad_filter_sizes(
                cfg['kernel_size'], cfg['strides'], cfg['dilation_rate']
            )
            f.write(
                f"\tlayer[{layer_id}] = model.hook("
                + f"{conv_type}({num_filters}, kernel{kernel}, "
                + f"stride{stride}, dilation{dilation}, "
                + f"PADDING_{cfg['padding']}, &{layer.name}_w, "
                + f"&{layer.name}_b), layer[{LI[inp][0]}]);\n"
            )

        # activations
        elif "activation" in layer.name or "re_lu" in layer.name:
            add_activation(layer, inp, layer_id, cfg)

        # pooling
        elif "pooling" in layer.name:
            pooling_type = "Avg" if "average" in layer.name else layer.name[:3].capitalize()
            if "global" in layer.name:
                # a global avg pool before softmax can be replace by sumpool in MCU (recommend)
                if pooling_type == "Avg" and layer == model.layers[-2] and "Softmax" in model.layers[-1].output.name:
                    if verbose:
                        print(layer.name, 'has been replaced by GlobalSumPool()')
                    f.write(f"\tlayer[{layer_id}] = model.hook(GlobalSumPool(), layer[{LI[inp][0]}]);\n")
                else:
                    f.write(f"\tlayer[{layer_id}] = model.hook(Global{pooling_type}Pool(), layer[{LI[inp][0]}]);\n")
            else:
                # Expand 1D Pooling params
                pool_size, strides = pad_filter_sizes(cfg["pool_size"], cfg["strides"])
                padding = cfg["padding"].upper()
                f.write(
                    f"\tlayer[{layer_id}] = model.hook("
                    + f"{pooling_type}Pool("
                    + f"kernel{pool_size}, stride{strides}, PADDING_{padding}"
                    + f"), layer[{LI[inp][0]}]);\n"
                )
        elif "up_sampling" in layer.name:
            size = pad_filter_sizes(cfg["size"])[0]
            f.write(f"\tlayer[{layer_id}] = model.hook(UpSample(kernel{size}), layer[{LI[inp][0]}]);\n")

        # Zero padding / Cropping
        elif "zero_padding" in layer.name or "cropping" in layer.name:
            is_padding = "zero_padding" in layer.name
            config_var = "padding" if is_padding else "cropping"
            func_name = "ZeroPadding" if is_padding else "Cropping"
            border_size = pad_filter_sizes(flatten(cfg[config_var]), pad_val=0, shape=4)[0]
            f.write(f"\tlayer[{layer_id}] = model.hook({func_name}(border{border_size}), layer[{LI[inp][0]}]);\n")

        # Flatten
        elif "flatten" in layer.name: # flatten is needed in CHW backend but not needed in HWC
            f.write(f"\tlayer[{layer_id}] = model.hook(Flatten(), layer[{LI[inp][0]}]);\n")

        # Multi-input layers
        elif any(merge_name in layer.name for merge_name in ["concatenate", "add", "subtract", "multiply"]):
            inps = [get_iname(input) for input in layer.input]
            inX = ", ".join([f"layer[{LI[inp][0]}]" for inp in inps])
            if "concatenate" in layer.name:
                f.write(f"\tlayer[{layer_id}] = model.mergex(Concat({cfg['axis']}), {len(inps)}, {inX});\n")
            else:
                func_name = "Mult" if "multiply" in layer.name else layer.name[:3].capitalize()
                if func_name == "Mult":
                    warnings.warn("Warning mutiply is under testing")
                f.write(
                    f"\tlayer[{layer_id}] = model.mergex("
                    + f"{func_name}({layer.name.upper()}_OUTPUT_RSHIFT), {len(inps)}{inX});\n"
                )

        # Dense
        elif "dense" in layer.name:
            f.write(
                f"\tlayer[{layer_id}] = model.hook("
                + f"Dense({cfg['units']}, &{layer.name}_w, &{layer.name}_b), layer[{LI[inp][0]}]);\n"
            )

        else:
            raise Exception("unsupported layer", layer.name, layer)

    # FIXME, test later.
    if (
        "softmax" in layer.name
        or len(layer.output.shape) == 2
        or ("activation" in layer.name and layer.get_config()["activation"] == "softmax")
    ):
        out_shape = (layer.output.shape[1], 1, 1)
    elif len(layer.output.shape) == 4:
        out_shape = layer.output.shape[1:]
    elif len(layer.output.shape) == 3:
        out_shape = (1, layer.output.shape[1], layer.output.shape[2])
    else:
        raise Exception("unsupported output shape of the last layer", layer.name, layer)
    f.write(f"\tlayer[{layer_id + 1}] = model.hook(Output(shape{out_shape}, nnom_output_data), layer[{layer_id}]);\n")
    f.write(f"\tmodel_compile(&model, layer[0], layer[{layer_id + 1}]);\n")
    if ID > 32:
        f.write("\tfree(layer);\n")
    f.write("\treturn &model;\n}\n")
    save_root, _ = os.path.split(name)
    with open(os.path.join(save_root, ".shift_list"), 'w') as file:
        file.write(str(shift_list))
    with open(name, 'w+', encoding="utf-8") as file:
        file.write(f.getvalue())


def evaluate_model(model, x_test, y_test, running_time=False, to_file='evaluation.txt'):
    # Score trained model.
    scores = model.evaluate(x_test, y_test, verbose=2)
    print('Test loss:', scores[0])
    print('Top 1:', scores[1])

    if(len(y_test.shape)>1):
        # predictions = model.predict(x_test)
        # output = tf.keras.metrics.top_k_categorical_accuracy(y_test, predictions, k=2)
        # # with tf.Session() as sess:
        # #     result = sess.run(output)
        # result =
        # print("Top 2:",result)

        predictions = model.predict(x_test)
        matrix = metrics.confusion_matrix(y_test.argmax(axis=1), predictions.argmax(axis=1))
        print(matrix)

    run_time = 0
    if running_time:
        # try to calculate the time
        T = time.time()
        for i in range(10):
            model.predict(x_test)
        T = time.time() - T
        run_time = round((T / 10 / x_test.shape[0] * 1000 * 1000), 2)
        print("Runing time:",run_time , "us" )
    #
    with open(to_file, 'w') as f:
        f.write("Runing time: "+ str(run_time) + "us" + "\n")
        f.write('Test loss:'+ str(scores[0]) + "\n")
        f.write('Top 1:'+ str(scores[1])+ "\n")
        if (len(y_test.shape) > 1):
            #f.write("Top 2:"+ str(result)+ "\n")
            #f.write(str(matrix))
            for row in matrix:
                row.tofile(f, sep=',')
                f.write("\n")

    # try to check the weight and bias dec ranges
    for layer in model.layers:
        if (not layer.weights):
            continue
        for var in layer.weights:
            var_name = str(var.name)
            if ("kernel" in var_name):
                var_values = layer.get_weights()[0]  # weight
            else:
                var_values = layer.get_weights()[1]  # bias
            min_value = np.min(var_values)
            max_value = np.max(var_values)
            intt = int(np.ceil(np.log2(max(abs(min_value), abs(max_value)))))
            dec = 7 - intt
            print(var_name, "Dec num:", dec)
    return scores

def f2q(d, Q):
    '''To convert a number from floating point to Qm.n format:
        1. Multiply the floating point number by 2n
        2. Round to the nearest integer
    '''
    return np.round(d*2**Q)


def q2f(d, Q):
    '''To convert a number from Qm.n format to floating point:
        1. Convert the number to floating point as if it were an integer, in other words remove the binary point
        2. Multiply by 2-n
    '''
    return d*2**-Q

def show_weights(w, name):
    sz = 1
    for s in w.shape:
        sz = sz*s
    aL = w.reshape(sz,)
    MIN,MAX=min(aL),max(aL)
    Q = int(np.ceil(np.log2(max(abs(MIN),abs(MAX)))))
    Q = 7-Q
    qL = f2q(aL,Q)
    qL = q2f(qL,Q)
    plt.figure(figsize=(18, 3))  
    plt.subplot(131)
    plt.title(name)
    plt.plot(aL)
    plt.grid()
    aL.sort()
    plt.plot(aL,'r')
    plt.grid()
    plt.subplot(132)
    plt.title('Q%s'%(Q))
    qL.sort()
    plt.plot(aL,'r')
    plt.plot(qL,'g')
    plt.grid()
    plt.subplot(133)
    plt.hist(aL,100)
    plt.title('hist')
    plt.grid()
    plt.show()

def compare(a,b,name):
    sz = 1
    for s in a.shape:
        sz = sz*s
    aL = a.reshape(sz,)
    bL = b.reshape(sz,)
    assert(len(aL) == len(bL))
    Z = list(zip(aL,bL))
    Z.sort(key=lambda x: x[0])
    aL1,bL1=zip(*Z)
    plt.figure(figsize=(18, 3))
    plt.subplot(131)
    plt.plot(aL)
    plt.plot(aL1,'r')
    plt.grid()
    plt.title('tf-%s'%(name))
    plt.subplot(133)
    plt.plot(bL1,'g')
    plt.plot(aL1,'r')
    plt.grid()
    plt.title('compare')
    plt.subplot(132)
    bL1=list(bL1)
    bL1.sort()
    plt.plot(bL)
    plt.plot(bL1,'g')
    plt.grid()
    plt.title('nn-%s'%(name))
    plt.show()
