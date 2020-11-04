#!/usr/bin/env python3
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf
import numpy as np
import sys
import random
import os
import math
import functools
import time
import argparse

import tflite_runtime.interpreter as tflite


def benchmark(model_file):
    interpreter = tflite.Interpreter(
        model_path=model_file,
        experimental_delegates=[tflite.load_delegate("libedgetpu.so.1", {})],
    )
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    start = time.perf_counter()
    MIN_TIME = 10  # 5 sec

    N = 0
    while time.perf_counter() < start + MIN_TIME:
        inp = np.zeros(input_details["shape"], dtype="int8")
        interpreter.set_tensor(input_details["index"], inp)
        interpreter.invoke()
        out = interpreter.get_tensor(output_details["index"])
        N += 1

    end = time.perf_counter()
    inference_time = end - start
    return inference_time / N / inp.shape[0]


def seq(layers):
    def run(inputs, **kwargs):
        for l in layers:
            inputs = l(inputs, **kwargs)
        return inputs

    return run


def make_model(layers):
    input_data = tf.keras.Input(
        name="the_input", shape=(None, 1), dtype="float32"
    )  # >>(?, max_batch_seq, 26)

    y_pred = tf.squeeze(seq(layers)(tf.expand_dims(input_data, axis=0)), axis=0)
    y_pred = tf.keras.layers.Layer(name="ident")(y_pred)
    return tf.keras.Model(inputs=input_data, outputs=y_pred)


def save(model, fname, shape):

    def representative_dataset_gen():
        for _ in range(20):
            x = np.random.rand(*shape) - 0.5
            yield [x.astype(np.float32)]

    run_model = tf.function(lambda x: model(x))

    # This is important, let's fix the input size.
    concrete_func = run_model.get_concrete_function(
        tf.TensorSpec(shape, dtype=tf.float32)
    )
    converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])

    converter.experimental_new_converter = True
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    converter.representative_dataset = representative_dataset_gen
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    tflite_quant_model = converter.convert()
    with open(fname, "wb") as f:
        f.write(tflite_quant_model)


class Noop:
    def init_args(parser):
        pass

    def make_block(filters, args):
        def identity(x, training=False):
            return x
        return identity

class FullConv:
    def init_args(parser):
        parser.add_argument("--ksize", required=True, type=int)

    def make_block(filters, args):
        ksize = args.ksize
        layers = [
            tf.keras.layers.Conv2D(filters=filters, kernel_size=(1, ksize),),
        ]
        return seq(layers)

class Separable:
    def init_args(parser):
        parser.add_argument("--ksize", required=True, type=int)

    def make_block(filters, args):
        return tf.keras.layers.SeparableConv2D(
            filters=filters,
            kernel_size=(1, args.ksize),
        )

class Depthwise:
    def init_args(parser):
        parser.add_argument("--ksize", required=True, type=int)

    def make_block(filters, args):
        return tf.keras.layers.DepthwiseConv2D(
            kernel_size=(1, args.ksize),
        )

class KSeparable:
    def init_args(parser):
        parser.add_argument("--ksize", required=True, type=int)
        parser.add_argument("--k", required=True, type=int)

    def make_block(filters, args):
        assert args.ksize % args.k == 0
        layers = [
            tf.keras.layers.Conv2D(
                filters=filters,
                kernel_size=(1, args.k),
            ),
            tf.keras.layers.DepthwiseConv2D(
                kernel_size=(1, args.ksize // args.k),
                dilation_rate=(1, args.k),
            ),
        ]
        return seq(layers)

class GluFullConv:
    def init_args(parser):
        parser.add_argument("--ksize", required=True, type=int)

    def make_block(filters, args):
        l1 = tf.keras.layers.Conv2D(filters=filters, kernel_size=(1, args.ksize))
        l2 = tf.keras.layers.Conv2D(filters=filters, kernel_size=(1, args.ksize), activation=tf.nn.sigmoid)
        def run(x, training=False):
            return l1(x, training=training) * l2(x, training=training)
        return run

BLOCK_DEF = {
    "baseline_io": Noop,
    "fullconv": FullConv,
    "depthwise": Depthwise,
    "separable": Separable,
    "kseparable": KSeparable,
    "glu": GluFullConv,
}

def str_to_int_array(s):
    return [int(x) for x in s.split(",")]

parser = argparse.ArgumentParser()
parser.add_argument("--repeat", type=int, default=10)
parser.add_argument("--shape", default=[4,1668], type=str_to_int_array)
parser.add_argument("--filters", default=[96, 112, 128, 144, 156, 196, 224, 256, 320], type=str_to_int_array)

subparsers = parser.add_subparsers(dest="block_type", required=True)

for key, block in BLOCK_DEF.items():
    subparser = subparsers.add_parser(key)
    block.init_args(subparser)

args = parser.parse_args()

block = BLOCK_DEF[args.block_type]
shape = args.shape + [1]



def eval_model(block, filters, args):
    model = make_model(
        [
            # expand 1-channel input to filters
            tf.keras.layers.Conv2D(kernel_size=1, filters=filters),
            # reapeat benchmarked block
            *[block.make_block(filters, args) for i in range(args.repeat)],
            # downscale to 1-channel
            tf.keras.layers.Conv2D(kernel_size=1, filters=1),
        ]
    )

    model.compile()

    try:
        save(model, "tmp.tflite", shape)
        os.system("rm tmp_edgetpu.tflite")
        os.system("edgetpu_compiler tmp.tflite > /dev/null")
        return benchmark("tmp_edgetpu.tflite") * 1000000 / args.repeat
    except Exception as e:
        print(e)

results = []
print ("Benchmarking", args.block_type, "on filters count", args.filters)
for filters in args.filters:
    res = eval_model(block, filters, args)
    results.append(res)
print(", ".join("-" if x is None else "%.1f" % x for x in results))
