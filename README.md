# Benchmark inference speed of tensors on Coral Edge TPU

## Installation

- Install tensorflow >=2.3 (we used 2.4 nightly) - `pip install tensorflow-gpu`
- Install Coral SDK and compiler: `edgetpu-compiler`, `libedgetpu1-max` from https://coral.ai/software/#debian-packages 

## Usage

`python op-benchmark.py [--repeat R] [--shape S] [--filters F] {baseline_io,fullconv,...} --ksize K`
with defaults `--repeat=10`, `--shape=[4,1668]` and `--filters=[96,112,...320]`.

This will benchmark a network that consists of
1) input of single channel => (pointwise) convolution which upscales this to `F` channels
2) `--repeat` times given block under test
3) pointwise convolution to single channel => output

and reports average running time for single block in microseconds.

Example:
First measure "I/O" overhead:
```
>python op_benchmark.py baseline_io
Benchmarking baseline_io on filters count [96, 112, 128, 144, 156, 196, 224, 256, 320]
17.0, 17.6, 17.9, 18.0, 18.4, 19.6, 19.6, 19.7, 22.1
```

Then measure the block
```
>python op_benchmark.py depthwise --ksize 5
Benchmarking depthwise on filters count [96, 112, 128, 144, 156, 196, 224, 256, 320]
50.5, 56.0, 61.5, 66.6, 70.6, 84.2, 93.1, 180.9, 211.4
```

This indicates that Coral takes roughly 61.5-17.9 = ~44us to run 128x5->128 convolution on tensor 4x1668x128.

## Pro tips:
Coral has currently some weird problems on USB3 (https://github.com/google-coral/edgetpu/issues/207) where it might run out of bandwidth reservation. 
It seems that this benchmark is quite good at triggering it :-)
If you come across this problem on linux:
1) find your usb hub device id using `lspci` (e.g. something like `00:14.0 USB controller: ...`)
2) hard-reset USB hub:
  ```
  echo "0000:00:14.0" > /sys/bus/pci/drivers/xhci_hcd/unbind
  echo "0000:00:14.0" > /sys/bus/pci/drivers/xhci_hcd/bind
  ```
  Note that simple unplugging and plugging device back does not help
