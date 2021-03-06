# Tensorflow C/C++ Example

Simple example of how to load a pretrained model and use it to predict with Tensorflow 2.3.0 C API (CPU only, but probably it also works with GPU). The libraries for Linux used can be downloaded [here](https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow-cpu-linux-x86_64-2.3.0.tar.gz). It has not been tested with [Windows](https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow-cpu-windows-x86_64-2.3.0.zip) or [MacOS](https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow-cpu-darwin-x86_64-2.3.0.tar.gz) libraries, but I guess it should work too. This program will run independently from any Python Tensorflow installation that you may have.

It uses a model pretained with a simple MNIST convolutional network, that can be obtained by saving (`model.save(save_path)`) the resulting model from following the tutorial [Simple MNIST convnet](https://keras.io/examples/vision/mnist_convnet/). It loads one image from the dataset and does a class prediction with it.

The saved model needs to be in the **SavedModel** format and not **Keras H5** format or the C API won't load it. Read more about that [here](https://www.tensorflow.org/guide/keras/save_and_serialize).

## Dependencies
Tensorflow shared libraries (.so files) need to be downloaded and placed into the `lib` folder. In order for the program to find these libraries in execution time you need to run in the same terminal:
```
export LD_LIBRARY_PATH=lib
```
or you can follow the instructions [here](https://www.tensorflow.org/install/lang_c).

## Compilation and execution
```
make
./tf_example
```