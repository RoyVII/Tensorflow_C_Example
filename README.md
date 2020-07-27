# Tensorflow C/C++ Example

Simple example of how to load a pretrained model and use it to predict with Tensorflow 2.2.0 C API (nightly, CPU only). The libraries for Linux  used can be downloaded [here](https://storage.googleapis.com/tensorflow-nightly/github/tensorflow/lib_package/libtensorflow-cpu-linux-x86_64.tar.gz). This program will run independently from any Python Tensorflow installation that you may have.

It uses a model pretained with a simple MNIST convolutional network, that can be obtained by saving (`model.save(save_path)`) the resulting model from following the tutorial [Simple MNIST convnet](https://keras.io/examples/vision/mnist_convnet/). It loads one image from the dataset and does a class prediction with it.

The saved model needs to be in the **SavedModel** format and not **Keras H5** format or the C API won't load it. See more about that [here](https://www.tensorflow.org/guide/keras/save_and_serialize).

## Dependencies
Tensorflow shared libraries (.so files) need to be downloaded and placed into the `lib` folder. In order for the program to find these libraries in execution time you need to run in the same terminal:
```
export LD_LIBRARY_PATH=lib
```
or add them to your shared libraries path permanently if you prefer.

## Compilation and execution
```
make
./tf_example
```