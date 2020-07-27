#include "../include/tf_functions.hpp"
#include <iostream>
#include <fstream>
#include <string>
#include <cmath>

void read_data(std::vector<float> * x_test_0, std::vector<float> * y_test_0) {
	int i, j;
	std::string line;
	std::string delimiter = " ";


	std::ifstream x_test_file("test_model/mnist_x_test_0.txt");

	if (x_test_file.is_open()) {
		i = 0;

		while (std::getline(x_test_file, line)) {
			j = 0;

			size_t pos = 0;
			std::string token;
			while ((pos = line.find(delimiter)) != std::string::npos) {
			    token = line.substr(0, pos);
			    (*x_test_0)[i*28 + j] = std::stof(token);
			    line.erase(0, pos + delimiter.length());

			    j++;
			}
			(*x_test_0)[i*28 + j] = std::stof(token);

			i++;
		}

		x_test_file.close();
	} else {
		std::cout << "Unable to open file"; 
	}


	std::ifstream y_test_file("test_model/mnist_y_test_0.txt");

	if (y_test_file.is_open()) {
		i = 0;

		while (std::getline(y_test_file, line)) {
			j = 0;

			size_t pos = 0;
			std::string token;
			while ((pos = line.find(delimiter)) != std::string::npos) {
			    token = line.substr(0, pos);
			    (*y_test_0)[i*10 + j] = std::stof(token);
			    line.erase(0, pos + delimiter.length());

			    j++;
			}
			(*y_test_0)[i*10 + j] = std::stof(token);

			i++;
		}

		y_test_file.close();
	} else {
		std::cout << "Unable to open file"; 
	}
}

int main () {
	std::vector<float> x_test_0(28*28*1);
	std::vector<float> y_test_0(10*1);

	/* Loads one image from the MNIST database.
	 * 
	 * The values are already normalized between [0, 1].
	 * It is loaded into a vector with 784 items (28*28*1), since the API
	 * takes the data from 1D vectors.
	 *
	 */

	read_data(&x_test_0, &y_test_0);

	/* For the moment it does not work with models saved in the .h5 format, just SavedModel format.
	 * See more here https://www.tensorflow.org/guide/keras/save_and_serialize
	 */
	char model_path[] = "test_model/mnist_test_model";
	TF_Graph * graph = nullptr;
	TF_Session * session = nullptr;
	TF_Tensor * input_tensor = nullptr, * output_tensor = nullptr;

	tf_functions::load_session(model_path, &graph, &session);


	/* HOW TO GET OPERATIONS NAMES (in Python):
	 *
	 * loaded = tf.saved_model.load(model_path)
	 * loaded.signatures # This returns _SignatureMap({'serving_default': <tensorflow.python.saved_model.load._WrapperFunction object at 0x7f63f5bce460>})
	 * infer = loaded.signatures["serving_default"] 
	 * infer.structured_input_signature # This returns ((), {'input_1': TensorSpec(shape=(None, 28, 28, 1), dtype=tf.float32, name='input_1')})
	 * 
	 * With this we form the name "serving_default_input_1" 
	 *
	 * We can also use
	 * infer.graph.get_operations()
	 * to get all the operations names. The first one is the input, and the penultimate the output.
	 *
	 */

	TF_Operation * input_op = TF_GraphOperationByName(graph, "serving_default_input_1");
	TF_Output input = TF_Output{input_op, 0};
	if (input.oper == nullptr) {
		std::cout << "Can't init input_op" << std::endl;
		return 2;
	}

	std::cout << "init input_op" << std::endl;

	TF_Operation * output_op = TF_GraphOperationByName(graph, "StatefulPartitionedCall");
	TF_Output output = TF_Output{output_op, 0};
	if (output.oper == nullptr) {
		std::cout << "Can't init output_op" << std::endl;
		return 2;
	}

	std::cout << "init output_op" << std::endl;

	/* The first dimension corresponds to the number of elements that will have to predict.
	 * The other dimensions correspond to the elements dimensions.
	 *
	 * In this example we just have 1 image with 28x28 dimensions. Since it is a black and
	 * white picture, its third value is just 1. If it was a color picture it would typically
	 * have 3, one per each RGB channel.
     *		
	 */

	std::vector<std::int64_t> dims = {1, 28, 28, 1};
	int num_dims = 4;
	tf_functions::create_tensor(TF_FLOAT, dims, num_dims, x_test_0, &input_tensor);

	tf_functions::run_session (session,
		&input, &input_tensor, 1,
		&output, &output_tensor, 1);


	// Check results
	auto tensor_data = static_cast<float*>(TF_TensorData(output_tensor));


	/* EXPECTED OUTPUT:
	 * 
	 * 1.1062757e-09, 9.3401142e-10, 4.5229385e-06, 1.4784278e-05,
     * 2.0455739e-13, 8.3552873e-11, 3.9019279e-16, 9.9997914e-01,
     * 1.8186158e-08, 1.5447515e-06
	 * 
	 * If rounded it is the same as y_test_0.
     */

	std::cout << std::endl << "Prediction:" << std::endl;

	for (int i = 0; i < 10; i++) {
		std::cout << tensor_data[i] << " ";
	}
	std::cout << std::endl << std::endl;

	for (int i = 0; i < 10; i++) {
		std::cout << "y_test: " << y_test_0[i] << " Prediction (rounded): " << round(tensor_data[i]) << std::endl;

	}

	tf_functions::delete_tensor(input_tensor);
	tf_functions::delete_tensor(output_tensor);
	tf_functions::delete_graph(graph);
	tf_functions::delete_session(session);

	return 0;
}