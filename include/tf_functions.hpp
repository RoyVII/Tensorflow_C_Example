#include <vector>
#include <cstdint>
#include <tensorflow/c/c_api.h> // TensorFlow C API header.
 
namespace tf_functions {
	int load_session (const char * model_path, TF_Graph ** graph, TF_Session ** session);

	void delete_graph (TF_Graph * graph);

	int delete_session (TF_Session * session);

	int run_session (TF_Session * session,
		const TF_Output * input_operations, TF_Tensor * const * input_tensors, std::size_t num_inputs,
		const TF_Output * output_operations, TF_Tensor ** output_tensors, std::size_t num_outputs
		);

	int run_session (TF_Session * session,
		const std::vector<TF_Output>& input_operations, const std::vector<TF_Tensor*>& input_tensors, std::size_t num_inputs,
		const std::vector<TF_Output>& output_operations, std::vector<TF_Tensor*>& output_tensors, std::size_t num_outputs
		);

	int create_tensor (TF_DataType data_type, const std::int64_t * dims, std::size_t num_dims, const void * data, TF_Tensor ** tensor);

	// Templates must be defined in the same place as the definition it seems
	template <typename T>
	int create_tensor (TF_DataType data_type, const std::vector<std::int64_t>& dims, std::size_t num_dims, const std::vector<T>& data, TF_Tensor ** tensor) {
		return create_tensor(data_type, dims.data(), num_dims, data.data(), tensor);
	}


	void delete_tensor(TF_Tensor * tensor);

	
}