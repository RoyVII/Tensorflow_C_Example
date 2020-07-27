#include "../include/tf_functions.hpp"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>


namespace tf_functions {

	int load_session (const char * model_path, TF_Graph ** graph, TF_Session ** session) {
		TF_Status * status = TF_NewStatus();
		TF_SessionOptions * session_opts = TF_NewSessionOptions();
		TF_Buffer * run_opts = NULL;
		const char * tags = "serve";
		int num_tags = 1;

		*graph = TF_NewGraph();

	    *session = TF_LoadSessionFromSavedModel(session_opts, run_opts, model_path, &tags, num_tags, *graph, NULL, status);
	    
	    TF_DeleteSessionOptions(session_opts);

	    if (TF_GetCode(status) != TF_OK) {
			fprintf(stderr, "ERROR: Unable to create session %s\n", TF_Message(status));
			TF_DeleteStatus(status);
			return -1;
		}
	    TF_DeleteStatus(status);

	    fprintf(stdout, "Successfully created session\n");
	    return 0;
	}

	void delete_graph (TF_Graph * graph) {
		TF_DeleteGraph(graph);
	}

	int delete_session (TF_Session * session) {
		TF_Status * status = TF_NewStatus();
		TF_CloseSession(session, status);
		if (TF_GetCode(status) != TF_OK) {
			fprintf(stderr, "ERROR: Unable to close session %s\n", TF_Message(status));
			TF_DeleteStatus(status);
			return -1;
		}

		TF_DeleteSession(session, status);
		if (TF_GetCode(status) != TF_OK) {
			fprintf(stderr, "ERROR: Unable to delete session %s\n", TF_Message(status));
			TF_DeleteStatus(status);
			return -1;
		}
		TF_DeleteStatus(status);

		return 0;
	}

	int run_session (TF_Session * session, const TF_Output * input_operations, TF_Tensor * const * input_tensors, std::size_t num_inputs, const TF_Output * output_operations, TF_Tensor ** output_tensors, std::size_t num_outputs) {
		TF_Status * status = TF_NewStatus();
		TF_SessionRun(session,
			nullptr, // no options
			input_operations, input_tensors, static_cast<int>(num_inputs), // Input tensors, input tensor values, number of inputs.
			output_operations, output_tensors, static_cast<int>(num_outputs), // Output tensors, output tensor values, number of outputs.
			nullptr, 0, // Target operations, number of targets.
			nullptr, // Run metadata.
			status // Output status.
		);
		if (TF_GetCode(status) != TF_OK) {
			fprintf(stderr, "ERROR: Unable to run session %s\n", TF_Message(status));
			TF_DeleteStatus(status);
			return -1;
		}
		TF_DeleteStatus(status);

		fprintf(stdout, "Successfully run session\n");

		return 0;
	}

	int run_session (TF_Session * session, const std::vector<TF_Output>& input_operations, const std::vector<TF_Tensor*>& input_tensors, std::size_t num_inputs, const std::vector<TF_Output>& output_operations, std::vector<TF_Tensor*>& output_tensors, std::size_t num_outputs) {
		return run_session(session,
			input_operations.data(), input_tensors.data(), input_tensors.size(),
			output_operations.data(), output_tensors.data(), output_tensors.size());
	}


	int create_tensor (TF_DataType data_type, const std::int64_t * dims, std::size_t num_dims, const void * data, TF_Tensor ** tensor) {
		std::size_t len = 1;
		for (std::size_t i = 0; i < num_dims; i++) len *= dims[i];
		len *= TF_DataTypeSize(data_type);

		*tensor = TF_AllocateTensor(data_type, dims, num_dims, len);

		if (*tensor == nullptr) {
			return -1;
		}

		void * tensor_data = TF_TensorData(*tensor);
		if (tensor_data == nullptr) {
			TF_DeleteTensor(*tensor);
			*tensor = nullptr;
			return -1;
		}

		if (data != nullptr && len > 0 && len <= TF_TensorByteSize(*tensor)) {
			std::memcpy(tensor_data, data, len);
		}

		fprintf(stdout, "Successfully created tensor\n");
		
		return 0;
	}

	void delete_tensor(TF_Tensor * tensor) {
		TF_DeleteTensor(tensor);
	}
}
