#include <torch/extension.h>

#include <pybind11/stl.h>


void grouped_linear_forward(
	at::Tensor input,
	at::Tensor weight,
	at::Tensor output,
	const std::vector<int> &input_group_sizes
);


void grouped_linear_backward_dx(
	at::Tensor input,
	at::Tensor weight,
    at::Tensor grad_output,
	at::Tensor grad_input,
	const std::vector<int> &input_group_sizes
);


void grouped_linear_backward_dw(
	at::Tensor input,
	at::Tensor weight,
    at::Tensor grad_output,
	at::Tensor grad_weight,
	const std::vector<int> &input_group_sizes
);


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
	m.def(
		"grouped_linear_forward",
		&grouped_linear_forward,
		pybind11::arg("input"),
		pybind11::arg("weight"),
		pybind11::arg("output"),
		pybind11::arg("input_group_sizes")
	);
	m.def(
		"grouped_linear_backward_dx",
		&grouped_linear_backward_dx,
		pybind11::arg("input"),
		pybind11::arg("weight"),
		pybind11::arg("grad_output"),
		pybind11::arg("grad_input"),
		pybind11::arg("input_group_sizes")
	);
    m.def(
		"grouped_linear_backward_dw",
		&grouped_linear_backward_dw,
		pybind11::arg("input"),
		pybind11::arg("weight"),
		pybind11::arg("grad_output"),
		pybind11::arg("grad_weight"),
		pybind11::arg("input_group_sizes")
	);
}
