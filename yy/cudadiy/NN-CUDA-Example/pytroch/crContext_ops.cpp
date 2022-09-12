#include <torch/extension.h>
#include "createContext.h"

void torch_launch_crContext(int dev) {
    crContext(dev);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("torch_launch_crContext",
          &torch_launch_crContext,
          "crContext kernel warpper");
}

TORCH_LIBRARY(crContext, m) {
    m.def("torch_launch_crContext", torch_launch_crContext);
}
