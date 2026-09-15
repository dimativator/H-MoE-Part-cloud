CXXFLAGS += -O3 -Wall -shared -std=c++17 -fPIC -fdiagnostics-color
PYBIND11_INCLUDES := $(shell python3 -m pybind11 --includes 2>/dev/null || python3 -c "import sysconfig, torch; print('-I' + torch.__path__[0] + '/include -I' + sysconfig.get_paths()['include'])")
CPPFLAGS += $(PYBIND11_INCLUDES)

LIBNAME = helpers_cpp
LIBEXT = $(shell python3-config --extension-suffix)

OUT = $(LIBNAME)$(LIBEXT)
SRC = helpers.cpp

default: $(OUT)

$(OUT): $(SRC)
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) $< -o $@
