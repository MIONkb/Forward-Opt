set(Python_ADDITIONAL_VERSIONS 3.11)
find_package(PythonLibs 3 REQUIRED)
set(PYBIND11_PYTHON_VERSION 3.11)

add_subdirectory(pymlir)
add_subdirectory(kld)
# add_subdirectory(pyruntime)
