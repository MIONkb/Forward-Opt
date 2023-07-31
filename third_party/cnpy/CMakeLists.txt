file(GLOB _sources *.cpp)


add_llvm_library(cnpy
  ${_sources}
)

target_compile_options(cnpy PRIVATE "-frtti")
target_compile_options(cnpy PRIVATE "-fexceptions")