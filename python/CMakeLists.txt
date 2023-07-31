include(AddMLIRPython)

set(FORWARD_MLIR_PYTHON_ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/mlir")
set(FORWARD_MLIR_PYTHON_PACKAGES_DIR "${CMAKE_BINARY_DIR}/python")

################################################################################
# Structural groupings.
################################################################################
declare_mlir_python_sources(ForwardMLIRPythonSources)
declare_mlir_python_sources(ForwardMLIRPythonSources.Dialects
  ROOT_DIR "${FORWARD_MLIR_PYTHON_ROOT_DIR}"
  ADD_TO_PARENT ForwardMLIRPythonSources
)

################################################################################
# Dialect bindings
################################################################################
# declare_mlir_dialect_python_bindings(
#  ADD_TO_PARENT ForwardMLIRPythonSources.Dialects
#  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/tpu_mlir"
#  TD_FILE dialects/TopOps.td
#  SOURCES dialects/top.py
#  DIALECT_NAME top)

################################################################################
# The fully assembled package of modules.
################################################################################
set(_source_components
  ForwardMLIRPythonSources
)
add_mlir_python_modules(ForwardMLIRPythonModules
  ROOT_PREFIX "${FORWARD_MLIR_PYTHON_PACKAGES_DIR}/mlir"
  INSTALL_PREFIX "python/mlir/"
  DECLARED_SOURCES ${_source_components}
  )


#add_subdirectory(tools)
install(
  DIRECTORY ./ DESTINATION python FILES_MATCHING
  USE_SOURCE_PERMISSIONS
  PATTERN "*.py"
  PATTERN "xlsx_to_py.py" EXCLUDE
  PATTERN "test_*" EXCLUDE
  PATTERN "__pycache__" EXCLUDE
  PATTERN ".pytest_cache" EXCLUDE
)
#install(
#  DIRECTORY ./ DESTINATION python FILES_MATCHING
#  USE_SOURCE_PERMISSIONS
#  PATTERN "test_onnx.py"
#)

set(PYTHON_INSTALL_PATH ${PROJECT_SOURCE_DIR}/build/bin)

set(TOOLS_SUBFILES
  utils/mlir_parser.py
  utils/misc.py
  utils/preprocess.py
  utils/log_setting.py
  transform/MLIRImporter.py
  transform/BaseConverter.py
  calibration/kld_calibrator.py
  calibration/data_selector.py)

message("Install python: ${PYTHON_INSTALL_PATH}")

set(TOOLS_FILES
  trans.py
  seperate_weights.py
  run_calibration.py
  utils/mlir_parser.py
  utils/misc.py
  utils/preprocess.py
  utils/log_setting.py
  transform/MLIRImporter.py
  transform/BaseConverter.py
  calibration/kld_calibrator.py
  calibration/data_selector.py)

set(TOOLS_TARGETS ${TOOLS_FILES})

# Stuff our python into the build area:
add_custom_target(seperate_weights.py ALL DEPENDS ${TOOLS_TARGETS})

# This chicanery is necessary to ensure executable permissions.
file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/copy_weightseperate.cmake"
"file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/seperate_weights.py
DESTINATION ${PROJECT_BINARY_DIR}/bin
FILE_PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_WRITE
GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE)")

add_custom_command(OUTPUT ${PROJECT_BINARY_DIR}/bin/seperate_weights.py COMMAND
${CMAKE_COMMAND} -P ${CMAKE_CURRENT_BINARY_DIR}/copy_weightseperate.cmake)

file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/copy_runcalibration.cmake"
"file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/run_calibration.py
DESTINATION ${PROJECT_BINARY_DIR}/bin
FILE_PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_WRITE
GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE)")

add_custom_command(OUTPUT ${PROJECT_BINARY_DIR}/bin/run_calibration.py COMMAND
${CMAKE_COMMAND} -P ${CMAKE_CURRENT_BINARY_DIR}/copy_runcalibration.cmake)

foreach(file ${TOOLS_SUBFILES})
add_custom_command(OUTPUT ${PROJECT_BINARY_DIR}/bin/${file}
                    COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/${file}
                   ${PROJECT_BINARY_DIR}/bin/${file}
                   DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/${file})
endforeach()

# Install our python stuff too.
install(PROGRAMS seperate_weights.py run_calibration.py trans.py DESTINATION ${PYTHON_INSTALL_PATH})
#foreach(file ${TOOLS_SUBFILES})
#  install(PROGRAMS ${file} DESTINATION ${PYTHON_INSTALL_PATH}/)
#endforeach()
