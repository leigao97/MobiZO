cmake_minimum_required(VERSION 3.29)
project(mobizo_train LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
find_package(executorch CONFIG REQUIRED)
add_executable(mobizo_train train.cpp)
target_include_directories(mobizo_train PRIVATE
  "${EXECUTORCH_ROOT}/.."
  "${EXECUTORCH_ROOT}/extension/llm/tokenizers/third-party/json/single_include")
target_compile_definitions(mobizo_train PRIVATE ET_EVENT_TRACER_ENABLED)
target_link_options(mobizo_train PRIVATE "-Wl,--no-as-needed")
target_link_libraries(mobizo_train PRIVATE
  extension_module extension_tensor executorch portable_ops_lib
  qnn_executorch_backend)
set_target_properties(mobizo_train PROPERTIES BUILD_RPATH "${CMAKE_PREFIX_PATH}/lib" INSTALL_RPATH "$ORIGIN")
