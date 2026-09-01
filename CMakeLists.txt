cmake_minimum_required(VERSION 3.16)

project(LANGER VERSION 1.0.0 LANGUAGES CXX)

option(LANGER_BUILD_TESTS "Build LANGER tests" ON)

add_library(langer src/langer.cpp)
add_library(LANGER::langer ALIAS langer)

target_include_directories(
  langer
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>)
target_compile_features(langer PUBLIC cxx_std_17)

if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
  target_compile_options(langer PRIVATE -Wall -Wextra -Wpedantic)
endif()

add_executable(langer_synthetic_smoke examples/synthetic_smoke.cpp)
target_link_libraries(langer_synthetic_smoke PRIVATE LANGER::langer)

add_executable(langer_cli examples/langer_cli.cpp)
target_link_libraries(langer_cli PRIVATE LANGER::langer)

if(LANGER_BUILD_TESTS)
  enable_testing()
  add_executable(langer_test tests/test_langer.cpp)
  target_link_libraries(langer_test PRIVATE LANGER::langer)
  add_test(NAME langer_test COMMAND langer_test)
endif()

include(GNUInstallDirs)
install(TARGETS langer EXPORT LANGERTargets)
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
