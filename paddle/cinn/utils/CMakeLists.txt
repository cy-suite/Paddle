core_gather_headers()

gather_srcs(
  cinnapi_src
  SRCS
  dot_lang.cc
  error.cc
  functional.cc
  sized_multi_set.cc
  small_vector.cc
  string.cc
  timer.cc
  profiler.cc
  event.cc
  multi_threading.cc
  random_engine.cc
  external_func_names.cc)

cinn_cc_test(test_sized_multi_set SRCS sized_multi_set_test.cc DEPS cinncore)
cinn_cc_test(test_multi_threading SRCS multi_threading_test.cc DEPS cinncore)
cinn_cc_test(
  test_functional
  SRCS
  string.cc
  functional.cc
  functional_test.cc
  DEPS
  absl
  Threads::Threads)
