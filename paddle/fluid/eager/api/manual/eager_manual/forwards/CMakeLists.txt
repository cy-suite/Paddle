set(eager_manual_functions
    ${PADDLE_SOURCE_DIR}/paddle/fluid/eager/api/manual/eager_manual/forwards/add_n_fwd_func.cc
    ${PADDLE_SOURCE_DIR}/paddle/fluid/eager/api/manual/eager_manual/forwards/conv2d_fwd_function.cc
    ${PADDLE_SOURCE_DIR}/paddle/fluid/eager/api/manual/eager_manual/forwards/sync_batch_norm_fwd_func.cc
    ${PADDLE_SOURCE_DIR}/paddle/fluid/eager/api/manual/eager_manual/forwards/multiply_fwd_func.cc
    ${PADDLE_SOURCE_DIR}/paddle/fluid/eager/api/manual/eager_manual/forwards/reshard_fwd_func.cc
    ${PADDLE_SOURCE_DIR}/paddle/fluid/eager/api/manual/eager_manual/forwards/dtensor_to_local_fwd_func.cc
    ${PADDLE_SOURCE_DIR}/paddle/fluid/eager/api/manual/eager_manual/forwards/dtensor_from_local_fwd_func.cc
    PARENT_SCOPE)
