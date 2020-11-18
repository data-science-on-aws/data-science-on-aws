package(default_visibility = ["//visibility:public"])

licenses(["notice"])  # Apache 2.0

py_library(
    name = "head_utils",
    srcs = [
        "head_utils.py",
    ],
    deps = [
        ":binary_class_head",
        ":multi_class_head",
    ],
)

py_library(
    name = "binary_class_head",
    srcs = [
        "binary_class_head.py",
    ],
    deps = [
        "//third_party/py/tensorflow",
        "//third_party/tensorflow/python:keras_lib",  # TODO(b/163395075): Remove when fixed.
        "//third_party/tensorflow_estimator",
    ],
)

py_library(
    name = "multi_class_head",
    srcs = [
        "multi_class_head.py",
    ],
    deps = [
        "//third_party/py/tensorflow",
        "//third_party/tensorflow/python:keras_lib",  # TODO(b/163395075): Remove when fixed.
        "//third_party/tensorflow_estimator",
    ],
)

py_library(
    name = "multi_label_head",
    srcs = [
        "multi_label_head.py",
    ],
    deps = [
        "//third_party/py/tensorflow",
        "//third_party/tensorflow/python:keras_lib",  # TODO(b/163395075): Remove when fixed.
        "//third_party/tensorflow_estimator",
    ],
)

py_library(
    name = "dnn",
    srcs = [
        "dnn.py",
    ],
    deps = [
        ":head_utils",
        "//third_party/py/tensorflow",
        "//third_party/tensorflow_estimator",
    ],
)

py_library(
    name = "test_utils",
    srcs = [
        "test_utils.py",
    ],
    deps = [
        "//third_party/py/tensorflow",
    ],
)

py_test(
    name = "binary_class_head_test",
    timeout = "long",
    srcs = ["binary_class_head_test.py"],
    python_version = "PY3",
    deps = [
        ":binary_class_head",
        ":test_utils",
        "//third_party/py/tensorflow",
        "//third_party/py/tensorflow_privacy/privacy/optimizers:dp_optimizer_keras",
    ],
)

py_test(
    name = "multi_class_head_test",
    timeout = "long",
    srcs = ["multi_class_head_test.py"],
    python_version = "PY3",
    deps = [
        ":multi_class_head",
        ":test_utils",
        "//third_party/py/tensorflow",
        "//third_party/py/tensorflow_privacy/privacy/optimizers:dp_optimizer_keras",
    ],
)

py_test(
    name = "multi_label_head_test",
    timeout = "long",
    srcs = ["multi_label_head_test.py"],
    python_version = "PY3",
    deps = [
        ":multi_label_head",
        ":test_utils",
        "//third_party/py/tensorflow",
        "//third_party/py/tensorflow_privacy/privacy/optimizers:dp_optimizer_keras",
    ],
)

py_test(
    name = "dnn_test",
    timeout = "long",
    srcs = ["dnn_test.py"],
    python_version = "PY3",
    deps = [
        ":dnn",
        ":test_utils",
        "//third_party/py/absl/testing:parameterized",
        "//third_party/py/tensorflow",
        "//third_party/py/tensorflow_privacy/privacy/optimizers:dp_optimizer_keras",
    ],
)
