package(default_visibility = ["//visibility:public"])

licenses(["notice"])

py_library(
    name = "head",
    srcs = [
        "head.py",
    ],
    deps = [
        "//third_party/py/tensorflow",
        "//third_party/tensorflow_estimator",
    ],
)

py_library(
    name = "dnn",
    srcs = [
        "dnn.py",
    ],
    deps = [
        ":head",
        "//third_party/py/tensorflow",
        "//third_party/tensorflow_estimator",
    ],
)

py_test(
    name = "head_test",
    timeout = "long",
    srcs = ["head_test.py"],
    python_version = "PY3",
    deps = [
        ":head",
        "//third_party/py/tensorflow",
        "//third_party/py/tensorflow_privacy/privacy/estimators:test_utils",
        "//third_party/py/tensorflow_privacy/privacy/optimizers:dp_optimizer",
    ],
)

py_test(
    name = "dnn_test",
    timeout = "long",
    srcs = ["dnn_test.py"],
    python_version = "PY3",
    deps = [
        ":dnn",
        "//third_party/py/tensorflow",
        "//third_party/py/tensorflow_privacy/privacy/estimators:test_utils",
        "//third_party/py/tensorflow_privacy/privacy/optimizers:dp_optimizer",
    ],
)
