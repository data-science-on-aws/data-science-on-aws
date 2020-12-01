package(default_visibility = ["//visibility:public"])

licenses(["notice"])  # Apache 2.0

py_library(
    name = "dp_query",
    srcs = ["dp_query.py"],
    deps = [
        "//third_party/py/distutils",
        "//third_party/py/tensorflow",
    ],
)

py_library(
    name = "gaussian_query",
    srcs = ["gaussian_query.py"],
    deps = [
        ":dp_query",
        ":normalized_query",
        "//third_party/py/distutils",
        "//third_party/py/tensorflow",
    ],
)

py_test(
    name = "gaussian_query_test",
    size = "small",
    srcs = ["gaussian_query_test.py"],
    python_version = "PY3",
    deps = [
        ":gaussian_query",
        ":test_utils",
        "//learning/brain/public:disable_tf2",  # build_cleaner: keep; go/disable_tf2
        "//third_party/py/absl/testing:parameterized",
        "//third_party/py/numpy",
        "//third_party/py/six",
        "//third_party/py/tensorflow",
    ],
)

py_library(
    name = "no_privacy_query",
    srcs = ["no_privacy_query.py"],
    deps = [
        ":dp_query",
        "//third_party/py/distutils",
        "//third_party/py/tensorflow",
    ],
)

py_test(
    name = "no_privacy_query_test",
    size = "small",
    srcs = ["no_privacy_query_test.py"],
    python_version = "PY3",
    deps = [
        ":no_privacy_query",
        ":test_utils",
        "//learning/brain/public:disable_tf2",  # build_cleaner: keep; go/disable_tf2
        "//third_party/py/absl/testing:parameterized",
        "//third_party/py/tensorflow",
    ],
)

py_library(
    name = "normalized_query",
    srcs = ["normalized_query.py"],
    deps = [
        ":dp_query",
        "//third_party/py/distutils",
        "//third_party/py/tensorflow",
    ],
)

py_test(
    name = "normalized_query_test",
    size = "small",
    srcs = ["normalized_query_test.py"],
    python_version = "PY3",
    deps = [
        ":gaussian_query",
        ":normalized_query",
        ":test_utils",
        "//learning/brain/public:disable_tf2",  # build_cleaner: keep; go/disable_tf2
        "//third_party/py/tensorflow",
    ],
)

py_library(
    name = "nested_query",
    srcs = ["nested_query.py"],
    deps = [
        ":dp_query",
        "//third_party/py/tensorflow",
        "//third_party/py/tree",
    ],
)

py_test(
    name = "nested_query_test",
    size = "small",
    srcs = ["nested_query_test.py"],
    python_version = "PY3",
    deps = [
        ":gaussian_query",
        ":nested_query",
        ":test_utils",
        "//learning/brain/public:disable_tf2",  # build_cleaner: keep; go/disable_tf2
        "//third_party/py/absl/testing:parameterized",
        "//third_party/py/distutils",
        "//third_party/py/numpy",
        "//third_party/py/tensorflow",
    ],
)

py_library(
    name = "quantile_adaptive_clip_sum_query",
    srcs = ["quantile_adaptive_clip_sum_query.py"],
    deps = [
        ":dp_query",
        ":gaussian_query",
        ":normalized_query",
        ":quantile_estimator_query",
        "//third_party/py/tensorflow",
    ],
)

py_test(
    name = "quantile_adaptive_clip_sum_query_test",
    srcs = ["quantile_adaptive_clip_sum_query_test.py"],
    python_version = "PY3",
    deps = [
        ":quantile_adaptive_clip_sum_query",
        ":test_utils",
        "//learning/brain/public:disable_tf2",  # build_cleaner: keep; go/disable_tf2
        "//third_party/py/absl/testing:parameterized",
        "//third_party/py/numpy",
        "//third_party/py/tensorflow",
        "//third_party/py/tensorflow_privacy/privacy/analysis:privacy_ledger",
    ],
)

py_library(
    name = "quantile_estimator_query",
    srcs = ["quantile_estimator_query.py"],
    deps = [
        ":dp_query",
        ":gaussian_query",
        ":no_privacy_query",
        "//third_party/py/tensorflow",
    ],
)

py_test(
    name = "quantile_estimator_query_test",
    srcs = ["quantile_estimator_query_test.py"],
    python_version = "PY3",
    deps = [
        ":quantile_estimator_query",
        ":test_utils",
        "//learning/brain/public:disable_tf2",  # build_cleaner: keep; go/disable_tf2
        "//third_party/py/absl/testing:parameterized",
        "//third_party/py/numpy",
        "//third_party/py/tensorflow",
    ],
)

py_library(
    name = "test_utils",
    srcs = ["test_utils.py"],
    deps = [],
)
