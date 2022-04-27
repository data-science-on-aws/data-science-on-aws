import re
import builtins
import json
import sys
import functools
import importlib.util
from itertools import chain
import importlib
import importlib_metadata
import yaml
import pkg_resources
from collections import namedtuple
from typing import NamedTuple, Optional
from sys import version_info
from packaging.version import Version, InvalidVersion

PYTHON_VERSION = "{major}.{minor}.{micro}".format(
    major=version_info.major, minor=version_info.minor, micro=version_info.micro
)


def _get_pip_version():
    """
    :return: The version of ``pip`` that is installed in the current environment,
             or ``None`` if ``pip`` is not currently installed / does not have a
             ``__version__`` attribute.
    """
    try:
        import pip

        return getattr(pip, "__version__")
    except ImportError:
        return None


_conda_header = """\
name: mlflow-env
channels:
  - conda-forge
"""


_CONDA_ENV_FILE_NAME = "conda.yaml"
_REQUIREMENTS_FILE_NAME = "requirements.txt"
_CONSTRAINTS_FILE_NAME = "constraints.txt"


def _mlflow_conda_env(
    path=None,
    additional_conda_deps=None,
    additional_pip_deps=None,
    additional_conda_channels=None,
    install_mlflow=True,
):
    """
    Creates a Conda environment with the specified package channels and dependencies. If there are
    any pip dependencies, including from the install_mlflow parameter, then pip will be added to
    the conda dependencies. This is done to ensure that the pip inside the conda environment is
    used to install the pip dependencies.
    :param path: Local filesystem path where the conda env file is to be written. If unspecified,
                 the conda env will not be written to the filesystem; it will still be returned
                 in dictionary format.
    :param additional_conda_deps: List of additional conda dependencies passed as strings.
    :param additional_pip_deps: List of additional pip dependencies passed as strings.
    :param additional_conda_channels: List of additional conda channels to search when resolving
                                      packages.
    :return: ``None`` if ``path`` is specified. Otherwise, the a dictionary representation of the
             Conda environment.
    """
    pip_deps = (["mlflow"] if install_mlflow else []) + (
        additional_pip_deps if additional_pip_deps else []
    )
    conda_deps = additional_conda_deps if additional_conda_deps else []
    if pip_deps:
        pip_version = _get_pip_version()
        if pip_version is not None:
            # When a new version of pip is released on PyPI, it takes a while until that version is
            # uploaded to conda-forge. This time lag causes `conda create` to fail with
            # a `ResolvePackageNotFound` error. As a workaround for this issue, use `<=` instead
            # of `==` so conda installs `pip_version - 1` when `pip_version` is unavailable.
            conda_deps.append(f"pip<={pip_version}")
        else:
            _logger.warning(
                "Failed to resolve installed pip version. ``pip`` will be added to conda.yaml"
                " environment spec without a version specifier."
            )
            conda_deps.append("pip")

    env = yaml.safe_load(_conda_header)
    env["dependencies"] = ["python={}".format(PYTHON_VERSION)]
    env["dependencies"] += conda_deps
    env["dependencies"].append({"pip": pip_deps})
    if additional_conda_channels is not None:
        env["channels"] += additional_conda_channels

    if path is not None:
        with open(path, "w") as out:
            yaml.safe_dump(env, stream=out, default_flow_style=False)
        return None
    else:
        return env


def _is_comment(line):
    return line.startswith("#")


def _is_empty(line):
    return line == ""


def _strip_inline_comment(line):
    return line[: line.find(" #")].rstrip() if " #" in line else line


def _is_requirements_file(line):
    return line.startswith("-r ") or line.startswith("--requirement ")


def _is_constraints_file(line):
    return line.startswith("-c ") or line.startswith("--constraint ")


def _join_continued_lines(lines):
    """
    Joins lines ending with '\\'.
    >>> _join_continued_lines["a\\", "b\\", "c"]
    >>> 'abc'
    """
    continued_lines = []

    for line in lines:
        if line.endswith("\\"):
            continued_lines.append(line.rstrip("\\"))
        else:
            continued_lines.append(line)
            yield "".join(continued_lines)
            continued_lines.clear()

    # The last line ends with '\'
    if continued_lines:
        yield "".join(continued_lines)


# Represents a pip requirement.
#
# :param req_str: A requirement string (e.g. "scikit-learn == 0.24.2").
# :param is_constraint: A boolean indicating whether this requirement is a constraint.
_Requirement = namedtuple("_Requirement", ["req_str", "is_constraint"])


def _parse_requirements(requirements, is_constraint):
    """
    A simplified version of `pip._internal.req.parse_requirements` which performs the following
    operations on the given requirements file and yields the parsed requirements.
    - Remove comments and blank lines
    - Join continued lines
    - Resolve requirements file references (e.g. '-r requirements.txt')
    - Resolve constraints file references (e.g. '-c constraints.txt')
    :param requirements: A string path to a requirements file on the local filesystem or
                         an iterable of pip requirement strings.
    :param is_constraint: Indicates the parsed requirements file is a constraint file.
    :return: A list of ``_Requirement`` instances.
    References:
    - `pip._internal.req.parse_requirements`:
      https://github.com/pypa/pip/blob/7a77484a492c8f1e1f5ef24eaf71a43df9ea47eb/src/pip/_internal/req/req_file.py#L118
    - Requirements File Format:
      https://pip.pypa.io/en/stable/cli/pip_install/#requirements-file-format
    - Constraints Files:
      https://pip.pypa.io/en/stable/user_guide/#constraints-files
    """
    if isinstance(requirements, str):
        base_dir = os.path.dirname(requirements)
        with open(requirements) as f:
            requirements = f.read().splitlines()
    else:
        base_dir = os.getcwd()

    lines = map(str.strip, requirements)
    lines = map(_strip_inline_comment, lines)
    lines = _join_continued_lines(lines)
    lines = filterfalse(_is_comment, lines)
    lines = filterfalse(_is_empty, lines)

    for line in lines:
        if _is_requirements_file(line):
            req_file = line.split(maxsplit=1)[1]
            # If `req_file` is an absolute path, `os.path.join` returns `req_file`:
            # https://docs.python.org/3/library/os.path.html#os.path.join
            abs_path = os.path.join(base_dir, req_file)
            yield from _parse_requirements(abs_path, is_constraint=False)
        elif _is_constraints_file(line):
            req_file = line.split(maxsplit=1)[1]
            abs_path = os.path.join(base_dir, req_file)
            yield from _parse_requirements(abs_path, is_constraint=True)
        else:
            yield _Requirement(line, is_constraint)


# https://www.python.org/dev/peps/pep-0508/#names
_PACKAGE_NAME_REGEX = re.compile(r"^(\w+|\w+[\w._-]*\w+)")

def _get_package_name(requirement):
    m = _PACKAGE_NAME_REGEX.match(requirement)
    return m and m.group(1)


_NORMALIZE_REGEX = re.compile(r"[-_.]+")


def _normalize_package_name(pkg_name):
    """
    Normalizes a package name using the rule defined in PEP 503:
    https://www.python.org/dev/peps/pep-0503/#normalized-names
    """
    return _NORMALIZE_REGEX.sub("-", pkg_name).lower()


def _get_requires_recursive(pkg_name, top_pkg_name=None) -> set:
    """
    Recursively yields both direct and transitive dependencies of the specified
    package.
    The `top_pkg_name` argument will track what's the top-level dependency for
    which we want to list all sub-dependencies.
    This ensures that we don't fall into recursive loops for packages with are
    dependant on each other.
    """
    if top_pkg_name is None:
        # Assume the top package
        top_pkg_name = pkg_name

    pkg_name = _normalize_package_name(pkg_name)
    if pkg_name not in pkg_resources.working_set.by_key:
        return

    package = pkg_resources.working_set.by_key[pkg_name]
    reqs = package.requires()
    if len(reqs) == 0:
        return

    for req in reqs:
        req_name = _normalize_package_name(req.name)
        if req_name == top_pkg_name:
            # If the top package ends up providing himself again through a
            # recursive dependency, we don't want to consider it as a
            # dependency
            continue

        yield req_name
        yield from _get_requires_recursive(req.name, top_pkg_name)


def _prune_packages(packages):
    """
    Prunes packages required by other packages. For example, `["scikit-learn", "numpy"]` is pruned
    to `["scikit-learn"]`.
    """
    packages = set(packages)
    requires = set(_flatten(map(_get_requires_recursive, packages)))
    return packages - requires


def _get_installed_version(package, module=None):
    """
    Obtains the installed package version using `importlib_metadata.version`. If it fails, use
    `__import__(module or package).__version__`.
    """
    try:
        version = importlib_metadata.version(package)
    except importlib_metadata.PackageNotFoundError:
        # Note `importlib_metadata.version(package)` is not necessarily equal to
        # `__import__(package).__version__`. See the example for pytorch below.
        #
        # Example
        # -------
        # $ pip install torch==1.9.0
        # $ python -c "import torch; print(torch.__version__)"
        # 1.9.0+cu102
        # $ python -c "import importlib_metadata; print(importlib_metadata.version('torch'))"
        # 1.9.0
        version = __import__(module or package).__version__

    # Strip the suffix from `dev` versions of PySpark, which are not available for installation
    # from Anaconda or PyPI
    if package == "pyspark":
        version = _strip_dev_version_suffix(version)

    return version


def _get_local_version_label(version):
    """
    Extracts a local version label from `version`.
    :param version: A version string.
    """
    try:
        return Version(version).local
    except InvalidVersion:
        return None


def _strip_local_version_label(version):
    """
    Strips a local version label in `version`.
    Local version identifiers:
    https://www.python.org/dev/peps/pep-0440/#local-version-identifiers
    :param version: A version string to strip.
    """

    class IgnoreLocal(Version):
        @property
        def local(self):
            return None

    try:
        return str(IgnoreLocal(version))
    except InvalidVersion:
        return version


def _get_pinned_requirement(package, version=None, module=None):
    """
    Returns a string representing a pinned pip requirement to install the specified package and
    version (e.g. 'mlflow==1.2.3').
    :param package: The name of the package.
    :param version: The version of the package. If None, defaults to the installed version.
    :param module: The name of the top-level module provided by the package . For example,
                   if `package` is 'scikit-learn', `module` should be 'sklearn'. If None, defaults
                   to `package`.
    """
    if version is None:
        version_raw = _get_installed_version(package, module)
        local_version_label = _get_local_version_label(version_raw)
        if local_version_label:
            version = _strip_local_version_label(version_raw)
            msg = (
                "Found {package} version ({version_raw}) contains a local version label "
                "(+{local_version_label}). MLflow logged a pip requirement for this package as "
                "'{package}=={version_logged}' without the local version label to make it "
                "installable from PyPI. To specify pip requirements containing local version "
                "labels, please use `conda_env` or `pip_requirements`."
            ).format(
                package=package,
                version_raw=version_raw,
                version_logged=version,
                local_version_label=local_version_label,
            )
            _logger.warning(msg)

        else:
            version = version_raw

    return f"{package}=={version}"


_MODULES_TO_PACKAGES = None
_PACKAGES_TO_MODULES = None


def _init_modules_to_packages_map():
    global _MODULES_TO_PACKAGES
    if _MODULES_TO_PACKAGES is None and _PACKAGES_TO_MODULES is None:
        # Note `importlib_metada.packages_distributions` only captures packages installed into
        # Python’s site-packages directory via tools such as pip:
        # https://importlib-metadata.readthedocs.io/en/latest/using.html#using-importlib-metadata
        _MODULES_TO_PACKAGES = importlib_metadata.packages_distributions()

def _init_packages_to_modules_map():
    _init_modules_to_packages_map()
    global _PACKAGES_TO_MODULES
    _PACKAGES_TO_MODULES = {}
    for module, pkg_list in _MODULES_TO_PACKAGES.items():
        for pkg_name in pkg_list:
            _PACKAGES_TO_MODULES[pkg_name] = module

# Represents the PyPI package index at a particular date
# :param date: The YYYY-MM-DD formatted string date on which the index was fetched.
# :param package_names: The set of package names in the index.
_PyPIPackageIndex = namedtuple("_PyPIPackageIndex", ["date", "package_names"])

def _load_pypi_package_index():
    pypi_index_path = './pypi_package_index.json'
    with open(pypi_index_path, "r") as f:
        index_dict = json.load(f)

    return _PyPIPackageIndex(
        date=index_dict["index_date"],
        package_names=set(index_dict["package_names"]),
    )


_PYPI_PACKAGE_INDEX = None


_init_modules_to_packages_map()
if _PYPI_PACKAGE_INDEX is None:
    _PYPI_PACKAGE_INDEX = _load_pypi_package_index()

def _flatten(iterable):
    return chain.from_iterable(iterable)


def load_module(file_name, module_name):
    spec = importlib.util.spec_from_file_location(module_name, file_name)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _get_top_level_module(full_module_name):
    return full_module_name.split(".")[0]


class _CaptureImportedModules:
    """
    A context manager to capture imported modules by temporarily applying a patch to
    `builtins.__import__` and `importlib.import_module`.
    """

    def __init__(self):
        self.imported_modules = set()
        self.original_import = None
        self.original_import_module = None

    def _wrap_import(self, original):
        # pylint: disable=redefined-builtin
        @functools.wraps(original)
        def wrapper(name, globals=None, locals=None, fromlist=(), level=0):
            is_absolute_import = level == 0
            if not name.startswith("_") and is_absolute_import:
                top_level_module = _get_top_level_module(name)
                self.imported_modules.add(top_level_module)
            return original(name, globals, locals, fromlist, level)

        return wrapper

    def _wrap_import_module(self, original):
        @functools.wraps(original)
        def wrapper(name, *args, **kwargs):
            if not name.startswith("_"):
                top_level_module = _get_top_level_module(name)
                self.imported_modules.add(top_level_module)
            return original(name, *args, **kwargs)

        return wrapper

    def __enter__(self):
        # Patch `builtins.__import__` and `importlib.import_module`
        self.original_import = builtins.__import__
        self.original_import_module = importlib.import_module
        builtins.__import__ = self._wrap_import(self.original_import)
        importlib.import_module = self._wrap_import_module(self.original_import_module)
        return self

    def __exit__(self, *_, **__):
        # Revert the patches
        builtins.__import__ = self.original_import
        importlib.import_module = self.original_import_module


with _CaptureImportedModules() as cap:
    # pylint: disable=unused-import,unused-variable
#    import math
#    __import__("pandas")
#    importlib.import_module("numpy")
#    import createS3Bucket

    load_module("createS3Bucket.py", "the_module")

print('******************************')
modules = cap.imported_modules 
print('modules: {}'.format(modules))

packages = _flatten([_MODULES_TO_PACKAGES.get(module, []) for module in modules])
print('flatten: {}'.format(packages))

packages = map(_normalize_package_name, packages)
print('normalize: {}'.format(packages))

packages = _prune_packages(packages)
print('prune: {}'.format(packages))

excluded_packages = [
    # Certain packages (e.g. scikit-learn 0.24.2) imports `setuptools` or `pkg_resources`
    # (a module provided by `setuptools`) to process or interact with package metadata.
    # It should be safe to exclude `setuptools` because it's rare to encounter a python
    # environment where `setuptools` is not pre-installed.
    "setuptools",
    # Exclude a package that provides the mlflow module (e.g. mlflow, mlflow-skinny).
    # Certain flavors (e.g. pytorch) import mlflow while loading a model, but mlflow should
    # not be counted as a model requirement.
    *_MODULES_TO_PACKAGES.get("mlflow", []),
]
print('excluded_packages: {}'.format(excluded_packages))

packages = packages - set(excluded_packages)
print('minus excluded_packages: {}'.format(packages))

unrecognized_packages = packages - _PYPI_PACKAGE_INDEX.package_names
print('unrecognized_packages: {}'.format(unrecognized_packages))

if unrecognized_packages:
    print(
        "The following packages were not found in the public PyPI package index as of"
        " %s; if these packages are not present in the public PyPI index, you must install"
        " them manually before loading your model: %s",
        _PYPI_PACKAGE_INDEX.date,
        unrecognized_packages,
    )

print('******************************')
pinned_requirements = sorted(map(_get_pinned_requirement, packages))
print(pinned_requirements)

with open('./requirements.txt', 'w') as f:
    for pinned_requirement in pinned_requirements:
        f.write(pinned_requirement + '\n')
print('./requirements.txt has been updated.')

_mlflow_conda_env(path='./conda.yaml',
                  additional_pip_deps=pinned_requirements,
                  install_mlflow=False)
print('./conda.yaml has been updated.')
print('******************************')

#nb = nbformat.read('createS3Bucket.ipynb', as_version=4)
#print(nb)
