# tests directory-specific settings - this file is run automatically by pytest before any tests are run

import sys
import os
import pytest
from os.path import abspath, dirname, join
import torch
import warnings

# allow having multiple repository checkouts and not needing to remember to rerun
# 'pip install -e .[dev]' when switching between checkouts and running tests.
git_repo_path = abspath(join(dirname(dirname(__file__)), "src"))
sys.path.insert(1, git_repo_path)


def pytest_addoption(parser):
    parser.addoption("--torch_ver", default=None, type=str)
    parser.addoption("--cuda_ver", default=None, type=str)
    parser.addoption(
        "--use_hpu",
        type=bool,
        default=False,
        help="Use HPU backend",
    )


def validate_version(expected, found):
    version_depth = expected.count('.') + 1
    found = '.'.join(found.split('.')[:version_depth])
    return found == expected


@pytest.fixture(scope="session", autouse=True)
def check_environment(pytestconfig):
    expected_torch_version = pytestconfig.getoption("torch_ver")
    expected_cuda_version = pytestconfig.getoption("cuda_ver")
    if expected_torch_version is None:
        warnings.warn(
            "Running test without verifying torch version, please provide an expected torch version with --torch_ver"
        )
    elif not validate_version(expected_torch_version, torch.__version__):
        pytest.exit(
            f"expected torch version {expected_torch_version} did not match found torch version {torch.__version__}",
            returncode=2)
    if expected_cuda_version is None:
        warnings.warn(
            "Running test without verifying cuda version, please provide an expected cuda version with --cuda_ver"
        )
    elif not validate_version(expected_cuda_version, torch.version.cuda):
        pytest.exit(
            f"expected cuda version {expected_cuda_version} did not match found cuda version {torch.version.cuda}",
            returncode=2)

def update_wa_env_var(key, value):
    if key not in os.environ.keys():
        os.environ[key] = value

def pytest_configure(config):
    pytest.use_hpu = config.getoption("--use_hpu")
    if (pytest.use_hpu):
        os.environ['DEEPSPEED_USE_HPU'] = "true"
        # TODO: SW-113485 need to remove the below WA once SW-113485 is unblocked
        update_wa_env_var("PT_HPU_LAZY_ACC_PAR_MODE", "0")
