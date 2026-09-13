import moescope


def test_package_exposes_version() -> None:
    assert moescope.__version__ == "0.1.0"
