"""M0 smoke tests: package imports, version metadata resolves, CLI stub runs."""

from importlib.metadata import version

import mimir
from mimir.cli import main


def test_version_matches_distribution_metadata():
    assert mimir.__version__ == version("mimisbrunnr")


def test_cli_main_prints_name_and_version(capsys):
    # main([]) not main(): with argv=None the CLI reads sys.argv, which under
    # pytest holds pytest's own arguments. The behavior contract is unchanged.
    assert main([]) == 0
    assert capsys.readouterr().out.strip() == f"mimir {mimir.__version__}"
