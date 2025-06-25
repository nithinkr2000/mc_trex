import pytest
import MDAnalysis
from pathlib import Path

@pytest.fixture(scope="session")
def test_universe():
    """
    Loads an MDAnalysis Universe from example1.prmtop and example1.nc.
    Assumes files are in docs/examples/ relative to the project root.
    """
    project_root = Path(__file__).parent.parent
    prmtop_path = project_root / "docs" / "examples" / "example1.prmtop"
    nc_path = project_root / "docs" / "examples" / "example1.nc"

    if not prmtop_path.exists():
        pytest.fail(f"Test topology not found: {prmtop_path}")
    if not nc_path.exists():
        pytest.fail(f"Test trajectory not found: {nc_path}")

    try:
        universe = MDAnalysis.Universe(str(prmtop_path), str(nc_path))
        return universe
    except Exception as e:
        pytest.fail(f"Failed to load MDAnalysis Universe: {e}")
