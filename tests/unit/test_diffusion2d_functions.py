"""
Tests for functions in class SolveDiffusion2D
"""

from diffusion2d import SolveDiffusion2D
import pytest


def test_initialize_domain():
    """
    Check function SolveDiffusion2D.initialize_domain
    """
    # Fixture
    w = 100.
    h = 1000.
    dx = 0.01
    dy = 0.02

    # Expected results
    expected_nx = 10000
    expected_ny = 50000

    # Actual results
    solver = SolveDiffusion2D()
    solver.initialize_domain(w, h, dx, dy)
    actual_nx = pytest.approx(solver.nx, abs=1e-12)
    actual_ny = pytest.approx(solver.ny, abs=1e-12)

    # Check
    assert actual_nx == expected_nx and actual_ny == expected_ny, "nx and ny are not correct"

def test_initialize_physical_parameters():
    """
    Checks function SolveDiffusion2D.initialize_physical_parameters
    """
    # Fixture
    d = 10.
    T_cold = 100.
    T_hot = 1000.

    # Expected results
    expected_dt = 0.00025

    # Actual results
    solver = SolveDiffusion2D()
    solver.initialize_domain()
    solver.initialize_physical_parameters(d, T_cold, T_hot)
    actual_dt = pytest.approx(solver.dt, abs=1e-12)
    actual_t_hot = solver.T_hot
    actual_t_cold = solver.T_cold

    # Check
    assert expected_dt == actual_dt, "dt is not correct"
    assert actual_t_cold == T_cold, "T_cold is not correct"
    assert actual_t_hot == T_hot, "T_hot is not correct"



def test_set_initial_condition():
    """
    Checks function SolveDiffusion2D.set_initial_function
    """
    # Actual results
    solver = SolveDiffusion2D()
    solver.initialize_domain()
    solver.initialize_physical_parameters()
    actual_u = solver.set_initial_condition()

    # Check
    assert actual_u.shape == (100, 100), "u does not have the correct shape"