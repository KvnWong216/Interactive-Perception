"""Prevent XML ordering and reset poses from leaking a scene across splits."""

import importlib.util
from pathlib import Path

import pytest

pytest.importorskip("h5py")
path = Path(__file__).resolve().parents[1] / "scripts/finalize_stage1.py"
spec = importlib.util.spec_from_file_location("finalize_stage1", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_scene_group_ignores_reset_pose_asset_order_and_install_location():
    first = """<mujoco><asset><mesh file='/a/libero/assets/cup.obj'/>
        <mesh file='/a/libero/assets/table.obj'/></asset><worldbody>
        <body name='table' pos='0 0 0'/><body name='cup' pos='1 2 3'/>
        <camera name='front' pos='0 0 1'/></worldbody></mujoco>"""
    second = """<mujoco><asset><mesh file='/b/libero/assets/table.obj'/>
        <mesh file='/b/libero/assets/cup.obj'/></asset><worldbody>
        <body name='cup' pos='4 5 6'/><body name='table' pos='0 0 0'/>
        <camera name='front' pos='0 0 2'/></worldbody></mujoco>"""
    assert module.asset_family(first) == module.asset_family(second)
    different = second.replace("cup.obj", "bowl.obj").replace(
        "name='cup'", "name='bowl'"
    )
    assert module.asset_family(first) != module.asset_family(different)
