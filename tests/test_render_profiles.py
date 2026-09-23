"""Renderer intervention must not change physics, cameras, or native defaults."""

import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from libero_probe_server import render_profile_xml


def test_native_is_unchanged_and_unknown_rejected():
    xml = '<mujoco><worldbody/><visual><quality offsamples="4"/></visual></mujoco>'
    assert render_profile_xml(xml, "native") == xml
    with pytest.raises(ValueError):
        render_profile_xml(xml, "typo")


def test_no_msaa_changes_only_sample_count():
    xml = '<mujoco><option timestep="0.002"/><visual><quality offsamples="4" shadowsize="4096"/></visual><worldbody><camera name="wrist" pos="1 2 3"/><body name="target"/></worldbody></mujoco>'
    expected = ET.fromstring(xml)
    expected.find("visual/quality").set("offsamples", "0")
    assert render_profile_xml(xml, "no_msaa") == ET.tostring(
        expected, encoding="unicode"
    )
    assert render_profile_xml(
        render_profile_xml(xml, "no_msaa"), "no_msaa"
    ) == render_profile_xml(xml, "no_msaa")


def test_no_msaa_adds_missing_visual_configuration():
    root = ET.fromstring(render_profile_xml("<mujoco><worldbody/></mujoco>", "no_msaa"))
    assert root.find("visual/quality").attrib == {"offsamples": "0"}
    assert root.find("worldbody") is not None
