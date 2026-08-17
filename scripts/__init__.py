"""Repository-local experiment entry points.

The marker keeps Python from resolving ``import scripts`` to unrelated system
packages (notably ROS) when tests import helpers from these entry points.
"""
