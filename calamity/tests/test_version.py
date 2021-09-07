# -*- coding: utf-8 -*-
# Copyright 2019 the HERA Project
# Licensed under the MIT License

"""Tests for version.py."""

import pytest
import sys
import os
import io
import subprocess
import json

from .. import version, __version__
from ..data import DATA_PATH


def test_get_gitinfo_file(tmpdir):
    calamity_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    git_file = os.path.join(calamity_dir, "GIT_INFO")
    if not os.path.exists(git_file):
        # write a file to read in
        tmp_path = tmpdir.strpath
        temp_git_file = os.path.join(tmp_path, "GIT_INFO")
        version_info = version.construct_version_info()
        data = [
            version_info["git_origin"],
            version_info["git_origin"],
            version_info["git_origin"],
            version_info["git_origin"],
        ]
        with open(temp_git_file, "w") as outfile:
            json.dump(data, outfile)
        git_file = temp_git_file

    with open(git_file) as data_file:
        data = [x for x in json.loads(data_file.read().strip())]
        git_origin = data[0]
        git_hash = data[1]
        git_description = data[2]
        git_branch = data[3]

    test_file_info = {
        "git_origin": git_origin,
        "git_hash": git_hash,
        "git_description": git_description,
        "git_branch": git_branch,
    }

    if "temp_git_file" in locals():
        file_info = version._get_gitinfo_file(git_file=temp_git_file)
        os.remove(temp_git_file)
    else:
        file_info = version._get_gitinfo_file()

    assert file_info == test_file_info


def test_history_string():
    hs = version.history_string()
    assert "function test_history_string() in test_version.py" in hs
    version_info = version.construct_version_info()
    for k, v in version_info.items():
        assert k in hs
        assert v in hs
    hs = version.history_string("stuff")
    assert "stuff" in hs
    assert "Notes" in hs


def test_main():
    version_info = version.construct_version_info()

    saved_stdout = sys.stdout
    try:
        out = io.StringIO()
        sys.stdout = out
        version.main()
        output = out.getvalue()
        assert output == (
            "Version = {v}\ngit origin = {o}\n"
            "git branch = {b}\ngit description = {d}\n".format(
                v=version_info["version"],
                o=version_info["git_origin"],
                b=version_info["git_branch"],
                d=version_info["git_description"],
            )
        )
    finally:
        sys.stdout = saved_stdout
