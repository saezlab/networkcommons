import pytest
from networkcommons.visual._aux import adjust_node_name


def test_replace_colon():
    assert adjust_node_name(
        "node:name"
    ) == "node_name"


def test_remove_provided_strings():
    # Remove 'COMPLEX' and 'ABC'
    assert adjust_node_name(
        "COMPLEX:ABC:node",
        remove_strings=["COMPLEX", "ABC"]
    ) == "node"


def test_special_character_replacement():
    # Replace special characters with '_'
    assert adjust_node_name(
        "node@name#1!"
    ) == "node_name_1"


def test_truncate_node_name():
    # Truncate node name to max_length
    assert adjust_node_name(
        "VeryLongNodeName",
        truncate=True,
        max_length=8
    ) == "VeryLong.."


def test_wrap_node_name():
    # Wrap node name into lines of wrap_length
    assert adjust_node_name(
        "VeryLongNodeName",
        wrap=True,
        wrap_length=4
    ) == "Very\nLong\nNode\nName"


def test_no_truncate_or_wrap():
    # Test without truncation or wrapping
    assert adjust_node_name(
        "node_name",
        truncate=False,
        wrap=False
    ) == "node_name"


def test_empty_node_name():
    # Handle empty node name
    assert adjust_node_name(
        "",
        remove_strings=["COMPLEX"]
    ) == ""


def test_strip_whitespace():
    # Ensure leading and trailing spaces are stripped
    assert adjust_node_name(
        "  COMPLEX:node  ",
        remove_strings=["COMPLEX"]
    ) == "node"


def test_multiple_underscores():
    # Test reducing multiple underscores to a single one
    assert adjust_node_name(
        "node___name"
    ) == "node_name"


def test_unique_node_name():
    # Test ensuring uniqueness by appending a number
    assert adjust_node_name(
        "node_name",
        ensure_unique=True,
        ensure_unique_list=["node_name", "node_name_1"]
    ) == "node_name_2"


def test_ensure_unique_with_empty_list():
    # Test ensuring uniqueness when the list is empty
    assert adjust_node_name(
        "node_name",
        ensure_unique=True,
        ensure_unique_list=[]
    ) == "node_name"


def test_empty_name_after_modifications():
    # Test if name becomes empty after all modifications
    assert adjust_node_name(
        "COMPLEX:ABC",
        remove_strings=["COMPLEX", "ABC"]
    ) == ""


def test_trailing_underscores():
    # Test if trailing underscores are removed after modifications
    assert adjust_node_name(
        "node___",
        remove_strings=["___"]
    ) == "node"


def test_leading_trailing_underscores_trimmed():
    # Test trimming of leading and trailing underscores
    assert adjust_node_name(
        "___node___",
        remove_strings=[]
    ) == "node"
