"""Profile loading and validation.

These tests exist because a malformed profile must fail loudly at load time. The
alternative is an empty graph three phases later with no obvious cause.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import pytest
import yaml
from pydantic import ValidationError

from ragtorio.config import (
    InfoboxConfig,
    Settings,
    TypeRule,
    WikiProfile,
    load_profile,
    profiles_dir,
)


def test_minimal_profile_validates(minimal_profile_data: dict[str, Any]) -> None:
    profile = WikiProfile.model_validate(minimal_profile_data)
    assert profile.wiki.id == "testwiki"
    assert profile.wiki.rate_limit_rps == 1.0


def test_crawl_namespaces_includes_extras_and_archive(
    minimal_profile_data: dict[str, Any],
) -> None:
    data = copy.deepcopy(minimal_profile_data)
    data["wiki"]["extra_namespaces"] = [3002, 0]
    data["wiki"]["archived"] = {"namespace_id": 3004}
    profile = WikiProfile.model_validate(data)
    assert profile.crawl_namespaces == [0, 3002, 3004]


@pytest.mark.parametrize(
    ("target", "ok"),
    [
        ("prop.stack_size", True),
        ("rel.CRAFTED_AT", True),
        ("recipe", True),
        ("prop.StackSize", False),
        ("rel.crafted_at", False),
        ("nonsense", False),
    ],
)
def test_field_mapping_target_grammar(
    minimal_profile_data: dict[str, Any], target: str, ok: bool
) -> None:
    data = copy.deepcopy(minimal_profile_data)
    data["infobox"]["fields"] = {"x": {"to": target}}
    if ok:
        assert WikiProfile.model_validate(data)
    else:
        with pytest.raises(ValidationError, match="field mapping target"):
            WikiProfile.model_validate(data)


def test_unknown_parser_is_rejected(minimal_profile_data: dict[str, Any]) -> None:
    data = copy.deepcopy(minimal_profile_data)
    data["infobox"]["fields"] = {"x": {"to": "recipe", "parser": "no_such_parser"}}
    with pytest.raises(ValidationError, match="unknown parser"):
        WikiProfile.model_validate(data)


def test_parser_and_type_are_mutually_exclusive(minimal_profile_data: dict[str, Any]) -> None:
    data = copy.deepcopy(minimal_profile_data)
    data["infobox"]["fields"] = {"x": {"to": "recipe", "parser": "plus_list", "type": "int"}}
    with pytest.raises(ValidationError, match="either 'parser' or 'type'"):
        WikiProfile.model_validate(data)


def test_unknown_label_is_rejected(minimal_profile_data: dict[str, Any]) -> None:
    data = copy.deepcopy(minimal_profile_data)
    data["infobox"]["type_map"] = {"widget": ["Gadget"]}
    with pytest.raises(ValidationError, match="unknown labels"):
        WikiProfile.model_validate(data)


def test_separate_namespace_requires_namespace_id(minimal_profile_data: dict[str, Any]) -> None:
    data = copy.deepcopy(minimal_profile_data)
    data["infobox"]["location"] = "separate_namespace"
    with pytest.raises(ValidationError, match="namespace_id is required"):
        WikiProfile.model_validate(data)


def test_type_field_requires_type_map(minimal_profile_data: dict[str, Any]) -> None:
    data = copy.deepcopy(minimal_profile_data)
    data["infobox"]["type_map"] = {}
    with pytest.raises(ValidationError, match="type_map must be non-empty"):
        WikiProfile.model_validate(data)


def test_profile_needs_some_way_to_classify(minimal_profile_data: dict[str, Any]) -> None:
    data = copy.deepcopy(minimal_profile_data)
    del data["infobox"]["type_field"]
    data["infobox"]["type_map"] = {}
    with pytest.raises(ValidationError, match="type_field \\+ type_map, or type_rules"):
        WikiProfile.model_validate(data)


def test_subpage_language_filter_requires_keep(minimal_profile_data: dict[str, Any]) -> None:
    data = copy.deepcopy(minimal_profile_data)
    data["wiki"]["language_filter"] = {"strategy": "subpage_suffix"}
    with pytest.raises(ValidationError, match="keep is required"):
        WikiProfile.model_validate(data)


def test_extra_keys_are_rejected(minimal_profile_data: dict[str, Any]) -> None:
    data = copy.deepcopy(minimal_profile_data)
    data["wiki"]["typoed_key"] = 1
    with pytest.raises(ValidationError):
        WikiProfile.model_validate(data)


class TestClassify:
    """``classify`` prefers the explicit type map and falls back to rules."""

    def _config(self) -> InfoboxConfig:
        return InfoboxConfig(
            location="inline",
            template="Infobox",
            type_field="kind",
            type_map={"widget": ["Item"], "scenery": []},
            type_rules=[
                TypeRule(when_present=["cost", "allows"], emit=["Unlock"]),
                TypeRule(when_present=["recipe"], emit=["Item"]),
            ],
        )

    def test_type_map_wins(self) -> None:
        assert self._config().classify("widget", {"recipe"}) == ["Item"]

    def test_empty_list_means_deliberately_ignored(self) -> None:
        assert self._config().classify("scenery", set()) == []

    def test_falls_back_to_rules_when_type_absent(self) -> None:
        assert self._config().classify(None, {"recipe"}) == ["Item"]

    def test_rules_are_ordered_first_match_wins(self) -> None:
        assert self._config().classify(None, {"cost", "allows", "recipe"}) == ["Unlock"]

    def test_rule_needs_every_named_param(self) -> None:
        assert self._config().classify(None, {"cost"}) is None

    def test_unclassifiable_returns_none(self) -> None:
        assert self._config().classify("unheard-of", set()) is None


def test_load_profile_missing_raises_with_available_list(tmp_path: Path) -> None:
    (tmp_path / "alpha.yaml").write_text("{}", encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="alpha"):
        load_profile("nope", directory=tmp_path)


def test_load_profile_roundtrip(tmp_path: Path, minimal_profile_data: dict[str, Any]) -> None:
    (tmp_path / "testwiki.yaml").write_text(yaml.safe_dump(minimal_profile_data), encoding="utf-8")
    assert load_profile("testwiki", directory=tmp_path).wiki.id == "testwiki"


def test_settings_user_agent_carries_contact() -> None:
    agent = Settings(contact_email="someone@example.org").user_agent
    assert "ragtorio" in agent
    assert "someone@example.org" in agent


class TestShippedFactorioProfile:
    """The committed profile must stay valid and keep its measured shape."""

    def test_it_loads(self) -> None:
        assert load_profile("factorio").wiki.id == "factorio"

    def test_profiles_dir_points_at_the_repo(self) -> None:
        assert (profiles_dir() / "factorio.yaml").is_file()

    def test_crawls_mainspace_infobox_and_archive(self) -> None:
        assert load_profile("factorio").crawl_namespaces == [0, 3002, 3004]

    def test_every_measured_type_is_mapped(self) -> None:
        # 96 distinct prototype-type values were measured across 553 pages.
        assert len(load_profile("factorio").infobox.type_map) == 96

    def test_technology_unlocks_come_from_effects_not_allows(self) -> None:
        fields = load_profile("factorio").infobox.fields
        assert fields["effects"].to == "rel.UNLOCKS"
        assert "allows" not in fields, "allows is the inverse of required-technologies"
