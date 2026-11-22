# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The tier table routes declared requirements to satisfying tiers, cheapest first."""

import pytest
from shellbox.machine import CommandSemantics, NetworkPolicy, PackageSpec
from shellbox.tiers import DeclaredSource, TaskRequirements, Tier, route_tiers

PINNED_IMAGE = "ghcr.io/example/task@sha256:" + "0" * 64
UNPINNED_IMAGE = "ghcr.io/example/task:latest"


def test_shell_simulator_task_with_no_source_routes_to_shellsim():
    requirements = TaskRequirements(
        command_semantics=CommandSemantics.SHELL_SIMULATOR,
        capabilities=("shell", "filesystem"),
    )
    assert route_tiers(requirements) == (Tier.SHELLSIM,)


def test_shell_simulator_task_with_packages_fails_up_to_local():
    """A shell_simulator task may run on a linux_process tier: its capabilities are a Linux subset."""
    requirements = TaskRequirements(
        command_semantics=CommandSemantics.SHELL_SIMULATOR,
        capabilities=("shell", "filesystem"),
        package_spec=PackageSpec(pypi_packages=("numpy==2.2.3",)),
    )
    assert route_tiers(requirements) == (Tier.SHELLSIM, Tier.LOCAL)


def test_shell_simulator_task_with_omitted_capabilities_keeps_todays_route():
    """An omitted capabilities field means "not stated": today's single tier, no fail-up."""
    requirements = TaskRequirements(
        command_semantics=CommandSemantics.SHELL_SIMULATOR,
        package_spec=PackageSpec(pypi_packages=("numpy==2.2.3",)),
    )
    assert route_tiers(requirements) == (Tier.SHELLSIM,)


def test_linux_process_task_never_runs_on_shellsim():
    requirements = TaskRequirements(
        command_semantics=CommandSemantics.LINUX_PROCESS,
        capabilities=("shell", "filesystem"),
        packages_lock="s3://bucket/task/uv.lock",
    )
    assert route_tiers(requirements) == (Tier.LOCAL,)


def test_linux_process_task_with_packages_routes_to_local():
    requirements = TaskRequirements(
        command_semantics=CommandSemantics.LINUX_PROCESS,
        capabilities=("shell", "filesystem"),
        package_spec=PackageSpec(pypi_packages=("numpy==2.2.3",)),
    )
    assert route_tiers(requirements) == (Tier.LOCAL,)


def test_pinned_image_with_declared_capabilities_routes_local_first_then_daytona():
    """A task that declares its capabilities tries the cheapest satisfying tier and fails up."""
    requirements = TaskRequirements(
        command_semantics=CommandSemantics.LINUX_PROCESS,
        capabilities=("shell", "filesystem"),
        docker_image=PINNED_IMAGE,
    )
    assert route_tiers(requirements) == (Tier.LOCAL, Tier.DAYTONA)


def test_image_with_omitted_capabilities_keeps_todays_image_tier():
    """Harbor-archive specs omit capabilities: the declared image keeps today's image tier."""
    requirements = TaskRequirements(command_semantics=CommandSemantics.LINUX_PROCESS, docker_image=PINNED_IMAGE)
    assert route_tiers(requirements) == (Tier.DAYTONA,)


def test_unpinned_image_skips_the_local_tier():
    requirements = TaskRequirements(
        command_semantics=CommandSemantics.LINUX_PROCESS,
        capabilities=("shell", "filesystem"),
        docker_image=UNPINNED_IMAGE,
    )
    assert route_tiers(requirements) == (Tier.DAYTONA,)


def test_native_task_without_a_declared_source_is_unsupported():
    """Native execution needs a pinned image or a package source; no tier satisfies it without one."""
    requirements = TaskRequirements(command_semantics=CommandSemantics.LINUX_PROCESS, capabilities=("shell",))
    assert route_tiers(requirements) == ()


def test_shell_simulator_task_beyond_simulator_capabilities_fails_up_to_linux_tiers():
    requirements = TaskRequirements(
        command_semantics=CommandSemantics.SHELL_SIMULATOR,
        capabilities=("shell", "filesystem", "process"),
        package_spec=PackageSpec(),
    )
    assert route_tiers(requirements) == (Tier.LOCAL,)


def test_network_need_skips_shellsim_for_both_semantics():
    for semantics in (CommandSemantics.SHELL_SIMULATOR, CommandSemantics.LINUX_PROCESS):
        requirements = TaskRequirements(
            command_semantics=semantics,
            capabilities=("shell", "filesystem", "network"),
            network=NetworkPolicy.ALLOW,
            package_spec=PackageSpec(pypi_packages=("httpx==0.28.0",)),
        )
        assert route_tiers(requirements) == (Tier.LOCAL,)


def test_network_capability_with_denied_networking_satisfies_no_tier():
    """Mirrors the backend checks: no tier runs a network-capable task with denied networking."""
    requirements = TaskRequirements(
        command_semantics=CommandSemantics.LINUX_PROCESS,
        capabilities=("shell", "filesystem", "network"),
        network=NetworkPolicy.DENY,
        docker_image=PINNED_IMAGE,
    )
    assert route_tiers(requirements) == ()


def test_capability_outside_every_tier_is_unsupported():
    """An agent-harness capability is not a shellbox capability: no tier satisfies it."""
    requirements = TaskRequirements(
        command_semantics=CommandSemantics.LINUX_PROCESS,
        capabilities=("shell", "nemotron-agent:swe_agent"),
        docker_image=PINNED_IMAGE,
    )
    assert route_tiers(requirements) == ()


def test_machine_less_requirements_have_no_route():
    with pytest.raises(ValueError, match="command semantics"):
        route_tiers(TaskRequirements())


def test_conflicting_sources_are_rejected():
    with pytest.raises(ValueError, match="at most one"):
        TaskRequirements(
            command_semantics=CommandSemantics.LINUX_PROCESS,
            docker_image=PINNED_IMAGE,
            packages_lock="s3://bucket/task/uv.lock",
        )


def test_declared_source_derivation():
    assert TaskRequirements().source is DeclaredSource.NONE
    assert TaskRequirements(package_spec=PackageSpec()).source is DeclaredSource.PACKAGE_SPEC
    assert TaskRequirements(packages_lock="s3://bucket/task/uv.lock").source is DeclaredSource.PACKAGES_LOCK
    assert TaskRequirements(docker_image=PINNED_IMAGE).source is DeclaredSource.IMAGE
