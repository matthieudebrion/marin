# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The tier table routes declared requirements to satisfying tiers, cheapest first."""

import pytest
from shellbox.machine import CommandSemantics, NetworkPolicy, PackageSpec
from shellbox.tiers import TaskRequirements, Tier, route_tiers

PINNED_IMAGE = "ghcr.io/example/task@sha256:" + "0" * 64
UNPINNED_IMAGE = "ghcr.io/example/task:latest"

SIMULATOR = CommandSemantics.SHELL_SIMULATOR
LINUX = CommandSemantics.LINUX_PROCESS


def _requirements(**fields) -> TaskRequirements:
    return TaskRequirements(**fields)


@pytest.mark.parametrize(
    "requirements, expected",
    [
        # A shell_simulator task with no declared source runs on the simulator's default packages.
        (
            _requirements(command_semantics=SIMULATOR, capabilities=("shell", "filesystem")),
            (Tier.SHELLSIM,),
        ),
        # A shell_simulator task may run on a linux_process tier: its capabilities are a Linux
        # subset, so the simulator task with packages fails up to local.
        (
            _requirements(
                command_semantics=SIMULATOR,
                capabilities=("shell", "filesystem"),
                package_spec=PackageSpec(pypi_packages=("numpy==2.2.3",)),
            ),
            (Tier.SHELLSIM, Tier.LOCAL),
        ),
        # An omitted capabilities field means "not stated": the spec routes by its declared
        # source alone, so the simulator task with packages binds the simulator only.
        (
            _requirements(command_semantics=SIMULATOR, package_spec=PackageSpec(pypi_packages=("numpy==2.2.3",))),
            (Tier.SHELLSIM,),
        ),
        # A linux_process task never runs on ShellSim.
        (
            _requirements(
                command_semantics=LINUX, capabilities=("shell", "filesystem"), packages_lock="s3://bucket/task/uv.lock"
            ),
            (Tier.LOCAL,),
        ),
        (
            _requirements(
                command_semantics=LINUX,
                capabilities=("shell", "filesystem"),
                package_spec=PackageSpec(pypi_packages=("numpy==2.2.3",)),
            ),
            (Tier.LOCAL,),
        ),
        # A task that declares its capabilities tries the cheapest satisfying tier and fails up.
        (
            _requirements(command_semantics=LINUX, capabilities=("shell", "filesystem"), docker_image=PINNED_IMAGE),
            (Tier.LOCAL, Tier.DAYTONA),
        ),
        # An omitted capabilities field routes by the declared source alone: the image tier.
        (
            _requirements(command_semantics=LINUX, docker_image=PINNED_IMAGE),
            (Tier.DAYTONA,),
        ),
        # The local tier accepts images only when digest-pinned.
        (
            _requirements(command_semantics=LINUX, capabilities=("shell", "filesystem"), docker_image=UNPINNED_IMAGE),
            (Tier.DAYTONA,),
        ),
        # Native execution needs a pinned image or a package source; no tier satisfies it without one.
        (
            _requirements(command_semantics=LINUX, capabilities=("shell",)),
            (),
        ),
        # A shell_simulator task beyond the simulator's capabilities fails up to the Linux tiers.
        (
            _requirements(
                command_semantics=SIMULATOR,
                capabilities=("shell", "filesystem", "process"),
                package_spec=PackageSpec(),
            ),
            (Tier.LOCAL,),
        ),
        # A task that needs the network skips ShellSim for both semantics.
        (
            _requirements(
                command_semantics=SIMULATOR,
                capabilities=("shell", "filesystem", "network"),
                network=NetworkPolicy.ALLOW,
                package_spec=PackageSpec(pypi_packages=("httpx==0.28.0",)),
            ),
            (Tier.LOCAL,),
        ),
        (
            _requirements(
                command_semantics=LINUX,
                capabilities=("shell", "filesystem", "network"),
                network=NetworkPolicy.ALLOW,
                package_spec=PackageSpec(pypi_packages=("httpx==0.28.0",)),
            ),
            (Tier.LOCAL,),
        ),
        # No tier runs a network-capable task with denied networking, matching every backend check.
        (
            _requirements(
                command_semantics=LINUX,
                capabilities=("shell", "filesystem", "network"),
                network=NetworkPolicy.DENY,
                docker_image=PINNED_IMAGE,
            ),
            (),
        ),
        # An agent-harness capability is not a shellbox capability: no tier satisfies it.
        (
            _requirements(
                command_semantics=LINUX,
                capabilities=("shell", "nemotron-agent:swe_agent"),
                docker_image=PINNED_IMAGE,
            ),
            (),
        ),
    ],
)
def test_declared_requirements_route_to_the_satisfying_tiers(requirements: TaskRequirements, expected: tuple):
    assert route_tiers(requirements) == expected
