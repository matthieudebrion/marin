# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The ordered task-machine tier table: declared task requirements to execution backends.

One table maps the requirements a task declares — command semantics, capabilities,
network policy, and its image or package source — to the execution tiers that
satisfy them, cheapest first: ShellSim (in-process), local bubblewrap, Daytona
(shared quota). Atlas ingest tests a task's oracle and negative controls on the
cheapest satisfying tier and fails up on control failure, recording the first tier
where both pass as the task's route; RL runs each task on its recorded route tier.
A ``shell_simulator`` task may run on a ``linux_process`` tier, because its
capabilities are a subset of Linux's; a ``linux_process`` task never runs on ShellSim.
A spec that omits its capabilities routes by its declared source alone: the single
tier that source requires, with the full Linux capability set assumed.
"""

from dataclasses import dataclass
from enum import StrEnum

from shellbox.image import is_digest_pinned_image
from shellbox.machine import (
    LINUX_PROCESS_CAPABILITIES,
    NETWORK_CAPABILITY,
    SHELL_SIMULATOR_CAPABILITIES,
    CommandSemantics,
    NetworkPolicy,
    PackageSpec,
    satisfies_semantics,
)


class DeclaredSource(StrEnum):
    """The software source a task declares, as routing sees it."""

    NONE = "none"
    """No image or package source declared."""
    PACKAGE_SPEC = "package_spec"
    """PyPI requirements resolved when the machine starts."""
    PACKAGES_LOCK = "packages_lock"
    """A resolved lockfile the selected backend materializes."""
    IMAGE = "image"
    """A registry image reference."""


class Tier(StrEnum):
    """Execution tiers, cheapest first.

    Each value matches the ``Backend`` that serves the tier, so a recorded route
    tier names the runtime backend directly.
    """

    SHELLSIM = "shellsim"
    """The in-process shell simulator."""
    LOCAL = "local"
    """Local bubblewrap sandboxes on this host."""
    DAYTONA = "daytona"
    """Daytona sandboxes on the shared provider quota."""


@dataclass(frozen=True)
class TaskRequirements:
    """The requirements a task declares, as the tier table reads them.

    Optional fields mirror the task spec: an unset field means "not stated", not
    "needs nothing". Docker build contexts must be resolved to pinned images, and
    machine-less tasks gated, before routing.
    """

    command_semantics: CommandSemantics | None = None
    capabilities: tuple[str, ...] = ()
    network: NetworkPolicy = NetworkPolicy.DENY
    docker_image: str | None = None
    packages_lock: str | None = None
    package_spec: PackageSpec | None = None

    def __post_init__(self) -> None:
        declared = sum(value is not None for value in (self.docker_image, self.packages_lock, self.package_spec))
        if declared > 1:
            raise ValueError("A task declares at most one of an image, a packages lock and a package spec")

    @property
    def source(self) -> DeclaredSource:
        """The declared software source."""
        if self.docker_image is not None:
            return DeclaredSource.IMAGE
        if self.packages_lock is not None:
            return DeclaredSource.PACKAGES_LOCK
        if self.package_spec is not None:
            return DeclaredSource.PACKAGE_SPEC
        return DeclaredSource.NONE


@dataclass(frozen=True)
class TierDefinition:
    """One tier of the table: the semantics, capabilities, sources and network policies it satisfies."""

    tier: Tier
    semantics: CommandSemantics
    capabilities: frozenset[str]
    sources: frozenset[DeclaredSource]
    network: frozenset[NetworkPolicy]
    digest_pinned_images: bool = False
    """Whether this tier accepts image sources only when digest-pinned."""


TIER_TABLE: tuple[TierDefinition, ...] = (
    TierDefinition(
        tier=Tier.SHELLSIM,
        semantics=CommandSemantics.SHELL_SIMULATOR,
        capabilities=SHELL_SIMULATOR_CAPABILITIES,
        sources=frozenset({DeclaredSource.NONE, DeclaredSource.PACKAGE_SPEC}),
        network=frozenset({NetworkPolicy.DENY}),
    ),
    TierDefinition(
        tier=Tier.LOCAL,
        semantics=CommandSemantics.LINUX_PROCESS,
        capabilities=LINUX_PROCESS_CAPABILITIES,
        sources=frozenset({DeclaredSource.PACKAGE_SPEC, DeclaredSource.PACKAGES_LOCK, DeclaredSource.IMAGE}),
        network=frozenset({NetworkPolicy.DENY, NetworkPolicy.ALLOW}),
        digest_pinned_images=True,
    ),
    TierDefinition(
        tier=Tier.DAYTONA,
        semantics=CommandSemantics.LINUX_PROCESS,
        capabilities=LINUX_PROCESS_CAPABILITIES,
        sources=frozenset({DeclaredSource.IMAGE}),
        network=frozenset({NetworkPolicy.DENY, NetworkPolicy.ALLOW}),
    ),
)
"""The ordered tier table, cheapest first: ShellSim, local bubblewrap, Daytona."""


def _declared_source_route(requirements: TaskRequirements) -> tuple[Tier, ...]:
    """The declared-source route: the single tier the spec's declared source requires.

    A spec that omits its capabilities routes by its declared source alone, with the
    full Linux capability set assumed — the route every published cohort took before
    the tier table existed. Fail-up needs a task to declare what it actually
    requires. A task with no declared source runs on the simulator's default
    packages, or is unsupported.
    """
    semantics = requirements.command_semantics
    assert semantics is not None
    source = requirements.source
    if source is DeclaredSource.IMAGE:
        return (Tier.DAYTONA,)
    if source is DeclaredSource.PACKAGES_LOCK:
        return (Tier.LOCAL,)
    if source is DeclaredSource.PACKAGE_SPEC:
        simulator = semantics is CommandSemantics.SHELL_SIMULATOR
        return (Tier.SHELLSIM,) if simulator else (Tier.LOCAL,)
    return (Tier.SHELLSIM,) if semantics is CommandSemantics.SHELL_SIMULATOR else ()


def satisfies(definition: TierDefinition, requirements: TaskRequirements) -> bool:
    """Whether one tier satisfies the declared requirements.

    Mirrors what the tier's backend check rejects, without acquiring resources:
    a refusal from a backend check is "tier does not satisfy", never an error.
    """
    if requirements.command_semantics is None:
        raise ValueError("A task without declared command semantics has no machine to route")
    if not satisfies_semantics(definition.semantics, requirements.command_semantics):
        return False
    if not set(requirements.capabilities) <= definition.capabilities:
        return False
    if NETWORK_CAPABILITY in requirements.capabilities and requirements.network is NetworkPolicy.DENY:
        # No tier runs a network-capable task with denied networking, matching every backend check.
        return False
    if requirements.network not in definition.network:
        return False
    source = requirements.source
    if source not in definition.sources:
        return False
    if source is DeclaredSource.IMAGE and definition.digest_pinned_images:
        return is_digest_pinned_image(requirements.docker_image)
    return True


def route_tiers(requirements: TaskRequirements) -> tuple[Tier, ...]:
    """The ordered tiers that satisfy the declared requirements, cheapest first.

    Empty means no tier in the table satisfies the requirements: an unsupported
    task, not an error. A spec that omits its capabilities routes by its declared
    source alone — the single tier that source requires, with the full Linux
    capability set assumed; no cheaper tier is tried. Callers gate machine-less
    tasks and resolve Docker build contexts before routing.
    """
    if requirements.command_semantics is None:
        raise ValueError("A task without declared command semantics has no machine to route")
    if not requirements.capabilities:
        return _declared_source_route(requirements)
    return tuple(definition.tier for definition in TIER_TABLE if satisfies(definition, requirements))
