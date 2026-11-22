# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Grade a source's control submissions through its configured grading boundary."""

import asyncio
import os
from dataclasses import dataclass, replace
from functools import partial
from typing import Any, Protocol

from shellbox.machine import Command, MachineFactory, MachineSpec, UnsupportedMachineSpec
from verifyit.grade import positive_candidate
from verifyit.modes.grade_judge import JudgeConnection
from verifyit.spec import ExactSpec, MathSpec, McqSpec, NumericSpec, StructureClosureSpec

from taskcompendium.chat import CONTROL_REPLY_STATUS
from taskcompendium.grading_result import Outcome
from taskcompendium.models import (
    CONVERSATION_ANSWERS,
    ConversationTrace,
    EnvironmentRequirements,
    GradingAttempt,
    MachineRoute,
    NoGrader,
    TaskSpec,
    TextMessage,
    VerifyitGrader,
    grades_in_process,
    grades_with_judge,
    require_resolved_environment,
    verifyit_spec,
)
from taskcompendium.pipeline.fingerprints import callable_identity
from taskcompendium.pipeline.models import (
    CheckResult,
    CheckStatus,
    CheckSuite,
    Controls,
    ControlSubmission,
    OracleCommand,
    Reply,
    TierRole,
    TierVerdict,
    VerificationReport,
    WorkspaceFiles,
)
from taskcompendium.pipeline.verification import answer_event, control_result
from taskcompendium.runtime.environment import prepare_machine_spec, setup_user, validate_machine_spec
from taskcompendium.runtime.grading import grade_empty_in_sandbox
from taskcompendium.runtime.oracle import (
    OracleAnswerError,
    oracle_answer_attempt,
    oracle_environment,
    read_oracle_answer,
    run_oracle_command,
)
from taskcompendium.runtime.shell import ShellEnvironment, upload_resources
from taskcompendium.runtime.task_grading import grade_task, sandbox_grade

CONTROLS_REVISION = "22"
ORACLE_TIMEOUT = 600.0
ORACLE_OUTPUT_LIMIT_BYTES = 1_048_576
FILE_SUBMISSION_MESSAGE = TextMessage(role="assistant", content="The submission is in the workspace.")
"""The final message of a file submission; the grader reads the captured files."""


class GradingMachines(Protocol):
    """Fresh grading machines for a campaign's verification backends."""

    def identity(self) -> dict[str, Any]:
        """Backends, worker image and other settings that can change a control's outcome."""
        ...

    def routes(self, environment: EnvironmentRequirements, memory_mb: int) -> tuple[MachineRoute, ...]:
        """The tier candidates for ``environment``, cheapest first; empty when no tier satisfies it."""
        ...


class OracleFailed(Exception):
    """The oracle command ran and did not produce a submission."""


def answer_reply(task: TaskSpec, answer: str) -> Reply:
    """A final assistant reply carrying ``answer`` in the task's answer format."""
    return Reply(answer_event(task, answer))


def reference_reply(task: TaskSpec) -> Reply | None:
    """The in-process grader's own reference answer as a reply, when its mode has one."""
    grader = task.grader
    if not isinstance(grader, VerifyitGrader):
        return None
    spec = verifyit_spec(grader)
    if isinstance(spec, MathSpec):
        return answer_reply(task, rf"\boxed{{{spec.expected}}}")
    if isinstance(spec, NumericSpec | McqSpec):
        return answer_reply(task, spec.expected)
    if isinstance(spec, StructureClosureSpec):
        return answer_reply(task, spec.expected[len(spec.prefix) :])
    if isinstance(spec, ExactSpec):
        candidate = positive_candidate(spec)
        return answer_reply(task, candidate) if candidate is not None else None
    return None


def controls_identity(controls: Controls) -> dict[str, Any]:
    """The control code and machine size that can change a control's outcome."""
    return {
        "revision": CONTROLS_REVISION,
        "golden": callable_identity(controls.golden) if controls.golden is not None else None,
        "negative": callable_identity(controls.negative) if controls.negative is not None else None,
        "negative_reward": callable_identity(controls.negative_reward) if controls.negative_reward is not None else None,
        "memory_mb": controls.memory_mb,
    }


def control_suite(
    controls: Controls,
    machines: GradingMachines | None,
    judge_connection: JudgeConnection | None = None,
) -> CheckSuite:
    """Check each sampled task with a declared positive or negative submission, falling back to empty."""
    return CheckSuite(
        id="controls",
        revision=CONTROLS_REVISION,
        parameters={
            **(machines.identity() if machines is not None else {}),
            **controls_identity(controls),
            **({"judge_base_url": judge_connection.base_url} if judge_connection is not None else {}),
        },
        run=partial(run_controls, controls=controls, machines=machines, judge_connection=judge_connection),
    )


def run_controls(
    task: TaskSpec,
    *,
    controls: Controls,
    machines: GradingMachines | None,
    judge_connection: JudgeConnection | None = None,
) -> VerificationReport:
    return asyncio.run(_control_checks(task, controls, machines, judge_connection))


def _empty_submission(task: TaskSpec) -> ControlSubmission:
    """The empty submission for an in-process grader, which scores it without a machine."""
    if task.answer_type in CONVERSATION_ANSWERS:
        return Reply(TextMessage(role="assistant", content=""))
    return WorkspaceFiles({})


async def _control_checks(
    task: TaskSpec,
    controls: Controls,
    machines: GradingMachines | None,
    judge_connection: JudgeConnection | None,
) -> VerificationReport:
    require_resolved_environment(task.environment_requirements)
    grader = task.grader
    if isinstance(grader, NoGrader):
        return VerificationReport(
            [
                CheckResult(
                    check="grader", status=CheckStatus.UNSUPPORTED, detail=f"A {grader.kind} grader has no controls"
                )
            ]
        )
    judge_task = grades_with_judge(task.grader)
    if judge_task and judge_connection is None:
        raise ValueError("Judge controls require an explicit remote connection")
    sandbox = None
    grader_environment = None
    if not grades_in_process(grader) and not (judge_task and isinstance(grader, VerifyitGrader)):
        grader_environment = grader.environment
        assert grader_environment is not None
        require_resolved_environment(grader_environment)
        if machines is None:
            raise ValueError("Sandbox controls require grading machines")
        sandbox = _Sandbox(
            machines,
            controls.memory_mb,
            grader_environment,
            _tier_candidates(machines, grader_environment, controls.memory_mb),
        )
    golden = controls.golden(task) if controls.golden is not None else None
    if golden is not None:
        check, verdicts, route_tier = await _control(task, "golden", golden, 1.0, sandbox, judge_connection)
    elif controls.negative is not None and (negative := controls.negative(task)) is not None:
        expected = controls.negative_reward(task) if controls.negative_reward is not None else 0.0
        check, verdicts, route_tier = await _control(task, "negative", negative, expected, sandbox, judge_connection)
    elif sandbox is not None:
        check, verdicts, route_tier = await _empty_sandbox_control(task, sandbox, judge_connection)
    else:
        check, verdicts, route_tier = await _control(task, "empty", _empty_submission(task), 0.0, None, judge_connection)
    return VerificationReport([check], verdicts=tuple(verdicts), route_tier=route_tier)


def _tier_candidates(
    machines: GradingMachines, environment: EnvironmentRequirements, memory_mb: int
) -> tuple[MachineRoute, ...]:
    """The ordered tier candidates for one environment, cheapest first.

    Raises:
        UnsupportedMachineSpec: No tier in the table satisfies the declared requirements.
    """
    require_resolved_environment(environment)
    routes = machines.routes(environment, memory_mb)
    if not routes:
        raise UnsupportedMachineSpec(
            f"No tier satisfies the declared requirements: {environment.model_dump_json(exclude_none=True)}"
        )
    return routes


def _verdict(role: TierRole, route: MachineRoute, status: CheckStatus, detail: str) -> TierVerdict:
    return TierVerdict(role=role, tier=route.tier.value, status=status, detail=detail)


def _exhausted(name: str, verdicts: list[TierVerdict]) -> CheckResult:
    """The final verdict after the ladder ends: the top tried tier's outcome."""
    last = verdicts[-1]
    return CheckResult(check=name, status=last.status, detail=last.detail)


@dataclass(frozen=True)
class _Sandbox:
    """Where a sandbox-graded task's controls run, with the verifier machine's tier candidates."""

    machines: GradingMachines
    memory_mb: int
    environment: EnvironmentRequirements
    """The grader environment every candidate serves."""
    grader_routes: tuple[MachineRoute, ...]


async def _oracle_submission(
    task: TaskSpec,
    command: OracleCommand,
    environment: EnvironmentRequirements,
    routes: tuple[MachineRoute, ...],
) -> tuple[GradingAttempt | None, list[TierVerdict], str | None]:
    """Run the oracle on the cheapest task tier that produces a submission, failing up on failure.

    An infrastructure error stops the ladder: it says nothing about a tier's
    suitability, and the suite retry reruns the whole ladder.
    """
    verdicts: list[TierVerdict] = []
    for route in routes:
        try:
            validate_machine_spec(environment, route.factory, route.spec)
            attempt = await _oracle_attempt(task, command, route.factory, route.spec, environment)
        except UnsupportedMachineSpec as error:
            verdicts.append(_verdict(TierRole.TASK, route, CheckStatus.UNSUPPORTED, str(error)))
            continue
        except OracleFailed as error:
            verdicts.append(_verdict(TierRole.TASK, route, CheckStatus.FAIL, str(error)))
            continue
        except (RuntimeError, OSError) as error:
            verdicts.append(_verdict(TierRole.TASK, route, CheckStatus.INFRA_ERROR, str(error)))
            return None, verdicts, None
        verdicts.append(_verdict(TierRole.TASK, route, CheckStatus.PASS, "The oracle produced a submission"))
        return attempt, verdicts, route.tier.value
    return None, verdicts, None


async def _grade_on_routes(
    task: TaskSpec,
    attempt: GradingAttempt,
    name: str,
    expected: float,
    environment: EnvironmentRequirements,
    routes: tuple[MachineRoute, ...],
    judge_connection: JudgeConnection | None,
) -> tuple[CheckResult, list[TierVerdict]]:
    """Grade one submission across the verifier tier ladder, failing up on control failure.

    A pass ends the ladder; an infrastructure error stops it for the suite retry.
    Every other rejecting outcome tries the next tier, and the top tier's outcome
    is the final verdict.
    """
    verdicts: list[TierVerdict] = []
    for route in routes:
        try:
            validate_machine_spec(environment, route.factory, route.spec)
        except UnsupportedMachineSpec as error:
            verdicts.append(_verdict(TierRole.VERIFIER, route, CheckStatus.UNSUPPORTED, str(error)))
            continue
        grade = await sandbox_grade(task, attempt, route.factory, route.spec, judge_connection=judge_connection)
        result = control_result(grade, name, expected)
        verdicts.append(_verdict(TierRole.VERIFIER, route, result.status, result.detail))
        if result.status is CheckStatus.PASS:
            return result, verdicts
        if result.status is CheckStatus.INFRA_ERROR:
            # An infrastructure error says nothing about the tier; the suite retry reruns the ladder.
            return result, verdicts
    return _exhausted(name, verdicts), verdicts


async def _control(
    task: TaskSpec,
    name: str,
    submission: ControlSubmission,
    expected: float,
    sandbox: _Sandbox | None,
    judge_connection: JudgeConnection | None,
) -> tuple[CheckResult, list[TierVerdict], str | None]:
    """Run one control across the tier ladders it needs.

    Returns the final check result, every tried tier's verdict in order, and the
    task route tier when the control passed through an oracle machine.
    """
    if isinstance(submission, OracleCommand):
        if sandbox is None:
            raise ValueError("An oracle command requires a sandbox grader")
        environment = oracle_environment(task, submission)
        task_routes = _tier_candidates(sandbox.machines, environment, sandbox.memory_mb)
        attempt, verdicts, task_tier = await _oracle_submission(task, submission, environment, task_routes)
        if attempt is None:
            return _exhausted(name, verdicts), verdicts, None
        result, verifier_verdicts = await _grade_on_routes(
            task, attempt, name, expected, sandbox.environment, sandbox.grader_routes, judge_connection
        )
        return result, [*verdicts, *verifier_verdicts], task_tier if result.status is CheckStatus.PASS else None
    if isinstance(submission, Reply):
        attempt = GradingAttempt(ConversationTrace(events=(*task.context.events, submission.event)))
    else:
        attempt = GradingAttempt(
            ConversationTrace(events=(*task.context.events, FILE_SUBMISSION_MESSAGE)), dict(submission.files)
        )
    if task.completion_protocol == "tool_free_single":
        attempt = replace(attempt, reply_status=CONTROL_REPLY_STATUS)
    if sandbox is None:
        grade = grade_task(task, attempt, judge_connection=judge_connection)
        if (
            grades_with_judge(task.grader)
            and grade.status in {Outcome.GRADED, Outcome.SUBMISSION_FAILURE}
            and "model" not in (grade.detail or {})
        ):
            return (
                CheckResult(
                    check=name,
                    status=CheckStatus.FAIL,
                    detail=f"Judge control was decided by a local gate without a remote verdict: {grade.detail}",
                ),
                [],
                None,
            )
        return control_result(grade, name, expected), [], None
    result, verdicts = await _grade_on_routes(
        task, attempt, name, expected, sandbox.environment, sandbox.grader_routes, judge_connection
    )
    return result, verdicts, None


async def _empty_sandbox_control(
    task: TaskSpec,
    sandbox: _Sandbox,
    judge_connection: JudgeConnection | None,
) -> tuple[CheckResult, list[TierVerdict], str | None]:
    """Run the grader on an empty answer across the verifier ladder; it must run cleanly on one tier.

    A zero reward or a rejected submission passes; a positive reward marks the task defective.
    """
    verdicts: list[TierVerdict] = []
    for route in sandbox.grader_routes:
        try:
            validate_machine_spec(sandbox.environment, route.factory, route.spec)
        except UnsupportedMachineSpec as error:
            verdicts.append(_verdict(TierRole.VERIFIER, route, CheckStatus.UNSUPPORTED, str(error)))
            continue
        try:
            grade = await grade_empty_in_sandbox(task, route.factory, route.spec, judge_connection=judge_connection)
        except (RuntimeError, OSError) as error:
            verdicts.append(_verdict(TierRole.VERIFIER, route, CheckStatus.INFRA_ERROR, str(error)))
            return CheckResult(check="empty", status=CheckStatus.INFRA_ERROR, detail=str(error)), verdicts, None
        result = control_result(grade, "empty", 0.0)
        verdicts.append(_verdict(TierRole.VERIFIER, route, result.status, result.detail))
        if result.status is CheckStatus.PASS:
            return result, verdicts, None
    return _exhausted("empty", verdicts), verdicts, None


async def _oracle_attempt(
    task: TaskSpec,
    command: OracleCommand,
    factory: MachineFactory,
    spec: MachineSpec,
    environment: EnvironmentRequirements,
) -> GradingAttempt:
    """Run the oracle in a fresh machine from ``factory`` with the worker and oracle files mounted."""
    async with asyncio.timeout(spec.startup_timeout):
        prepared = await asyncio.to_thread(
            prepare_machine_spec,
            environment,
            factory,
            spec,
            dict(os.environ),
        )
        machine = await factory.create(prepared)
    workspace = prepared.workdir
    try:
        await upload_resources(
            machine,
            (*task.resources.all, *task.resources.worker, *task.resources.oracle),
            ORACLE_TIMEOUT,
            user=setup_user(factory.backend),
        )
        commands = [Command(("mkdir", "-p", workspace), timeout=ORACLE_TIMEOUT)] if workspace else []
        commands.extend(
            Command(("sh", "-c", setup), cwd=workspace, timeout=ORACLE_TIMEOUT, user=setup_user(factory.backend))
            for setup in environment.setup_commands
        )
        for step in commands:
            result = await machine.run(step)
            if result.exit_code != 0:
                raise OracleFailed(
                    f"Oracle command exited {result.exit_code}: {result.stderr.decode(errors='replace')[-2000:]}"
                )
        result = await run_oracle_command(
            machine, command, cwd=workspace, timeout=ORACLE_TIMEOUT, output_limit_bytes=ORACLE_OUTPUT_LIMIT_BYTES
        )
        if result.exit_code != 0:
            raise OracleFailed(
                f"Oracle command exited {result.exit_code}: {result.stderr.decode(errors='replace')[-2000:]}"
            )
        if command.answer_file is None:
            evidence = await ShellEnvironment(
                machine, task.output_paths, ORACLE_TIMEOUT, ORACLE_OUTPUT_LIMIT_BYTES
            ).evidence()
            return GradingAttempt(
                ConversationTrace(events=(*task.context.events, FILE_SUBMISSION_MESSAGE)), evidence.files
            )
        try:
            answer = await read_oracle_answer(
                machine,
                command.answer_file,
                cwd=workspace,
                timeout=ORACLE_TIMEOUT,
                output_limit_bytes=ORACLE_OUTPUT_LIMIT_BYTES,
            )
        except OracleAnswerError as error:
            raise OracleFailed(str(error)) from error
        return oracle_answer_attempt(task, answer)
    finally:
        await machine.close()
