# Owner(s): ["module: inductor"]

"""End-to-end coverage for opt-in translated staged-reduction fusion."""

from __future__ import annotations

import dataclasses
from contextlib import nullcontext

import torch
import torch._inductor.config as inductor_config
from torch._inductor import metrics
from torch._inductor.choices import InductorChoices
from torch._inductor.scheduler import (
    FusedNestedReductions,
    FusedStagedReduction,
    NestedReduction,
)
from torch._inductor.test_case import TestCase, run_tests
from torch._inductor.utils import fresh_inductor_cache
from torch._inductor.virtualized import V
from torch.testing._internal.inductor_utils import GPU_TYPE, HAS_GPU


HEAD_DIM = 192
QK_ROPE_A = 64


def _shifted_mla_indexer(x, ln_w, ln_b, cos, sin, rope_width):
    mean = x.mean(-1, keepdim=True)
    var = ((x - mean) ** 2).mean(-1, keepdim=True)
    normed = (x - mean) / torch.sqrt(var + 1e-5) * ln_w + ln_b
    k_rot, k_pass = torch.split(
        normed.unsqueeze(2),
        [rope_width, x.shape[-1] - rope_width],
        dim=-1,
    )
    return k_rot * cos + k_rot * sin, k_pass


def shifted_mla_indexer(x, ln_w, ln_b, cos, sin):
    return _shifted_mla_indexer(x, ln_w, ln_b, cos, sin, QK_ROPE_A)


def equal_split_mla_indexer(x, ln_w, ln_b):
    mean = x.mean(-1, keepdim=True)
    var = ((x - mean) ** 2).mean(-1, keepdim=True)
    normed = (x - mean) / torch.sqrt(var + 1e-5) * ln_w + ln_b
    return torch.split(normed.unsqueeze(2), [HEAD_DIM // 2, HEAD_DIM // 2], dim=-1)


def _make_mla_inputs(
    *, batch_size: int, seq_len: int, head_dim: int = HEAD_DIM, seed: int = 0
):
    torch.manual_seed(seed)
    dtype = torch.bfloat16
    return (
        torch.randn(batch_size, seq_len, head_dim, device=GPU_TYPE, dtype=dtype),
        torch.randn(head_dim, device=GPU_TYPE, dtype=dtype),
        torch.randn(head_dim, device=GPU_TYPE, dtype=dtype),
        torch.randn(
            batch_size, seq_len, 1, QK_ROPE_A, device=GPU_TYPE, dtype=dtype
        ),
        torch.randn(
            batch_size, seq_len, 1, QK_ROPE_A, device=GPU_TYPE, dtype=dtype
        ),
    )


@dataclasses.dataclass(frozen=True)
class _Observation:
    outputs: tuple[torch.Tensor, ...]
    generated_kernel_count: int
    translated_codegen_count: int
    staged_fusion_count: int
    translations: tuple[tuple[object, ...], ...]


def _capture_staged_plans(nodes, staged_plans):
    for node in nodes:
        if not isinstance(node, FusedStagedReduction) or isinstance(
            node, FusedNestedReductions
        ):
            continue
        reductions = [
            candidate for candidate in node.get_nodes() if candidate.is_reduction()
        ]
        if not reductions:
            continue
        _, (parent_numel, parent_rnumel) = reductions[0].group
        plan = NestedReduction.sub_parent_epilogue_plan(
            node.get_nodes(), parent_numel, parent_rnumel
        )
        if plan is not None:
            staged_plans.append(plan)


def _choices_context(force_persistent: bool | None):
    if force_persistent is None:
        return nullcontext()

    class _Choices(InductorChoices):
        @staticmethod
        def should_use_cooperative_reduction(*args, **kwargs):
            return False

        @staticmethod
        def should_use_persistent_reduction(*args, **kwargs):
            return force_persistent

    return V.set_choices_handler(_Choices())


def _observe(
    fn,
    inputs: tuple[torch.Tensor, ...],
    *,
    polyhedral_fusion: bool,
    force_persistent: bool | None = None,
    nested_reduction: bool = True,
) -> _Observation:
    torch._dynamo.reset()
    metrics.reset()
    staged_plans = []

    def capture(nodes):
        _capture_staged_plans(nodes, staged_plans)
        return nodes

    with (
        inductor_config.patch(
            polyhedral_fusion=polyhedral_fusion,
            _post_fusion_custom_pass=capture,
            fx_graph_cache=False,
        ),
        inductor_config.patch("triton.nested_reduction", nested_reduction),
        fresh_inductor_cache(),
        _choices_context(force_persistent),
    ):
        compiled = torch.compile(fn, fullgraph=True)
        outputs = compiled(*inputs)

    if not isinstance(outputs, tuple):
        raise AssertionError("translated MLA fixture must return a tuple")
    translations = tuple(
        relation.translation
        for plan in staged_plans
        for stage in plan.sub_parent_stages
        for relation in stage.access_relations
    )
    return _Observation(
        outputs=tuple(outputs),
        generated_kernel_count=metrics.generated_kernel_count,
        translated_codegen_count=metrics.codegen_translated_staged_reduction,
        staged_fusion_count=len(staged_plans),
        translations=translations,
    )


def _assert_outputs_match(expected, actual) -> None:
    if len(expected) != len(actual):
        raise AssertionError(f"expected {len(expected)} outputs, got {len(actual)}")
    for expected_output, actual_output in zip(expected, actual):
        torch.testing.assert_close(
            actual_output, expected_output, atol=4e-2, rtol=2e-2
        )


class PolyhedralMLAFusionTest(TestCase):
    __unittest_skip__ = not HAS_GPU

    def assert_translated_plan(self, observation: _Observation) -> None:
        self.assertGreaterEqual(observation.translated_codegen_count, 1)
        self.assertGreaterEqual(observation.staged_fusion_count, 1)
        self.assertEqual(set(observation.translations), {(0, 0), (0, QK_ROPE_A)})

    def test_static_shape_matrix(self):
        for batch_size, seq_len in ((2, 8),):
            inputs = _make_mla_inputs(batch_size=batch_size, seq_len=seq_len)
            eager = tuple(shifted_mla_indexer(*inputs))
            disabled = _observe(
                shifted_mla_indexer,
                inputs,
                polyhedral_fusion=False,
            )
            enabled = _observe(
                shifted_mla_indexer,
                inputs,
                polyhedral_fusion=True,
            )
            _assert_outputs_match(eager, disabled.outputs)
            _assert_outputs_match(eager, enabled.outputs)
            self.assertEqual(disabled.translated_codegen_count, 0)
            self.assertEqual(disabled.staged_fusion_count, 0)
            self.assert_translated_plan(enabled)

    def test_looped_declines_and_persistent_fuses(self):
        inputs = _make_mla_inputs(batch_size=2, seq_len=8)
        eager = tuple(shifted_mla_indexer(*inputs))
        for force_persistent in (False, True):
            disabled = _observe(
                shifted_mla_indexer,
                inputs,
                polyhedral_fusion=False,
                force_persistent=force_persistent,
            )
            enabled = _observe(
                shifted_mla_indexer,
                inputs,
                polyhedral_fusion=True,
                force_persistent=force_persistent,
            )
            _assert_outputs_match(eager, disabled.outputs)
            _assert_outputs_match(eager, enabled.outputs)
            self.assertEqual(disabled.translated_codegen_count, 0)
            if force_persistent:
                self.assert_translated_plan(enabled)
            else:
                self.assertEqual(enabled.translated_codegen_count, 0)
                self.assertEqual(enabled.staged_fusion_count, 0)
                self.assertEqual(enabled.translations, ())

    def test_nested_reduction_gate(self):
        inputs = _make_mla_inputs(batch_size=2, seq_len=8)
        eager = tuple(shifted_mla_indexer(*inputs))
        observation = _observe(
            shifted_mla_indexer,
            inputs,
            polyhedral_fusion=True,
            nested_reduction=False,
        )
        _assert_outputs_match(eager, observation.outputs)
        self.assertEqual(observation.translated_codegen_count, 0)
        self.assertEqual(observation.staged_fusion_count, 0)
        self.assertEqual(observation.translations, ())

    def test_legal_but_unsupported_split_declines(self):
        inputs = _make_mla_inputs(batch_size=2, seq_len=8)[:3]
        eager = tuple(equal_split_mla_indexer(*inputs))
        disabled = _observe(
            equal_split_mla_indexer,
            inputs,
            polyhedral_fusion=False,
        )
        enabled = _observe(
            equal_split_mla_indexer,
            inputs,
            polyhedral_fusion=True,
        )
        _assert_outputs_match(eager, disabled.outputs)
        _assert_outputs_match(eager, enabled.outputs)
        self.assertEqual(disabled.translated_codegen_count, 0)
        self.assertEqual(enabled.translated_codegen_count, 0)
        self.assertEqual(disabled.staged_fusion_count, 0)
        self.assertEqual(enabled.staged_fusion_count, 0)
        self.assertEqual(enabled.translations, ())


if __name__ == "__main__":
    run_tests()
