# Copyright 2026 The RLinf Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     https://www.apache.org/licenses/LICENSE-2.0

"""StarVLA rollout/training contracts with a small external VLM fixture."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
from starVLA.model.framework.share_tools import add_discretized_state_to_instruction

from rlinf.algorithms.advantages import compute_grpo_advantages
from rlinf.algorithms.losses import compute_grpo_actor_loss_fn
from rlinf.models.embodiment.starvla.starvla_action_model import (
    StarVLAForRLActionPrediction,
)


class TinyVLM(torch.nn.Module):
    """Deterministic context-sensitive replacement for the external Qwen model."""

    def __init__(self):
        super().__init__()
        self.model = torch.nn.Linear(1, 1)
        self.model.config = SimpleNamespace(hidden_size=4, pad_token_id=0)
        self.prompts = []
        self.forward_shapes = []

    def build_qwenvl_inputs(self, images, instructions):
        self.prompts = instructions
        self.image_sizes = [[image.size for image in views] for views in images]
        tokens = [torch.tensor([ord(c) for c in text]) for text in instructions]
        ids = torch.nn.utils.rnn.pad_sequence(tokens, batch_first=True)
        return {"input_ids": ids, "attention_mask": ids.ne(0).long()}

    def forward(self, input_ids, attention_mask, **kwargs):
        self.forward_shapes.append(tuple(input_ids.shape))
        assert set(kwargs) <= {
            "use_cache",
            "output_attentions",
            "output_hidden_states",
            "return_dict",
        }
        context = (input_ids.float() * attention_mask).sum(-1) / 1000000
        hidden = context[:, None, None].expand(-1, input_ids.shape[1], 4)
        return SimpleNamespace(hidden_states=(hidden,))


class TinyHead(torch.nn.Linear):
    def predict_action(self, hidden):
        return self(hidden)


class TinyOFT(torch.nn.Module):
    """Use the actual SFT state tokenizer with a tiny differentiable action head."""

    add_discretized_state_to_instruction = staticmethod(
        add_discretized_state_to_instruction
    )

    def __init__(self):
        super().__init__()
        self.framework_name = "QwenOFT"
        self.config = OmegaConf.create(
            {
                "framework": {
                    "name": "QwenOFT",
                    "action_model": {
                        "state_dim": 35,
                        "action_horizon": 30,
                        "past_action_window_size": 0,
                        "future_action_window_size": 8,
                    },
                },
                "datasets": {"vla_data": {"obs_image_size": [8, 8]}},
            }
        )
        self.qwen_vl_interface = TinyVLM()
        self.action_model = TinyHead(4, 3)
        self.action_token = "🔍"
        self.action_token_id = ord(self.action_token)
        self.chunk_len = 30
        self.norm_stats = {
            "pipette": {"action": {"q01": [-0.01] * 3, "q99": [0.01] * 3}}
        }

    def _gather_action_token_embeddings(self, hidden, input_ids, action_token_id):
        return hidden[input_ids == action_token_id].reshape(len(input_ids), 30, 4)


def make_policy(with_state=True, require_single_sample=False):
    return StarVLAForRLActionPrediction(
        TinyOFT(),
        3,
        30,
        add_value_head=False,
        unnorm_key="pipette",
        action_stats_source="q01q99",
        enable_state_input=with_state,
        state_normalizer=(lambda raw: raw / 2) if with_state else None,
        action_denormalizer=lambda raw: raw * 0.01,
        policy_setup="pipette",
        require_single_sample=require_single_sample,
    ).eval()


def observation(value=0):
    return {
        "main_images": np.zeros((1, 8, 8, 3), np.uint8),
        "extra_view_images": np.zeros((1, 2, 8, 8, 3), np.uint8),
        "task_descriptions": ["Attach a green pipette tip."],
        "states": np.full((1, 35), value, np.float32),
    }


def test_oft_state_reaches_sft_prompt_and_changes_prediction():
    policy = make_policy()
    actions, cached = policy.predict_action_batch(
        observation(), mode="eval", calculate_values=False
    )
    prompt = policy.starvla_model.qwen_vl_interface.prompts[0]
    assert "[STATE] " + " ".join(["128"] * 35) + " [ACTION]" in prompt
    assert actions.shape == (1, 30, 3)  # action_horizon overrides legacy 8+1
    changed, other = policy.predict_action_batch(
        observation(1), mode="eval", calculate_values=False
    )
    assert not np.array_equal(actions, changed)
    np.testing.assert_allclose(other["forward_inputs"]["state"].numpy(), 0.5)
    assert not torch.equal(
        cached["forward_inputs"]["input_ids"], other["forward_inputs"]["input_ids"]
    )


def test_rollout_logprob_recompute_and_actor_gradient():
    policy = make_policy()
    _, cached = policy.predict_action_batch(
        observation(0.2), mode="train", calculate_values=False
    )
    recomputed = policy.default_forward(cached["forward_inputs"], compute_logprobs=True)
    shapes = policy.starvla_model.qwen_vl_interface.forward_shapes
    assert shapes[-1] == shapes[-2]
    torch.testing.assert_close(
        recomputed["logprobs"], cached["prev_logprobs"], rtol=1e-5, atol=1e-5
    )
    (-recomputed["logprobs"].mean()).backward()
    assert policy.starvla_model.action_model.weight.grad.abs().sum() > 0
    assert policy.actor_logstd.grad is not None


def test_single_sample_contract_rejects_rollout_and_training_batches():
    policy = make_policy(require_single_sample=True)
    obs = observation()
    batch = {
        k: v * 2 if isinstance(v, list) else np.concatenate([v, v])
        for k, v in obs.items()
    }
    with pytest.raises(ValueError, match="micro-batch=1"):
        policy.predict_action_batch(batch, calculate_values=False)
    _, cached = policy.predict_action_batch(obs, calculate_values=False)
    combined = {k: torch.cat([v, v]) for k, v in cached["forward_inputs"].items()}
    with pytest.raises(ValueError, match="micro-batch=1"):
        policy.default_forward(combined, compute_logprobs=True)


@pytest.mark.parametrize(
    "states", [None, np.zeros((1, 32)), np.zeros((1, 2, 35)), np.full((1, 35), np.nan)]
)
def test_state_conditioned_oft_rejects_missing_or_incompatible_state(states):
    obs = observation()
    obs["states"] = states
    with pytest.raises(ValueError):
        make_policy().predict_action_batch(obs, calculate_values=False)


def test_state_free_oft_keeps_original_prompt():
    policy = make_policy(with_state=False)
    policy.predict_action_batch(observation(), mode="eval", calculate_values=False)
    assert "[STATE]" not in policy.starvla_model.qwen_vl_interface.prompts[0]


def test_oft_image_size_uses_native_height_width_and_view_order():
    policy = make_policy()
    policy.starvla_model.config.datasets.vla_data.obs_image_size = [
        [8, 12],
        [16, 20],
        [24, 28],
    ]
    policy.predict_action_batch(observation(), mode="eval", calculate_values=False)
    assert policy.starvla_model.qwen_vl_interface.image_sizes == [
        [(12, 8), (20, 16), (28, 24)]
    ]


def test_state_conditioned_oft_requires_checkpoint_normalizer():
    with pytest.raises(ValueError, match="state_normalizer"):
        StarVLAForRLActionPrediction(
            TinyOFT(), 3, 30, unnorm_key="pipette", action_stats_source="q01q99"
        )


def test_explicit_full_checkpoint_config_keeps_state_metadata(tmp_path):
    from starVLA.model.framework.share_tools import read_mode_config

    checkpoints = tmp_path / "checkpoints"
    checkpoints.mkdir()
    checkpoint = checkpoints / "steps_10000_pytorch_model.pt"
    checkpoint.touch()
    (tmp_path / "config.yaml").write_text("framework:\n  name: QwenOFT\n")
    full = tmp_path / "config.full.yaml"
    full.write_text(
        "framework:\n  name: QwenOFT\n  action_model:\n"
        "    action_horizon: 30\n    state_dim: 35\n"
    )
    (tmp_path / "dataset_statistics.json").write_text('{"new_embodiment": {}}')
    cfg, stats = read_mode_config(str(checkpoint), config_path=str(full))
    assert cfg["framework"]["action_model"]["state_dim"] == 35
    assert list(stats) == ["new_embodiment"]


@pytest.mark.parametrize("head_dtype", [torch.float32, torch.bfloat16])
def test_sft_oft_head_accepts_bf16_backbone_and_retains_gradient(head_dtype):
    from starVLA.model.modules.action_model.MLP_ActionHeader import (
        L1RegressionActionHead,
    )

    head = L1RegressionActionHead(
        input_dim=4, hidden_dim=8, action_dim=3, NUM_ACTIONS_CHUNK=30
    ).to(dtype=head_dtype)
    hidden = torch.randn(1, 30, 4, dtype=torch.bfloat16)
    actions = head.predict_action(hidden)
    assert actions.shape == (1, 30, 3)
    assert actions.dtype == head_dtype
    actions.float().square().mean().backward()
    assert head.model.fc2.weight.grad.abs().sum() > 0


def test_serial_group_can_update_oft_with_rlinf_grpo():
    torch.manual_seed(42)
    policy = make_policy()
    cache = [
        policy.predict_action_batch(
            observation(), mode="train", calculate_values=False
        )[1]
        for _ in range(2)
    ]
    # One success/failure group, with four executed primitive rows per decision.
    advantages, _ = compute_grpo_advantages(
        rewards=torch.tensor([[10.0, -0.01]]),
        loss_mask=torch.ones((1, 2)),
        group_size=2,
    )
    old = torch.stack([item["prev_logprobs"][:, :4].sum() for item in cache])[None]
    current = torch.stack(
        [
            policy.default_forward(item["forward_inputs"], compute_logprobs=True)[
                "logprobs"
            ][:, :4].sum()
            for item in cache
        ]
    )[None]
    loss, _ = compute_grpo_actor_loss_fn(
        logprobs=current,
        old_logprobs=old.detach(),
        advantages=advantages,
        clip_ratio_low=0.1,
        clip_ratio_high=0.1,
    )
    before = policy.starvla_model.action_model.weight.detach().clone()
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-5)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    assert torch.isfinite(loss)
    assert not torch.equal(before, policy.starvla_model.action_model.weight)
