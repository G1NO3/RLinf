# QwenOFT state-conditioned pipette policy

The StarVLA wrapper supports the pipette QwenOFT checkpoint with three camera
views, 35-dimensional proprioceptive state, and 30 rows of 3-dimensional actions.
Use `examples/embodiment/config/model/starvla_pipette_oft.yaml` as the model
configuration and supply the checkpoint path and its full SFT configuration.
This fragment configures a model; hardware collection and episode handling
belong to the separate VLAPolicyBridge actor.

## Inputs and normalization

Set `starvla.enable_state_input` and `starvla.use_policy_norm_processor` to true.
The observation's `states` field contains 29 measured joint angles, the measured
end-effector XYZ position, and the commanded reference FK XYZ position. The
checkpoint's state normalization and discretized state tokenizer are applied
before building the Qwen prompt. Missing or incorrectly shaped state is rejected.
The three views use the native checkpoint's resize settings. Actions are
denormalized with the same `PolicyNormProcessor` used by SFT deployment.

`starvla.config_path` selects the full configuration when the checkpoint's
adjacent `config.yaml` omits deployment fields. Dataset statistics still come
from the checkpoint's run directory; `unnorm_key` must select its actual key.
The matching StarVLA checkout must support this explicit configuration override.

## Training and replay

`starvla.train_action_head_only` freezes the backbone and keeps the OFT head in
FP32. The wrapper also exposes Gaussian exploration with
`starvla.actor_logstd_init`; likelihood arithmetic remains FP32. Rollout prompt
padding is applied before sampling, and cached inputs retain the original state
tokens and sampled action for likelihood replay.

For the pipette checkpoint, keep `starvla.require_single_sample` enabled and
use micro-batches of one. Different batch shapes can change BF16 Qwen outputs
enough to affect likelihood ratios. Larger effective batches can accumulate
gradients across samples. VLAPolicyBridge's off-policy actor-critic instead
caches frozen features and trains the existing OFT head with separate critics;
it does not need behavior likelihood ratios.

## Validation

With the compatible StarVLA checkout on `PYTHONPATH`, run:

```bash
python -m pytest tests/unit_tests/test_starvla.py -q
```

The tests use a small external VLM fixture to check state conditioning,
normalization, cached replay, native image sizes, head gradients, and GRPO loss.
VLAPolicyBridge additionally provides opt-in real-checkpoint GPU tests with
synthetic observations. These tests do not actuate hardware.
