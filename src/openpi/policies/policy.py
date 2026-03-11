
from collections.abc import Sequence
import logging
import pathlib
from typing import Any, TypeAlias
import threading
import time

import flax
import flax.traverse_util
import jax
import jax.numpy as jnp
try:
    # only used for onboard testing
    from pynput import keyboard
except ImportError:
    pass
import numpy as np
from openpi_client import base_policy as _base_policy
from typing_extensions import override

from openpi import transforms as _transforms
from openpi.models import model as _model
from openpi.models.pi0_atomic import Pi0Atomic
from openpi.shared import array_typing as at
from openpi.shared import nnx_utils

BasePolicy: TypeAlias = _base_policy.BasePolicy


class Policy(BasePolicy):
    def __init__(
        self,
        model: _model.BaseModel,
        *,
        rng: at.KeyArrayLike | None = None,
        transforms: Sequence[_transforms.DataTransformFn] = (),
        output_transforms: Sequence[_transforms.DataTransformFn] = (),
        sample_kwargs: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        pytorch_device: str = "cpu",
        is_pytorch: bool = False,
    ):
        self._sample_actions = nnx_utils.module_jit(model.sample_actions)
        self._input_transform = _transforms.compose(transforms)
        self._output_transform = _transforms.compose(output_transforms)
        self._rng = rng or jax.random.key(0)
        self._sample_kwargs = sample_kwargs or {}
        self._metadata = metadata or {}

    @override
    def infer(self, obs: dict) -> dict:  # type: ignore[misc]
        # Make a copy since transformations may modify the inputs in place.
        inputs = jax.tree.map(lambda x: x, obs)
        inputs = self._input_transform(inputs)
        # Make a batch and convert to jax.Array.
        inputs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)
        start_time = time.monotonic()

        self._rng, sample_rng = jax.random.split(self._rng)
        outputs = {
            "state": inputs["state"],
            "actions": self._sample_actions(sample_rng, _model.AtomicObservation.from_dict(inputs), **self._sample_kwargs),
        }
        model_time = time.monotonic() - start_time

        # Unbatch and convert to np.ndarray.
        outputs = jax.tree.map(lambda x: np.asarray(x[0, ...]), outputs)
        outputs = self._output_transform(outputs)
        
        outputs["policy_timing"] = {
            "infer_ms": model_time * 1000,
        }
        return outputs

    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata


class ReasoningPolicy(BasePolicy):
    def __init__(
        self,
        model: Pi0Atomic,
        *,
        rng: at.KeyArrayLike | None = None,
        transforms: Sequence[_transforms.DataTransformFn] = (),
        output_transforms: Sequence[_transforms.DataTransformFn] = (),
        sample_kwargs: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        use_ref_img: bool = False,
        initial_scene_plan: str = '',
    ):
        self._prefill = nnx_utils.module_jit(model.prefill,
                                             static_argnames=('temprature',))
        self._act = nnx_utils.module_jit(model.act)
        self._reason = nnx_utils.module_jit(model.reason,
                                           static_argnames=('temprature',))
        self._input_transform = _transforms.compose(transforms)
        self._output_transform = _transforms.compose(output_transforms)
        self._rng = rng or jax.random.key(0)
        self._sample_kwargs = sample_kwargs or {}
        self._metadata = metadata or {}

        self._is_thinking = False
        self._lock = threading.Lock()

        self._use_ref_img = use_ref_img

        self._thought: str | None = None
        self._ref_img: np.ndarray | None = None
        self._initial_scene_plan: str = initial_scene_plan
        self._scene_plan: str | None = self._initial_scene_plan

        self._action_placehoser = np.zeros((model.action_horizon, model.action_dim))

        self._instruction: str | None = None

        self._intervention: str | None = None
        self._intervention_added: bool = False
        
        self._robot_question: str | None = None
        self._user_answer: str | None = None

        self._temperature = 0
        self._temperature = float(self._temperature)

    def reset(self):
        """
        Start a new rollout.
        1. Reset the thinking flag.
        2. Reset the thought and ref_img.
        """
        self._reset_thought()

        print("Start a new rollout.")
    
    def _reset_thought(self):
        self.is_thinking = False
        self._thought = None
        self._ref_img = None
        self._scene_plan = self._initial_scene_plan
        self._instruction = None
        self._intervention = None
        self._intervention_added = False
        self._robot_question = None
        self._user_answer = None

    def _prepare_obs(self, obs: dict) -> dict:
        """
        Update the `obs` dict's 
        'thought' and 'ref_img' fields.
        """
        assert 'prompt' in obs, \
            "The observation must contain a 'prompt' key."
        instruction = obs['prompt']

        if self._thought is None:
            self._instruction = instruction
            obs['thought'] = [instruction]
        else:
            obs['thought'] = [instruction,self._thought]
        
        return obs

    def _update_thought(self, model_output: str, obs: dict):
        """
        Update the thought and ref_img fields.
        """
        self._scene_plan = model_output.rsplit(None, 1)[-1]  
        self._thought = (self._scene_plan)


    @override
    def infer(self, obs: dict) -> dict:

        inputs = jax.tree.map(lambda x: x, obs)
        warmup = inputs.pop('is_warm_up', False)
        if warmup:
            return self._warmup(inputs)

        inputs = self._prepare_obs(inputs)
        prefix = inputs['thought'][0]
        inputs = self._input_transform(inputs)
        # Make a batch and convert to jax.Array.
        inputs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)
        
        prefill_rng, think_rng, act_rng, self._rng = jax.random.split(self._rng, 4)

        # prefill and decide to act or think

        # defaults:
        actions = self._action_placehoser[np.newaxis, ...]
        # default: '<eos>' token
        start_time = time.monotonic()
        suffix_tokens = np.ones((1, 1), dtype=np.int32)
        inputs = _model.AtomicObservation.from_dict(inputs)
        (
            processed_inputs,
            prefix_cache,
            first_suffix_token,
            last_logit,
            prefix_mask,
            prefix_positions,
            to_act
        ) = self._prefill(prefill_rng, inputs, temprature=self._temperature)
        assert jnp.size(to_act) == 1
        to_act = to_act.item()
        to_think = not to_act
        if self.is_thinking == True:
            to_act = True
            to_think = False

        self.is_thinking = to_think
        if to_act:
            print('Acting')
            actions = self._act(act_rng, processed_inputs,
                                prefix_cache, prefix_mask, prefix_positions)
            # self.is_thinking = True
        else:
            print('Thinking...')
            suffix_tokens = self._reason(think_rng, last_logit, prefix_cache, prefix_mask,
                                        prefix_positions, temprature=self._temperature)
        model_time = time.monotonic() - start_time
        # print(model_time)
        outputs = {
            "state": inputs.state,
            "actions": actions,
            "tokenized_suffix": suffix_tokens,
        }

        # Unbatch and convert to np.ndarray.
        outputs = jax.tree.map(lambda x: np.asarray(x[0, ...]), outputs)
        transformed = self._output_transform(outputs)

        if to_think:
            print(transformed['thoughts'])
            self._update_thought(transformed['thoughts'], obs)
            # self.infer(obs)
        transformed["policy_timing"] = {
            "infer_ms": model_time * 1000,
        }
        transformed = {"actions":transformed["actions"]}
        # print(transformed)
        return transformed
    
    @property
    def is_thinking(self) -> bool:
        with self._lock:
            return self._is_thinking
    
    @is_thinking.setter
    def is_thinking(self, value: bool):
        with self._lock:
            self._is_thinking = value

    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata


class PolicyRecorder(_base_policy.BasePolicy):
    """Records the policy's behavior to disk."""

    def __init__(self, policy: _base_policy.BasePolicy, record_dir: str):
        self._policy = policy

        logging.info(f"Dumping policy records to: {record_dir}")
        self._record_dir = pathlib.Path(record_dir)
        self._record_dir.mkdir(parents=True, exist_ok=True)
        self._record_step = 0

    @override
    def infer(self, obs: dict) -> dict:  # type: ignore[misc]
        results = self._policy.infer(obs)

        data = {"inputs": obs, "outputs": results}
        data = flax.traverse_util.flatten_dict(data, sep="/")

        output_path = self._record_dir / f"step_{self._record_step}"
        self._record_step += 1

        np.save(output_path, np.asarray(data))
        return results
