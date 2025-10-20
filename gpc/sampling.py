from typing import Any, Callable, Tuple

import jax
import jax.numpy as jnp
from hydrax.alg_base import Trajectory
from hydrax.algs.predictive_sampling import PredictiveSampling
from mujoco import mjx

from gpc.policy import Policy


class BootstrappedPredictiveSampling(PredictiveSampling):
    """Perform predictive sampling, but add samples from a generative policy."""

    def __init__(
        self,
        policy: Policy,
        observation_fn: Callable[[mjx.Data], jax.Array],
        num_policy_samples: int,
        warm_start_level: float = 0.0,
        inference_timestep: float = 0.1,
        **kwargs,
    ):
        """Initialize the controller.

        Args:
            policy: The generative policy to sample from.
            observation_fn: A function that produces an observation vector.
            num_policy_samples: The number of samples to take from the policy.
            warm_start_level: The warm start level in [0, 1] to use for the
                policy samples. 0.0 generates samples from scratch, while 1.0
                seed all samples from the previous action sequence.
            inference_timestep: The timestep dt for flow matching inference.
            **kwargs: Constructor arguments for PredictiveSampling.
        """
        self.observation_fn = observation_fn
        self.policy = policy.replace(dt=inference_timestep)
        self.policy.model.eval()  # Don't update batch statistics
        self.warm_start_level = jnp.clip(warm_start_level, 0.0, 1.0)
        self.num_policy_samples = num_policy_samples

        super().__init__(**kwargs)

    def optimize(self, state: mjx.Data, params: Any) -> Tuple[Any, Trajectory]:
        """Perform an optimization step to update the policy parameters.

        In addition to sampling random control sequences, also sample control
        sequences from the generative policy.

        Args:
            state: The initial state x₀.
            params: The current policy parameters, U ~ π(params).

        Returns:
            Updated policy parameters
            Rollouts used to update the parameters
        """
        # Warm-start spline by advancing knot times by sim dt, then recomputing
        # the mean knots by evaluating the old spline at those times
        tk = params.tk
        new_tk = (
            jnp.linspace(0.0, self.plan_horizon, self.num_knots) + state.time
        )
        new_mean = self.interp_func(new_tk, tk, params.mean[None, ...])[0]
        params = params.replace(tk=new_tk, mean=new_mean)
        
        rng, policy_rng, dr_rng = jax.random.split(params.rng, 3)

        # Sample random control knots
        knots, params = self.sample_knots(params)
        knots = jnp.clip(knots, self.task.u_min, self.task.u_max)

        # Update sensor readings and get an observation
        state = mjx.forward(self.task.model, state)
        y = self.observation_fn(state)

        # Sample from the generative policy, which is conditioned on the latest
        # observation.
        policy_rngs = jax.random.split(policy_rng, self.num_policy_samples)
        policy_knots = jax.vmap(
            self.policy.apply, in_axes=(None, None, 0, None)
        )(
            params.mean,
            y,
            policy_rngs,
            self.warm_start_level,
        )

        # Combine the random and policy samples
        knots = jnp.concatenate([knots, policy_knots], axis=0)

        # Roll out the control sequences, applying domain randomizations and
        # combining costs using self.risk_strategy.
        rollouts = self.rollout_with_randomizations(state, new_tk, knots, dr_rng)

        # Update the policy parameters based on the combined costs
        params = params.replace(rng=rng)
        params = self.update_params(params, rollouts)
        return params, rollouts
