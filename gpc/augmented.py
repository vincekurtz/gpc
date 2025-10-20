from typing import Any, Tuple

import jax
import jax.numpy as jnp
from flax.struct import dataclass
from hydrax.alg_base import SamplingBasedController, Trajectory
from mujoco import mjx


@dataclass
class PACParams:
    """Parameters for the policy-augmented controller.

    Attributes:
        base_params: The parameters for the base controller.
        policy_samples: Control sequences sampled from the policy.
        rng: Random number generator key for domain randomization.
    """

    base_params: Any
    policy_samples: jax.Array
    rng: jax.Array


class PolicyAugmentedController(SamplingBasedController):
    """An SPC generalization where samples are augmented by a learned policy."""

    def __init__(
        self,
        base_ctrl: SamplingBasedController,
        num_policy_samples: int,
    ) -> None:
        """Initialize the policy-augmented controller.

        Args:
            base_ctrl: The base controller to augment.
            num_policy_samples: The number of samples to draw from the policy.
        """
        self.base_ctrl = base_ctrl
        self.num_policy_samples = num_policy_samples
        
        # Expose num_samples from base controller (handle Evosax case)
        if hasattr(base_ctrl, 'num_samples'):
            self.num_samples = base_ctrl.num_samples
        elif hasattr(base_ctrl, 'strategy') and hasattr(base_ctrl.strategy, 'population_size'):
            self.num_samples = base_ctrl.strategy.population_size
        else:
            self.num_samples = 0
        
        super().__init__(
            base_ctrl.task,
            base_ctrl.num_randomizations,
            base_ctrl.risk_strategy,
            seed=0,
            plan_horizon=base_ctrl.plan_horizon,
            spline_type=base_ctrl.spline_type,
            num_knots=base_ctrl.num_knots,
            iterations=base_ctrl.iterations,
        )

    def init_params(self) -> PACParams:
        """Initialize the controller parameters."""
        base_params = self.base_ctrl.init_params()
        base_rng, our_rng = jax.random.split(base_params.rng)
        base_params = base_params.replace(rng=base_rng)
        policy_samples = jnp.zeros(
            (
                self.num_policy_samples,
                self.base_ctrl.num_knots,
                self.task.model.nu,
            )
        )
        return PACParams(
            base_params=base_params,
            policy_samples=policy_samples,
            rng=our_rng,
        )

    def optimize(self, state: mjx.Data, params: PACParams) -> Tuple[PACParams, Trajectory]:
        """Perform an optimization step to update the policy parameters.

        This method samples from both the base controller's distribution and the
        policy, rolls them out together, but only updates the base controller
        using its own rollouts.

        Args:
            state: The initial state x₀.
            params: The current policy parameters.

        Returns:
            Updated policy parameters
            Rollouts used to update the parameters (includes both base and policy samples)
        """
        # Warm-start spline by advancing knot times by sim dt, then recomputing
        # the mean knots by evaluating the old spline at those times
        tk = params.base_params.tk
        new_tk = (
            jnp.linspace(0.0, self.plan_horizon, self.num_knots) + state.time
        )
        new_mean = self.interp_func(new_tk, tk, params.base_params.mean[None, ...])[0]
        base_params = params.base_params.replace(tk=new_tk, mean=new_mean)
        params = params.replace(base_params=base_params)

        def _optimize_scan_body(params: PACParams, _: Any):
            # Sample control knots from both base controller and policy
            knots, params = self.sample_knots(params)
            knots = jnp.clip(knots, self.task.u_min, self.task.u_max)

            # Roll out the control sequences
            rng, dr_rng = jax.random.split(params.rng)
            rollouts = self.rollout_with_randomizations(state, new_tk, knots, dr_rng)
            base_params = self.base_ctrl.update_params(params.base_params, rollouts)
            params = params.replace(base_params=base_params, rng=rng)

            return params, rollouts

        params, rollouts = jax.lax.scan(
            f=_optimize_scan_body, init=params, xs=jnp.arange(self.iterations)
        )

        rollouts_final = jax.tree.map(lambda x: x[-1], rollouts)
        return params, rollouts_final

    def sample_knots(self, params: PACParams) -> Tuple[jax.Array, PACParams]:
        """Sample control knots from the base controller and the policy."""
        base_samples, base_params = self.base_ctrl.sample_knots(
            params.base_params
        )
        samples = jnp.concatenate([base_samples, params.policy_samples], axis=0)
        return samples, params.replace(base_params=base_params)

    def update_params(
        self, params: PACParams, rollouts: Trajectory
    ) -> PACParams:
        """Update the policy parameters according to the base controller."""
        base_params = self.base_ctrl.update_params(params.base_params, rollouts)
        return params.replace(base_params=base_params)

    def get_action(self, params: PACParams, t: float) -> jax.Array:
        """Get the action from the base controller at a given time."""
        return self.base_ctrl.get_action(params.base_params, t)