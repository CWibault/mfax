from dataclasses import fields

from mfax.envs.pushforward.macro.endogenous import (
    PushforwardEndogenousEnvironment,
    PushforwardEndogenousEnvParams,
)
from mfax.envs.pushforward.toy.beach_bar_1d import (
    PushforwardBeachBar1DEnvironment,
    PushforwardBeachBar1DEnvParams,
)
from mfax.envs.pushforward.toy.linear_quadratic import (
    PushforwardLinearQuadraticEnvironment,
    PushforwardLinearQuadraticEnvParams,
)

from mfax.envs.sample.macro.endogenous import (
    SampleEndogenousEnvironment,
    SampleEndogenousEnvParams,
)
from mfax.envs.sample.toy.beach_bar_1d import (
    SampleBeachBar1DEnvironment,
    SampleBeachBar1DEnvParams,
)
from mfax.envs.sample.toy.linear_quadratic import (
    SampleLinearQuadraticEnvironment,
    SampleLinearQuadraticEnvParams,
)

"""
Environments for mean field games. 

pushforward: exact mean-field updates
sample: approximate mean-field updates 
"""

_ENV_BUILDERS = {
    "pushforward/beach_bar_1d": (
        PushforwardBeachBar1DEnvironment,
        PushforwardBeachBar1DEnvParams,
    ),
    "pushforward/linear_quadratic": (
        PushforwardLinearQuadraticEnvironment,
        PushforwardLinearQuadraticEnvParams,
    ),
    "pushforward/endogenous": (
        PushforwardEndogenousEnvironment,
        PushforwardEndogenousEnvParams,
    ),
    "sample/beach_bar_1d": (SampleBeachBar1DEnvironment, SampleBeachBar1DEnvParams),
    "sample/linear_quadratic": (
        SampleLinearQuadraticEnvironment,
        SampleLinearQuadraticEnvParams,
    ),
    "sample/endogenous": (SampleEndogenousEnvironment, SampleEndogenousEnvParams),
}


def _filter_params(params_cls, param_overrides: dict) -> dict:
    """Keep only overrides that match the dataclass fields."""
    valid_fields = {f.name for f in fields(params_cls)}
    unknown = set(param_overrides) - valid_fields
    if unknown:
        raise ValueError(
            f"Invalid params for {params_cls.__name__}: {sorted(unknown)}; "
            f"allowed: {sorted(valid_fields)}"
        )
    return {k: v for k, v in param_overrides.items() if k in valid_fields}


def make_env(task: str, **param_kwargs):
    """Instantiate the environment matching the given task name."""
    task_key = str(task).lower()
    try:
        env_cls, params_cls = _ENV_BUILDERS[task_key]
    except KeyError as exc:
        raise ValueError(f"Invalid task: {task}") from exc

    filtered_kwargs = _filter_params(params_cls, param_kwargs)
    return env_cls(params_cls(**filtered_kwargs))
