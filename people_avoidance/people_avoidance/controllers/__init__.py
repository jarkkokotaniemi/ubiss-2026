"""
controllers — Stage 3 of the people-avoidance pipeline.

Input : List[Track], robot pose (x, y, theta), destination (goal_x, goal_y)
Output: geometry_msgs/Twist  published on /cmd_vel

Provides make_controller(), a small factory that builds either the reactive
CBF-QP controller or the predictive MPC-CBF controller behind the common
BaseController.compute() interface, selected by the `controller_type` ROS
parameter ("cbf" or "mpc").
"""
from __future__ import annotations

import inspect
from typing import Any, Dict, Type

from .base import BaseController, PERSON_CLEARANCE, obstacle_radius
from .cbf_controller import CBFController
from .mpc_controller import MPCController

_REGISTRY: Dict[str, Type[BaseController]] = {
    "cbf": CBFController,
    "mpc": MPCController,
}


def make_controller(kind: str, **params: Any) -> BaseController:
    """
    Build the avoidance controller selected by `kind` ("cbf" or "mpc").

    `params` is typically the node's full parameter dict; each controller's
    constructor only picks out the keys it declares, so callers can pass one
    combined dict regardless of which controller is selected.
    """
    try:
        cls = _REGISTRY[kind]
    except KeyError:
        raise ValueError(
            f"Unknown controller_type '{kind}'; expected one of {sorted(_REGISTRY)}"
        ) from None

    sig = inspect.signature(cls.__init__)
    kwargs = {k: v for k, v in params.items() if k in sig.parameters}
    return cls(**kwargs)


__all__ = [
    "BaseController",
    "CBFController",
    "MPCController",
    "make_controller",
    "obstacle_radius",
    "PERSON_CLEARANCE",
]
