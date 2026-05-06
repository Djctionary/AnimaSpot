"""Trajectory-level soft-constrained IK for Spot retargeting."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import numpy as np
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation
from tqdm import tqdm

from .config import HIP_ATTACHMENTS, HIP_X_OFFSET, JOINT_LIMITS, LEG_ORDER, LEG_SIDE, L_LOWER, L_UPPER, RetargetConfig
from .ik_solver import forward_kinematics

if TYPE_CHECKING:
    from .retarget import RetargetContext

LOGGER = logging.getLogger(__name__)


def _joint_bounds(n_frames: int) -> list[tuple[float, float]]:
    per_frame = [JOINT_LIMITS[key] for key in ["hx", "hy", "kn"] * len(LEG_ORDER)]
    return per_frame * n_frames


def _trajectory_bounds(n_frames: int) -> list[tuple[float | None, float | None]]:
    q_bounds = _joint_bounds(n_frames)
    root_pos_bounds = [(None, None)] * (n_frames * 3)
    root_rot_bounds = [(None, None)] * (n_frames * 3)
    return q_bounds + root_pos_bounds + root_rot_bounds


def _pack_variables(q: np.ndarray, root_pos: np.ndarray, root_rotvec: np.ndarray) -> np.ndarray:
    return np.concatenate([q.reshape(-1), root_pos.reshape(-1), root_rotvec.reshape(-1)])


def _unpack_variables(flat: np.ndarray, n_frames: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    q_size = n_frames * 12
    root_pos_size = n_frames * 3
    q = flat[:q_size].reshape(n_frames, 12)
    root_pos = flat[q_size : q_size + root_pos_size].reshape(n_frames, 3)
    root_rotvec = flat[q_size + root_pos_size :].reshape(n_frames, 3)
    return q, root_pos, root_rotvec


def _unpack_torch_variables(flat: Any, n_frames: int) -> tuple[Any, Any, Any]:
    q_size = n_frames * 12
    root_pos_size = n_frames * 3
    q = flat[:q_size].reshape(n_frames, 12)
    root_pos = flat[q_size : q_size + root_pos_size].reshape(n_frames, 3)
    root_rotvec = flat[q_size + root_pos_size :].reshape(n_frames, 3)
    return q, root_pos, root_rotvec


def _diff_costs(values: np.ndarray) -> tuple[float, float]:
    if values.shape[0] > 1:
        d_values = values[1:] - values[:-1]
        velocity = float(np.sum(d_values * d_values))
    else:
        velocity = 0.0
    if values.shape[0] > 2:
        dd_values = values[2:] - 2.0 * values[1:-1] + values[:-2]
        acceleration = float(np.sum(dd_values * dd_values))
    else:
        acceleration = 0.0
    return velocity, acceleration


def _torch_diff_costs(values: Any) -> tuple[Any, Any]:
    if values.shape[0] > 1:
        d_values = values[1:] - values[:-1]
        velocity = (d_values * d_values).sum()
    else:
        velocity = values.new_tensor(0.0)
    if values.shape[0] > 2:
        dd_values = values[2:] - 2.0 * values[1:-1] + values[:-2]
        acceleration = (dd_values * dd_values).sum()
    else:
        acceleration = values.new_tensor(0.0)
    return velocity, acceleration


def _paw_body_positions(q: np.ndarray) -> np.ndarray:
    """Return FK paw positions relative to each hip attachment, shape (T, 4, 3)."""
    paw = np.zeros((q.shape[0], len(LEG_ORDER), 3), dtype=np.float64)
    for frame_idx in range(q.shape[0]):
        for leg_idx, leg in enumerate(LEG_ORDER):
            hx, hy, kn = q[frame_idx, 3 * leg_idx : 3 * leg_idx + 3]
            paw[frame_idx, leg_idx] = forward_kinematics(
                hx,
                hy,
                kn,
                HIP_X_OFFSET,
                L_UPPER,
                L_LOWER,
                LEG_SIDE[leg],
            )
    return paw


def _torch_paw_body_positions(q: Any, side_signs: Any, torch: Any) -> Any:
    """Vectorized torch FK paw positions relative to each hip attachment."""
    q_leg = q.reshape(q.shape[0], len(LEG_ORDER), 3)
    hx = q_leg[..., 0]
    hy = q_leg[..., 1]
    kn = q_leg[..., 2]
    hx_raw = side_signs[None, :] * hx

    x_sag = -(L_UPPER * hy.sin() + L_LOWER * (hy + kn).sin())
    y_sag = q.new_full(hx.shape, HIP_X_OFFSET)
    z_sag = -(L_UPPER * hy.cos() + L_LOWER * (hy + kn).cos())

    c = hx_raw.cos()
    s = hx_raw.sin()
    y_out = c * y_sag - s * z_sag
    pz = s * y_sag + c * z_sag
    py = side_signs[None, :] * y_out
    return torch.stack((x_sag, py, pz), dim=-1)


def _torch_rotvec_to_matrix(root_rotvec: Any, torch: Any) -> Any:
    """Differentiable Rodrigues formula for batched rotation vectors."""
    x = root_rotvec[:, 0]
    y = root_rotvec[:, 1]
    z = root_rotvec[:, 2]
    zeros = torch.zeros_like(x)
    skew = torch.stack(
        (
            torch.stack((zeros, -z, y), dim=-1),
            torch.stack((z, zeros, -x), dim=-1),
            torch.stack((-y, x, zeros), dim=-1),
        ),
        dim=-2,
    )

    theta2 = (root_rotvec * root_rotvec).sum(dim=-1, keepdim=True)
    theta = torch.sqrt(torch.clamp(theta2, min=1e-16))
    small = theta2 < 1e-8
    theta4 = theta2 * theta2
    a_taylor = 1.0 - theta2 / 6.0 + theta4 / 120.0
    b_taylor = 0.5 - theta2 / 24.0 + theta4 / 720.0
    a_direct = torch.sin(theta) / theta
    b_direct = (1.0 - torch.cos(theta)) / (theta * theta)
    a = torch.where(small, a_taylor, a_direct).reshape(-1, 1, 1)
    b = torch.where(small, b_taylor, b_direct).reshape(-1, 1, 1)

    eye = torch.eye(3, dtype=root_rotvec.dtype, device=root_rotvec.device).expand(root_rotvec.shape[0], 3, 3)
    return eye + a * skew + b * (skew @ skew)


def _ground_penetration_cost(
    paw_body: np.ndarray,
    root_rotvec: np.ndarray,
    root_pos: np.ndarray,
    ground_level: float,
) -> float:
    rotations = Rotation.from_rotvec(root_rotvec).as_matrix()
    hip_offsets = np.stack([HIP_ATTACHMENTS[leg] for leg in LEG_ORDER], axis=0)
    penalty = 0.0
    for frame_idx in range(paw_body.shape[0]):
        paw_world = (rotations[frame_idx] @ (hip_offsets + paw_body[frame_idx]).T).T + root_pos[frame_idx]
        penetration = np.maximum(ground_level - paw_world[:, 2], 0.0)
        penalty += float(np.sum(penetration * penetration))
    return penalty


def _import_torch() -> Any | None:
    try:
        import torch  # type: ignore[import-not-found]
    except ImportError:
        return None
    return torch


def _resolve_torch_device(torch: Any, requested: str) -> Any:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        LOGGER.warning("Requested TrajectoryIK autograd device 'cuda' is unavailable; falling back to CPU.")
        return torch.device("cpu")
    return torch.device(requested)


def solve_trajectory_ik(
    context: "RetargetContext",
    config: RetargetConfig,
    q_init: np.ndarray,
    root_quat: np.ndarray,
    root_pos: np.ndarray,
) -> np.ndarray:
    """Optimize a full joint-angle sequence with four soft objective terms."""
    q_init = np.asarray(q_init, dtype=np.float64)
    if q_init.ndim != 2 or q_init.shape[1] != 12:
        raise ValueError(f"Expected q_init shape (T, 12), got {q_init.shape}")
    root_pos_init = np.asarray(root_pos, dtype=np.float64)
    root_quat_init = np.asarray(root_quat, dtype=np.float64)

    n_frames = q_init.shape[0]
    if root_pos_init.shape != (n_frames, 3):
        raise ValueError(f"Expected root_pos shape ({n_frames}, 3), got {root_pos_init.shape}")
    if root_quat_init.shape != (n_frames, 4):
        raise ValueError(f"Expected root_quat shape ({n_frames}, 4), got {root_quat_init.shape}")

    targets = np.asarray(context.scaled_targets, dtype=np.float64)
    stable_joint_indices = tuple(
        idx for idx in config.trajectory_stable_joint_indices
        if 0 <= idx < q_init.shape[1]
    )
    root_rotvec_init = Rotation.from_quat(root_quat_init).as_rotvec()

    def objective(flat_q: np.ndarray) -> float:
        q, root_pos_opt, root_rotvec = _unpack_variables(flat_q, n_frames)
        paw_body = _paw_body_positions(q)

        track = float(np.sum((paw_body - targets) ** 2))

        q_velocity, q_acceleration = _diff_costs(q)
        root_pos_velocity, root_pos_acceleration = _diff_costs(root_pos_opt)
        root_rot_velocity, root_rot_acceleration = _diff_costs(root_rotvec)
        velocity = (
            q_velocity
            + config.trajectory_smooth_root_pos_weight * root_pos_velocity
            + config.trajectory_smooth_root_rot_weight * root_rot_velocity
        )
        acceleration = (
            q_acceleration
            + config.trajectory_smooth_root_pos_weight * root_pos_acceleration
            + config.trajectory_smooth_root_rot_weight * root_rot_acceleration
        )
        smooth = (
            config.trajectory_smooth_velocity_weight * velocity
            + config.trajectory_smooth_acceleration_weight * acceleration
        )

        ground = _ground_penetration_cost(
            paw_body,
            root_rotvec,
            root_pos_opt,
            config.trajectory_ground_level,
        )

        if stable_joint_indices:
            stable_delta = q[:, stable_joint_indices]
            stable = float(np.sum(stable_delta * stable_delta))
        else:
            stable = 0.0

        return (
            config.trajectory_w_track * track
            + config.trajectory_w_smooth * smooth
            + config.trajectory_w_ground * ground
            + config.trajectory_w_stable * stable
        )

    torch = _import_torch() if config.trajectory_use_autograd_jac else None
    autograd_device = None
    targets_t = None
    hip_offsets_t = None
    side_signs_t = None
    if torch is not None:
        autograd_device = _resolve_torch_device(torch, config.trajectory_autograd_device)
        targets_t = torch.as_tensor(targets, dtype=torch.float64, device=autograd_device)
        hip_offsets_t = torch.as_tensor(
            np.stack([HIP_ATTACHMENTS[leg] for leg in LEG_ORDER], axis=0),
            dtype=torch.float64,
            device=autograd_device,
        )
        side_signs_t = torch.as_tensor(
            [1.0 if LEG_SIDE[leg] == "left" else -1.0 for leg in LEG_ORDER],
            dtype=torch.float64,
            device=autograd_device,
        )
        LOGGER.info("TrajectoryIK using PyTorch autograd Jacobian on %s.", autograd_device)
    elif config.trajectory_use_autograd_jac:
        LOGGER.warning("PyTorch is not available; TrajectoryIK is falling back to SciPy finite-difference gradients.")
    else:
        LOGGER.info("TrajectoryIK autograd Jacobian disabled; using SciPy finite-difference gradients.")

    def objective_and_grad(flat_q: np.ndarray) -> tuple[float, np.ndarray]:
        assert torch is not None
        assert autograd_device is not None
        assert targets_t is not None
        assert hip_offsets_t is not None
        assert side_signs_t is not None

        flat_t = torch.as_tensor(flat_q, dtype=torch.float64, device=autograd_device).clone().detach()
        flat_t.requires_grad_(True)
        q, root_pos_opt, root_rotvec = _unpack_torch_variables(flat_t, n_frames)
        paw_body = _torch_paw_body_positions(q, side_signs_t, torch)

        track = ((paw_body - targets_t) ** 2).sum()

        q_velocity, q_acceleration = _torch_diff_costs(q)
        root_pos_velocity, root_pos_acceleration = _torch_diff_costs(root_pos_opt)
        root_rot_velocity, root_rot_acceleration = _torch_diff_costs(root_rotvec)
        velocity = (
            q_velocity
            + config.trajectory_smooth_root_pos_weight * root_pos_velocity
            + config.trajectory_smooth_root_rot_weight * root_rot_velocity
        )
        acceleration = (
            q_acceleration
            + config.trajectory_smooth_root_pos_weight * root_pos_acceleration
            + config.trajectory_smooth_root_rot_weight * root_rot_acceleration
        )
        smooth = (
            config.trajectory_smooth_velocity_weight * velocity
            + config.trajectory_smooth_acceleration_weight * acceleration
        )

        rotations = _torch_rotvec_to_matrix(root_rotvec, torch)
        paw_world = torch.einsum("tij,tlj->tli", rotations, hip_offsets_t[None, :, :] + paw_body) + root_pos_opt[:, None, :]
        penetration = torch.clamp(config.trajectory_ground_level - paw_world[..., 2], min=0.0)
        ground = (penetration * penetration).sum()

        if stable_joint_indices:
            stable_delta = q[:, stable_joint_indices]
            stable = (stable_delta * stable_delta).sum()
        else:
            stable = q.new_tensor(0.0)

        total = (
            config.trajectory_w_track * track
            + config.trajectory_w_smooth * smooth
            + config.trajectory_w_ground * ground
            + config.trajectory_w_stable * stable
        )
        total.backward()
        grad = flat_t.grad.detach().cpu().numpy().astype(np.float64, copy=False)
        return float(total.detach().cpu().item()), grad

    initial_flat = _pack_variables(q_init, root_pos_init, root_rotvec_init)
    initial_cost = objective(initial_flat)
    progress = tqdm(
        total=config.trajectory_maxiter,
        desc="TrajectoryIK",
        unit="iter",
        dynamic_ncols=True,
    )

    def callback(flat_q: np.ndarray) -> None:
        cost = objective(flat_q)
        progress.update(1)
        progress.set_postfix(cost=f"{cost:.6g}")

    try:
        minimize_fun = objective_and_grad if torch is not None else objective
        result = minimize(
            minimize_fun,
            initial_flat,
            method="L-BFGS-B",
            jac=True if torch is not None else None,
            bounds=_trajectory_bounds(n_frames),
            callback=callback,
            options={
                "maxiter": config.trajectory_maxiter,
                "maxfun": config.trajectory_maxfun,
                "ftol": config.trajectory_ftol,
            },
        )
    finally:
        progress.close()

    if not result.success:
        LOGGER.warning("TrajectoryIK did not fully converge: %s", result.message)
    LOGGER.info(
        "TrajectoryIK objective %.6f -> %.6f in %d iteration(s).",
        initial_cost,
        float(result.fun),
        int(result.nit),
    )
    q_result, root_pos_result, root_rotvec_result = _unpack_variables(result.x, n_frames)
    root_quat_result = Rotation.from_rotvec(root_rotvec_result).as_quat()
    if stable_joint_indices:
        stable_delta = q_result[:, stable_joint_indices]
        LOGGER.info(
            "TrajectoryIK stable joints %s relative to zero: squared cost %.6g, RMS angle %.6g rad, max abs angle %.6g rad.",
            stable_joint_indices,
            float(np.sum(stable_delta * stable_delta)),
            float(np.sqrt(np.mean(stable_delta * stable_delta))),
            float(np.max(np.abs(stable_delta))),
        )
    else:
        LOGGER.info("TrajectoryIK stable joint penalty disabled.")
    root_pos_delta = root_pos_result - root_pos_init
    root_rot_delta = root_rotvec_result - root_rotvec_init
    LOGGER.info(
        "TrajectoryIK root pose delta: pos RMS %.6g m, pos max %.6g m, rot RMS %.6g rad, rot max %.6g rad.",
        float(np.sqrt(np.mean(root_pos_delta * root_pos_delta))),
        float(np.max(np.abs(root_pos_delta))),
        float(np.sqrt(np.mean(root_rot_delta * root_rot_delta))),
        float(np.max(np.abs(root_rot_delta))),
    )
    return q_result, root_pos_result, root_quat_result
