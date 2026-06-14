import argparse
import base64
import io
import json
import math
import os
import queue
import re
import threading
from contextlib import nullcontext
from getpass import getpass
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

import mujoco
import numpy as np
import requests
from PIL import Image
from sshtunnel import SSHTunnelForwarder

from raccoon_env import SyncSimRaccoonEnv


CYLINDER_BODY_BY_COLOR = {
    "red": "target_object",
    "blue": "target_object_blue",
    "green": "target_object_green",
    "yellow": "target_object_yellow",
    "white": "target_object_cube",
}
CYLINDER_COLORS = tuple(CYLINDER_BODY_BY_COLOR.keys())

# Dataset collection code와 동일한 원호 배치 조건.
DEFAULT_OBJECT_X_RANGE = (-0.11, 0.11)
DEFAULT_OBJECT_Y_RANGE = (0.19, 0.21)
DEFAULT_MIN_OBJECT_DISTANCE = 0.045
DEFAULT_YAW_RANGE = (-math.pi / 4, math.pi / 4)
DEFAULT_INSTRUCTION_TEMPLATE = "grasp the {color} cylinder"


def image_to_b64(image_rgb: np.ndarray) -> str:
    buffer = io.BytesIO()
    Image.fromarray(image_rgb).save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def request_action(
    server_url: str,
    instruction: str,
    image_rgb: np.ndarray,
    unnorm_key: Optional[str],
    timeout: float = 60.0,
) -> Dict[str, Any]:
    payload = {
        "instruction": instruction,
        "image_b64": image_to_b64(image_rgb),
        "unnorm_key": unnorm_key,
        "do_sample": False,
    }
    response = requests.post(f"{server_url.rstrip('/')}/predict", json=payload, timeout=timeout)
    if not response.ok:
        print(f"[SERVER ERROR] {response.status_code} | {response.text}")
        response.raise_for_status()
    return response.json()


def resolve_ssh_password(args: argparse.Namespace) -> Optional[str]:
    if args.ssh_password:
        return args.ssh_password
    env_password = os.environ.get("OPENVLA_SSH_PASSWORD")
    if env_password:
        return env_password
    if args.use_ssh_tunnel and args.ssh_ask_password:
        return getpass("SSH password: ")
    return None


def open_ssh_tunnel(args: argparse.Namespace) -> SSHTunnelForwarder:
    ssh_password = resolve_ssh_password(args)
    tunnel = SSHTunnelForwarder(
        ssh_address_or_host=(args.ssh_host, args.ssh_port),
        ssh_username=args.ssh_user,
        ssh_password=ssh_password,
        remote_bind_address=(args.remote_server_host, args.remote_server_port),
        local_bind_address=(args.local_server_host, args.local_server_port),
    )
    tunnel.start()
    return tunnel


def build_server_url(args: argparse.Namespace, tunnel: Optional[SSHTunnelForwarder]) -> str:
    if tunnel is not None:
        return f"http://{args.local_server_host}:{tunnel.local_bind_port}"
    if not args.server_url:
        raise ValueError("--server_url is required when --use_ssh_tunnel is not enabled.")
    return args.server_url


def maybe_tunnel_context(args: argparse.Namespace):
    if args.use_ssh_tunnel:
        return open_ssh_tunnel(args)
    return nullcontext(None)


def print_success_log(step_idx: int, exec_info: Dict[str, Any]) -> None:
    final_delta_xyz = [round(float(v), 4) for v in exec_info["final_delta_xyz"]]
    move_xyz = [round(float(v), 4) for v in exec_info["actual_move_xyz"]]
    target_xyz = [round(float(v), 4) for v in exec_info["target_xyz"]]
    gripper = float(exec_info["gripper_cmd"])
    retries = int(exec_info["retry_count"])
    print(
        f"[{step_idx:03d}] OK | final_delta={final_delta_xyz} | "
        f"move={move_xyz} | target={target_xyz} | "
        f"gripper={gripper:.1f} | retries={retries}"
    )


def print_fail_log(step_idx: int, exc: Exception) -> None:
    print(f"[{step_idx:03d}] FAIL | {exc}")


def gate_gripper_close_action(
    action: Any,
    env: SyncSimRaccoonEnv,
    object_specs: Dict[str, Dict[str, float]],
    target_color: str,
    task_type: str,
    enabled: bool,
    close_latched: bool,
    max_xy_distance: float,
    max_z: float,
) -> Any:
    if not enabled or task_type not in ("grasp", "lift", "pick_place_location"):
        return action
    if close_latched:
        return action
    if len(action) < 7 or float(action[6]) <= 0.5:
        return action

    ee_x, ee_y, ee_z = env.get_ee_pose()
    target_spec = object_specs[target_color]
    target_xy = np.array([float(target_spec["x"]), float(target_spec["y"])], dtype=np.float64)
    ee_xy = np.array([ee_x, ee_y], dtype=np.float64)
    xy_distance = float(np.linalg.norm(ee_xy - target_xy))

    if xy_distance <= max_xy_distance and ee_z <= max_z:
        return action

    gated_action = list(action)
    gated_action[6] = 0.0
    print(
        f"[GRIPPER GATE] close delayed | xy_dist={xy_distance:.4f}m "
        f"(limit={max_xy_distance:.4f}) | ee_z={ee_z:.4f}m (limit={max_z:.4f})"
    )
    return gated_action


def apply_gripper_hold_after_close(
    action: Any,
    task_type: str,
    enabled: bool,
    close_latched: bool,
) -> Any:
    if not enabled or task_type not in ("lift", "pick_place_location") or not close_latched or len(action) < 7:
        return action
    if float(action[6]) > 0.5:
        return action

    held_action = list(action)
    held_action[6] = 1.0
    print("[GRIPPER HOLD] keeping gripper closed after grasp")
    return held_action


def apply_lift_assist_after_close(
    action: Any,
    env: SyncSimRaccoonEnv,
    task_type: str,
    enabled: bool,
    close_latched: bool,
    min_dz: float,
    until_z: float,
) -> Any:
    if not enabled or task_type not in ("lift", "pick_place_location") or not close_latched or len(action) < 7:
        return action

    _, _, ee_z = env.get_ee_pose()
    if ee_z >= until_z:
        return action

    assisted_action = list(action)
    raw_dz = float(assisted_action[2])
    if raw_dz < min_dz:
        assisted_action[2] = float(min_dz)
        assisted_action[6] = 1.0
        print(
            f"[LIFT ASSIST] dz {raw_dz:.4f} -> {min_dz:.4f} "
            f"until ee_z reaches {until_z:.4f}m | current ee_z={ee_z:.4f}m"
        )
    return assisted_action


def apply_pick_place_location_carry_assist(
    action: Any,
    env: SyncSimRaccoonEnv,
    task_type: str,
    enabled: bool,
    close_latched: bool,
    goal_lane: int,
    lane_count: int,
    move_start_z: float,
    min_xy_step: float,
    place_xy_threshold: float,
    place_release_z: float,
) -> Any:
    if not enabled or task_type != "pick_place_location" or not close_latched or len(action) < 7:
        return action

    ee_x, ee_y, ee_z = env.get_ee_pose()
    goal_x, goal_y = arc_lane_xy(goal_lane, lane_count=lane_count)
    goal_vec = np.array([goal_x - ee_x, goal_y - ee_y], dtype=np.float64)
    goal_dist = float(np.linalg.norm(goal_vec))
    near_place = goal_dist <= place_xy_threshold and ee_z <= place_release_z

    assisted_action = list(action)

    # Carry with a stable horizontal wrist. Without this, a noisy dpitch can unlock
    # Link4 and twist the grasped cylinder out of the fingers.
    if not near_place:
        assisted_action[4] = 0.0
        assisted_action[6] = 1.0

    if ee_z >= move_start_z and goal_dist > place_xy_threshold:
        direction = goal_vec / max(goal_dist, 1e-6)
        raw_xy = np.array([float(assisted_action[0]), float(assisted_action[1])], dtype=np.float64)
        projected = float(np.dot(raw_xy, direction))
        if projected < min_xy_step:
            assisted_action[0] = float(direction[0] * min_xy_step)
            assisted_action[1] = float(direction[1] * min_xy_step)
            print(
                f"[CARRY ASSIST] xy toward lane {goal_lane} | "
                f"goal_dist={goal_dist:.4f}m | step={min_xy_step:.4f}m"
            )

    return assisted_action


def estimate_task_phase(
    task_type: str,
    env: SyncSimRaccoonEnv,
    current_phase: str,
    close_latched: bool,
    release_latched: bool,
    gripper_cmd: float,
    actual_move_xyz: Any,
    carry_move_start_z: float,
) -> Tuple[str, bool]:
    move = np.asarray(actual_move_xyz, dtype=np.float64)
    if move.size < 3:
        move = np.zeros(3, dtype=np.float64)
    dx, dy, dz = float(move[0]), float(move[1]), float(move[2])
    xy_step = float(np.linalg.norm(np.array([dx, dy], dtype=np.float64)))
    xy_motion_threshold = 0.0010
    lower_xy_threshold = 0.0030
    z_motion_threshold = 0.0005

    if task_type == "grasp":
        return "grasp", release_latched

    if task_type == "lift":
        if not close_latched:
            return "grasp", release_latched
        if dz > z_motion_threshold or current_phase == "lift":
            return "lift", release_latched
        return "grasp", release_latched

    if task_type != "pick_place_location":
        return task_type, release_latched

    if not close_latched:
        return "grasp", release_latched

    if release_latched or (current_phase in ("lower", "retreat") and float(gripper_cmd) <= 0.5):
        return "retreat", True

    _, _, ee_z = env.get_ee_pose()
    sufficiently_lifted = ee_z >= carry_move_start_z

    if current_phase in ("move", "lower") and xy_step <= lower_xy_threshold and dz < -z_motion_threshold:
        return "lower", release_latched
    if sufficiently_lifted and xy_step > xy_motion_threshold:
        return "move", release_latched
    if dz > z_motion_threshold or current_phase == "lift":
        return "lift", release_latched
    return current_phase, release_latched


def color_matches_from_instruction(instruction: Optional[str]) -> list:
    """Return color words in the order they appear in an instruction."""
    if not instruction:
        return []

    text = instruction.lower()
    matches = []
    for color in CYLINDER_COLORS:
        match = re.search(rf"\b{re.escape(color)}\b", text)
        if match:
            matches.append((match.start(), color))

    return [color for _, color in sorted(matches)]


def make_object_phrase(color: str) -> str:
    if color == "white":
        return "white cube"
    return f"{color} cylinder"


def make_instruction(task_type: str, target_color: str, instruction_template: str) -> str:
    if task_type == "lift":
        return f"lift the {make_object_phrase(target_color)}"
    if task_type == "pick_place_location":
        return f"pick the {make_object_phrase(target_color)} and place it at position four"
    if target_color == "white":
        return "grasp the white cube"
    return instruction_template.format(color=target_color)


def infer_task_type(instruction: Optional[str], task_type_arg: str) -> str:
    if task_type_arg != "auto":
        return task_type_arg
    if not instruction:
        return "grasp"
    text = instruction.lower()
    if "pick" in text and "place" in text and "position four" in text:
        return "pick_place_location"
    if "pick" in text and "place" in text:
        return "pick_place_location"
    if "lift" in text:
        return "lift"
    return "grasp"


def resolve_target_color_and_instruction(
    instruction: Optional[str],
    target_color_arg: str,
    task_type_arg: str,
    rng: np.random.Generator,
    instruction_template: str,
) -> Tuple[str, Optional[str], str, str]:
    """
    Keep the OpenVLA prompt and the physical target color synchronized.

    Priority:
      1. Infer task_type from --task_type or --instruction.
      2. Use --target_color or sample random colors.
    """
    task_type = infer_task_type(instruction=instruction, task_type_arg=task_type_arg)
    instruction_colors = color_matches_from_instruction(instruction)

    def choose_target_color() -> str:
        if target_color_arg in CYLINDER_COLORS:
            return target_color_arg
        if target_color_arg in ("auto", "random"):
            return str(rng.choice(CYLINDER_COLORS))
        raise ValueError(f"지원하지 않는 --target_color 값입니다: {target_color_arg}")

    place_color = None
    if task_type == "pick_place_location":
        if len(instruction_colors) > 1:
            raise ValueError(
                f"pick_place_location instruction에는 source 색상 하나만 들어가야 합니다: "
                f"{instruction_colors} | {instruction!r}"
            )
        if len(instruction_colors) == 1:
            target_color = instruction_colors[0]
            if target_color_arg in CYLINDER_COLORS and target_color_arg != target_color:
                raise ValueError(
                    f"--instruction 색상({target_color})과 --target_color({target_color_arg})가 다릅니다."
                )
        elif target_color_arg in ("auto", "random"):
            target_color = "red"
        else:
            target_color = choose_target_color()
    else:
        if len(instruction_colors) > 1:
            raise ValueError(f"{task_type} instruction에 여러 색상이 들어 있습니다: {instruction_colors} | {instruction!r}")
        if len(instruction_colors) == 1:
            target_color = instruction_colors[0]
            if target_color_arg in CYLINDER_COLORS and target_color_arg != target_color:
                raise ValueError(
                    f"--instruction 색상({target_color})과 --target_color({target_color_arg})가 다릅니다. "
                    "OpenVLA prompt와 실제 target이 어긋나지 않도록 둘 중 하나를 수정하세요."
                )
        else:
            target_color = choose_target_color()

    if instruction is None or instruction.strip() == "":
        instruction = make_instruction(
            task_type=task_type,
            target_color=target_color,
            instruction_template=instruction_template,
        )

    return target_color, place_color, task_type, instruction


def make_default_object_specs() -> Dict[str, Dict[str, float]]:
    """Deterministic fallback used when randomization is disabled."""
    x_values = np.linspace(
        DEFAULT_OBJECT_X_RANGE[0] * 0.75,
        DEFAULT_OBJECT_X_RANGE[1] * 0.75,
        len(CYLINDER_COLORS),
    )
    y_center = float(sum(DEFAULT_OBJECT_Y_RANGE) / 2.0)
    return {
        color: {
            "body_name": CYLINDER_BODY_BY_COLOR[color],
            "x": float(x_values[idx]),
            "y": y_center,
            "yaw": 0.0,
        }
        for idx, color in enumerate(CYLINDER_COLORS)
    }


def sample_object_specs(
    rng: np.random.Generator,
    x_range: Tuple[float, float] = DEFAULT_OBJECT_X_RANGE,
    y_range: Tuple[float, float] = DEFAULT_OBJECT_Y_RANGE,
    yaw_range: Tuple[float, float] = DEFAULT_YAW_RANGE,
    min_distance: float = DEFAULT_MIN_OBJECT_DISTANCE,
    max_tries: int = 1000,
    target_color: Optional[str] = None,
    allowed_target_lanes: Optional[Tuple[int, ...]] = None,
) -> Dict[str, Dict[str, float]]:
    """Dataset collection code와 동일한 조건으로 물체를 원호 위에 배치한다."""
    if x_range[0] >= x_range[1] or y_range[0] >= y_range[1]:
        raise ValueError(f"잘못된 spawn range입니다: x_range={x_range}, y_range={y_range}")

    radius = float(sum(y_range) / 2.0)
    max_abs_x = float(min(abs(x_range[0]), abs(x_range[1])))
    if radius <= 0.0 or max_abs_x <= 0.0:
        raise ValueError(f"원호 배치를 만들 수 없는 spawn range입니다: x_range={x_range}, y_range={y_range}")

    angle_limit = float(min(0.55, math.asin(min(0.99, max_abs_x / radius))))
    if len(CYLINDER_COLORS) > 1:
        arc_spacing = 2.0 * radius * math.sin(angle_limit / float(len(CYLINDER_COLORS) - 1))
        if arc_spacing < min_distance:
            raise ValueError(
                f"원호 배치 간격이 너무 좁습니다: arc_spacing={arc_spacing:.3f}, "
                f"min_distance={min_distance}, x_range={x_range}, y_range={y_range}"
            )
        arc_angles = np.linspace(-angle_limit, angle_limit, len(CYLINDER_COLORS))
    else:
        arc_angles = np.array([0.0])

    if target_color is not None and allowed_target_lanes is not None:
        if target_color not in CYLINDER_COLORS:
            raise ValueError(f"지원하지 않는 target_color입니다: {target_color}")
        allowed_lane_indices = [int(lane) - 1 for lane in allowed_target_lanes]
        allowed_lane_indices = [idx for idx in allowed_lane_indices if 0 <= idx < len(CYLINDER_COLORS)]
        if not allowed_lane_indices:
            raise ValueError(f"allowed_target_lanes가 비어 있습니다: {allowed_target_lanes}")

        chosen_lane_idx = int(rng.choice(allowed_lane_indices))
        remaining_lane_indices = [idx for idx in range(len(CYLINDER_COLORS)) if idx != chosen_lane_idx]
        rng.shuffle(remaining_lane_indices)

        lane_order = []
        for color in CYLINDER_COLORS:
            if color == target_color:
                lane_order.append(chosen_lane_idx)
            else:
                lane_order.append(remaining_lane_indices.pop())
    else:
        lane_order = list(range(len(CYLINDER_COLORS)))
        rng.shuffle(lane_order)

    specs: Dict[str, Dict[str, float]] = {}
    for color, lane_idx in zip(CYLINDER_COLORS, lane_order):
        angle = float(arc_angles[lane_idx])
        specs[color] = {
            "body_name": CYLINDER_BODY_BY_COLOR[color],
            "x": float(radius * math.sin(angle)),
            "y": float(radius * math.cos(angle)),
            "yaw": float(rng.uniform(yaw_range[0], yaw_range[1])),
        }

    return {color: specs[color] for color in CYLINDER_COLORS}


def arc_lane_xy(
    lane_index: int,
    lane_count: int = 5,
    x_range: Tuple[float, float] = DEFAULT_OBJECT_X_RANGE,
    y_range: Tuple[float, float] = DEFAULT_OBJECT_Y_RANGE,
) -> Tuple[float, float]:
    lane_index = int(lane_index)
    lane_count = int(lane_count)
    if lane_index < 1 or lane_index > lane_count:
        raise ValueError(f"lane_index는 1~{lane_count} 범위여야 합니다: {lane_index}")

    radius = float(sum(y_range) / 2.0)
    max_abs_x = float(min(abs(x_range[0]), abs(x_range[1])))
    angle_limit = float(min(0.55, math.asin(min(0.99, max_abs_x / radius))))
    arc_angles = np.linspace(-angle_limit, angle_limit, lane_count) if lane_count > 1 else np.array([0.0])
    angle = float(arc_angles[lane_index - 1])
    return float(radius * math.sin(angle)), float(radius * math.cos(angle))


def make_pick_place_location_object_specs(
    rng: np.random.Generator,
    target_color: str = "red",
    source_lane: int = 2,
    lane_count: int = 5,
    x_range: Tuple[float, float] = DEFAULT_OBJECT_X_RANGE,
    y_range: Tuple[float, float] = DEFAULT_OBJECT_Y_RANGE,
    source_y_offset: float = 0.0,
) -> Dict[str, Dict[str, float]]:
    if target_color not in CYLINDER_BODY_BY_COLOR:
        raise ValueError(f"지원하지 않는 target_color입니다: {target_color}")
    x, y = arc_lane_xy(source_lane, lane_count=lane_count, x_range=x_range, y_range=y_range)
    y += float(source_y_offset)
    return {
        target_color: {
            "body_name": CYLINDER_BODY_BY_COLOR[target_color],
            "x": x,
            "y": y,
            "yaw": float(rng.uniform(DEFAULT_YAW_RANGE[0], DEFAULT_YAW_RANGE[1])),
        }
    }


def reset_freejoint_body_pose(env: SyncSimRaccoonEnv, body_name: str, x: float, y: float, z: float, yaw: float) -> None:
    """Set a MuJoCo freejoint body pose directly through env.model/env.data."""
    if not hasattr(env, "model") or not hasattr(env, "data"):
        raise AttributeError("SyncSimRaccoonEnv에 model/data 속성이 필요합니다.")

    body_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    if body_id == -1:
        raise ValueError(f"body not found: {body_name}. XML이 Raccoon_colored_cylinder.xml인지 확인하세요.")

    jnt_adr = int(env.model.body_jntadr[body_id])
    jnt_num = int(env.model.body_jntnum[body_id])
    if jnt_num < 1:
        raise ValueError(f"{body_name} has no joint")

    joint_id = jnt_adr
    qpos_adr = int(env.model.jnt_qposadr[joint_id])

    # freejoint qpos = [x, y, z, qw, qx, qy, qz]
    qw = math.cos(yaw / 2.0)
    qz = math.sin(yaw / 2.0)
    env.data.qpos[qpos_adr:qpos_adr + 7] = np.array([x, y, z, qw, 0.0, 0.0, qz], dtype=np.float64)

    qvel_adr = int(env.model.jnt_dofadr[joint_id])
    env.data.qvel[qvel_adr:qvel_adr + 6] = 0.0


def reset_multicolor_scene(
    env: SyncSimRaccoonEnv,
    object_specs: Dict[str, Dict[str, float]],
    target_color: str,
) -> None:
    """
    Reset the robot using the existing env.reset_episode(), then place all four
    colored cylinders in the scene. The prompted color is stored as env.active_object_body_name
    when the env supports that attribute, but inference only needs the rendered image.
    """
    if target_color not in object_specs:
        raise ValueError(f"target_color={target_color}가 object_specs에 없습니다.")

    target_spec = object_specs[target_color]

    # Existing raccoon_env expects a single target pose for reset_episode().
    # We use the prompted target pose to reset the robot/home state, then override
    # all four cylinder poses below.
    env.reset_episode(float(target_spec["x"]), float(target_spec["y"]), float(target_spec["yaw"]))

    hidden_idx = 0
    for color, body_name in CYLINDER_BODY_BY_COLOR.items():
        if color in object_specs:
            continue
        reset_freejoint_body_pose(
            env=env,
            body_name=body_name,
            x=0.45,
            y=-0.35 - 0.04 * hidden_idx,
            z=0.02,
            yaw=0.0,
        )
        hidden_idx += 1

    for color, spec in object_specs.items():
        reset_freejoint_body_pose(
            env=env,
            body_name=str(spec["body_name"]),
            x=float(spec["x"]),
            y=float(spec["y"]),
            z=0.02,
            yaw=float(spec["yaw"]),
        )

    target_body_name = str(target_spec["body_name"])
    if hasattr(env, "active_object_body_name"):
        env.active_object_body_name = target_body_name
    if hasattr(env, "target_body_name"):
        env.target_body_name = target_body_name

    mujoco.mj_forward(env.model, env.data)


def object_specs_to_meta(object_specs: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, Any]]:
    return {
        color: {
            "body_name": str(spec["body_name"]),
            "xy": [float(spec["x"]), float(spec["y"])],
            "yaw": float(spec["yaw"]),
        }
        for color, spec in object_specs.items()
    }


def write_rollout_meta(
    out_dir: Path,
    instruction: str,
    task_type: str,
    target_color: str,
    place_color: Optional[str],
    object_specs: Dict[str, Dict[str, float]],
    args: Dict[str, Any],
) -> None:
    meta = {
        "instruction": instruction,
        "task_type": task_type,
        "target_color": target_color,
        "target_body_name": CYLINDER_BODY_BY_COLOR[target_color],
        "all_object_init_poses": object_specs_to_meta(object_specs),
        "args": args,
    }
    if place_color is not None:
        meta["place_color"] = place_color
        meta["place_body_name"] = CYLINDER_BODY_BY_COLOR[place_color]
    with open(out_dir / "rollout_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)


def rollout(
    xml_path: str,
    server_url: str,
    instruction: Optional[str],
    unnorm_key: str,
    output_dir: str,
    episode_id: int = 1,
    max_steps: int = 1000000,
    use_viewer: bool = True,
    camera_name: str = "front_view",
    image_size: Tuple[int, int] = (256, 256),
    speed: int = 70,
    settle_seconds_per_action: float = 0.8,
    initial_settle_seconds: float = 0.3,
    delta_scale: float = 1.0,
    randomize_objects: bool = True,
    request_timeout: float = 60.0,
    max_delta_xyz: float = 0.005,
    target_color_arg: str = "auto",
    task_type_arg: str = "auto",
    instruction_template: str = DEFAULT_INSTRUCTION_TEMPLATE,
    seed: Optional[int] = None,
    object_x_range: Tuple[float, float] = DEFAULT_OBJECT_X_RANGE,
    object_y_range: Tuple[float, float] = DEFAULT_OBJECT_Y_RANGE,
    min_object_distance: float = DEFAULT_MIN_OBJECT_DISTANCE,
    gate_gripper_close: bool = False,
    close_xy_threshold: float = 0.018,
    close_z_threshold: float = 0.023,
    hold_gripper_after_close: bool = False,
    assist_lift_after_close: bool = False,
    lift_assist_min_dz: float = 0.006,
    lift_assist_until_z: float = 0.070,
    assist_pick_place_carry: bool = False,
    carry_goal_lane: int = 4,
    carry_move_start_z: float = 0.070,
    carry_min_xy_step: float = 0.010,
    carry_place_xy_threshold: float = 0.020,
    carry_place_release_z: float = 0.040,
    save_images: bool = True,
    frame_callback: Optional[Callable[[np.ndarray], None]] = None,
    preview_image_size: Optional[Tuple[int, int]] = None,
    status_callback: Optional[Callable[[str], None]] = None,
    phase_callback: Optional[Callable[[str], None]] = None,
    stop_event: Optional[threading.Event] = None,
) -> None:
    out_dir = Path(output_dir) / f"episode_{episode_id:06d}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 기존 이미지 삭제 후 새로 저장 시작
    clear_existing_images(out_dir)

    rng = np.random.default_rng(seed)
    target_color, place_color, task_type, instruction = resolve_target_color_and_instruction(
        instruction=instruction,
        target_color_arg=target_color_arg,
        task_type_arg=task_type_arg,
        rng=rng,
        instruction_template=instruction_template,
    )

    if task_type == "pick_place_location":
        object_specs = make_pick_place_location_object_specs(
            rng=rng,
            target_color=target_color,
            source_lane=2,
            lane_count=5,
            x_range=object_x_range,
            y_range=object_y_range,
        )
    elif randomize_objects:
        allowed_target_lanes = (2, 3, 4) if task_type in ("grasp", "lift") else None
        object_specs = sample_object_specs(
            rng=rng,
            x_range=object_x_range,
            y_range=object_y_range,
            min_distance=min_object_distance,
            target_color=target_color,
            allowed_target_lanes=allowed_target_lanes,
        )
    else:
        object_specs = make_default_object_specs()

    env = SyncSimRaccoonEnv(
        xml_path=xml_path,
        image_size=image_size,
        camera_name=camera_name,
        use_viewer=use_viewer,
    )

    try:
        reset_multicolor_scene(
            env=env,
            object_specs=object_specs,
            target_color=target_color,
        )

        env.lockh()
        env.debug_check_current_ee_reachable()

        # Dataset collector와 동일하게 첫 observation 전에 free-joint cylinder를 안정화한다.
        if initial_settle_seconds > 0:
            env.settle_steps(seconds=initial_settle_seconds)

        write_rollout_meta(
            out_dir=out_dir,
            instruction=instruction,
            task_type=task_type,
            target_color=target_color,
            place_color=place_color,
            object_specs=object_specs,
            args={
                "xml_path": xml_path,
                "unnorm_key": unnorm_key,
                "camera_name": camera_name,
                "image_size": list(image_size),
                "speed": speed,
                "settle_seconds_per_action": settle_seconds_per_action,
                "initial_settle_seconds": initial_settle_seconds,
                "delta_scale": delta_scale,
                "max_delta_xyz": max_delta_xyz,
                "seed": seed,
                "object_x_range": list(object_x_range),
                "object_y_range": list(object_y_range),
                "min_object_distance": min_object_distance,
                "task_type": task_type,
                "place_color": place_color,
                "gate_gripper_close": gate_gripper_close,
                "close_xy_threshold": close_xy_threshold,
                "close_z_threshold": close_z_threshold,
                "hold_gripper_after_close": hold_gripper_after_close,
                "assist_lift_after_close": assist_lift_after_close,
                "lift_assist_min_dz": lift_assist_min_dz,
                "lift_assist_until_z": lift_assist_until_z,
                "assist_pick_place_carry": assist_pick_place_carry,
                "carry_goal_lane": carry_goal_lane,
                "carry_move_start_z": carry_move_start_z,
                "carry_min_xy_step": carry_min_xy_step,
                "carry_place_xy_threshold": carry_place_xy_threshold,
                "carry_place_release_z": carry_place_release_z,
                "save_images": save_images,
            },
        )

        scene_log = (
            f"[SCENE] instruction={instruction!r} | task_type={task_type!r} | "
            f"target_color={target_color!r} | place_color={place_color!r} | "
            f"target_xy=({object_specs[target_color]['x']:.3f}, {object_specs[target_color]['y']:.3f}) | "
            f"objects={object_specs_to_meta(object_specs)}"
        )
        print(scene_log)
        if status_callback is not None:
            status_callback(scene_log)

        obs = env.get_observation()
        if frame_callback is not None:
            preview_image = env.render_rgb_size(preview_image_size) if preview_image_size is not None else obs["image"]
            frame_callback(preview_image)
        step_idx = 0
        close_latched = False
        release_latched = False
        current_phase = "grasp"
        if phase_callback is not None:
            phase_callback(current_phase)

        while True:
            if stop_event is not None and stop_event.is_set():
                print("[STOP] UI stop requested")
                if status_callback is not None:
                    status_callback("[STOP] UI stop requested")
                break

            response = request_action(
                server_url=server_url,
                instruction=instruction,
                image_rgb=obs["image"],
                unnorm_key=unnorm_key,
                timeout=request_timeout,
            )
            action = response["action"]
            action = gate_gripper_close_action(
                action=action,
                env=env,
                object_specs=object_specs,
            target_color=target_color,
            task_type=task_type,
            enabled=gate_gripper_close,
            close_latched=close_latched,
            max_xy_distance=close_xy_threshold,
            max_z=close_z_threshold,
        )
            close_allowed_this_step = len(action) >= 7 and float(action[6]) > 0.5
            if close_allowed_this_step:
                close_latched = True
            action = apply_gripper_hold_after_close(
                action=action,
                task_type=task_type,
                enabled=hold_gripper_after_close,
                close_latched=close_latched,
            )
            action = apply_lift_assist_after_close(
                action=action,
                env=env,
                task_type=task_type,
                enabled=assist_lift_after_close,
                close_latched=close_latched,
                min_dz=lift_assist_min_dz,
                until_z=lift_assist_until_z,
            )
            action = apply_pick_place_location_carry_assist(
                action=action,
                env=env,
                task_type=task_type,
                enabled=assist_pick_place_carry,
                close_latched=close_latched,
                goal_lane=carry_goal_lane,
                lane_count=5,
                move_start_z=carry_move_start_z,
                min_xy_step=carry_min_xy_step,
                place_xy_threshold=carry_place_xy_threshold,
                place_release_z=carry_place_release_z,
            )

            try:
                exec_info = env.execute_delta_action7(
                    action=action,
                    speed=speed,
                    delta_scale=delta_scale,
                    max_delta_xyz=max_delta_xyz,
                )
                phase, release_latched = estimate_task_phase(
                    task_type=task_type,
                    env=env,
                    current_phase=current_phase,
                    close_latched=close_latched,
                    release_latched=release_latched,
                    gripper_cmd=float(exec_info["gripper_cmd"]),
                    actual_move_xyz=exec_info["actual_move_xyz"],
                    carry_move_start_z=carry_move_start_z,
                )
                current_phase = phase
                if phase_callback is not None:
                    phase_callback(phase)
                print_success_log(step_idx, exec_info)
                if status_callback is not None:
                    final_delta_xyz = [round(float(v), 4) for v in exec_info["final_delta_xyz"]]
                    target_xyz = [round(float(v), 4) for v in exec_info["target_xyz"]]
                    gripper = float(exec_info["gripper_cmd"])
                    status_callback(
                        f"[{step_idx:03d}] OK | delta={final_delta_xyz} | "
                        f"target={target_xyz} | gripper={gripper:.1f}"
                    )

                env.settle_steps(seconds=settle_seconds_per_action)
                obs = env.get_observation()
                if frame_callback is not None:
                    preview_image = env.render_rgb_size(preview_image_size) if preview_image_size is not None else obs["image"]
                    frame_callback(preview_image)

                if save_images:
                    frame_name = f"frame_{step_idx:06d}.png"
                    Image.fromarray(obs["image"]).save(out_dir / frame_name)

            except Exception as exc:
                print_fail_log(step_idx, exc)
                if status_callback is not None:
                    status_callback(f"[{step_idx:03d}] FAIL | {exc}")
                obs = env.get_observation()
                if frame_callback is not None:
                    preview_image = env.render_rgb_size(preview_image_size) if preview_image_size is not None else obs["image"]
                    frame_callback(preview_image)

                if save_images:
                    frame_name = f"frame_{step_idx:06d}_skipped.png"
                    Image.fromarray(obs["image"]).save(out_dir / frame_name)

                step_idx += 1
                if step_idx >= max_steps:
                    print("[STOP] max_steps reached")
                    break
                continue

            step_idx += 1
            if step_idx >= max_steps:
                print("[STOP] max_steps reached")
                break

    except KeyboardInterrupt:
        print("\n[STOP] interrupted by user")

    finally:
        env.close()


def clear_existing_images(out_dir: Path) -> None:
    image_exts = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}

    deleted_count = 0
    for file_path in out_dir.iterdir():
        if file_path.is_file() and file_path.suffix.lower() in image_exts:
            file_path.unlink()
            deleted_count += 1

    print(f"[CLEANUP] removed {deleted_count} existing image files from {out_dir}")


def run_control_ui(args: argparse.Namespace, server_url: str) -> None:
    try:
        import tkinter as tk
        import tkinter.font as tkfont
        from tkinter import ttk
        from PIL import ImageTk
    except ImportError as exc:
        raise ImportError("UI 모드에는 tkinter와 Pillow ImageTk가 필요합니다.") from exc

    event_queue: queue.Queue[Tuple[str, Any]] = queue.Queue()
    stop_event = threading.Event()
    worker: Optional[threading.Thread] = None
    latest_photo: Optional[Any] = None
    panel_width = 430
    preview_size = 640

    root = tk.Tk()
    root.title("Raccoon OpenVLA Control")
    root.geometry(f"{preview_size + panel_width + 80}x780")
    default_font = tkfont.nametofont("TkDefaultFont")
    default_font.configure(size=12)
    text_font = tkfont.nametofont("TkTextFont")
    text_font.configure(size=12)
    fixed_font = tkfont.nametofont("TkFixedFont")
    fixed_font.configure(size=11)
    style = ttk.Style(root)
    style.configure("TLabel", font=default_font)
    style.configure("TButton", font=default_font, padding=(12, 6))
    style.configure("TRadiobutton", font=default_font)
    style.configure("TCombobox", font=default_font)
    style.configure("TEntry", font=default_font)
    root.columnconfigure(0, weight=1)
    root.columnconfigure(1, weight=0)
    root.rowconfigure(0, weight=1)

    preview_frame = ttk.Frame(root, padding=10)
    preview_frame.grid(row=0, column=0, sticky="nsew")
    preview_frame.columnconfigure(0, weight=1)
    preview_frame.rowconfigure(0, weight=1)

    black_image = Image.new("RGB", (preview_size, preview_size), color=(0, 0, 0))
    latest_photo = ImageTk.PhotoImage(black_image)
    preview_label = ttk.Label(preview_frame, image=latest_photo, anchor="center")
    preview_label.grid(row=0, column=0, sticky="nsew")

    panel = ttk.Frame(root, width=panel_width, padding=16)
    panel.grid(row=0, column=1, sticky="ns")
    panel.grid_propagate(False)
    panel.columnconfigure(0, weight=1)

    ttk.Label(panel, text="Task").grid(row=0, column=0, sticky="w")
    task_var = tk.StringVar(value=args.task_type if args.task_type != "auto" else "pick_place_location")
    task_box = ttk.Frame(panel)
    task_box.grid(row=1, column=0, sticky="ew", pady=(4, 12))
    for idx, task in enumerate(("grasp", "lift", "pick_place_location")):
        ttk.Radiobutton(task_box, text=task, value=task, variable=task_var).grid(row=idx, column=0, sticky="w")

    ttk.Label(panel, text="Target Color").grid(row=2, column=0, sticky="w")
    target_var = tk.StringVar(value=args.target_color if args.target_color not in ("auto", "random") else "red")
    target_combo = ttk.Combobox(panel, textvariable=target_var, values=CYLINDER_COLORS, state="readonly")
    target_combo.grid(row=3, column=0, sticky="ew", pady=(4, 12))

    ttk.Label(panel, text="Instruction").grid(row=4, column=0, sticky="w")
    instruction_var = tk.StringVar()
    instruction_entry = ttk.Entry(panel, textvariable=instruction_var, width=36)
    instruction_entry.grid(row=5, column=0, sticky="ew", pady=(4, 12))

    status_var = tk.StringVar(value="Select a task, edit instruction, then press Enter.")
    ttk.Label(panel, textvariable=status_var, wraplength=panel_width - 36).grid(row=6, column=0, sticky="ew", pady=(0, 12))

    log_frame = ttk.Frame(panel)
    log_frame.grid(row=7, column=0, sticky="nsew", pady=(0, 12))
    log_frame.columnconfigure(0, weight=1)
    log_frame.rowconfigure(0, weight=1)
    log_text = tk.Text(log_frame, width=42, height=18, state="disabled", wrap="none", font=fixed_font)
    log_text.grid(row=0, column=0, sticky="nsew")
    log_y = ttk.Scrollbar(log_frame, orient="vertical", command=log_text.yview)
    log_y.grid(row=0, column=1, sticky="ns")
    log_x = ttk.Scrollbar(log_frame, orient="horizontal", command=log_text.xview)
    log_x.grid(row=1, column=0, sticky="ew")
    log_text.configure(yscrollcommand=log_y.set, xscrollcommand=log_x.set)

    phase_var = tk.StringVar(value="Phase: idle")
    phase_label = ttk.Label(panel, textvariable=phase_var, font=("TkDefaultFont", 15, "bold"))
    phase_label.grid(row=8, column=0, sticky="ew", pady=(0, 12))

    button_box = ttk.Frame(panel)
    button_box.grid(row=9, column=0, sticky="ew")
    run_button = ttk.Button(button_box, text="Run")
    run_button.grid(row=0, column=0, padx=(0, 8))
    stop_button = ttk.Button(button_box, text="Stop")
    stop_button.grid(row=0, column=1)

    def append_log(message: str) -> None:
        log_text.configure(state="normal")
        log_text.insert("end", message + "\n")
        log_text.see("end")
        log_text.configure(state="disabled")
        status_var.set(message)

    def build_instruction() -> str:
        return make_instruction(
            task_type=task_var.get(),
            target_color=target_var.get(),
            instruction_template=args.instruction_template,
        )

    def refresh_instruction(*_: Any) -> None:
        if task_var.get() == "pick_place_location":
            target_var.set("red")
            target_combo.configure(state="disabled")
        else:
            target_combo.configure(state="readonly")
        instruction_var.set(build_instruction())

    def put_frame(image_rgb: np.ndarray) -> None:
        event_queue.put(("frame", image_rgb))

    def put_status(message: str) -> None:
        event_queue.put(("status", message))

    def put_phase(phase: str) -> None:
        event_queue.put(("phase", phase))

    def start_rollout(*_: Any) -> None:
        nonlocal worker
        if worker is not None and worker.is_alive():
            append_log("[UI] rollout already running")
            return

        stop_event.clear()
        run_button.configure(state="disabled")
        append_log("[UI] starting rollout")

        rollout_kwargs = dict(
            xml_path=args.xml_path,
            server_url=server_url,
            instruction=instruction_var.get().strip() or None,
            unnorm_key=args.unnorm_key,
            output_dir=args.output_dir,
            episode_id=args.episode_id,
            max_steps=args.max_steps,
            use_viewer=False,
            camera_name=args.camera_name,
            image_size=tuple(args.image_size),
            speed=args.speed,
            settle_seconds_per_action=args.settle_seconds_per_action,
            initial_settle_seconds=args.initial_settle_seconds,
            delta_scale=args.delta_scale,
            randomize_objects=not (args.no_randomize_box or args.no_randomize_objects),
            request_timeout=args.request_timeout,
            max_delta_xyz=args.max_delta_xyz,
            target_color_arg=target_var.get(),
            task_type_arg=task_var.get(),
            instruction_template=args.instruction_template,
            seed=args.seed,
            object_x_range=tuple(args.object_x_range),
            object_y_range=tuple(args.object_y_range),
            min_object_distance=args.min_object_distance,
            gate_gripper_close=args.gate_gripper_close,
            close_xy_threshold=args.close_xy_threshold,
            close_z_threshold=args.close_z_threshold,
            hold_gripper_after_close=args.hold_gripper_after_close,
            assist_lift_after_close=args.assist_lift_after_close,
            lift_assist_min_dz=args.lift_assist_min_dz,
            lift_assist_until_z=args.lift_assist_until_z,
            assist_pick_place_carry=args.assist_pick_place_carry,
            carry_goal_lane=args.carry_goal_lane,
            carry_move_start_z=args.carry_move_start_z,
            carry_min_xy_step=args.carry_min_xy_step,
            carry_place_xy_threshold=args.carry_place_xy_threshold,
            carry_place_release_z=args.carry_place_release_z,
            save_images=not args.no_save_images,
            frame_callback=put_frame,
            preview_image_size=tuple(args.ui_render_size),
            status_callback=put_status,
            phase_callback=put_phase,
            stop_event=stop_event,
        )

        def worker_main() -> None:
            try:
                rollout(**rollout_kwargs)
            except Exception as exc:
                put_status(f"[UI ERROR] {exc}")
            finally:
                event_queue.put(("done", None))

        worker = threading.Thread(target=worker_main, daemon=True)
        worker.start()

    def stop_rollout() -> None:
        stop_event.set()
        append_log("[UI] stop requested")

    def pump_events() -> None:
        nonlocal latest_photo
        try:
            while True:
                kind, payload = event_queue.get_nowait()
                if kind == "frame":
                    image = Image.fromarray(payload).resize((preview_size, preview_size), Image.Resampling.BILINEAR)
                    latest_photo = ImageTk.PhotoImage(image)
                    preview_label.configure(image=latest_photo)
                elif kind == "status":
                    append_log(str(payload))
                elif kind == "phase":
                    phase_var.set(f"Phase: {payload}")
                elif kind == "done":
                    run_button.configure(state="normal")
                    append_log("[UI] rollout finished")
        except queue.Empty:
            pass
        root.after(50, pump_events)

    def on_close() -> None:
        stop_event.set()
        root.destroy()

    for var in (task_var, target_var):
        var.trace_add("write", refresh_instruction)
    run_button.configure(command=start_rollout)
    stop_button.configure(command=stop_rollout)
    instruction_entry.bind("<Return>", start_rollout)
    root.protocol("WM_DELETE_WINDOW", on_close)

    refresh_instruction()
    instruction_entry.focus_set()
    pump_events()
    root.mainloop()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--xml_path", type=str, default="Raccoon_colored_cylinder.xml")
    parser.add_argument("--server_url", type=str, default=None, help="Direct HTTP URL, e.g. http://127.0.0.1:8000")
    parser.add_argument(
        "--instruction",
        type=str,
        default=None,
        help="OpenVLA prompt. If omitted, generated as 'grasp the {color} cylinder'.",
    )
    parser.add_argument(
        "--target_color",
        type=str,
        default="auto",
        choices=["auto", "random", *CYLINDER_COLORS],
        help="Target color. 'auto' uses the color in --instruction, or random if instruction has no color.",
    )
    parser.add_argument(
        "--task_type",
        type=str,
        default="auto",
        choices=["auto", "grasp", "lift", "pick_place_location"],
        help="Task type. 'auto' infers from --instruction or defaults to grasp.",
    )
    parser.add_argument("--instruction_template", type=str, default=DEFAULT_INSTRUCTION_TEMPLATE)
    parser.add_argument("--unnorm_key", type=str, default="raccoon_pick_place")
    parser.add_argument("--output_dir", type=str, default="rollout_outputs")
    parser.add_argument("--episode_id", type=int, default=1)
    parser.add_argument("--max_steps", type=int, default=1000000)
    parser.add_argument("--speed", type=int, default=70)
    parser.add_argument("--settle_seconds_per_action", type=float, default=0.8)
    parser.add_argument("--initial_settle_seconds", type=float, default=0.3)
    parser.add_argument("--delta_scale", type=float, default=1.0)
    parser.add_argument("--max_delta_xyz", type=float, default=0.005)
    parser.add_argument("--request_timeout", type=float, default=60.0)
    parser.add_argument("--use_viewer", action="store_true")
    parser.add_argument("--image_size", type=int, nargs=2, default=(256, 256), help="MuJoCo render size for non-UI mode: width height")
    parser.add_argument(
        "--use_ui",
        action="store_true",
        help="Launch a simple control UI. The left panel shows MuJoCo rendered frames and the right panel selects task/colors/instruction.",
    )
    parser.add_argument("--ui_render_size", type=int, nargs=2, default=(640, 640), help="MuJoCo render size for UI preview: width height")
    parser.add_argument("--no_save_images", action="store_true", help="Do not save rollout frame PNGs; faster for live inference.")
    parser.add_argument("--camera_name", type=str, default="front_view")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--object_x_range", type=float, nargs=2, default=DEFAULT_OBJECT_X_RANGE)
    parser.add_argument("--object_y_range", type=float, nargs=2, default=DEFAULT_OBJECT_Y_RANGE)
    parser.add_argument("--min_object_distance", type=float, default=DEFAULT_MIN_OBJECT_DISTANCE)
    parser.add_argument(
        "--gate_gripper_close",
        action="store_true",
        help="Delay close commands until the EE is near and low enough for grasp/lift.",
    )
    parser.add_argument("--close_xy_threshold", type=float, default=0.018)
    parser.add_argument("--close_z_threshold", type=float, default=0.023)
    parser.add_argument(
        "--hold_gripper_after_close",
        action="store_true",
        help="For lift, keep gripper closed after the first accepted close command.",
    )
    parser.add_argument(
        "--assist_lift_after_close",
        action="store_true",
        help="For lift, force a minimum upward dz after the first accepted close command.",
    )
    parser.add_argument("--lift_assist_min_dz", type=float, default=0.006)
    parser.add_argument("--lift_assist_until_z", type=float, default=0.070)
    parser.add_argument(
        "--assist_pick_place_carry",
        action="store_true",
        help="For pick_place_location, keep the wrist stable and assist motion toward the place lane after grasp.",
    )
    parser.add_argument("--carry_goal_lane", type=int, default=4)
    parser.add_argument("--carry_move_start_z", type=float, default=0.070)
    parser.add_argument("--carry_min_xy_step", type=float, default=0.010)
    parser.add_argument("--carry_place_xy_threshold", type=float, default=0.020)
    parser.add_argument("--carry_place_release_z", type=float, default=0.040)
    parser.add_argument(
        "--no_randomize_box",
        action="store_true",
        help="Legacy name. Disables randomization for all four colored cylinders.",
    )
    parser.add_argument(
        "--no_randomize_objects",
        action="store_true",
        help="Disables randomization for all four colored cylinders.",
    )

    parser.add_argument("--use_ssh_tunnel", action="store_true", help="Connect to the inference server through SSH local port forwarding")
    parser.add_argument("--ssh_host", type=str, default="qlak315.iptime.org")
    parser.add_argument("--ssh_port", type=int, default=24100)
    parser.add_argument("--ssh_user", type=str, default="root")
    parser.add_argument("--ssh_password", type=str, default=None, help="Prefer OPENVLA_SSH_PASSWORD or --ssh_ask_password")
    parser.add_argument("--ssh_ask_password", action="store_true", help="Prompt for the SSH password interactively")
    parser.add_argument("--remote_server_host", type=str, default="127.0.0.1")
    parser.add_argument("--remote_server_port", type=int, default=8000)
    parser.add_argument("--local_server_host", type=str, default="127.0.0.1")
    parser.add_argument("--local_server_port", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    with maybe_tunnel_context(args) as tunnel:
        server_url = build_server_url(args, tunnel)

        if tunnel is not None:
            print(
                f"[SSH] {args.local_server_host}:{tunnel.local_bind_port} -> "
                f"{args.remote_server_host}:{args.remote_server_port}"
            )

        if args.use_ui:
            run_control_ui(args=args, server_url=server_url)
            return

        rollout(
            xml_path=args.xml_path,
            server_url=server_url,
            instruction=args.instruction,
            unnorm_key=args.unnorm_key,
            output_dir=args.output_dir,
            episode_id=args.episode_id,
            max_steps=args.max_steps,
            use_viewer=args.use_viewer,
            camera_name=args.camera_name,
            image_size=tuple(args.image_size),
            speed=args.speed,
            settle_seconds_per_action=args.settle_seconds_per_action,
            initial_settle_seconds=args.initial_settle_seconds,
            delta_scale=args.delta_scale,
            randomize_objects=not (args.no_randomize_box or args.no_randomize_objects),
            request_timeout=args.request_timeout,
            max_delta_xyz=args.max_delta_xyz,
            target_color_arg=args.target_color,
            task_type_arg=args.task_type,
            instruction_template=args.instruction_template,
            seed=args.seed,
            object_x_range=tuple(args.object_x_range),
            object_y_range=tuple(args.object_y_range),
            min_object_distance=args.min_object_distance,
            gate_gripper_close=args.gate_gripper_close,
            close_xy_threshold=args.close_xy_threshold,
            close_z_threshold=args.close_z_threshold,
            hold_gripper_after_close=args.hold_gripper_after_close,
            assist_lift_after_close=args.assist_lift_after_close,
            lift_assist_min_dz=args.lift_assist_min_dz,
            lift_assist_until_z=args.lift_assist_until_z,
            assist_pick_place_carry=args.assist_pick_place_carry,
            carry_goal_lane=args.carry_goal_lane,
            carry_move_start_z=args.carry_move_start_z,
            carry_min_xy_step=args.carry_min_xy_step,
            carry_place_xy_threshold=args.carry_place_xy_threshold,
            carry_place_release_z=args.carry_place_release_z,
            save_images=not args.no_save_images,
        )


if __name__ == "__main__":
    main()
