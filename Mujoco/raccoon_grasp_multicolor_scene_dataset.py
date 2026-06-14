import os
import json
import math
import shutil
from pathlib import Path

import os
os.environ["MUJOCO_GL"] = "egl"

import mujoco
import mujoco.viewer
import numpy as np
from PIL import Image


class DatasetLogger:
    """
    Raw dataset logger.
    Saves:
      dataset_root/
        episode_000001/
          frame_000000.png
          frame_000001.png
          ...
          meta.json
    """
    def __init__(self, root_dir="dataset_raw", keep_failed=False):
        self.root_dir = Path(root_dir)
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self.keep_failed = keep_failed
        self.episode_dir = None
        self.meta = None

    def start_episode(
        self,
        episode_id,
        instruction,
        goal_xy,
        box_init_xy,
        box_init_yaw,
        task_type="pick",
        target_color=None,
        target_body_name=None,
        place_color=None,
        place_body_name=None,
        all_object_init_poses=None,
    ):
        episode_name = f"episode_{episode_id:06d}"
        self.episode_dir = self.root_dir / episode_name
        if self.episode_dir.exists():
            shutil.rmtree(self.episode_dir, ignore_errors=True)
        self.episode_dir.mkdir(parents=True, exist_ok=True)

        self.meta = {
            "episode_id": int(episode_id),
            "instruction": str(instruction),
            "task_type": str(task_type),
            # grasp-only에서는 별도 place goal이 없으므로 초기 box 위치를 goal_xy로 둔다.
            # 기존 intermediate/RLDS 변환 코드와 호환되도록 2차원 필드는 유지한다.
            "goal_xy": [float(goal_xy[0]), float(goal_xy[1])],
            "box_init_xy": [float(box_init_xy[0]), float(box_init_xy[1])],
            "box_init_yaw": float(box_init_yaw),
            "success": False,
            "steps": []
        }

        if target_color is not None:
            self.meta["target_color"] = str(target_color)
        if target_body_name is not None:
            self.meta["target_body_name"] = str(target_body_name)
        if place_color is not None:
            self.meta["place_color"] = str(place_color)
        if place_body_name is not None:
            self.meta["place_body_name"] = str(place_body_name)
        if all_object_init_poses is not None:
            self.meta["all_object_init_poses"] = all_object_init_poses

    def log_step(
        self,
        step_idx,
        image_rgb,
        joint_angles,
        gripper_state,
        object_pose,
        ee_pose,
        action,
        is_first=False,
        is_last=False,
    ):
        image_file = f"frame_{step_idx:06d}.png"
        image_path = self.episode_dir / image_file
        Image.fromarray(image_rgb).save(image_path)

        step_data = {
            "t": int(step_idx),
            "image_file": image_file,
            "joint_angles": [float(x) for x in joint_angles],
            "gripper_state": float(gripper_state),
            "object_pose": [float(x) for x in object_pose],
            "ee_pose": [float(x) for x in ee_pose],
            "action": [float(x) for x in action],
            "is_first": bool(is_first),
            "is_last": bool(is_last),
        }
        self.meta["steps"].append(step_data)

    def finalize_episode(self, success, exception_text=None):
        self.meta["success"] = bool(success)
        if exception_text is not None:
            self.meta["exception"] = str(exception_text)

        meta_path = self.episode_dir / "meta.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(self.meta, f, indent=2, ensure_ascii=False)

        if (not success) and (not self.keep_failed):
            shutil.rmtree(self.episode_dir, ignore_errors=True)

    def abort_episode(self):
        if self.episode_dir is not None and self.episode_dir.exists():
            shutil.rmtree(self.episode_dir, ignore_errors=True)


class SyncSimRaccoonDataset:
    """
    Synchronous MuJoCo dataset collector for RaccoonBot.

    Key design choices:
    - No background simulation thread
    - No real-time sleep-based settling
    - Main loop only: command -> run N mj_step -> render/save
    - Safe with viewer=False (physics still advances)
    """

    MAX_SPEEDS = [2.2, 2.3, 2.3, 2.3]
    GRIPPER_SPEED = 15.0

    # Uploaded move_to code style uses centimeter-scale IK constants.
    L1, L2, L3, L4 = 8.25, 10.0, 10.0, 8.0

    MODE_POSITION = 0
    MODE_VELOCITY = 1

    GRIP_OPEN = 0.15701
    GRIP_CLOSE = -0.85

    GRIP_MODE_FREE = 0
    GRIP_MODE_HORZ = 1
    GRIP_MODE_VERT = 2

    CYLINDER_BODY_BY_COLOR = {
        "red": "target_object",
        "blue": "target_object_blue",
        "green": "target_object_green",
        "yellow": "target_object_yellow",
        "white": "target_object_cube",
    }
    CYLINDER_COLORS = tuple(CYLINDER_BODY_BY_COLOR.keys())

    # Reachable front arc used when all colored objects are visible at once.
    # This keeps objects inside the arm workspace while leaving side-to-side
    # clearance so the gripper does not pass through other objects.
    DEFAULT_OBJECT_X_RANGE = (-0.11, 0.11)
    DEFAULT_OBJECT_Y_RANGE = (0.19, 0.21)
    DEFAULT_MIN_OBJECT_DISTANCE = 0.045

    def __init__(self, xml_path, image_size=(256, 256), camera_name=None, use_viewer=False):
        if not os.path.exists(xml_path):
            raise FileNotFoundError(f"xml 파일을 찾을 수 없습니다: {xml_path}")

        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        self.renderer = mujoco.Renderer(self.model, height=image_size[1], width=image_size[0])
        self.camera_name = camera_name
        self.use_viewer = use_viewer

        self.viewer = None
        if self.use_viewer:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)

        self.target_angles = [0.0] * 4
        self.current_setpoints = [0.0] * 5
        self.joint_velocities = [0.0] * 4
        self.joint_control_mode = [self.MODE_POSITION] * 4
        self.gripper_target = self.GRIP_OPEN
        self.gripper_mode = self.GRIP_MODE_FREE
        self.active_object_body_name = self.CYLINDER_BODY_BY_COLOR["red"]

        for i in range(4):
            self.joint_velocities[i] = self.MAX_SPEEDS[i] * 0.7

        # Initialize all colored cylinders in the scene. Dataset collection will
        # randomize these positions for every episode.
        self.reset_episode(
            object_specs=self.make_default_object_specs(),
            target_color="red",
        )

    # ---------- kinematics / commands ----------

    def _calc_inv_kinematics(self, x, y, z, pitch=None):
        """
        Inputs are in centimeters, matching the uploaded move_to code style.
        pitch is the gripper pitch in radians, where 0 keeps the gripper horizontal.
        Returns [j1, j2, j3, j4] in degrees.
        """
        if isinstance(x, (int, float)) and isinstance(y, (int, float)) and isinstance(z, (int, float)):
            if (-28.0 <= x <= 28.0) and (-15 <= y <= 28.0) and (0 <= z <= 36.25):
                x, y = y, -x
                th1 = math.atan2(y, x)
                c1 = math.cos(th1)
                s1 = math.sin(th1)
                x = x - self.L4 * c1
                y = y - self.L4 * s1
                zL1 = z - self.L1
                c3 = (x * x + y * y + zL1 * zL1 - self.L2 * self.L2 - self.L3 * self.L3) / (2 * self.L2 * self.L3)
                c32 = c3 * c3
                if c32 > 1:
                    c32 = 1
                s3 = -math.sqrt(1 - c32)
                th3 = math.atan2(s3, c3)
                M1 = c3 * self.L3 + self.L2
                M2 = z - self.L1
                M3 = s3 * self.L3
                M4 = c1 * x + s1 * y
                c2 = M1 * M2 - M3 * M4
                s2 = -M2 * M3 - M1 * M4
                th2 = math.atan2(s2, c2)
                th1 = math.degrees(th1)
                th2 = math.degrees(th2)
                th3 = math.degrees(th3)
                th4 = -(th2 + th3) - 90
                if pitch is not None:
                    th4 = math.degrees(float(pitch)) - th2 - th3 - 90

                if th1 < -120 or th1 > 120:
                    return None
                if th2 < -90 or th2 > 30:
                    return None
                if th3 < -150 or th3 > 0:
                    return None
                if pitch is not None and (th4 < -105 or th4 > 105):
                    return None

                return [th1, th2, th3, th4]
            return None
        return None

    def degree_to(self, joints, degrees, speed=70):
        j_list = joints if isinstance(joints, (list, tuple)) else [joints]
        d_list = degrees if isinstance(degrees, (list, tuple)) else [degrees]

        if len(d_list) == 1 and len(j_list) > 1:
            d_list = d_list * len(j_list)

        for j, deg in zip(j_list, d_list):
            idx = j - 1
            if 0 <= idx < 4:
                self.joint_control_mode[idx] = self.MODE_POSITION
                self.target_angles[idx] = np.radians(deg)
                percent = np.clip(speed, 0.0, 100.0)
                self.joint_velocities[idx] = (percent / 100.0) * self.MAX_SPEEDS[idx]

    def move_to(self, x_cm, y_cm, z_cm, speed=70, pitch=None):
        angles = self._calc_inv_kinematics(x_cm, y_cm, z_cm, pitch=pitch)
        if angles is None:
            raise ValueError(f"도달할 수 없는 좌표입니다: ({x_cm:.2f}, {y_cm:.2f}, {z_cm:.2f}) cm")
        if pitch is None:
            self.degree_to([1, 2, 3], angles[:3], speed)
        else:
            self.degree_to([1, 2, 3, 4], angles[:4], speed)

    def open_gripper(self):
        self.gripper_target = self.GRIP_OPEN

    def close_gripper(self):
        self.gripper_target = self.GRIP_CLOSE

    def lockh(self):
        self.gripper_mode = self.GRIP_MODE_HORZ

    def lockv(self):
        self.gripper_mode = self.GRIP_MODE_VERT

    def unlock(self):
        if self.gripper_mode != self.GRIP_MODE_FREE:
            self.target_angles[3] = self.data.qpos[3]
            self.gripper_mode = self.GRIP_MODE_FREE

    def execute_action(self, action, speed=70):
        """
        action = [target_x_m, target_y_m, target_z_m, gripper]
              or [target_x_m, target_y_m, target_z_m, target_pitch_rad, gripper]
        """
        if len(action) >= 5:
            target_x, target_y, target_z, target_pitch, gripper = action[:5]
            self.unlock()
        else:
            target_x, target_y, target_z, gripper = action[:4]
            target_pitch = None
            self.unlock()

        # move_to convention is centimeters.
        self.move_to(
            target_x * 100.0,
            target_y * 100.0,
            target_z * 100.0,
            speed=speed,
            pitch=target_pitch,
        )

        if gripper >= 0.5:
            self.close_gripper()
        else:
            self.open_gripper()

    # ---------- synchronous stepping ----------

    def _apply_controls_once(self):
        dt = self.model.opt.timestep

        for i in range(4):
            if i == 3 and self.gripper_mode != self.GRIP_MODE_FREE:
                base_angle = -(self.current_setpoints[1] + self.current_setpoints[2])
                if self.gripper_mode == self.GRIP_MODE_HORZ:
                    desired = base_angle - np.radians(90)
                else:
                    desired = base_angle - np.radians(180)

                error = desired - self.current_setpoints[i]
                speed_rad_s = self.MAX_SPEEDS[i]
                limit_step = speed_rad_s * dt
                step = np.clip(error, -limit_step, limit_step)
                self.current_setpoints[i] += step
            else:
                if self.joint_control_mode[i] == self.MODE_VELOCITY:
                    self.current_setpoints[i] += self.joint_velocities[i] * dt
                else:
                    error = self.target_angles[i] - self.current_setpoints[i]
                    if abs(error) > 1e-4:
                        max_step = abs(self.joint_velocities[i]) * dt
                        step_val = np.clip(error, -max_step, max_step)
                        self.current_setpoints[i] += step_val

            joint_id = self.model.actuator_trnid[i, 0]
            rng = self.model.jnt_range[joint_id]
            self.current_setpoints[i] = np.clip(self.current_setpoints[i], rng[0], rng[1])
            self.data.ctrl[i] = self.current_setpoints[i]

        # Gripper stop-on-contact logic from uploaded code.
        try:
            touch_L = self.data.sensor("sensor_L").data[0]
            touch_R = self.data.sensor("sensor_R").data[0]
            is_touched = (touch_L > 0.1) and (touch_R > 0.1)
        except Exception:
            is_touched = False

        if self.gripper_target == self.GRIP_CLOSE and is_touched:
            self.gripper_target = self.data.qpos[4] - 0.028

        g_err = self.gripper_target - self.current_setpoints[4]
        if abs(g_err) > 1e-4:
            g_step = self.GRIPPER_SPEED * dt
            g_move = np.clip(g_err, -g_step, g_step)
            self.current_setpoints[4] += g_move

        self.data.ctrl[4] = self.current_setpoints[4]

    def step_n(self, n_steps):
        for _ in range(int(n_steps)):
            self._apply_controls_once()
            mujoco.mj_step(self.model, self.data)
            if self.viewer is not None and self.viewer.is_running():
                self.viewer.sync()

    def steps_for_seconds(self, seconds):
        return max(1, int(round(seconds / self.model.opt.timestep)))

    def settle_steps(self, seconds=2.0):
        self.step_n(self.steps_for_seconds(seconds))

    # ---------- rendering / state ----------

    def get_robot_state(self):
        joint_angles = [float(self.data.qpos[i]) for i in range(4)]
        gripper_state = float(self.data.qpos[4])
        return {
            "joint_angles": joint_angles,
            "gripper_state": gripper_state
        }

    @staticmethod
    def ee_pitch_from_joint_angles(joint_angles):
        return float(joint_angles[1] + joint_angles[2] + joint_angles[3] + math.pi / 2.0)

    def get_object_pose(self, body_name="target_object"):
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if body_id == -1:
            raise ValueError(f"body not found: {body_name}")

        pos = self.data.xpos[body_id].copy()
        xmat = self.data.xmat[body_id].reshape(3, 3).copy()
        yaw = math.atan2(xmat[1, 0], xmat[0, 0])

        return np.array([pos[0], pos[1], pos[2], yaw], dtype=np.float32)

    def render_rgb(self):
        cam_id = self.camera_name if self.camera_name is not None else -1
        self.renderer.update_scene(self.data, camera=cam_id)
        image = self.renderer.render()
        return image.copy()

    def get_observation(self, object_body_name=None):
        if object_body_name is None:
            object_body_name = self.active_object_body_name

        rs = self.get_robot_state()
        obj = self.get_object_pose(object_body_name)
        img = self.render_rgb()

        # EE pose: Link4 position plus gripper pitch.
        link4_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "Link4")
        if link4_id != -1:
            ee_pos = self.data.xpos[link4_id].copy()
            ee_pose_list = [
                float(ee_pos[0]),
                float(ee_pos[1]),
                float(ee_pos[2]),
                self.ee_pitch_from_joint_angles(rs["joint_angles"]),
            ]
        else:
            ee_pose_list = [0.0, 0.0, 0.0, 0.0]

        return {
            "image": img,
            "joint_angles": rs["joint_angles"],
            "gripper_state": rs["gripper_state"],
            "object_pose": obj,
            "ee_pose": ee_pose_list,
        }

    # ---------- reset / success ----------

    def reset_object_pose(self, body_name="target_object", x=0.15, y=0.15, z=0.02, yaw=0.0):
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if body_id == -1:
            raise ValueError(f"body not found: {body_name}")

        jnt_adr = self.model.body_jntadr[body_id]
        jnt_num = self.model.body_jntnum[body_id]
        if jnt_num < 1:
            raise ValueError(f"{body_name} has no joint")

        joint_id = jnt_adr
        qpos_adr = self.model.jnt_qposadr[joint_id]

        # freejoint qpos = [x, y, z, qw, qx, qy, qz]
        qw = math.cos(yaw / 2.0)
        qz = math.sin(yaw / 2.0)
        self.data.qpos[qpos_adr:qpos_adr + 7] = np.array([x, y, z, qw, 0.0, 0.0, qz], dtype=np.float64)

        # Zero object joint velocities if present.
        qvel_adr = self.model.jnt_dofadr[joint_id]
        self.data.qvel[qvel_adr:qvel_adr + 6] = 0.0

    @classmethod
    def make_default_object_specs(cls):
        """
        Deterministic fallback placement for initialization only.
        Dataset collection uses sample_object_specs() for randomized positions.
        """
        x_values = np.linspace(
            cls.DEFAULT_OBJECT_X_RANGE[0] * 0.75,
            cls.DEFAULT_OBJECT_X_RANGE[1] * 0.75,
            len(cls.CYLINDER_COLORS),
        )
        y_center = float(sum(cls.DEFAULT_OBJECT_Y_RANGE) / 2.0)
        return {
            color: {
                "body_name": cls.CYLINDER_BODY_BY_COLOR[color],
                "x": float(x_values[idx]),
                "y": y_center,
                "yaw": 0.0,
            }
            for idx, color in enumerate(cls.CYLINDER_COLORS)
        }

    @classmethod
    def sample_object_specs(
        cls,
        rng,
        colors=None,
        x_range=None,
        y_range=None,
        yaw_range=(-np.pi / 4, np.pi / 4),
        min_distance=None,
        max_tries=1000,
    ):
        """Place objects on a reachable front arc so the gripper path stays clear."""
        colors = tuple(colors or cls.CYLINDER_COLORS)
        x_range = x_range or cls.DEFAULT_OBJECT_X_RANGE
        y_range = y_range or cls.DEFAULT_OBJECT_Y_RANGE
        min_distance = cls.DEFAULT_MIN_OBJECT_DISTANCE if min_distance is None else min_distance

        if len(colors) == 0:
            raise ValueError("colors는 비어 있을 수 없습니다.")
        if x_range[0] >= x_range[1] or y_range[0] >= y_range[1]:
            raise ValueError(f"잘못된 spawn range입니다: x_range={x_range}, y_range={y_range}")

        radius = float(sum(y_range) / 2.0)
        max_abs_x = float(min(abs(x_range[0]), abs(x_range[1])))
        if radius <= 0.0 or max_abs_x <= 0.0:
            raise ValueError(f"원호 배치를 만들 수 없는 spawn range입니다: x_range={x_range}, y_range={y_range}")

        angle_limit = float(min(0.55, math.asin(min(0.99, max_abs_x / radius))))
        if len(colors) > 1:
            arc_spacing = 2.0 * radius * math.sin(angle_limit / float(len(colors) - 1))
            if arc_spacing < min_distance:
                raise ValueError(
                    f"원호 배치 간격이 너무 좁습니다: arc_spacing={arc_spacing:.3f}, "
                    f"min_distance={min_distance}, x_range={x_range}, y_range={y_range}"
                )
            arc_angles = np.linspace(-angle_limit, angle_limit, len(colors))
        else:
            arc_angles = np.array([0.0])

        lane_order = list(range(len(colors)))
        rng.shuffle(lane_order)
        specs = {}
        for color, lane_idx in zip(colors, lane_order):
            if color not in cls.CYLINDER_BODY_BY_COLOR:
                raise ValueError(f"지원하지 않는 색상입니다: {color}")

            angle = float(arc_angles[lane_idx])
            x = float(radius * math.sin(angle))
            y = float(radius * math.cos(angle))

            specs[color] = {
                "body_name": cls.CYLINDER_BODY_BY_COLOR[color],
                "x": x,
                "y": y,
                "yaw": float(rng.uniform(yaw_range[0], yaw_range[1])),
            }

        xy_values = np.array([[spec["x"], spec["y"]] for spec in specs.values()], dtype=np.float64)
        for i in range(len(xy_values)):
            for j in range(i + 1, len(xy_values)):
                dist = float(np.linalg.norm(xy_values[i] - xy_values[j]))
                if dist < min_distance:
                    raise ValueError(
                        f"원호 배치 후 물체 간격이 너무 좁습니다: distance={dist:.3f}, "
                        f"min_distance={min_distance}"
                    )

        # Return in canonical color order for stable metadata.
        return {color: specs[color] for color in colors}

    @classmethod
    def arc_lane_xy(cls, lane_index, lane_count=5, x_range=None, y_range=None):
        """Return a fixed front-arc lane position. lane_index is 1-based, left to right."""
        x_range = x_range or cls.DEFAULT_OBJECT_X_RANGE
        y_range = y_range or cls.DEFAULT_OBJECT_Y_RANGE
        lane_index = int(lane_index)
        lane_count = int(lane_count)
        if lane_count < 1:
            raise ValueError("lane_count는 1 이상이어야 합니다.")
        if lane_index < 1 or lane_index > lane_count:
            raise ValueError(f"lane_index는 1~{lane_count} 범위여야 합니다: {lane_index}")

        radius = float(sum(y_range) / 2.0)
        max_abs_x = float(min(abs(x_range[0]), abs(x_range[1])))
        angle_limit = float(min(0.55, math.asin(min(0.99, max_abs_x / radius))))
        arc_angles = np.linspace(-angle_limit, angle_limit, lane_count) if lane_count > 1 else np.array([0.0])
        angle = float(arc_angles[lane_index - 1])
        return float(radius * math.sin(angle)), float(radius * math.cos(angle))

    @classmethod
    def fixed_lane_object_specs(
        cls,
        rng,
        lane_by_color,
        lane_count=5,
        x_range=None,
        y_range=None,
        yaw_range=(-np.pi / 4, np.pi / 4),
    ):
        specs = {}
        for color, lane_index in lane_by_color.items():
            if color not in cls.CYLINDER_BODY_BY_COLOR:
                raise ValueError(f"지원하지 않는 색상입니다: {color}")
            x, y = cls.arc_lane_xy(
                lane_index=lane_index,
                lane_count=lane_count,
                x_range=x_range,
                y_range=y_range,
            )
            specs[color] = {
                "body_name": cls.CYLINDER_BODY_BY_COLOR[color],
                "x": x,
                "y": y,
                "yaw": float(rng.uniform(yaw_range[0], yaw_range[1])),
            }
        return specs

    @staticmethod
    def specs_to_meta(object_specs):
        return {
            color: {
                "body_name": str(spec["body_name"]),
                "xy": [float(spec["x"]), float(spec["y"])],
                "yaw": float(spec["yaw"]),
            }
            for color, spec in object_specs.items()
        }

    def reset_colored_objects(self, object_specs, target_color):
        """
        Place every colored cylinder in the scene. The target color controls
        which body is used for object_pose logging and grasp trajectory target.
        """
        if target_color not in object_specs:
            raise ValueError(f"target_color={target_color}가 object_specs에 없습니다.")

        self.active_object_body_name = object_specs[target_color]["body_name"]

        hidden_idx = 0
        for color, body_name in self.CYLINDER_BODY_BY_COLOR.items():
            if color in object_specs:
                continue
            body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
            if body_id == -1:
                continue
            self.reset_object_pose(
                body_name,
                x=0.45,
                y=-0.35 - 0.04 * hidden_idx,
                z=0.02,
                yaw=0.0,
            )
            hidden_idx += 1

        for color, spec in object_specs.items():
            body_name = spec["body_name"]
            body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
            if body_id == -1:
                raise ValueError(f"body not found for color '{color}': {body_name}")

            self.reset_object_pose(
                body_name,
                x=spec["x"],
                y=spec["y"],
                z=0.02,
                yaw=spec["yaw"],
            )

    def reset_episode(self, object_specs, target_color="red"):
        home = np.radians([0.0, -10.0, -140.0, 60.0])

        for i in range(4):
            self.data.qpos[i] = home[i]
            self.data.ctrl[i] = home[i]
            self.current_setpoints[i] = home[i]
            self.target_angles[i] = home[i]
            self.joint_control_mode[i] = self.MODE_POSITION

        self.data.qvel[:] = 0.0

        self.data.qpos[4] = self.GRIP_OPEN
        self.data.ctrl[4] = self.GRIP_OPEN
        self.current_setpoints[4] = self.GRIP_OPEN
        self.gripper_target = self.GRIP_OPEN
        self.gripper_mode = self.GRIP_MODE_FREE

        self.reset_colored_objects(object_specs=object_specs, target_color=target_color)
        mujoco.mj_forward(self.model, self.data)

        # Short stabilization after reset.
        self.step_n(20)

    def get_gripper_touch_state(self):
        """
        Return whether the left/right gripper touch sensors are in contact.
        If the XML does not expose these sensors, this returns False for both sides.
        """
        try:
            touch_l = float(self.data.sensor("sensor_L").data[0])
            touch_r = float(self.data.sensor("sensor_R").data[0])
        except Exception:
            touch_l = 0.0
            touch_r = 0.0

        return touch_l, touch_r

    def is_grasp_success(self, touch_threshold=0.1, require_closed=True):
        """
        Grasp-only success criterion.
        The episode is considered successful when both gripper touch sensors detect contact.
        Optionally also require the gripper to have moved away from its fully-open position.
        """
        touch_l, touch_r = self.get_gripper_touch_state()
        both_touched = (touch_l > touch_threshold) and (touch_r > touch_threshold)

        if not require_closed:
            return bool(both_touched)

        # Make sure this is not just an accidental touch while the gripper is still fully open.
        gripper_is_closing_or_closed = float(self.data.qpos[4]) < (self.GRIP_OPEN - 0.01)
        return bool(both_touched and gripper_is_closing_or_closed)

    def is_body_touching_robot(self, body_name, ignored_geom_names=("floor",)):
        """
        Return True when the requested object body is in contact with a non-floor,
        non-cylinder body. This makes success target-specific when all four
        colored cylinders are present: touching the wrong color does not count.
        """
        target_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if target_body_id == -1:
            raise ValueError(f"body not found: {body_name}")

        cylinder_body_ids = set()
        for cylinder_body_name in self.CYLINDER_BODY_BY_COLOR.values():
            cylinder_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, cylinder_body_name)
            if cylinder_body_id != -1:
                cylinder_body_ids.add(cylinder_body_id)

        ignored_geom_names = set(ignored_geom_names or [])

        for contact_idx in range(int(self.data.ncon)):
            contact = self.data.contact[contact_idx]
            geom1 = int(contact.geom1)
            geom2 = int(contact.geom2)
            body1 = int(self.model.geom_bodyid[geom1])
            body2 = int(self.model.geom_bodyid[geom2])

            if target_body_id not in (body1, body2):
                continue

            other_geom = geom2 if body1 == target_body_id else geom1
            other_body = body2 if body1 == target_body_id else body1

            other_geom_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, other_geom) or ""
            if other_geom_name in ignored_geom_names:
                continue

            # Do not count target-object contact with another colored cylinder
            # as a grasp. We only want contacts against the robot/gripper.
            if other_body in cylinder_body_ids:
                continue

            return True

        return False

    def is_target_grasp_success(self, target_body_name, touch_threshold=0.1, require_closed=True):
        """
        Success for the multi-cylinder scene. Both gripper touch sensors must be
        active, the gripper must be closing/closed, and the prompted target body
        must be the object contacting the robot.
        """
        return bool(
            self.is_grasp_success(touch_threshold=touch_threshold, require_closed=require_closed)
            and self.is_body_touching_robot(target_body_name)
        )

    def is_target_lift_success(
        self,
        target_body_name,
        initial_object_z,
        lift_height=0.05,
        min_lift_ratio=0.6,
        touch_threshold=0.1,
    ):
        object_pose = self.get_object_pose(target_body_name)
        lifted_enough = float(object_pose[2]) >= float(initial_object_z) + float(lift_height) * float(min_lift_ratio)
        return bool(
            lifted_enough
            and self.is_target_grasp_success(
                target_body_name=target_body_name,
                touch_threshold=touch_threshold,
            )
        )

    def is_target_place_success(
        self,
        target_body_name,
        place_body_name,
        xy_threshold=0.035,
        min_height_above=0.012,
    ):
        target_pose = self.get_object_pose(target_body_name)
        place_pose = self.get_object_pose(place_body_name)
        xy_close = float(np.linalg.norm(target_pose[:2] - place_pose[:2])) <= float(xy_threshold)
        above_place_object = float(target_pose[2]) >= float(place_pose[2]) + float(min_height_above)
        gripper_is_open = float(self.data.qpos[4]) > (self.GRIP_OPEN - 0.03)
        return bool(xy_close and above_place_object and gripper_is_open)

    def is_target_place_location_success(
        self,
        target_body_name,
        place_xy,
        xy_threshold=0.035,
        max_resting_z=0.030,
    ):
        target_pose = self.get_object_pose(target_body_name)
        xy_close = float(np.linalg.norm(target_pose[:2] - np.asarray(place_xy, dtype=np.float64))) <= float(xy_threshold)
        resting_low = float(target_pose[2]) <= float(max_resting_z)
        gripper_is_open = float(self.data.qpos[4]) > (self.GRIP_OPEN - 0.03)
        return bool(xy_close and resting_low and gripper_is_open)

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None

    # ---------- grasp-only plan ----------

    @staticmethod
    def grasp_z_for_color(target_color):
        # The cube tends to be held more securely when the gripper closes a bit
        # lower on the body instead of near the top edge.
        return 0.018 if target_color == "white" else 0.020

    @staticmethod
    def _minimum_jerk(t):
        t = float(np.clip(t, 0.0, 1.0))
        return 10.0 * t**3 - 15.0 * t**4 + 6.0 * t**5

    @staticmethod
    def _action_xyz(action):
        return np.array([float(action[0]), float(action[1]), float(action[2])], dtype=np.float64)

    @staticmethod
    def _action_gripper(action):
        return float(action[-1])

    @classmethod
    def interpolate_plan(cls, plan, max_cartesian_step=0.008, min_segment_steps=2):
        """
        Densify Cartesian waypoint plans with minimum-jerk interpolation.

        Open/close waypoints are kept discrete. Move waypoints are split into
        small Cartesian targets so the logged EE deltas are smoother and less
        dominated by slow/fast transients near the object.
        """
        if not plan:
            return plan

        interpolated = [plan[0]]
        for start, end in zip(plan[:-1], plan[1:]):
            start_gripper = cls._action_gripper(start)
            end_gripper = cls._action_gripper(end)
            start_xyz = cls._action_xyz(start)
            end_xyz = cls._action_xyz(end)
            distance = float(np.linalg.norm(end_xyz - start_xyz))

            gripper_changes = abs(end_gripper - start_gripper) >= 0.5
            if gripper_changes or distance <= 1e-6:
                interpolated.append(end)
                continue

            steps = max(int(min_segment_steps), int(math.ceil(distance / float(max_cartesian_step))))
            for step_idx in range(1, steps + 1):
                alpha = cls._minimum_jerk(step_idx / float(steps))
                xyz = start_xyz + (end_xyz - start_xyz) * alpha

                # Preserve pitch only for segments where both endpoints carry it.
                if len(start) >= 5 and len(end) >= 5:
                    pitch = float(start[3]) + (float(end[3]) - float(start[3])) * alpha
                    interp_action = [float(xyz[0]), float(xyz[1]), float(xyz[2]), pitch, end_gripper]
                elif len(end) >= 5:
                    interp_action = [float(xyz[0]), float(xyz[1]), float(xyz[2]), float(end[3]), end_gripper]
                else:
                    interp_action = [float(xyz[0]), float(xyz[1]), float(xyz[2]), end_gripper]
                interpolated.append(interp_action)

        return interpolated

    def make_grasp_plan(self, box_x, box_y, target_color=None):
        z_above = 0.10
        z_grasp = self.grasp_z_for_color(target_color)
        pitch_grasp = 0.0

        return [
            [box_x, box_y, z_above, 0],                # Move above object without forcing pitch.
            [box_x, box_y, z_grasp, pitch_grasp, 0],   # Move down to grasp height.
            [box_x, box_y, z_grasp, pitch_grasp, 1],   # Close gripper and finish once grasped.
        ]

    def make_lift_plan(self, box_x, box_y, lift_height=0.05, target_color=None):
        z_above = 0.10
        z_grasp = self.grasp_z_for_color(target_color)
        z_lift = min(z_above, z_grasp + float(lift_height))
        pitch_grasp = 0.0

        return [
            [box_x, box_y, z_above, 0],                # Move above object without forcing pitch.
            [box_x, box_y, z_grasp, pitch_grasp, 0],   # Move down to grasp height.
            [box_x, box_y, z_grasp, pitch_grasp, 1],   # Close gripper on the object.
            [box_x, box_y, z_lift, pitch_grasp, 1],    # Lift first while staying horizontal to avoid floor contact.
            [box_x, box_y, z_lift, 1],                 # Hold lifted object without forcing a new pitch target.
        ]

    def make_post_grasp_lift_plans(self, box_x, box_y, lift_height=0.05, target_color=None):
        z_above = 0.10
        z_grasp = self.grasp_z_for_color(target_color)
        z_lift = min(z_above, z_grasp + float(lift_height))
        pitch_grasp = 0.0

        pre_grasp_plan = [
            [box_x, box_y, z_above, 0],
            [box_x, box_y, z_grasp, pitch_grasp, 0],
            [box_x, box_y, z_grasp, pitch_grasp, 1],
        ]
        logged_lift_plan = [
            [box_x, box_y, z_lift, pitch_grasp, 1],
            [box_x, box_y, z_lift, 1],
            [box_x, box_y, z_lift, 1],
        ]
        return pre_grasp_plan, logged_lift_plan

    def make_pick_place_plan(self, pick_x, pick_y, place_x, place_y, lift_height=0.05, target_color=None):
        z_above = 0.10
        z_grasp = self.grasp_z_for_color(target_color)
        z_lift = min(z_above, z_grasp + float(lift_height))
        z_place = 0.040
        pitch_grasp = 0.0

        return [
            [pick_x, pick_y, z_above, 0],                  # Approach source without forcing pitch.
            [pick_x, pick_y, z_grasp, pitch_grasp, 0],     # Move down with gripper horizontal.
            [pick_x, pick_y, z_grasp, pitch_grasp, 1],     # Close on source object.
            [pick_x, pick_y, z_lift, pitch_grasp, 1],      # Lift source while staying horizontal.
            [place_x, place_y, z_lift, pitch_grasp, 1],    # Carry with stable horizontal pitch.
            [place_x, place_y, z_place, pitch_grasp, 1],   # Lower onto place target while horizontal.
            [place_x, place_y, z_place, pitch_grasp, 0],   # Open gripper to release.
            [place_x, place_y, z_lift, pitch_grasp, 0],    # Retreat upward while keeping wrist stable.
        ]

    def make_pick_place_location_plan(self, pick_x, pick_y, place_x, place_y, lift_height=0.05, target_color=None):
        z_above = 0.10
        z_grasp = self.grasp_z_for_color(target_color)
        z_lift = min(z_above, z_grasp + float(lift_height))
        pitch_grasp = 0.0

        return [
            [pick_x, pick_y, z_above, 0],
            [pick_x, pick_y, z_grasp, pitch_grasp, 0],
            [pick_x, pick_y, z_grasp, pitch_grasp, 1],
            [pick_x, pick_y, z_lift, pitch_grasp, 1],
            [place_x, place_y, z_lift, pitch_grasp, 1],
            [place_x, place_y, z_grasp, pitch_grasp, 1],
            [place_x, place_y, z_grasp, pitch_grasp, 0],
            [place_x, place_y, z_lift, pitch_grasp, 0],
        ]

    def make_pick_place_location_stage_plans(
        self,
        stage,
        pick_x,
        pick_y,
        place_x,
        place_y,
        lift_height=0.05,
        target_color=None,
    ):
        z_above = 0.10
        z_grasp = self.grasp_z_for_color(target_color)
        z_lift = min(z_above, z_grasp + float(lift_height))
        pitch_grasp = 0.0

        approach_and_grasp = [
            [pick_x, pick_y, z_above, 0],
            [pick_x, pick_y, z_grasp, pitch_grasp, 0],
            [pick_x, pick_y, z_grasp, pitch_grasp, 1],
        ]
        lift = [
            [pick_x, pick_y, z_lift, pitch_grasp, 1],
            [pick_x, pick_y, z_lift, pitch_grasp, 1],
        ]
        move_above_place = [
            [place_x, place_y, z_lift, pitch_grasp, 1],
        ]
        lower_to_place = [
            [place_x, place_y, z_grasp, pitch_grasp, 1],
            [place_x, place_y, z_grasp, pitch_grasp, 1],
        ]
        release = [
            [place_x, place_y, z_grasp, pitch_grasp, 0],
        ]
        retreat = [
            [place_x, place_y, z_lift, pitch_grasp, 0],
        ]

        if stage == "post_grasp_lift":
            return approach_and_grasp, lift
        if stage == "post_lift_move":
            return approach_and_grasp + lift, move_above_place
        if stage == "post_move_lower":
            return approach_and_grasp + lift + move_above_place, lower_to_place
        if stage == "post_place_retreat":
            return approach_and_grasp + lift + move_above_place + lower_to_place + release, retreat
        raise ValueError(f"지원하지 않는 pick_place_location stage입니다: {stage}")


def run_episode_and_record(
    rc: SyncSimRaccoonDataset,
    logger: DatasetLogger,
    episode_id: int,
    instruction: str,
    object_specs: dict,
    target_color: str = "red",
    place_color: str = None,
    place_xy=None,
    task_type: str = "grasp",
    lift_height: float = 0.05,
    speed: int = 70,
    settle_seconds_per_action: float = 2.0,
    interpolated_settle_seconds_per_action: float = 0.12,
    initial_settle_seconds: float = 0.3,
    hz: int = 10,
    touch_threshold: float = 0.1,
    interpolate_motion: bool = False,
    interpolation_max_step: float = 0.008,
    interpolation_min_segment_steps: int = 2,
):
    pick_place_location_stages = {
        "pick_place_location_post_grasp_lift": "post_grasp_lift",
        "pick_place_location_post_lift_move": "post_lift_move",
        "pick_place_location_post_move_lower": "post_move_lower",
        "pick_place_location_post_place_retreat": "post_place_retreat",
    }
    is_pick_place_location_task = task_type == "pick_place_location" or task_type in pick_place_location_stages

    if target_color not in object_specs:
        raise ValueError(f"target_color={target_color}가 object_specs에 없습니다.")

    target_spec = object_specs[target_color]
    target_body_name = target_spec["body_name"]
    target_x = float(target_spec["x"])
    target_y = float(target_spec["y"])
    target_yaw = float(target_spec["yaw"])
    place_spec = None
    place_body_name = None
    place_x = target_x
    place_y = target_y

    if task_type == "pick_place":
        if place_color is None:
            raise ValueError("pick_place task에는 place_color가 필요합니다.")
        if place_color == target_color:
            raise ValueError("pick_place task에서 target_color와 place_color는 달라야 합니다.")
        if place_color not in object_specs:
            raise ValueError(f"place_color={place_color}가 object_specs에 없습니다.")
        place_spec = object_specs[place_color]
        place_body_name = place_spec["body_name"]
        place_x = float(place_spec["x"])
        place_y = float(place_spec["y"])
    elif is_pick_place_location_task:
        if place_xy is None:
            raise ValueError(f"{task_type} task에는 place_xy가 필요합니다.")
        place_x = float(place_xy[0])
        place_y = float(place_xy[1])

    rc.reset_episode(object_specs=object_specs, target_color=target_color)
    rc.unlock()

    # Let newly reset free-joint cylinders fall/settle before capturing frame_000000.
    # Without this, the first saved image can show cylinders slightly floating while
    # later frames look normal after one physics step.
    if initial_settle_seconds > 0:
        rc.settle_steps(seconds=initial_settle_seconds)
    initial_object_z = float(rc.get_object_pose(target_body_name)[2])

    if task_type == "post_grasp_lift":
        pre_grasp_plan, plan = rc.make_post_grasp_lift_plans(
            target_x,
            target_y,
            lift_height=lift_height,
            target_color=target_color,
        )
        if interpolate_motion:
            pre_grasp_plan = rc.interpolate_plan(
                pre_grasp_plan,
                max_cartesian_step=interpolation_max_step,
                min_segment_steps=interpolation_min_segment_steps,
            )
            plan = rc.interpolate_plan(
                plan,
                max_cartesian_step=interpolation_max_step,
                min_segment_steps=interpolation_min_segment_steps,
            )

        try:
            for action in pre_grasp_plan:
                rc.execute_action(action, speed=speed)
                rc.settle_steps(
                    seconds=interpolated_settle_seconds_per_action if interpolate_motion else settle_seconds_per_action
                )

            if not rc.is_target_grasp_success(
                target_body_name=target_body_name,
                touch_threshold=touch_threshold,
            ):
                return False
        except Exception as e:
            logger.abort_episode()
            raise e
    elif task_type in pick_place_location_stages:
        precondition_plan, plan = rc.make_pick_place_location_stage_plans(
            stage=pick_place_location_stages[task_type],
            pick_x=target_x,
            pick_y=target_y,
            place_x=place_x,
            place_y=place_y,
            lift_height=lift_height,
            target_color=target_color,
        )
        if interpolate_motion:
            precondition_plan = rc.interpolate_plan(
                precondition_plan,
                max_cartesian_step=interpolation_max_step,
                min_segment_steps=interpolation_min_segment_steps,
            )
            plan = rc.interpolate_plan(
                plan,
                max_cartesian_step=interpolation_max_step,
                min_segment_steps=interpolation_min_segment_steps,
            )

        try:
            for action in precondition_plan:
                rc.execute_action(action, speed=speed)
                rc.settle_steps(
                    seconds=interpolated_settle_seconds_per_action if interpolate_motion else settle_seconds_per_action
                )
        except Exception as e:
            logger.abort_episode()
            raise e

    logger.start_episode(
        episode_id=episode_id,
        instruction=instruction,
        task_type=task_type,
        goal_xy=[place_x, place_y],
        box_init_xy=[target_x, target_y],
        box_init_yaw=target_yaw,
        target_color=target_color,
        target_body_name=target_body_name,
        place_color=place_color,
        place_body_name=place_body_name,
        all_object_init_poses=SyncSimRaccoonDataset.specs_to_meta(object_specs),
    )

    try:
        # The prompt decides which object to manipulate. All objects are visible,
        # but the trajectory is aimed only at the prompted target.
        if task_type == "grasp":
            plan = rc.make_grasp_plan(target_x, target_y, target_color=target_color)
        elif task_type == "lift":
            plan = rc.make_lift_plan(target_x, target_y, lift_height=lift_height, target_color=target_color)
        elif task_type == "post_grasp_lift":
            pass
        elif task_type == "pick_place":
            plan = rc.make_pick_place_plan(
                pick_x=target_x,
                pick_y=target_y,
                place_x=place_x,
                place_y=place_y,
                lift_height=lift_height,
                target_color=target_color,
            )
        elif task_type == "pick_place_location":
            plan = rc.make_pick_place_location_plan(
                pick_x=target_x,
                pick_y=target_y,
                place_x=place_x,
                place_y=place_y,
                lift_height=lift_height,
                target_color=target_color,
            )
        elif task_type in pick_place_location_stages:
            pass
        else:
            raise ValueError(f"지원하지 않는 task_type입니다: {task_type}")

        if interpolate_motion and task_type not in ("post_grasp_lift", *pick_place_location_stages.keys()):
            plan = rc.interpolate_plan(
                plan,
                max_cartesian_step=interpolation_max_step,
                min_segment_steps=interpolation_min_segment_steps,
            )

        # Initial observation.
        obs = rc.get_observation()
        dt = 1.0 / hz
        step_counter = 0
        action_settle_seconds = (
            interpolated_settle_seconds_per_action if interpolate_motion else settle_seconds_per_action
        )

        for action in plan:
            # Set control target to current waypoint.
            rc.execute_action(action, speed=speed)

            # Capture continuous observations at specified Hz while moving toward the target.
            num_frames = max(1, int(action_settle_seconds * hz))

            for _ in range(num_frames):
                logger.log_step(
                    step_idx=step_counter,
                    image_rgb=obs["image"],
                    joint_angles=obs["joint_angles"],
                    gripper_state=obs["gripper_state"],
                    object_pose=obs["object_pose"],
                    ee_pose=obs["ee_pose"],
                    action=action,
                    is_first=(step_counter == 0),
                    is_last=False,
                )

                # Advance physics by dt seconds.
                rc.settle_steps(seconds=dt)

                # Observe after stepping.
                obs = rc.get_observation()
                step_counter += 1

        # Record terminal observation.
        logger.log_step(
            step_idx=step_counter,
            image_rgb=obs["image"],
            joint_angles=obs["joint_angles"],
            gripper_state=obs["gripper_state"],
            object_pose=obs["object_pose"],
            ee_pose=obs["ee_pose"],
            action=plan[-1],
            is_first=False,
            is_last=True,
        )

        if task_type in ("lift", "post_grasp_lift"):
            success = rc.is_target_lift_success(
                target_body_name=target_body_name,
                initial_object_z=initial_object_z,
                lift_height=lift_height,
                touch_threshold=touch_threshold,
            )
        elif task_type == "pick_place":
            success = rc.is_target_place_success(
                target_body_name=target_body_name,
                place_body_name=place_body_name,
            )
        elif task_type == "pick_place_location":
            success = rc.is_target_place_location_success(
                target_body_name=target_body_name,
                place_xy=[place_x, place_y],
            )
        elif task_type == "pick_place_location_post_grasp_lift":
            success = rc.is_target_lift_success(
                target_body_name=target_body_name,
                initial_object_z=initial_object_z,
                lift_height=lift_height,
                touch_threshold=touch_threshold,
            )
        elif task_type == "pick_place_location_post_lift_move":
            target_pose = rc.get_object_pose(target_body_name)
            xy_close = float(np.linalg.norm(target_pose[:2] - np.asarray([place_x, place_y], dtype=np.float64))) <= 0.035
            success = bool(
                xy_close
                and rc.is_target_lift_success(
                    target_body_name=target_body_name,
                    initial_object_z=initial_object_z,
                    lift_height=lift_height,
                    touch_threshold=touch_threshold,
                )
            )
        elif task_type == "pick_place_location_post_move_lower":
            target_pose = rc.get_object_pose(target_body_name)
            xy_close = float(np.linalg.norm(target_pose[:2] - np.asarray([place_x, place_y], dtype=np.float64))) <= 0.035
            low_enough = float(target_pose[2]) <= 0.045
            gripper_is_closing_or_closed = float(rc.data.qpos[4]) < (rc.GRIP_OPEN - 0.03)
            success = bool(xy_close and low_enough and gripper_is_closing_or_closed)
        elif task_type == "pick_place_location_post_place_retreat":
            success = rc.is_target_place_location_success(
                target_body_name=target_body_name,
                place_xy=[place_x, place_y],
            )
        else:
            success = rc.is_target_grasp_success(
                target_body_name=target_body_name,
                touch_threshold=touch_threshold,
            )
        logger.finalize_episode(success=success)
        return success

    except Exception as e:
        logger.abort_episode()
        raise e


def _balanced_target_counts(num_episodes, items):
    """
    Return per-color episode targets. If num_episodes is divisible by the
    number of items, the split is exactly equal. Otherwise the remainder is
    distributed one-by-one to the first items.
    """
    base = num_episodes // len(items)
    remainder = num_episodes % len(items)
    return {
        item: base + (1 if idx < remainder else 0)
        for idx, item in enumerate(items)
    }


def _sample_remaining_item(rng, target_counts, success_counts):
    remaining_items = []
    remaining_weights = []

    for item, target_count in target_counts.items():
        remaining = target_count - success_counts[item]
        if remaining > 0:
            remaining_items.append(item)
            remaining_weights.append(remaining)

    if not remaining_items:
        return None

    remaining_weights = np.asarray(remaining_weights, dtype=np.float64)
    remaining_weights /= remaining_weights.sum()
    choice_idx = int(rng.choice(len(remaining_items), p=remaining_weights))
    return remaining_items[choice_idx]


def make_object_phrase(target_color):
    if target_color == "white":
        return "white cube"
    return f"{target_color} cylinder"


def make_instruction(target_color, task_type, cylinder_instruction_template, place_color=None):
    if task_type == "pick_place":
        if place_color is None:
            raise ValueError("pick_place instruction에는 place_color가 필요합니다.")
        return f"pick the {make_object_phrase(target_color)} and place it on the {make_object_phrase(place_color)}"
    if task_type in (
        "pick_place_location",
        "pick_place_location_post_grasp_lift",
        "pick_place_location_post_lift_move",
        "pick_place_location_post_move_lower",
        "pick_place_location_post_place_retreat",
    ):
        return f"pick the {make_object_phrase(target_color)} and place it at position four"
    if task_type in ("lift", "post_grasp_lift"):
        return f"lift the {make_object_phrase(target_color)}"
    if target_color == "white":
        return "grasp the white cube"
    return cylinder_instruction_template.format(color=target_color)


def build_target_items(task_types, colors, pick_place_pairs=None):
    items = []
    for task_type in task_types:
        if task_type == "pick_place":
            if pick_place_pairs is None:
                items.extend(
                    (task_type, target_color, place_color)
                    for target_color in colors
                    for place_color in colors
                    if place_color != target_color
                )
            else:
                items.extend(
                    (task_type, target_color, place_color)
                    for target_color, place_color in pick_place_pairs
                )
        elif task_type in (
            "pick_place_location",
            "pick_place_location_post_grasp_lift",
            "pick_place_location_post_lift_move",
            "pick_place_location_post_move_lower",
            "pick_place_location_post_place_retreat",
        ):
            items.extend((task_type, target_color, None) for target_color in colors)
        else:
            items.extend((task_type, target_color, None) for target_color in colors)
    return tuple(items)


def collect_dataset(
    xml_path="Raccoon_colored_cylinder.xml",
    dataset_root="raccoon_grasp_lift_colored_objects",
    num_episodes=100,
    colors=("red", "blue", "green", "yellow", "white"),
    task_types=("grasp", "lift"),
    instruction_template="grasp the {color} cylinder",
    lift_height=0.05,
    keep_failed=False,
    use_viewer=False,
    camera_name="front_view",
    speed=150,
    settle_seconds_per_action=0.8,
    interpolated_settle_seconds_per_action=0.12,
    initial_settle_seconds=0.3,
    hz=10,
    touch_threshold=0.1,
    seed=None,
    max_attempts=None,
    object_x_range=None,
    object_y_range=None,
    min_object_distance=None,
    interpolate_motion=False,
    interpolation_max_step=0.008,
    interpolation_min_segment_steps=2,
    pick_place_pairs=None,
    fixed_object_lanes=None,
    fixed_place_lane=None,
    fixed_lane_count=5,
):
    """
    Collect a balanced grasp/lift/pick_place dataset for colored cylinders and the white cube.

    Each episode contains all four colored cylinders at randomized positions.
    The instruction selects which object is the target, and task_type selects
    whether the robot grasps it, lifts it, or places it on another object.

    Default behavior with keep_failed=False:
    - Saves exactly num_episodes successful episodes when possible.
    - Balances successful episodes across colors according to target_counts.
      For num_episodes=500 and 4 colors, this yields 125 episodes per color.
    - Failed episodes are discarded and retried with the remaining color quota.
    - Before frame_000000 is captured, the scene is stepped for
      initial_settle_seconds so free-joint cylinders are already resting on the table.

    Position defaults place objects on a reachable front arc:
    - x range: -0.11~0.11
    - y range:  0.19~0.21
    - minimum spacing: 0.045

    If keep_failed=True, failed episodes are also saved, so the final folder can
    contain more than num_episodes attempts and the all-attempt ratio may differ.
    """
    colors = tuple(colors)
    task_types = tuple(task_types)
    valid_colors = set(SyncSimRaccoonDataset.CYLINDER_BODY_BY_COLOR.keys())
    unknown_colors = [color for color in colors if color not in valid_colors]
    if unknown_colors:
        raise ValueError(f"지원하지 않는 색상입니다: {unknown_colors}. 지원 색상: {sorted(valid_colors)}")

    if len(colors) == 0:
        raise ValueError("colors는 비어 있을 수 없습니다.")
    if pick_place_pairs is not None:
        pick_place_pairs = tuple((str(src), str(dst)) for src, dst in pick_place_pairs)
        for src, dst in pick_place_pairs:
            if src == dst:
                raise ValueError(f"pick_place pair의 source/place가 같습니다: {(src, dst)}")
            if src not in valid_colors or dst not in valid_colors:
                raise ValueError(
                    f"지원하지 않는 pick_place pair입니다: {(src, dst)}. "
                    f"지원 색상: {sorted(valid_colors)}"
                )
    valid_task_types = {
        "grasp",
        "lift",
        "post_grasp_lift",
        "pick_place",
        "pick_place_location",
        "pick_place_location_post_grasp_lift",
        "pick_place_location_post_lift_move",
        "pick_place_location_post_move_lower",
        "pick_place_location_post_place_retreat",
    }
    unknown_task_types = [task_type for task_type in task_types if task_type not in valid_task_types]
    if unknown_task_types:
        raise ValueError(f"지원하지 않는 task_type입니다: {unknown_task_types}. 지원 task: {sorted(valid_task_types)}")
    if len(task_types) == 0:
        raise ValueError("task_types는 비어 있을 수 없습니다.")

    target_items = build_target_items(task_types, colors, pick_place_pairs=pick_place_pairs)
    target_counts = _balanced_target_counts(num_episodes, target_items)
    rng = np.random.default_rng(seed)

    if max_attempts is None:
        # Prevent infinite loops if grasp repeatedly fails.
        max_attempts = max(num_episodes * 20, num_episodes + 100)

    rc = SyncSimRaccoonDataset(
        xml_path=xml_path,
        image_size=(256, 256),
        camera_name=camera_name,
        use_viewer=use_viewer,
    )
    logger = DatasetLogger(root_dir=dataset_root, keep_failed=keep_failed)

    success_counts = {item: 0 for item in target_items}
    attempt_count = 0

    print(f"Target task/object counts: {target_counts}")

    try:
        while sum(success_counts.values()) < num_episodes and attempt_count < max_attempts:
            attempt_count += 1

            target_item = _sample_remaining_item(rng, target_counts, success_counts)
            if target_item is None:
                break
            task_type, target_color, place_color = target_item

            instruction = make_instruction(
                target_color=target_color,
                task_type=task_type,
                cylinder_instruction_template=instruction_template,
                place_color=place_color,
            )
            if fixed_object_lanes is not None:
                object_specs = SyncSimRaccoonDataset.fixed_lane_object_specs(
                    rng=rng,
                    lane_by_color=fixed_object_lanes,
                    lane_count=fixed_lane_count,
                    x_range=object_x_range,
                    y_range=object_y_range,
                )
            else:
                object_specs = SyncSimRaccoonDataset.sample_object_specs(
                    rng=rng,
                    colors=colors,
                    x_range=object_x_range,
                    y_range=object_y_range,
                    min_distance=min_object_distance,
                )

            place_xy = None
            if task_type in {
                "pick_place_location",
                "pick_place_location_post_grasp_lift",
                "pick_place_location_post_lift_move",
                "pick_place_location_post_move_lower",
                "pick_place_location_post_place_retreat",
            }:
                if fixed_place_lane is None:
                    raise ValueError(f"{task_type}에는 fixed_place_lane이 필요합니다.")
                place_xy = SyncSimRaccoonDataset.arc_lane_xy(
                    lane_index=fixed_place_lane,
                    lane_count=fixed_lane_count,
                    x_range=object_x_range,
                    y_range=object_y_range,
                )

            # With keep_failed=False, failed attempts are deleted, so reusing the
            # next successful episode id keeps folder numbering compact.
            episode_id = attempt_count if keep_failed else (sum(success_counts.values()) + 1)

            try:
                success = run_episode_and_record(
                    rc=rc,
                    logger=logger,
                    episode_id=episode_id,
                    instruction=instruction,
                    object_specs=object_specs,
                    target_color=target_color,
                    place_color=place_color,
                    place_xy=place_xy,
                    task_type=task_type,
                    lift_height=lift_height,
                    speed=speed,
                    settle_seconds_per_action=settle_seconds_per_action,
                    interpolated_settle_seconds_per_action=interpolated_settle_seconds_per_action,
                    initial_settle_seconds=initial_settle_seconds,
                    hz=hz,
                    touch_threshold=touch_threshold,
                    interpolate_motion=interpolate_motion,
                    interpolation_max_step=interpolation_max_step,
                    interpolation_min_segment_steps=interpolation_min_segment_steps,
                )

                if success:
                    success_counts[target_item] += 1

                print(
                    f"[Attempt {attempt_count:04d}] episode_id={episode_id:06d} | "
                    f"task_type='{task_type}' | color='{target_color}' | "
                    f"place_color='{place_color}' | "
                    f"target_xy=({object_specs[target_color]['x']:.3f}, {object_specs[target_color]['y']:.3f}) | "
                    f"instruction='{instruction}' | success={success} | "
                    f"success_counts={success_counts}"
                )
            except Exception as e:
                print(
                    f"[Attempt {attempt_count:04d}] task_type='{task_type}' | "
                    f"color='{target_color}' | place_color='{place_color}' | exception: {e}"
                )

    finally:
        rc.close()

    total_success = sum(success_counts.values())
    print(f"완료: success episodes = {total_success}/{num_episodes}, attempts = {attempt_count}")
    print(f"task/object별 성공 episode 수: {success_counts}")

    if total_success < num_episodes:
        print(
            "주의: max_attempts에 도달해서 목표 episode 수를 모두 채우지 못했습니다. "
            "max_attempts를 늘리거나 grasp 성공 조건/동작 파라미터를 확인하세요."
        )


if __name__ == "__main__":
    collect_dataset(
        xml_path="Raccoon_colored_cylinder.xml",
        dataset_root="raccoon_grasp_lift_colored_objects",
        num_episodes=10,
        colors=("red", "blue", "green", "yellow", "white"),
        task_types=("grasp", "lift"),
        instruction_template="grasp the {color} cylinder",
        lift_height=0.05,
        keep_failed=False,
        use_viewer=False,
        camera_name="front_view",
        initial_settle_seconds=0.1,
        object_x_range=None,
        object_y_range=None,
        min_object_distance=None,
    )
