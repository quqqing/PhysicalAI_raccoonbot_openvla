# Raccoonbot_Openvla

기본 베이스는 https://github.com/KWU-FAIR-LAB/Raccoonbot_Openvla.git 를 참고하여 환경설정 진행함.

**기본 베이스 라인<br>**
  물체 : 색깔 원통 4개<br>
  task : grasp<br>
  언어 명령 : grasp the {color} cylinder<br>
  action : dx, dy, dz, gripper<br>

**확장 버전<br>**
  물체 : 색깔 원통 4개 + 2cm x 2cm 의 흰색 정육면체 1개  <br>
  task : grasp, lift, pick and place  <br>
  언어 명령 : grasp the {color} {cylinder or cube}, lift the {color} {cylinder or cube}, pick the red cylinder and place it at position four  <br>
  action : dx, dy, dz, dpitch, gripper (IK를 통해 로봇 제어 진행) <br>
  
## 데이터셋 확장<br>

### Grasp 데이터<br>
처음에 pitch를 포함하여 grasp을 시도할 때 발생한 문제들<br>
  1. 다른 물체와의 접촉을 방지하기 위해 물체 접근 시 pitch 각을 90도로 설정하여 지면과 수직이 되도록 설정함<br>
     <img width="256" height="256" alt="image" src="https://github.com/user-attachments/assets/28a70938-9d64-4d7f-b534-69a7f4029eb3" /><br>
  
  2. 90도로 설정하니 물체 접근 시 로봇에 무리가 가는 것을 확인. 이후 접근 각도를 완화시킴<br>
     <img width="256" height="256" alt="image" src="https://github.com/user-attachments/assets/5405e751-6f17-4778-b706-1cf8c73b89b9" /><br>
     
  3. 각도를 눕히니 다른 물체와의 접촉 발생. 로봇 중심 기준 원호를 그리는 선 위에 물체를 배치하여 다른 물체와의 접촉을 방지함.<br>
     <img width="256" height="256" alt="image" src="https://github.com/user-attachments/assets/dec98886-45ce-46ff-a950-a41885300fe5" /><br>

     원호 거리 설정<br>
      ```
      DEFAULT_OBJECT_X_RANGE = (-0.11, 0.11)
      DEFAULT_OBJECT_Y_RANGE = (0.19, 0.21)
      DEFAULT_MIN_OBJECT_DISTANCE = 0.045
      ```

  5. 이번엔 로봇이 바닥에 부딪치며 지면과의 접촉 문제가 발생하여 물체를 집기 전 pitch 값을 0으로 고정<br>
     <img width="256" height="256" alt="image" src="https://github.com/user-attachments/assets/37ae9fda-7d98-4f50-b7bc-e20d82df0442" /><br>

  6. 접근 자체는 안정적이나 물체에 도달하기 전에 그리퍼를 닫아버림. 최종적으로 z축 방향으로 -2cm 더 내려가서 그리퍼를 동작시키도록 함.<br>
     <img width="256" height="256" alt="image" src="https://github.com/user-attachments/assets/fadc2c28-3bc9-4aab-a7f6-326cbfb90169" /><br>

grasp 시 물체는 일정 간격을 기준으로 원호를 그리며 생성되며, 각 데이터 생성마다 랜덤 배치 시켜 여러 데이터를 수집하도록 함. <br>
모든 데이터셋은 각각의 물체를 똑같은 비율로 수집함<br>

### Lift 데이터<br>
  1. 물체를 grasp 한 후 지면 위로 +5cm하는 것이 기준<br>
     <img width="256" height="256" alt="image" src="https://github.com/user-attachments/assets/43325b45-6e03-43ad-b310-4c58a0b81dbe" /><br>

  2. 물체를 잘 들어올리긴 하나 들어올릴 때 로봇팔과 지면의 접촉 위험이 있어, 물체를 접촉하기 직전과 접촉한 후 일정 구간동안 pitch 값을 0으로 고정시킴<br>
     <img width="256" height="256" alt="image" src="https://github.com/user-attachments/assets/7a98fa3a-1487-42aa-a567-2436d50920db" /><br>

lift 또한 물체는 일정 간격을 기준으로 원호를 그리며 생성되며, 각 데이터 생성마다 랜덤 배치 시켜 여러 데이터를 수집하도록 함. <br>

lift 데이터 1000개를 수집하여 모델을 학습시켜주니 접근 후 grasp 까지는 진행하나 이후 들어올리지 못하는 문제가 발생<br>
-> lift 데이터 이외에 물체를 집은 상태에서 들어올리기만 하는 데이터를 따로 모아 학습에 포함시켜 행동을 강화함.<br>
lift 전체 과정 1000개 + lift 구간 과정 1000개 = 총 2000개의 데이터 수집 진행.<br>

### Pick and Place 데이터 <br>
  1. 기존의 5가지 물체를 사용해 물체를 집고 다른 물체 위에 올리는 task 를 생성하여 학습<br>
     [Screencast from 2026-06-02 03-43-58.webm](https://github.com/user-attachments/assets/4fa8f37b-df8d-4e2b-9731-2385d7ee6d76)<br>

  2. 물체를 제대로 들어올리지 못하는 문제가 발생하여, task를 단순화 하여 하나의 물체만을 고정된 위치에 생성하여 다른 위치에 놓도록 함. <br>
     데이터 2000개 기준<br>
     결과 동영상 : [Screencast from 2026-06-04 23-00-15.webm](https://github.com/user-attachments/assets/bb8e1a35-9a2c-4862-a93a-b5c4d50b36ea)<br>

  3. 하나의 물체만을 생성했음에도 불구하고,제대로 들어올리지 못하는 문제가 발생함. 이후 전체 과정을 구간별로 나누어 수집 진행<br>
     전체 pick and place : 1000개<br>
     lift 구간 : 500개<br>
     move 구간 : 500개<br>
     lower 구간 : 500개<br>
     retreat 구간 : 500개<br>
     <img width="256" height="256" alt="image" src="https://github.com/user-attachments/assets/0dac10b9-00e7-40f3-9181-ffd2cfb3f3bd" />
     <img width="256" height="256" alt="image" src="https://github.com/user-attachments/assets/7f38c331-77df-48ef-8bf4-15b4812146b8" />

  4. move하는 과정 중 물체를 놓치는 문제가 발생하여 데이터 수집 시에만 pitch를 고정시켜 안정성을 강화함<br>
     전체 pick and place : 2000개<br>
     lift 구간 : 1000개<br>
     move 구간 : 1000개<br>
     lower 구간 : 1000개<br>
     retreat 구간 : 1000개<br>

### 데이터 수집
데이터 수집 시 더 부드러운 움직임 구현을 위해 전체적으로 interpolation을 적용하여 수집함.<br>
최종 데이터 수집<br>
grasp: 500<br>
```
python -c "from raccoon_grasp_multicolor_scene_dataset import collect_dataset; collect_dataset(xml_path='Raccoon_colored_cylinder.xml', dataset_root='raccoon_interp_grasp_500', num_episodes=500, colors=('red','blue','green','yellow','white'), task_types=('grasp',), lift_height=0.05, keep_failed=False, use_viewer=False, camera_name='front_view', speed=180, initial_settle_seconds=0.1, settle_seconds_per_action=1.0, interpolated_settle_seconds_per_action=0.30, interpolate_motion=True, interpolation_max_step=0.010, interpolation_min_segment_steps=2, max_attempts=10000, seed=10)"
```
lift full: 1000<br>
```
python -c "from raccoon_grasp_multicolor_scene_dataset import collect_dataset; collect_dataset(xml_path='Raccoon_colored_cylinder.xml', dataset_root='raccoon_interp_lift_full_1000', num_episodes=1000, colors=('red','blue','green','yellow','white'), task_types=('lift',), lift_height=0.05, keep_failed=False, use_viewer=False, camera_name='front_view', speed=180, initial_settle_seconds=0.1, settle_seconds_per_action=1.0, interpolated_settle_seconds_per_action=0.30, interpolate_motion=True, interpolation_max_step=0.010, interpolation_min_segment_steps=2, max_attempts=20000, seed=11)"
```
post-grasp lift: 1000<br>
```
python -c "from raccoon_grasp_multicolor_scene_dataset import collect_dataset; collect_dataset(xml_path='Raccoon_colored_cylinder.xml', dataset_root='raccoon_interp_post_grasp_lift_1000', num_episodes=1000, colors=('red','blue','green','yellow','white'), task_types=('post_grasp_lift',), lift_height=0.05, keep_failed=False, use_viewer=False, camera_name='front_view', speed=180, initial_settle_seconds=0.1, settle_seconds_per_action=1.0, interpolated_settle_seconds_per_action=0.30, interpolate_motion=True, interpolation_max_step=0.010, interpolation_min_segment_steps=2, max_attempts=20000, seed=12)"
```
red lane2 -> lane4 full pick place: 2000<br>
post-grasp-lift: 1000<br>
post-lift-move: 1000<br>
post-move-lower: 1000<br>
post-place-retreat: 1000<br>
```
python - <<'PY'
from raccoon_grasp_multicolor_scene_dataset import collect_dataset

COMMON = dict(
    xml_path="Raccoon_colored_cylinder.xml",
    colors=("red",),
    fixed_object_lanes={"red": 2},
    fixed_place_lane=4,
    fixed_lane_count=5,
    lift_height=0.05,
    keep_failed=False,
    use_viewer=False,
    camera_name="front_view",
    speed=180,
    initial_settle_seconds=0.1,
    settle_seconds_per_action=1.0,
    interpolated_settle_seconds_per_action=0.30,
    interpolate_motion=True,
    interpolation_max_step=0.010,
    interpolation_min_segment_steps=2,
)

jobs = [
    dict(
        dataset_root="raccoon_pitch_fixed_red_lane2_to_lane4_full_pick_place_2000",
        num_episodes=2000,
        task_types=("pick_place_location",),
        max_attempts=40000,
        seed=130,
    ),
    dict(
        dataset_root="raccoon_pitch_fixed_red_lane2_to_lane4_post_grasp_lift_1000",
        num_episodes=1000,
        task_types=("pick_place_location_post_grasp_lift",),
        max_attempts=20000,
        seed=131,
    ),
    dict(
        dataset_root="raccoon_pitch_fixed_red_lane2_to_lane4_post_lift_move_1000",
        num_episodes=1000,
        task_types=("pick_place_location_post_lift_move",),
        max_attempts=20000,
        seed=132,
    ),
    dict(
        dataset_root="raccoon_pitch_fixed_red_lane2_to_lane4_post_move_lower_1000",
        num_episodes=1000,
        task_types=("pick_place_location_post_move_lower",),
        max_attempts=20000,
        seed=133,
    ),
    dict(
        dataset_root="raccoon_pitch_fixed_red_lane2_to_lane4_post_place_retreat_1000",
        num_episodes=1000,
        task_types=("pick_place_location_post_place_retreat",),
        max_attempts=20000,
        seed=134,
    ),
]

for idx, job in enumerate(jobs, start=1):
    print(f"\n=== [{idx}/{len(jobs)}] collecting {job['dataset_root']} ===")
    collect_dataset(**COMMON, **job)

print("\nDone. Counts:")
for job in jobs:
    from pathlib import Path
    root = Path(job["dataset_root"])
    count = len(list(root.glob("episode_*")))
    print(f"{root}: {count}")
PY
```
total: 8500개<br>

### 데이터셋 합치기<br>
```
python - <<'PY'
from pathlib import Path
import shutil

sources = [
    Path("raccoon_interp_grasp_500"),
    Path("raccoon_interp_lift_full_1000"),
    Path("raccoon_interp_post_grasp_lift_1000"),
    Path("raccoon_pitch_fixed_red_lane2_to_lane4_full_pick_place_2000"),
    Path("raccoon_pitch_fixed_red_lane2_to_lane4_post_grasp_lift_1000"),
    Path("raccoon_pitch_fixed_red_lane2_to_lane4_post_lift_move_1000"),
    Path("raccoon_pitch_fixed_red_lane2_to_lane4_post_move_lower_1000"),
    Path("raccoon_pitch_fixed_red_lane2_to_lane4_post_place_retreat_1000"),
]

out = Path("raccoon_pitch_fixed_all_tasks_8500")
out.mkdir(exist_ok=True)

episode_id = 1
for src in sources:
    if not src.exists():
        raise FileNotFoundError(src)

    episodes = sorted(src.glob("episode_*"))
    print(f"{src}: {len(episodes)} episodes")

    for ep in episodes:
        dst = out / f"episode_{episode_id:06d}"
        if dst.exists():
            raise FileExistsError(dst)
        shutil.copytree(ep, dst)
        episode_id += 1

print(f"merged {episode_id - 1} episodes into {out}")
PY
```
### RLDS 형식으로 변환 후 build<br>
변환하기<br>
```
cd /home/min/vla_lab/Raccoonbot_Openvla/Mujoco/raccoon_dataset

python convert_raw_to_openvla_rlds_intermediate.py \
  --raw_root /home/min/vla_lab/Raccoonbot_Openvla/Mujoco/raccoon_pitch_fixed_all_tasks_8500 \
  --out_root /home/min/vla_lab/Raccoonbot_Openvla/Mujoco/raccoon_dataset/openvla_rlds_pitch_fixed_all_tasks_8500 \
  --val_ratio 0.1
```
데이터셋 경로 <br>
```
INTERMEDIATE_ROOT = Path(
    "/home/min/vla_lab/Raccoonbot_Openvla/Mujoco/raccoon_dataset/openvla_rlds_pitch_fixed_all_tasks_8500"
)
```
build<br>
```
cd /home/min/vla_lab/Raccoonbot_Openvla/Mujoco/rlds_dataset_builder/raccoon_pick_place
tfds build --overwrite
```

### train 하기(서버에서 진행)<br>
총 11시간 32분 소요<br>
<img width="800" height="256" alt="Screenshot from 2026-06-03 10-00-47" src="https://github.com/user-attachments/assets/b709143a-2367-445a-940e-d853611c739e" />

```
cd /data/Raccoonbot_Openvla/openvla

WANDB_MODE=disabled CUDA_VISIBLE_DEVICES=0 \
torchrun --standalone --nnodes 1 --nproc-per-node 1 vla-scripts/finetune.py \
  --vla_path openvla/openvla-7b \
  --data_root_dir /data/Raccoonbot_Openvla/tensorflow_datasets_pitch_fixed_all_tasks_8500 \
  --dataset_name raccoon_pick_place \
  --run_root_dir /data/Raccoonbot_Openvla/openvla/openvla-runs \
  --adapter_tmp_dir /data/Raccoonbot_Openvla/openvla/openvla-adapter-tmp \
  --lora_rank 32 \
  --batch_size 4 \
  --grad_accumulation_steps 2 \
  --learning_rate 5e-4 \
  --max_steps 30000 \
  --save_steps 10000 \
  --run_id_note raccoon-all-tasks-8500
```

### inferenece<br>
<img width="800" height="256" alt="Screenshot from 2026-06-04 23-56-26" src="https://github.com/user-attachments/assets/45f5706e-cbc0-4b34-80a4-a72ee45335ee" />

**서버 측**<br>
```
CUDA_VISIBLE_DEVICES=0 python openvla_server.py \
  --model_path "/data/Raccoonbot_Openvla/openvla/openvla-runs/openvla-7b+raccoon_pick_place+b8+lr-0.0005+lora-r32+dropout-0.0--raccoon-all-tasks-8500--image_aug" \
  --default-unnorm-key raccoon_pick_place \
  --host 0.0.0.0 \
  --port 8000 \
  --device cuda
```

**로컬 측**<br>
grasp<br>
```
python openvla_multicolor_client.py \
  --server_url http://127.0.0.1:8000 \
  --xml_path Raccoon_colored_cylinder.xml \
  --task_type grasp \
  --target_color red \
  --use_viewer \
  --speed 180 \
  --settle_seconds_per_action 0.12 \
  --delta_scale 1.0 \
  --max_delta_xyz 0.02 \
  --max_steps 220 \
```

lift<br>
```
python openvla_multicolor_client.py \
  --server_url http://127.0.0.1:8000 \
  --xml_path Raccoon_colored_cylinder.xml \
  --task_type lift \
  --target_color red \
  --use_viewer \
  --speed 180 \
  --settle_seconds_per_action 0.12 \
  --delta_scale 1.4 \
  --max_delta_xyz 0.02 \
  --max_steps 300 \
```

pick and place<br>
```
python openvla_multicolor_client.py \
  --server_url http://127.0.0.1:8000 \
  --xml_path Raccoon_colored_cylinder.xml \
  --task_type pick_place_location \
  --target_color red \
  --use_viewer \
  --speed 180 \
  --settle_seconds_per_action 0.12 \
  --delta_scale 3.0 \
  --max_delta_xyz 0.02 \
  --max_steps 500
```

최종 결과 동영상 <br>

### 최종 데이터셋 확장 목록
1. 새 객체 추가 : white cube
2. 새 task 추가 : lift, pick and place
3. 언어 instruction 다양화 : grasp the {color} {cylinder or cube}, lift the {color} {cylinder or cube}, pick the red cylinder and place it at position four

**Visualize one episode**<br>
grasp
<img width="256" height="256" alt="frame_000000" src="https://github.com/user-attachments/assets/f043ba9c-b45f-4083-853f-6f615dbe6f42" />
<img width="256" height="256" alt="frame_000010" src="https://github.com/user-attachments/assets/21470747-109c-4170-b4e7-229392ca9b7e" />
<img width="256" height="256" alt="frame_000020" src="https://github.com/user-attachments/assets/4ba1a35f-e7b9-42c4-8262-31ead9d3e81d" />
<img width="256" height="256" alt="frame_000030" src="https://github.com/user-attachments/assets/329a4af3-4f10-4a6d-b8e6-b6c4ecb4efbf" />
<img width="256" height="256" alt="frame_000033" src="https://github.com/user-attachments/assets/8d550ef8-d527-4fea-8ca6-33c445134093" />

lift
<img width="256" height="256" alt="frame_000000" src="https://github.com/user-attachments/assets/0373694b-75b5-4597-8048-85dad1f1eab4" />
<img width="256" height="256" alt="frame_000010" src="https://github.com/user-attachments/assets/9a46cb85-a2fe-4f88-a3b6-36bd4ae888b4" />
<img width="256" height="256" alt="frame_000020" src="https://github.com/user-attachments/assets/eff1be54-da36-4f22-b802-3f6db0179000" />
<img width="256" height="256" alt="frame_000030" src="https://github.com/user-attachments/assets/5ba80738-f928-475b-af57-b23c15328c9c" />
<img width="256" height="256" alt="frame_000040" src="https://github.com/user-attachments/assets/0fa8e241-c2ae-4a8e-9fde-01c400ee3221" />
<img width="256" height="256" alt="frame_000048" src="https://github.com/user-attachments/assets/95691486-55d8-4c6f-8759-cce3529175e7" />

pick and place
<img width="256" height="256" alt="frame_000000" src="https://github.com/user-attachments/assets/487cba4e-153c-48a8-aa9e-9afd30ab2148" />
<img width="256" height="256" alt="frame_000010" src="https://github.com/user-attachments/assets/a56aa6dc-3fc3-4cf0-bfda-d091ebc95185" />
<img width="256" height="256" alt="frame_000020" src="https://github.com/user-attachments/assets/f0fa8082-8e31-4266-a818-3cbc270cf693" />
<img width="256" height="256" alt="frame_000030" src="https://github.com/user-attachments/assets/54ab7949-4e50-4499-8c79-177735df4d0e" />
<img width="256" height="256" alt="frame_000040" src="https://github.com/user-attachments/assets/a6ce8ada-c7e3-49f2-b636-a5888937bc5e" />
<img width="256" height="256" alt="frame_000050" src="https://github.com/user-attachments/assets/c5f62184-317b-4d7d-b0c9-1cd8555f5d4c" />
<img width="256" height="256" alt="frame_000060" src="https://github.com/user-attachments/assets/390e9879-0528-422d-92d5-8a582ac02e20" />
<img width="256" height="256" alt="frame_000070" src="https://github.com/user-attachments/assets/4e5beb1a-6173-43e8-a2ff-cf9aa3af488d" />
<img width="256" height="256" alt="frame_000080" src="https://github.com/user-attachments/assets/9884d195-1131-4c0b-920d-e11c630c4e23" />
<img width="256" height="256" alt="frame_000090" src="https://github.com/user-attachments/assets/50a1defb-fbe2-4789-a609-d83866b4f248" />
<img width="256" height="256" alt="frame_000090" src="https://github.com/user-attachments/assets/99db2045-1123-4aa1-a799-f8eb46d36381" />
<img width="256" height="256" alt="frame_000110" src="https://github.com/user-attachments/assets/29cf7d70-d47a-445c-a9f0-8ad4a799eb34" />


