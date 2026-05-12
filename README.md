# Raccoonbot_Openvla

0~3번 server에서 실행, 4번 local-server 실행<br>


## 0. Dependencies
```
pip install -r requirments.txt
```

## 1. Dataset 생성


## 2. rlds 파일 변환
raw data를 rlds builder에 맞게 변경
아래 명렁문 그대로 실행
```
cd /data/physicalai_workspace/Mujoco/raccoon_dataset
python convert_raw_to_openvla_rlds_intermediate.py \
--raw_root /data/physicalai_workspace/Mujoco/raccoon_dataset/raccoon_grasp/grasp_random_color_cylinder \
--out_root /data/physicalai_workspace/Mujoco/raccoon_dataset/raccoon_grasp/openvla_rlds_intermediate \
--val_ratio 0.1
```

## 2-1. rlds builder
rlds builder 실행
아래 명렁문 그대로 실행
```
cd /data/physicalai_workspace/Mujoco/rlds_dataset_builder/raccoon_grasp
tfds build --overwrite
```
실행하면 root 하위에 tensorflow_datasets 폴더 생성됨
```
mv /root/tensorflow_datasets /data/physicalai_workspace/Mujoco/
```

## 3. Raccoonbot 기반 OpenVLA finetuning
아래 명령어 그대로 실행 <br>
(max_steps, save_steps 수정 가능)
```
cd /data/physicalai_workspace/Mujoco/openvla
export PYTHONPATH=/data/physicalai_workspace/Mujoco/openvla:$PYTHONPATH

WANDB_MODE=disabled CUDA_VISIBLE_DEVICES=0 \
torchrun --standalone --nnodes 1 --nproc-per-node 1 vla-scripts/finetune.py \
  --vla_path openvla/openvla-7b \
  --data_root_dir /data/physicalai_workspace/Mujoco/tensorflow_datasets \
  --dataset_name raccoon_grasp \
  --run_root_dir /data/physicalai_workspace/Mujoco/raccoon_dataset/raccoon_grasp/openvla-runs \
  --adapter_tmp_dir /data/physicalai_workspace/Mujoco/raccoon_dataset/raccoon_grasp/openvla-adapter-tmp \
  --lora_rank 32 \
  --batch_size 8 \
  --grad_accumulation_steps 2 \
  --learning_rate 5e-4 \
  --max_steps 30000 \
  --save_steps 30000 \
  --run_id_note raccoon-eef-v100
```

## 4. Mujoco 환경 Inference (local-server)
[local 실행 코드] (구글드라이브링크) download <br>

server 실행 명령문 (수정 필요)
```
cd /data/physicalai_workspace/Mujoco/openvla
CUDA_VISIBLE_DEVICES=0 python openvla_server.py \
  --model_path /data/physicalai_workspace/Mujoco/raccoon_dataset/raccoon_grasp/openvla-runs/[하위폴더명] \
  --default-unnorm-key raccoon_grasp \
  --host 0.0.0.0 \
  --port 8000 \
  --device cuda
```

