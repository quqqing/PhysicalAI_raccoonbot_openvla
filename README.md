# Raccoonbot_Openvla

 1~3번은 직접 finetuning을 진행하는 내용입니다. 체크포인트를 불러와서 사용하는 경우 0번과 4번만 진행하면 됩니다.

- 0~3번: server에서 실행
- 4번: local-server에서 실행

---

## 0. Dependencies

 clone

```bash
git clone https://github.com/KWU-FAIR-LAB/Raccoonbot_Openvla.git
```

 패키지 설치

```bash
cd Raccoonbot_Openvla/openvla
pip install -e .
```

---

## 1. Dataset 생성

MuJoCo 가상환경에서 finetuning을 위한 데이터를 수집합니다.

```bash
python raccoon_grasp_multicolor_scene_dataset.py
```

---

## 2. RLDS 파일 변환

raw data를 RLDS builder에 맞게 변경합니다.

 명령문을 그대로 실행하세요.

```bash
cd /data/Raccoonbot_Openvla/raccoon_dataset

python convert_raw_to_openvla_rlds_intermediate.py \
  --raw_root /data/Raccoonbot_Openvla/raccoon_dataset/raccoon_grasp/grasp_random_color_cylinder \
  --out_root /data/Raccoonbot_Openvla/raccoon_dataset/raccoon_grasp/openvla_rlds_intermediate \
  --val_ratio 0.1
```

---

## 2-1. RLDS Builder 실행

RLDS builder를 실행합니다.

```bash
cd /data/Raccoonbot_Openvla/rlds_dataset_builder/raccoon_grasp
tfds build --overwrite
```

 후 `/root` 하위에 `tensorflow_datasets` 폴더가 생성됩니다. 해당 폴더를 프로젝트 경로로 이동합니다.

```bash
mv /root/tensorflow_datasets /data/Raccoonbot_Openvla/
```

---

## 3. Raccoonbot 기반 OpenVLA Finetuning

 명령어를 그대로 실행하세요.

`max_steps`, `save_steps`는 필요에 따라 변경할 수 있습니다.

```bash
cd /data/Raccoonbot_Openvla/openvla
export PYTHONPATH=/data/Raccoonbot_Openvla/openvla:$PYTHONPATH

WANDB_MODE=disabled CUDA_VISIBLE_DEVICES=0 \
torchrun --standalone --nnodes 1 --nproc-per-node 1 vla-scripts/finetune.py \
  --vla_path openvla/openvla-7b \
  --data_root_dir /data/Raccoonbot_Openvla/tensorflow_datasets \
  --dataset_name raccoon_grasp \
  --run_root_dir /data/Raccoonbot_Openvla/raccoon_dataset/raccoon_grasp/openvla-runs \
  --adapter_tmp_dir /data/Raccoonbot_Openvla/raccoon_dataset/raccoon_grasp/openvla-adapter-tmp \
  --lora_rank 32 \
  --batch_size 8 \
  --grad_accumulation_steps 2 \
  --learning_rate 5e-4 \
  --max_steps 30000 \
  --save_steps 30000 \
  --run_id_note raccoon-eef-v100
```

---

## 4. MuJoCo 환경 Inference

1~3번을 진행했다면 4-1은 건너뛰고, 이후 명령어에서 본인이 finetuning한 모델 경로로 `model_path`를 변경하여 진행하면 됩니다.

---

## 4-1. Hugging Face에서 RaccoonBot Finetuned OpenVLA 모델 다운로드

 터미널에서 아래 명령어를 입력하여 모델을 다운로드합니다.

```bash
pip install -U huggingface_hub

hf download fair-lab/openvla-7b-finetuned-raccoonbot \
  --local-dir /data/openvla-runs/openvla-7b-finetuned-raccoonbot
```

---

## 4-2. 서버측 코드 실행

server 실행 명령문입니다.

```bash
cd /data/Raccoonbot_Openvla/openvla

CUDA_VISIBLE_DEVICES=0 python openvla_server.py \
  --model_path /data/openvla-runs/openvla-7b-finetuned-raccoonbot \
  --default-unnorm-key raccoon_pick_place \
  --host 0.0.0.0 \
  --port 8000 \
  --device cuda
```

---

## 4-3. 클라이언트측 환경 설정

 코드와 MuJoCo XML 파일을 아래 링크에서 다운로드한 후 압축을 풉니다.

 링크:  
https://drive.google.com/drive/folders/1xrH3FoTfKC9CiUE-kDRorxTKMMq0O7Px?usp=sharing

 파일:

- `openvla_multicolor_client.py`
- `raccoon_env.py`
- `Raccoon_colored_cylinder.xml`
- `RaccoonBot_S.xml`
- `requirements.txt`

VSCode로 압축을 푼 상위 폴더를 열고, terminal에서 환경설정을 진행합니다.

```bash
pip install -r requirements.txt
```

---

## 4-4. 클라이언트측 코드 실행

`target_color`를 `[red, blue, green, yellow]` 중 하나로 수정하면 그에 맞게 prompt가 변경됩니다.

local 실행 명령문입니다.

```bash
python openvla_multicolor_client.py \
  --server_url http://127.0.0.1:8000 \
  --xml_path Raccoon_colored_cylinder.xml \
  --target_color red \
  --use_viewer
```
