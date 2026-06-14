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
  action : dx, dy, dz, dpitch, gripper<br>
  
## 추가 기능 구현 목록 정리<br>

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

최종 결과 동영상 <br>
[grasp the cube.webm](https://github.com/user-attachments/assets/9a6d5ac7-f95b-4f2d-8142-03b3815c19a2)<br>


### Lift 데이터<br>
물체를 grasp 한 후 지면 위로 +5cm하는 것이 기준<br>
<img width="256" height="256" alt="image" src="https://github.com/user-attachments/assets/43325b45-6e03-43ad-b310-4c58a0b81dbe" /><br>

물체를 잘 들어올리긴 하나 들어올릴 때 로봇팔과 지면의 접촉 위험이 있어, 물체를 접촉하기 직전과 접촉한 후 일정 구간동안 pitch 값을 0으로 고정시킴<br>
<img width="256" height="256" alt="image" src="https://github.com/user-attachments/assets/7a98fa3a-1487-42aa-a567-2436d50920db" /><br>

lift 또한 물체는 일정 간격을 기준으로 원호를 그리며 생성되며, 각 데이터 생성마다 랜덤 배치 시켜 여러 데이터를 수집하도록 함. <br>

lift 데이터 1000개를 수집하여 모델을 학습시켜주니 접근 후 grasp 까지는 진행하나 이후 들어올리지 못하는 문제가 발생<br>
-> lift 데이터 이외에 물체를 집은 상태에서 들어올리기만 하는 데이터를 따로 모아 학습에 포함시켜 행동을 강화함.<br>
lift 전체 과정 1000개 + lift 구간 과정 1000개 = 총 2000개의 데이터 수집 진행.<br>

최종 결과 동영상 <br>
[llift.webm](https://github.com/user-attachments/assets/c633d2ef-cedc-4158-8983-d76b310eec75)<br>

### Pick and Place<br>
기존의 5가지 물체를 사용해 물체를 집고 다른 물체 위에 올리는 task 를 생성하여 학습<br>
[Screencast from 2026-06-02 03-43-58.webm](https://github.com/user-attachments/assets/4fa8f37b-df8d-4e2b-9731-2385d7ee6d76)<br>

물체를 제대로 들어올리지 못하는 문제가 발생하여, task를 단순화 하여 하나의 물체만을 고정된 위치에 생성하여 다른 위치에 놓도록 함. <br>

