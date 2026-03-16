import sys
import time
import cv2
import numpy as np
import mediapipe as mp
import threading

# 修正为你当前正确的绝对路径
sys.path.append("/home/wdsdz/z1_sdk/lib")
import unitree_arm_interface

# ================= 1. 高频无阻塞摄像头采集线程 =================
class CameraStream:
    def __init__(self, src=0):
        self.cap = cv2.VideoCapture(src)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        self.ret, self.frame = self.cap.read()
        self.running = True
        self.thread = threading.Thread(target=self.update, args=())
        self.thread.daemon = True
        self.thread.start()

    def update(self):
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                self.frame = frame
            time.sleep(0.005)

    def read(self):
        return self.ret, self.frame.copy() if self.ret else None

    def stop(self):
        self.running = False
        self.thread.join()
        self.cap.release()

# ================= 2. 握拳检测算法 =================
def detect_fist(hand_landmarks):
    """
    通过比较指尖和指关节到手腕的距离，判断是否为握拳状态。
    """
    wrist = np.array([hand_landmarks.landmark[0].x, hand_landmarks.landmark[0].y])
    fingers =[(8, 5), (12, 9), (16, 13), (20, 17)]
    
    is_fist = True
    for tip_idx, mcp_idx in fingers:
        tip = np.array([hand_landmarks.landmark[tip_idx].x, hand_landmarks.landmark[tip_idx].y])
        mcp = np.array([hand_landmarks.landmark[mcp_idx].x, hand_landmarks.landmark[mcp_idx].y])
        
        dist_tip = np.linalg.norm(tip - wrist)
        dist_mcp = np.linalg.norm(mcp - wrist)
        
        # 指尖离手腕比关节离手腕更远，说明手指伸直了，不是握拳
        if dist_tip > dist_mcp:
            is_fist = False
            break
            
    return is_fist

# ================= 3. 主程序 =================
def main():
    print("正在连接机械臂...")
    arm = unitree_arm_interface.ArmInterface(hasGripper=True)
    arm.loopOn()
    arm.backToStart()

    # --- 机械臂固定坐标参数 ---
    fixed_x = 0.3 
    current_y, current_z = 0.0, 0.3
    
    print("移动到初始观察点...")
    init_pose = np.array([0.0, 0.0, 0.0, fixed_x, current_y, current_z], dtype=np.float64)
    arm.MoveL(init_pose, 0.3)
    time.sleep(2)

    cam = CameraStream(0)
    img_center_x, img_center_y = 320, 240
    
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7)
    mp_draw = mp.solutions.drawing_utils

    # --- 映射参数与阈值 ---
    pixel_to_m_xy = 0.0008   
    dead_zone = 35           

    # --- 状态机控制变量 ---
    STATE_LOOKING = 0 
    STATE_MOVING = 1  
    STATE_COOLDOWN = 2 
    
    current_state = STATE_LOOKING
    move_start_time = 0
    move_duration = 0
    
    # 【优化点1】：握拳防误触控制变量
    is_tracking_enabled = False
    fist_counter = 0           # 握拳连续帧计数器
    fist_trigger_frames = 10   # 连续识别10帧(约0.3秒)才触发切换
    is_fist_locked = False     # 触发后锁定，必须松手才能解锁下一次触发
    
    buffer_ex, buffer_ey = [],[]
    frames_needed_to_confirm = 4

    print("========================================")
    print("✅ 系统已启动！(严格限位 + 防抖模式)")
    print("✊ 请对准摄像头【握拳并保持】来开启或暂停跟踪。")
    print("========================================")
    
    while True:
        ret, frame = cam.read()
        if not ret: continue

        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)
        
        hand_detected = False
        cx, cy = 0, 0

        # ---- AI 识别与手势检测阶段 ----
        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            hand_detected = True
            
            h, w, c = frame.shape
            lm_mid = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP]
            cx, cy = int(lm_mid.x * w), int(lm_mid.y * h)
            
            # 【优化点2】：边缘触发机制的握拳逻辑
            if detect_fist(hand_landmarks):
                if not is_fist_locked:
                    fist_counter += 1
                    # 屏幕上显示“蓄力”进度条
                    cv2.putText(frame, f"Switching: {'#' * fist_counter}", (180, 240), 
                                cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 165, 255), 2)
                    
                    if fist_counter >= fist_trigger_frames:
                        is_tracking_enabled = not is_tracking_enabled
                        is_fist_locked = True # 触发后立刻上锁
                        buffer_ex.clear()
                        buffer_ey.clear()
                        print(f"Tracking state changed to: {is_tracking_enabled}")
                else:
                    cv2.putText(frame, "RELEASE HAND TO RESET", (180, 240), 
                                cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 0, 255), 2)
            else:
                # 只有完全张开手，才会清空计数器并解锁
                fist_counter = 0
                is_fist_locked = False
        else:
            fist_counter = 0
            is_fist_locked = False

        # ---- 状态机逻辑阶段 ----
        if current_state == STATE_LOOKING:
            if not is_tracking_enabled:
                cv2.putText(frame, "TRACKING: OFF (Make a Fist)", (10, 40), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                buffer_ex.clear()
                buffer_ey.clear()
            
            elif hand_detected:
                cv2.putText(frame, "TRACKING: ON (Make a Fist)", (10, 40), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.circle(frame, (cx, cy), 15, (0, 255, 0), cv2.FILLED)
                
                ex = cx - img_center_x
                ey = cy - img_center_y
                
                if abs(ex) > dead_zone or abs(ey) > dead_zone:
                    buffer_ex.append(ex)
                    buffer_ey.append(ey)
                    
                    cv2.putText(frame, f"Locking... {len(buffer_ex)}/{frames_needed_to_confirm}", 
                                (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
                    
                    if len(buffer_ex) >= frames_needed_to_confirm:
                        avg_ex = np.mean(buffer_ex)
                        avg_ey = np.mean(buffer_ey)
                        
                        delta_y = -avg_ex * pixel_to_m_xy
                        delta_z = -avg_ey * pixel_to_m_xy
                        
                        current_y += delta_y
                        current_z += delta_z
                        
                        # 【优化点3】：收紧绝对安全的工作空间边界，彻底消灭 IK Error！
                        # 对于 Z1 机械臂在 [0,0,0] 姿态下，安全的矩形空间如下：
                        current_y = np.clip(current_y, -0.26, 0.26) # 左右最远26cm
                        current_z = np.clip(current_z, 0.20, 0.45)  # 最低20cm，绝不允许压到10cm
                        
                        max_dist = max(abs(delta_y), abs(delta_z))
                        arm_speed = 0.15 
                        move_duration = (max_dist / arm_speed) + 0.15 
                        
                        target_pose = np.array([0.0, 0.0, 0.0, fixed_x, current_y, current_z], dtype=np.float64)
                        arm.MoveL(target_pose, arm_speed)
                        
                        current_state = STATE_MOVING
                        move_start_time = time.time()
                        buffer_ex.clear()
                        buffer_ey.clear()
                else:
                    buffer_ex.clear()
                    buffer_ey.clear()
                    cv2.putText(frame, "Target Centered", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        elif current_state == STATE_MOVING:
            cv2.putText(frame, "MOVING...", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 165, 255), 3)
            if time.time() - move_start_time > move_duration:
                current_state = STATE_COOLDOWN
                cooldown_start = time.time()

        elif current_state == STATE_COOLDOWN:
            cv2.putText(frame, "COOLDOWN...", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            if time.time() - cooldown_start > 0.3:
                current_state = STATE_LOOKING

        cv2.drawMarker(frame, (img_center_x, img_center_y), (255, 0, 0), cv2.MARKER_CROSS, 20, 2)
        cv2.imshow("Look-And-Move Servoing", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cam.stop()
    cv2.destroyAllWindows()
    hands.close()
    
    print("正在安全复位...")
    arm.backToStart() 
    arm.loopOff()
    print("程序已结束。")

if __name__ == "__main__":
    main()
