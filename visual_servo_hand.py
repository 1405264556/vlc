import sys
import time
import cv2
import numpy as np
import mediapipe as mp

# 修正为你当前正确的绝对路径
sys.path.append("/home/wdsdz/z1_sdk/lib")
import unitree_arm_interface

def main():
    # ---------------- 1. 初始化机械臂 ----------------
    print("正在连接机械臂...")
    arm = unitree_arm_interface.ArmInterface(hasGripper=True)
    arm.loopOn() # 开启后台通信线程
    
    print("机械臂归零中...")
    arm.backToStart()
    
    # 【修复核心1】：将姿态(RPY)和位置(XYZ)合并为一个包含6个元素的数组
    # 格式为:[Roll, Pitch, Yaw, X, Y, Z]
    print("移动到观察点 (笛卡尔直线插补)...")
    init_pose = np.array([0.0, 0.0, 0.0, 0.3, 0.0, 0.3], dtype=np.float64)
    
    # MoveL 参数变成了2个：(包含6个元素的位姿数组, 速度)
    arm.MoveL(init_pose, 0.2) 
    time.sleep(2)

    # ---------------- 2. 初始化视觉与AI模型 ----------------
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    img_center_x, img_center_y = 320, 240
    
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        max_num_hands=1, 
        min_detection_confidence=0.7, 
        min_tracking_confidence=0.7
    )
    mp_draw = mp.solutions.drawing_utils

    # ---------------- 3. 伺服控制参数 ----------------
    Kp = 0.0001     # 比例系数（控制跟随灵敏度）
    dead_zone = 30  # 像素死区（手在画面中心30像素内机械臂不动，防抖）
    
    # 状态记录（与上方观察点坐标保持一致）
    current_x = 0.3
    current_y = 0.0
    current_z = 0.3

    print("========================================")
    print("✅ AI 视觉伺服已启动！")
    print("✋ 请将手掌放到摄像头前。")
    print("⏹️  在电脑画面选中时，按下 'q' 键安全退出。")
    print("⚠️  如遇异常，立刻按硬件急停按钮！")
    print("========================================")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("无法获取摄像头画面！")
            break
            
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)
        
        target_center = None
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # 绘制手部骨骼（方便在电脑端监视AI识别状态）
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                # 提取掌心特征点（中指根部）
                landmark_9 = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP]
                
                h, w, c = frame.shape
                cx, cy = int(landmark_9.x * w), int(landmark_9.y * h)
                target_center = (cx, cy)
                
                # 在掌心画实心圆
                cv2.circle(frame, (cx, cy), 15, (0, 255, 0), cv2.FILLED)
        
        # ---------------- 4. 机械臂运动控制 ----------------
        if target_center:
            tx, ty = target_center
            error_x = tx - img_center_x
            error_y = ty - img_center_y
            
            if abs(error_x) > dead_zone or abs(error_y) > dead_zone:
                # 计算增量
                delta_y = -error_x * Kp  
                delta_z = -error_y * Kp  
                
                current_y += delta_y
                current_z += delta_z
                
                # 【安全护城河】严格限制活动范围，防止过热或砸桌子
                current_y = np.clip(current_y, -0.25, 0.25) # 左右限制 ±25cm
                current_z = np.clip(current_z, 0.15, 0.45)  # 上下限制 15cm ~ 45cm
                
                # 【修复核心2】：伺服循环内的指令也合并为 6个元素的数组
                target_pose = np.array([0.0, 0.0, 0.0, current_x, current_y, current_z], dtype=np.float64)
                
                # 执行微调移动 (速度降低至 0.1 保证平滑)
                arm.MoveL(target_pose, 0.1)
                
        # 电脑端监视画面美化
        cv2.drawMarker(frame, (img_center_x, img_center_y), (255, 0, 0), cv2.MARKER_CROSS, 20, 2)
        cv2.putText(frame, f"Target Y: {current_y:.2f} Z: {current_z:.2f}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        cv2.imshow("AI Hand Tracking Monitor", frame)
        
        # 按下 'q' 键退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("收到退出指令...")
            break

    # ---------------- 5. 清理与安全复位 ----------------
    cap.release()
    cv2.destroyAllWindows()
    hands.close()
    
    print("正在让机械臂安全折叠复位...")
    arm.backToStart() 
    arm.loopOff()
    print("程序已安全结束，可断电。")

if __name__ == "__main__":
    main()
