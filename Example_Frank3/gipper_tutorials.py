import franky

gripper = franky.Gripper("192.168.1.11")

# 设置夹爪移动的速度和
speed = 0.02  # [m/s]
force = 20.0  # [N]

# 将夹爪移动到指定宽度：5cm
success = gripper.move(0.05, speed)

# 抓取一个宽度未知的物体。
# grasp() 方法会让夹爪闭合直到检测到物体被夹住。
# 参数：
# - 第一个参数是目标宽度（0.0 表示完全闭合）
# - speed: 移动速度
# - force: 夹爪施加的最大力
# - epsilon_outer: 外部位置容差，用于判断是否夹住物体
success &= gripper.grasp(0.0, speed, force, epsilon_outer=1.0)

# 获取当前被抓物体的宽度
width = gripper.width

# 打开夹爪，释放物体
gripper.open(speed)

###################### 异步操作 ######################
# 移动夹爪到5cm宽度
success_future = gripper.move_async(0.05, speed)

# 等待一秒让异步动作完成
if success_future.wait(1):
    print(f"Success: {success_future.get()}")
else:
    # 如果超时没有完成，则停止夹爪运动
    gripper.stop()
    success_future.wait()
    print("Gripper motion timed out.")