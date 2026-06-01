import dynamixel_sdk as dxl
import init_pose
import time
import torch
import numpy as np
from infer_student import load_student_policy, get_action
import argparse

init_pose.init_pose()

PORT_NAME = '/dev/ttyUSB0'
BAUDRATE = 1000000
PROTOCOL_VERSION = 1.0
MOTOR_IDS = list(range(1, 13))


DOF_POSITION = {
        1: 512,
        2: 512,
        3: 480,
        4: 410,
        5: 582,
        6: 512,
        7: 512,
        8: 512,
        9: 544,
        10: 614,
        11: 442,
        12: 512,
    }


ADDR_TORQUE_ENABLE = 24
ADDR_GOAL_POSITION = 30
ADDR_MOVING_SPEED = 32
ADDR_PRESENT_POSITION = 36


LEN_1BYTE = 1
LEN_2BYTE = 2

portHandler = dxl.PortHandler(PORT_NAME)
packetHandler = dxl.PacketHandler(PROTOCOL_VERSION)

portHandler.openPort()
portHandler.setBaudRate(BAUDRATE)



def get_present_position(motor_id):
    pos, dxl_comm_result, dxl_error = packetHandler.read2ByteTxRx(
        portHandler, motor_id, ADDR_PRESENT_POSITION
    )
    if dxl_comm_result != dxl.COMM_SUCCESS:
        print(f"모터 {motor_id} 현재 위치 읽기 오류: {packetHandler.getTxRxResult(dxl_comm_result)}")
        return 512
    elif dxl_error != 0:
        print(f"모터 {motor_id} 에러: {packetHandler.getRxPacketError(dxl_error)}")
        return 512
    return pos


def close_port():
    try:
        portHandler.closePort()
    except Exception:
        pass


def enable_torque_for(motor_ids):
    ok = True
    for motor_id in motor_ids:
        dxl_comm_result, dxl_error = packetHandler.write1ByteTxRx(
            portHandler, motor_id, ADDR_TORQUE_ENABLE, 1
        )
        if dxl_comm_result != dxl.COMM_SUCCESS:
            print(f"모터 {motor_id} 통신 오류: {packetHandler.getTxRxResult(dxl_comm_result)}")
            ok = False
        elif dxl_error != 0:
            print(f"모터 {motor_id} 오류: {packetHandler.getRxPacketError(dxl_error)}")
            ok = False
    return ok


def move_motors_slow(motor_ids, goal_positions, moving_speed):
    group_speed = dxl.GroupSyncWrite(portHandler, packetHandler, ADDR_MOVING_SPEED, LEN_2BYTE)
    for i, motor_id in enumerate(motor_ids):
        speed = max(1, min(1023, int(moving_speed[i])))
        param_speed = [speed & 0xFF, (speed >> 8) & 0xFF]
        if not group_speed.addParam(motor_id, param_speed):
            print(f"속도 파라미터 추가 실패: ID {motor_id}")
    dxl_comm_result = group_speed.txPacket()
    if dxl_comm_result != dxl.COMM_SUCCESS:
        print(f"속도 전송 오류: {packetHandler.getTxRxResult(dxl_comm_result)}")
    group_speed.clearParam()

    group_pos = dxl.GroupSyncWrite(portHandler, packetHandler, ADDR_GOAL_POSITION, LEN_2BYTE)
    for motor_id in motor_ids:
        pos = int(goal_positions.get(motor_id, DOF_POSITION.get(motor_id, 512)))
        pos = max(0, min(1023, pos))
        param_pos = [pos & 0xFF, (pos >> 8) & 0xFF]
        if not group_pos.addParam(motor_id, param_pos):
            print(f"포지션 파라미터 추가 실패: ID {motor_id}")
    dxl_comm_result = group_pos.txPacket()
    if dxl_comm_result != dxl.COMM_SUCCESS:
        print(f"포지션 전송 오류: {packetHandler.getTxRxResult(dxl_comm_result)}")
    group_pos.clearParam()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--file_path", dest = "file_path")
    parser.add_argument( "--freq", dest = "frequency")
    args = parser.parse_args()
    idx_id=[0, 2, 4, 6, 8, 10, 1, 3, 5, 7, 9, 11]
    csv_data = np.loadtxt(args.file_path, delimiter=',', skiprows=1)
    #csv_data = csv_data[:120]
    joint_positions = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    np_dof_pos = np.array([512 for i in range(12)])
    target_ids = [1,2,3,4,5,6,7,8,9,10,11,12]
    
    enable_torque_for(target_ids)
    
    group_speed = dxl.GroupSyncWrite(portHandler, packetHandler, ADDR_MOVING_SPEED, LEN_2BYTE)
    for motor_id in target_ids:
        speed = 1000
        param_speed = [speed & 0xFF, (speed >> 8) & 0xFF]
        group_speed.addParam(motor_id, param_speed)
    group_speed.txPacket()
    group_speed.clearParam()
    
    group_pos = dxl.GroupSyncWrite(portHandler, packetHandler, ADDR_GOAL_POSITION, LEN_2BYTE)
    for data in csv_data:
        start = time.time()
        action = data[1:13]
        #action[4] = - action[4]
        action[5] = - action[5]
        action[6] = - action[6]
        action[8] = - action[8]
        print("actions",action)
        action = action * (1024/300) *(180/ np.pi) + np_dof_pos[np.argsort(idx_id)]
        print("actions_position", action)
        
        goal_positions = {
            i+1 : action[idx_id[i]] for i in range(12)
        }
        print("goal positions:", goal_positions)
        
        group_pos.clearParam()
        for motor_id in target_ids:
            pos = int(goal_positions.get(motor_id, DOF_POSITION.get(motor_id, 512)))
            pos = max(0, min(1023, pos))
            param_pos = [pos & 0xFF, (pos >> 8) & 0xFF]
            group_pos.addParam(motor_id, param_pos)
        
        dxl_comm_result = group_pos.txPacket()
        if dxl_comm_result != dxl.COMM_SUCCESS:
            print(f"포지션 전송 오류: {packetHandler.getTxRxResult(dxl_comm_result)}")
        
        print("moved")
        elapsed = time.time() - start
        print(f"Loop time: {elapsed*1000:.1f}ms, Freq: {1/elapsed:.1f}Hz")
        
        time.sleep(max(0, 1/int(args.frequency) - elapsed))

    for motor_id in target_ids:
        packetHandler.write1ByteTxRx(portHandler, motor_id, ADDR_TORQUE_ENABLE, 0)

    close_port()
