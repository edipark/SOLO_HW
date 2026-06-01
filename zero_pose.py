import dynamixel_sdk as dxl

def init_pose():
    # check the port : ls /dev/ttyUSB*
    PORT_NAME = '/dev/ttyUSB0' 
    BAUDRATE = 1000000
    PROTOCOL_VERSION = 1.0

    MOTOR_IDS = list(range(1, 13))  # ID 1~12 for lower bodies
    INIT_POSITION = {
        1: 512,
        2: 512,
        3: 512,
        4: 512,
        5: 512,
        6: 512,
        7: 512,
        8: 512,
        9: 512,
        10: 512,
        11: 512,
        12: 512,
    }

    # Memory address for goal position digit
    ADDR_GOAL_POSITION = 30  


    portHandler = dxl.PortHandler(PORT_NAME)
    packetHandler = dxl.PacketHandler(PROTOCOL_VERSION)

    if not portHandler.openPort():
        print("포트 열기 실패")
        exit()

    if not portHandler.setBaudRate(BAUDRATE):
        print("보레이트 설정 실패")
        exit()

    for motor_id in MOTOR_IDS:
        goal_position = INIT_POSITION.get(motor_id, 512)
        dxl_comm_result, dxl_error = packetHandler.write4ByteTxRx(
            portHandler, motor_id, ADDR_GOAL_POSITION, goal_position
        )
        if dxl_comm_result != dxl.COMM_SUCCESS:
            print(f"모터 {motor_id} 통신 오류: {packetHandler.getTxRxResult(dxl_comm_result)}")
        elif dxl_error != 0:
            print(f"모터 {motor_id} 오류: {packetHandler.getRxPacketError(dxl_error)}")
        else:
            print(f"모터 {motor_id} 초기화 완료")

    portHandler.closePort()
if __name__ == "__main__":
    init_pose()
