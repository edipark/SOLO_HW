#!/bin/bash

# 대상 파일 경로 설정
TARGET="/sys/bus/usb-serial/devices/ttyUSB0/latency_timer"

# 1. 파일 존재 여부 확인
if [ ! -f "$TARGET" ]; then
    echo "오류: $TARGET 파일을 찾을 수 없습니다."
    echo "장치가 연결되어 있는지, 혹은 경로가 정확한지 확인하세요."
    exit 1
fi

# 2. 권한 변경 (744: 소유자 읽기/쓰기/실행, 그룹/기타 읽기)
echo "권한을 744로 변경 중..."
sudo chmod 744 "$TARGET"

# 3. 값 변경 (16 -> 1)
echo "latency_timer 값을 1로 변경 중..."
echo 1 | sudo tee "$TARGET" > /dev/null

# 4. 결과 확인
RESULT=$(cat "$TARGET")
echo "설정 완료. 현재 latency_timer 값: $RESULT"
