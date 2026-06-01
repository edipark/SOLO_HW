import numpy as np
import time
from inference.onnx_policy import TeacherPolicyONNX

def check_teacher_latency(onnx_path, iterations=1000):
    # 1. 모델 로드
    print(f"--- Teacher Policy 로드 중: {onnx_path} ---")
    try:
        policy = TeacherPolicyONNX(onnx_path)
    except Exception as e:
        print(f"모델 로드 실패: {e}")
        return

    # 2. 더미 입력 생성 (OBS_DIM = 43)
    # 실제 deploy 코드와 동일한 float32 타입 필수
    dummy_obs = np.random.randn(43).astype(np.float32)

    # 3. 워밍업 (Cold-start 방지)
    # ONNX Runtime이 내부 최적화를 마칠 때까지 초기 실행은 느릴 수 있음
    print("워밍업(Warm-up) 중...")
    for _ in range(20):
        _ = policy.predict(dummy_obs)

    # 4. 순수 추론 시간 측정
    print(f"측정 시작 ({iterations}회 반복)...")
    latencies = []
    
    for i in range(iterations):
        t0 = time.perf_counter()
        
        # --- 측정 구간 ---
        _ = policy.predict(dummy_obs)
        # ----------------
        
        dt = (time.perf_counter() - t0) * 1000  # ms 단위
        latencies.append(dt)

    # 5. 결과 분석
    avg_ms = np.mean(latencies)
    max_ms = np.max(latencies)
    min_ms = np.min(latencies)
    std_ms = np.std(latencies)

    print("\n" + "="*30)
    print(f"Teacher Policy Latency Result")
    print("="*30)
    print(f"평균(Avg): {avg_ms:.4f} ms")
    print(f"최대(Max): {max_ms:.4f} ms")
    print(f"최소(Min): {min_ms:.4f} ms")
    print(f"표준편차(Std): {std_ms:.4f} ms")
    print("-" * 30)
    print(f"이론적 처리량: {1000/avg_ms:.2f} FPS")
    print("="*30)

if __name__ == "__main__":
    # 실제 사용 중인 ONNX 파일 경로를 넣어주세요
    TEACHER_ONNX_PATH = "models/teacher_policy.onnx" 
    check_teacher_latency(TEACHER_ONNX_PATH)