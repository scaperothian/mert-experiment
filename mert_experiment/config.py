MODEL_ID = "m-a-p/MERT-v1-95M"
TARGET_SR = 24_000          # MERT expects 24 kHz mono
WINDOW_SEC = 1.0            # sliding window length over MERT frames
HOP_SEC = 0.5               # 50% overlap
LAYERS_TO_PROBE = [4, 6, 8, 10, 12]  # MERT-95M: 12 transformer layers + 1 conv
MERT_FRAME_RATE = 75        # MERT outputs ~75 frames per second
