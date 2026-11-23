"""Generates static/alarm.wav (two-tone siren). Run once if the file is missing."""
import math, struct, wave, os
import config

def main(path=config.ALARM_PATH, sr=22050):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    frames = []
    for i in range(int(sr * 1.0)):
        f = 1200 if (i // (sr // 4)) % 2 == 0 else 800
        frames.append(struct.pack("<h", int(24000 * math.sin(2 * math.pi * f * i / sr))))
    with wave.open(path, "w") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr)
        w.writeframes(b"".join(frames))
    print("Saved", path)

if __name__ == "__main__":
    main()
