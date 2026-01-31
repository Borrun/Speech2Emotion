import json
import time
import argparse


def load_code(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav", required=False, help="optional, for logging only")
    ap.add_argument("--code", required=True, help="outputs/emotion_codes/*.json")
    ap.add_argument("--start_delay", type=float, default=0.5, help="seconds")
    args = ap.parse_args()

    obj = load_code(args.code)
    fps = int(obj.get("fps", 30))
    frames = obj.get("frames", [])

    print("wav:", args.wav)
    print("code:", args.code)
    print("fps:", fps, "frames:", len(frames))
    print("type_map:", obj.get("type_map"))

    t0 = time.time() + float(args.start_delay)
    for fr in frames:
        target = t0 + float(fr["i"]) / float(fps)
        now = time.time()
        dt = target - now
        if dt > 0:
            time.sleep(dt)

        # Replace with your local actuator calls:
        print(f'[t={fr["t"]:.3f}] type_id={fr["type_id"]} level_id={fr["level_id"]}')

    print("done.")


if __name__ == "__main__":
    main()
