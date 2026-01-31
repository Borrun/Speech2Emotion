import argparse
import json
import time

TYPE_MAP = ["happy", "sad", "angry", "fear", "calm", "confused"]

def led_set(type_id: int, level_id: int):
    # TODO: replace with your hardware interface
    print("LED:", TYPE_MAP[type_id], "level", level_id)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--code", required=True)
    args = ap.parse_args()

    with open(args.code, "r", encoding="utf-8") as f:
        obj = json.load(f)

    fps = int(obj.get("fps", 30))
    t0 = time.time() + 0.3
    for fr in obj["frames"]:
        target = t0 + fr["i"] / fps
        dt = target - time.time()
        if dt > 0:
            time.sleep(dt)
        led_set(fr["type_id"], fr["level_id"])

if __name__ == "__main__":
    main()
