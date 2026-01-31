import argparse
import json

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--code", required=True)
    args = ap.parse_args()

    with open(args.code, "r", encoding="utf-8") as f:
        obj = json.load(f)

    for fr in obj["frames"][:30]:
        print(fr)

if __name__ == "__main__":
    main()
