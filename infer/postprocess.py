from typing import List, Dict, Any


def apply_boundary_hold(frames: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Optional postprocess: hold previous state unless boundary==1.
    If your inference doesn't output boundary, this is a no-op.
    """
    if not frames:
        return frames

    out = []
    cur_type = frames[0]["type_id"]
    cur_lvl = frames[0]["level_id"]

    for f in frames:
        b = f.get("boundary", None)
        if b is None:
            out.append(f)
            continue
        if int(b) == 1:
            cur_type = f["type_id"]
            cur_lvl = f["level_id"]
        nf = dict(f)
        nf["type_id"] = cur_type
        nf["level_id"] = cur_lvl
        out.append(nf)
    return out
