from config import PERSON_ALIASES, REQUIRED_CLASSES, CLASS_SYNONYMS, CONF_THRESH, CONTAINMENT_RATIO, PERSON_PAD_PX
from geometry import containment_ratio, pad_box

def normalize_name(name: str) -> str:
    n = name.lower().strip()
    return CLASS_SYNONYMS.get(n, n)

def parse_detections(yolo_result, class_names):
    dets = []
    for x1, y1, x2, y2, conf, cls_id in yolo_result.boxes.data.tolist():
        if conf < CONF_THRESH:
            continue
        name = class_names[int(cls_id)]
        cls_l = normalize_name(name)
        dets.append({
            "cls": name,
            "cls_l": cls_l,
            "bbox": [int(x1), int(y1), int(x2), int(y2)],
            "conf": conf
        })
    return dets

def split_person_ppe(dets):
    persons = [d for d in dets if d["cls_l"] in PERSON_ALIASES]
    ppes    = [d for d in dets if d["cls_l"] in REQUIRED_CLASSES]
    return persons, ppes

def assign_ppes_to_persons(persons, ppes):
    results = []
    for p in persons:
        pb = pad_box(p["bbox"], PERSON_PAD_PX)
        found = set()
        for q in ppes:
            if containment_ratio(q["bbox"], pb) >= CONTAINMENT_RATIO:
                found.add(q["cls_l"])
        missing = list(REQUIRED_CLASSES - found)
        results.append({"person": p, "found": found, "missing": missing})
    return results
