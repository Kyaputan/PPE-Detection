import cv2

def draw_ppes(frame, ppes):
    for d in ppes:
        x1, y1, x2, y2 = d["bbox"]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 1)
        cv2.putText(frame, d["cls"], (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,0), 1)

def draw_person_status(frame, person_results):
    for i, r in enumerate(person_results):
        p = r["person"]
        x1, y1, x2, y2 = p["bbox"]
        ok = (len(r["missing"]) == 0)
        color = (0,255,0) if ok else (0,0,255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        txt = f"Person {i+1}: " + ("OK" if ok else "Missing: " + ", ".join(r["missing"]))
        cv2.putText(frame, txt, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
