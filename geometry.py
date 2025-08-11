def area(b):
    return max(0, b[2]-b[0]) * max(0, b[3]-b[1])

def intersection(b1, b2):
    x1 = max(b1[0], b2[0]); y1 = max(b1[1], b2[1])
    x2 = min(b1[2], b2[2]); y2 = min(b1[3], b2[3])
    if x2 <= x1 or y2 <= y1:
        return 0
    return (x2-x1) * (y2-y1)

def containment_ratio(inner, outer):
    inter = intersection(inner, outer)
    ai = area(inner)
    return inter / (ai + 1e-6)

def pad_box(b, px=10):
    return [b[0]-px, b[1]-px, b[2]+px, b[3]+px]
