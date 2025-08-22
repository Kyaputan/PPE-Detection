import cv2
from detection import load_model, infer
from ppe_logic import parse_detections, split_person_ppe, assign_ppes_to_persons
from drawing import draw_ppes, draw_person_status
from camera import VideoSource, should_infer

def main():
    model, class_names = load_model()
    cam = VideoSource('rtsp://root01:12345678@192.168.1.102:554/stream1', cv2.CAP_FFMPEG)        
    infer_every_n = 5           
    last_person_results = []
    last_ppes = []

    frame_idx = 0
    while True:
        ok, frame = cam.read()
        if not ok:
            break
        frame = cv2.resize(frame, (640, 640))
        if should_infer(frame_idx, infer_every_n):
            yres = infer(model, frame)
            dets = parse_detections(yres, class_names)
            persons, ppes = split_person_ppe(dets)
            person_results = assign_ppes_to_persons(persons, ppes)
            last_person_results, last_ppes = person_results, ppes
        else:
            person_results, ppes = last_person_results, last_ppes

        draw_ppes(frame, ppes)
        draw_person_status(frame, person_results)

        cv2.imshow("PPE Detection (Per Person)", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        frame_idx += 1

    cam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
