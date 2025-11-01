# webcam_fullscreen.py
import cv2
import time
from rfdetr import RFDETRNano

# ================== THIẾT LẬP CỐ ĐỊNH ==================
# Đường dẫn checkpoint bạn muốn test:
WEIGHTS_PATH = r"D:\Downloads\IOT Lab2\IOT Lab2\train_4\checkpoint_best_total.pth"

# Chỉ số webcam (0 = webcam mặc định; nếu có nhiều cam, thử 1, 2, ...)
CAM_INDEX = 0

# Ngưỡng score ban đầu
SCORE_THR = 0.50

# Mapping nhãn theo COCO sau khi bạn đã fix:
# 0: without_mask, 1: with_mask, 2: mask_weared_incorrect
CLASSES = {0: "with_mask", 1: "without_mask", 2: "mask_weared_incorrect"}

# Tùy chọn: ép độ phân giải capture (0 = giữ mặc định driver)
FRAME_W, FRAME_H = 0, 0
# ========================================================

# (Tùy chọn) dùng supervision nếu có để vẽ box/label đẹp
try:
    import supervision as sv
    USE_SV = True
except Exception:
    USE_SV = False


def main():
    print("[INFO] Loading model...")
    t0 = time.time()
    model = RFDETRNano(pretrain_weights=WEIGHTS_PATH)
    print(f"[INFO] Model loaded in {time.time() - t0:.2f}s")

    cap = cv2.VideoCapture(CAM_INDEX)
    if FRAME_W > 0:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
    if FRAME_H > 0:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)

    if not cap.isOpened():
        print(f"[ERROR] Không mở được webcam (index={CAM_INDEX}).")
        return

    win_name = "Webcam RF-DETR (Fullscreen)"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    # Fullscreen ngay khi mở
    cv2.setWindowProperty(
        win_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    if USE_SV:
        box_annotator = sv.BoxAnnotator()
        label_annotator = sv.LabelAnnotator()

    score_thr = SCORE_THR
    frame_id = 0

    print("[INFO] Phím tắt: 'q' thoát | '+'/'-' tăng/giảm threshold | 's' lưu ảnh | 'f' bật/tắt fullscreen")

    fullscreen = True
    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            print("[WARN] Mất khung hình từ webcam.")
            break

        frame_id += 1

        # Model nhận RGB
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        detections = model.predict(frame_rgb, threshold=score_thr)

        vis = frame_bgr.copy()

        if USE_SV:
            labels = [
                f"{CLASSES.get(int(cid), str(int(cid)))} {conf:.2f}"
                for cid, conf in zip(detections.class_id, detections.confidence)
            ]
            vis = box_annotator.annotate(vis, detections)
            vis = label_annotator.annotate(vis, detections, labels)
        else:
            # Vẽ thủ công nếu không có supervision
            for (x1, y1, x2, y2), cid, conf in zip(
                detections.xyxy, detections.class_id, detections.confidence
            ):
                x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
                cls_name = CLASSES.get(int(cid), str(int(cid)))
                color = (255, 0, 255) if int(cid) == 2 else (
                    0, 200, 0)  # tô khác màu cho incorrect
                cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
                cv2.putText(
                    vis, f"{cls_name} {conf:.2f}",
                    (x1, max(20, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA
                )

        # HUD nhỏ
        cv2.putText(
            vis, f"thr={score_thr:.2f} det={len(detections)}  (q: quit, +/-: thr, s: save, f: fullscreen)",
            (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2, cv2.LINE_AA
        )

        cv2.imshow(win_name, vis)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        elif key in (ord('+'), ord('=')):
            score_thr = min(0.99, score_thr + 0.05)
        elif key == ord('-'):
            score_thr = max(0.05, score_thr - 0.05)
        elif key == ord('s'):
            out_path = f"webcam_{frame_id}.jpg"
            cv2.imwrite(out_path, vis)
            print(f"[INFO] Saved {out_path}")
        elif key == ord('f'):
            # Toggle fullscreen
            fullscreen = not fullscreen
            cv2.setWindowProperty(
                win_name,
                cv2.WND_PROP_FULLSCREEN,
                cv2.WINDOW_FULLSCREEN if fullscreen else cv2.WINDOW_NORMAL
            )
            if not fullscreen:
                # Nếu thoát fullscreen, phóng to tối đa cửa sổ
                cv2.resizeWindow(win_name, 1600, 900)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
