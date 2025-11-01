import cv2
import supervision as sv
from rfdetr import RFDETRNano
import time

# Định nghĩa nhãn của bạn
MY_CLASSES = {0: "with_mask", 1: "without_mask", 2: "mask_weared_incorrect"}

if __name__ == "__main__":
    print("[INFO] Start loading model...")
    start_time = time.time()
    model = RFDETRNano(
        pretrain_weights=r"D:\Downloads\IOT Lab2\IOT Lab2\train_4\checkpoint_best_total.pth")
    print(
        f"[INFO] Model loaded successfully in {time.time() - start_time:.2f} seconds!")

    url = "http://192.168.1.45/stream"
    cap = cv2.VideoCapture(url, cv2.CAP_ANY)

    if not cap.isOpened():
        print("[ERROR] Cannot open RTSP stream!")
        exit(1)
    else:
        print("[INFO] RTSP stream opened successfully.")

    frame_count = 0
    while True:
        success, frame = cap.read()
        if not success:
            print("[WARNING] Failed to read frame. Exiting...")
            break
        frame_count += 1
        print(f"[INFO] Processing frame {frame_count}")

        # Dự đoán
        detections = model.predict(frame[:, :, ::-1].copy(), threshold=0.5)
        print(f"[INFO] Detected {len(detections)} objects")

        labels = [
            f"{MY_CLASSES[class_id]} {confidence:.2f}"
            for class_id, confidence
            in zip(detections.class_id, detections.confidence)
        ]
        print(f"[DEBUG] Labels: {labels}")

        # Vẽ bbox và nhãn
        annotated_frame = frame.copy()
        annotated_frame = sv.BoxAnnotator().annotate(annotated_frame, detections)
        annotated_frame = sv.LabelAnnotator().annotate(
            annotated_frame, detections, labels)
        print("[INFO] Frame annotated")

        # Hiển thị
        cv2.imshow("RTSP Stream", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("[INFO] 'q' pressed. Exiting...")
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"[INFO] Total frames processed: {frame_count}")
    print("[INFO] Program ended successfully.")
