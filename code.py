# Install YOLOv8 (Ultralytics)
!pip install ultralytics opencv-python pandas

# Import libraries
import cv2
import pandas as pd
from ultralytics import YOLO

# Load YOLOv8 pretrained model (COCO dataset: includes 'person')
model = YOLO("yolov8n.pt")  # small + fast model (use yolov8s.pt for more accuracy)

# Open video source (0 = webcam, or give path to video file)
cap = cv2.VideoCapture(0)

# Excel file to save crowd count
crowd_data = []

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run detection
    results = model(frame, stream=True)

    person_count = 0
    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])  # class ID
            if cls == 0:  # class 0 = 'person'
                person_count += 1
                # Draw bounding box
                x1, y1, x2, y2 = box.xyxy[0]
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(frame, "Person", (int(x1), int(y1)-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Show count on screen
    cv2.putText(frame, f"People Count: {person_count}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

    # Save data
    crowd_data.append({"People_Count": person_count})

    # Display video
    cv2.imshow("Crowd Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# Save results to Excel
df = pd.DataFrame(crowd_data)
df.to_excel("crowd_count.xlsx", index=False)

print("✅ Crowd detection finished. Data saved in crowd_count.xlsx")
