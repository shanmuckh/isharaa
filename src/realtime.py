import os
import cv2
import torch
from torch import load
from model import DETR
import albumentations as A
from utils.boxes import rescale_bboxes
from utils.setup import get_classes, get_colors
from utils.logger import get_logger
from utils.rich_handlers import DetectionHandler, create_detection_live_display
import sys
import time
import signal

# Flag for graceful shutdown
running = True

def signal_handler(signum, frame):
    global running
    running = False
    print("\n[realtime] Received termination signal. Cleaning up...", flush=True)

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

# Initialize logger and handlers
logger = get_logger("realtime")
detection_handler = DetectionHandler()

logger.print_banner()
logger.realtime("Initializing real-time sign language detection...")
sys.stdout.flush()

transforms = A.Compose(
        [   
            A.Resize(224,224),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            A.ToTensorV2()
        ]
    )

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_path = os.path.join(base_dir, 'pretrained', '4426_model.pt')
if not os.path.exists(model_path):
    model_path = 'pretrained/4426_model.pt'

CLASSES = get_classes() 
COLORS = get_colors() 

model = DETR(num_classes=len(CLASSES))
model.eval()
model.load_pretrained(model_path) 

logger.realtime("Starting camera capture...")
sys.stdout.flush()

# Attempt camera open (try index 0, then index 1)
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    logger.warning("Camera index 0 unavailable, attempting index 1...")
    sys.stdout.flush()
    cap = cv2.VideoCapture(1)

if not cap.isOpened():
    logger.error("Failed to open camera device (tested index 0 and 1).")
    logger.error("Permission tip: On macOS, verify Camera access is granted in System Settings -> Privacy & Security -> Camera.")
    sys.stdout.flush()
    sys.exit(1)

# Initialize performance tracking
frame_count = 0
fps_start_time = time.time()

try:
    while cap.isOpened() and running: 
        ret, frame = cap.read()
        if not ret:
            logger.error("Failed to read frame from camera")
            sys.stdout.flush()
            break
            
        frame_h, frame_w = frame.shape[:2]
        
        # Time the inference
        inference_start = time.time()
        transformed = transforms(image=frame)
        result = model(torch.unsqueeze(transformed['image'], dim=0))
        inference_time = (time.time() - inference_start) * 1000  # Convert to ms

        probabilities = result['pred_logits'].softmax(-1)[:,:,:-1] 
        max_probs, max_classes = probabilities.max(-1)
        keep_mask = max_probs > 0.8

        batch_indices, query_indices = torch.where(keep_mask) 

        bboxes = rescale_bboxes(result['pred_boxes'][batch_indices, query_indices,:], (frame_w, frame_h))
        classes = max_classes[batch_indices, query_indices]
        probas = max_probs[batch_indices, query_indices]

        # Prepare detection results for logging
        detections = []
        for bclass, bprob, bbox in zip(classes, probas, bboxes): 
            bclass_idx = int(bclass.detach().numpy())
            bprob_val = bprob.detach().numpy() 
            x1,y1,x2,y2 = bbox.detach().numpy()
            
            class_name = CLASSES[bclass_idx] if bclass_idx < len(CLASSES) else f"Class {bclass_idx}"
            box_color = COLORS[bclass_idx % len(COLORS)]

            detections.append({
                'class': class_name,
                'confidence': float(bprob_val),
                'bbox': [float(x1), float(y1), float(x2), float(y2)]
            })
            
            # Draw bounding boxes on frame
            frame = cv2.rectangle(frame, (int(x1),int(y1)), (int(x2),int(y2)), box_color, 6)
            frame_text = f"{class_name} - {round(float(bprob_val),2)}"
            frame = cv2.rectangle(frame, (int(x1),max(0, int(y1)-40)), (min(frame_w, int(x1)+300),int(y1)), box_color, -1)
            frame = cv2.putText(frame, frame_text, (int(x1)+5,max(25, int(y1)-10)), cv2.FONT_HERSHEY_DUPLEX, 0.9, (255,255,255), 2, cv2.LINE_AA)

        # Calculate FPS
        frame_count += 1
        if frame_count % 30 == 0:  # Log every 30 frames
            elapsed_time = time.time() - fps_start_time
            fps = 30 / elapsed_time if elapsed_time > 0 else 0
            
            # Log detection results and performance
            if detections:
                detection_handler.log_detections(detections, frame_id=frame_count)
            detection_handler.log_inference_time(inference_time, fps)
            sys.stdout.flush()
            
            # Reset FPS counter
            fps_start_time = time.time()

        cv2.imshow('Frame', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'): 
            logger.realtime("Stopping real-time detection...")
            sys.stdout.flush()
            break
finally:
    cap.release() 
    cv2.destroyAllWindows() 
    logger.realtime("Camera released and windows closed.")
    sys.stdout.flush()
 
