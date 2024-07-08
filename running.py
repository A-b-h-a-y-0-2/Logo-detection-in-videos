import torch
import av
import json
import argparse
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from utils.datasets import letterbox
from utils.general import non_max_suppression, scale_coords


def process_frame(img, model, device):
    # Convert to numpy array
    img = np.array(img)
    h0, w0 = img.shape[:2]  # original shape
    img, ratio, pad = letterbox(img, new_shape=640)
    h1, w1 = img.shape[:2]  # new shape

    # Convert to CHW format
    img = img.transpose((2, 0, 1))  # HWC to CHW
    img = np.expand_dims(img, axis=0)
    img = np.ascontiguousarray(img)

    # Run inference
    img = torch.from_numpy(img).to(device)
    img = img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0

    with torch.no_grad():
        pred = model(img)[0]

    # Apply NMS
    pred = non_max_suppression(pred, 0.25, 0.45, classes=None, agnostic=False)
    return pred, h0, w0, h1, w1, ratio, pad


def draw_boxes(image, boxes, labels):
    draw = ImageDraw.Draw(image)
    for (xyxy, label) in zip(boxes, labels):
        draw.rectangle(xyxy, outline='red', width=3)
        draw.text((xyxy[0], xyxy[1]), label, fill='red')
    return image


def main(video_path, output_json, model_path, max_frames, frame_interval):
    # YOLOv7 model load
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = torch.load(model_path, map_location=device)['model'].float().eval()
    model.to(device)

    # Video file path
    container = av.open(video_path)
    frame_rate = float(container.streams.video[0].average_rate)

    # Initialize results dictionary
    results = {
        "Pepsi_pts": [],
        "CocaCola_pts": []
    }

    frame_count = 0

    # Create a directory to save annotated frames
    output_dir = Path('output/annotated_frames')
    output_dir.mkdir(parents=True, exist_ok=True)

    # Loop through video frames
    for frame in container.decode(video=0):
        if frame_count % frame_interval == 0:
            # Convert the frame to an image
            img = frame.to_image()

            # Run inference
            results_frame, h0, w0, h1, w1, ratio, pad = process_frame(
                img, model, device)

            # Get current timestamp
            timestamp = frame_count / frame_rate

            # Parse results
            boxes = []
            labels = []
            for det in results_frame:
                if len(det):
                    # Rescale boxes from img1 to img0
                    det[:, :4] = scale_coords(
                        (h1, w1), det[:, :4], (h0, w0)).round()
                    for *xyxy, conf, cls in det:
                        label = int(cls)
                        if label == 1:  # Assuming 0 is the label for Pepsi
                            results["Pepsi_pts"].append(timestamp)
                            labels.append('Pepsi')
                        elif label == 0:  # Assuming 1 is the label for CocaCola
                            results["CocaCola_pts"].append(timestamp)
                            labels.append('CocaCola')
                        boxes.append(xyxy)

            # Annotate frame
            annotated_frame = draw_boxes(img, boxes, labels)

            # Save the annotated frame
            annotated_frame.save(output_dir / f'frame_{frame_count}.jpg')

        frame_count += 1
        # Print progress
        if frame_count % (frame_interval * 10) == 0:
            print(f"Processed {frame_count} frames")

        # Break the loop if the maximum frame count is reached
        if frame_count >= max_frames:
            break

    # Save results to JSON
    with open(output_json, 'w') as f:
        json.dump(results, f)

    print("Inference complete. Results saved to results.json and annotated frames saved in 'annotated_frames' directory")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Detect Pepsi and CocaCola logos in a video.")
    parser.add_argument("video_path", type=str,
                        help="Path to the input video file.")
    parser.add_argument("output_json", type=str,
                        help="Path to the output JSON file.")
    parser.add_argument("--model_path", type=str,
                        default='models/best.pt', help="Path to the YOLOv7 model file.")
    parser.add_argument("--max_frames", type=int, default=1000,
                        help="Maximum number of frames to process.")
    parser.add_argument("--frame_interval", type=int, default=10,
                        help="Process every nth frame to save computation time.")

    args = parser.parse_args()
    main(args.video_path, args.output_json, args.model_path,
         args.max_frames, args.frame_interval)
