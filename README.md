# Logo Detection in Videos
This project uses YOLOv7 to detect Pepsi and CocaCola logos in a video file and outputs the timestamps where these logos appear. To recreate the inference over a custom input, please refer to the instructions provided below.

---

## README.md

### Setup Instructions
Prerequisites
1. Python 3.8 or later
2. PyTorch (compatible with your CUDA version, if available)


1. **Clone Repository**

   ```bash
   git clone https://github.com/A-b-h-a-y-0-2/Logo-detection-in-videos
   cd Logo-detection-in-videos
   ```

2. **Install Dependencies**

   Ensure you have atleast Python 3.8 installed. Install the required Python packages using pip:
  
   ```bash
   pip install -r requirements.txt
   ```
  
   This will install all necessary libraries for running the pipeline.

3. Ensure you have the YOLOv7 model file `best.pt` in the repository directory.

## Running the Pipeline

To run the pipeline and detect logos in a video file, use the following command:

```sh
python running.py path/to/video.mp4 output.json --model_path path/to/best.pt --max_frames 1000 --frame_interval 10
```

### Command-line Arguments

- `video_path`: Path to the input video file.
- `output_json`: Path to the output JSON file where results will be saved.
- `--model_path`: Path to the YOLOv7 model file (default is `best.pt`).
- `--max_frames`: Maximum number of frames to process (default is 1000).
- `--frame_interval`: Process every nth frame to save computation time (default is 10).

### Example

```sh
python running.py videoplayback.mp4 results.json 
```

This will process the video file `videoplayback.mp4`, detect Pepsi and CocaCola logos, and save the timestamps in `results.json`.
The output contains 2 items:
1. result.json
2. Annotated frames

## Demo Video

A [loom.com](http://loom.com) video recording of the demo, showing the pipeline processing an audio file and returning the desired list of action items, can be found [here](https://www.loom.com/share/e98a5d3ad5634de1b1f0b0632a6c0915?sid=0d9c8a05-a200-40c2-b353-f016a6142459).


### Demo Video (loom.com)

Watch the demo video on [Loom](http://loom.com) demonstrating the pipeline in action, processing an audio file and displaying the extracted action items.
