# Logo Detection in Videos
This project uses YOLOv7 to detect Pepsi and CocaCola logos in a video file and outputs the timestamps where these logos appear. To recreate the inference over a custom input, please refer to the instructions provided below.

---

## README.md

### Setup Instructions
####Prerequisites
1. Python 3.8 or later
2. PyTorch (compatible with your CUDA version, if available)


1. **Clone Repository**

   ```bash
   git clone <repository_url>
   cd <repository_name>
   ```

2. **Install Dependencies**

   Ensure you have atleast Python 3.8 installed. Install the required Python packages using pip:
  
   ```bash
   pip install -r requirements.txt
   ```
  
   This will install all necessary libraries for running the pipeline.

### Running the Pipeline

1. **Environment Setup**

   
2. **Executing the Pipeline**

   Navigate to the project directory and run the following command to execute the pipeline:

   ```bash
   python pipeline.py --input_file path/to/your/audio_file.wav
   ```

   Replace `path/to/your/audio_file.wav` with the path to your desired audio file.

3. **Output**

   - The pipeline will process the video file and print the extracted action items to the console, one per line.
   - Example output:
     ```
     Action Item 1
     Action Item 2
     Action Item 3
     ```

### Demo Video (loom.com)

Watch the demo video on [Loom](http://loom.com) demonstrating the pipeline in action, processing an audio file and displaying the extracted action items.
