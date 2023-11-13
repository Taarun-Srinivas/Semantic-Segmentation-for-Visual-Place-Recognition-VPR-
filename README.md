# Semantic-Segmentation-for-Visual-Place-Recognition-VPR-

# Idea
The main idea of this work is to generate semantic data for training an autonomous car, detect and have a good understanding of it's environment. In this work, I have generated semantic data for different seasonal modalities. These seasonal modalities are clear night, clear sunset, rain noon, and wet cloudy weather. This algorithm is built on top of [STEGO](https://github.com/mhamilton723/STEGO) algorithm. The objective of this work is to use this semantic data to achieve [Visual Place Recognition (VPR)](https://arxiv.org/abs/2303.03281)

# requirements
- Python - 3.9
- OpenCV - 4.6.0
- Pytorch - 1.13.1 + cu116

# Workflow
the main code or rather, the starting point can be found at `stego_data_gen.py`. This file saves the semantic data in an output folder. The `video_writer.py` file generates the visualization similar to the one given below.

# Visualization
 - Clear Night 
<div align = "center">
<img src="https://github.com/Taarun-Srinivas/Semantic-Segmentation-for-Visual-Place-Recognition-VPR-/assets/52371207/95683482-b9a2-4344-abad-48d021cf1016" width="400" height = "400" alt="clear_night_input">
<img src="https://github.com/Taarun-Srinivas/Semantic-Segmentation-for-Visual-Place-Recognition-VPR-/assets/52371207/a67697b1-9598-4ab3-b787-1446e7db82c8" width="400" height = "400" alt="clear_night_output">
</div>
