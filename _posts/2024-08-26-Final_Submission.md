---
layout: post
title: "Final Submission"
subtitle: "A new Beginning. End of my GSoC project"
date: 2024-08-26
background: "/img/main1.png"
tags: gsoc
---

# GSoC-2024: Final Report

<div align="center">
<img width="500" alt="intro" src="https://gist.github.com/user-attachments/assets/ad53f921-c611-4040-96e7-a8de4e1540f7">
</div>

|              |                                                                                                                                                       |
| ------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------- |
| Contributor  | [Aryan Nanda](https://github.com/AryanNanda17)                                                                                                        |
| Organization | [BeagleBoard.org](https://www.beagleboard.org/)                                                                                                       |
| Mentor       | [Deepak Khatri](https://github.com/lorforlinux), [Kumar Abhishek](https://github.com/abhishek-kakkar), [Jason Kridner](https://github.com/jadonk)     |
| Project      | [Enhanced Media Experience with AI-Powered Commercial Detection and Replacement](https://summerofcode.withgoogle.com/programs/2024/projects/UOX7iDEU) |

## Important Links:

- Main Code Repository - [Gitlab Link](https://openbeagle.org/gsoc/2024/commercial-detection)
- Mirror Code Repository - [Github Link](https://github.com/AryanNanda17/GSoC_2024-Enhanced_Media_Experience_with_AI-Powered_Commercial_Detection_and_Replacement)
- Pre GSoC - [Project Description and Discussion](https://forum.beagleboard.org/t/enhanced-media-experience-with-ai-powered-commercial-detection-and-replacement/37358)
- Proposal Link - [Commercial Detection and Replacement Proposal](https://gsoc.beagleboard.io/proposals/commercial_detection_and_replacement.html)
- Project Intro Video - [Youtube Link](https://www.youtube.com/watch?v=Kagg8JycOfo)
- Weekly Updates - [Forum Thread](https://forum.beagleboard.org/t/weekly-progress-report-thread-enhanced-media-experience-with-ai-powered-commercial-detection-and-replacement/38487)
- Blogs Site - [Biweekly Blogs](https://aryannanda17.github.io/Blogs/)
- Discord Discussion Link - [Discord Project Thread](https://discord.com/channels/1108795636956024986/1246133338343866420)

---

### **I have given very detailed weekly updates [here](https://forum.beagleboard.org/t/weekly-progress-report-thread-enhanced-media-experience-with-ai-powered-commercial-detection-and-replacement/38487)**

### **Also, I have written very detailed blogs on the approaches I used and errors I encountered [here](https://aryannanda17.github.io/Blogs/)**

---

## Pre-GSoC Period

Getting into GSoC was one of my goal which I aimed to achieve before finishing college. On May 1, I was thrilled to be accepted into my preferred project with a highly respected open-source organization. I’ll always cherish the hours of research, the extensive documentation I reviewed, and the meaningful discussions I had during that time. I interacted frequently with my mentors and community in the process, which you can see [here](https://forum.beagleboard.org/t/enhanced-media-experience-with-ai-powered-commercial-detection-and-replacement/37358). These interactions were invaluable, as they helped resolve many potential blockers that could have caused issues later on.

## Community-bonding period

Contributors use this Period to connect with their mentors, other Gsoc Candidates and the organisation as a whole!

- In BeagleBoard we usually have weekly meetings to discuss about the projects and solve the difficulties of everyone. So attending those meetings are mandatory as it involves your communication with mentors and other gsoc contributors. Also the round table discussion solves all queries which you have as all the students and mentors give solutions and guidelines on it. Additionally we can also set a meeting with the mentor if required.

- In BeagleBoard, we have a specific idea of presenting our projects to the community and others who are interested so we can get familier with other's project as well and by that time our concrete strategy of project is also ready.

Here is my presentation for reference:- [Project Presentation](https://www.youtube.com/watch?v=Kagg8JycOfo)

- Lastly, I explored [Youtube 8M-dataset](https://github.com/google/youtube-8m) and read the documentation of [BeagleBone AI-64](https://www.beagleboard.org/boards/beaglebone-ai-64) to give me a boost before the start of Coding period.

## Coding Period

The goal of my GSoC project was to make a system that is capable of detecting commercials from media stream and accelerate the system using the C7x DSPs and DL accelerators present in BeagleBone AI-64.

### Dataset Collection and Feature Extraction

The dataset Collection was in itself a huge task because videos occupies a lot of space and having a large dataset of videos is not feasible. For projects like “Enhanced Media Experience with AI-Powered Commercial Detection and Replacement,” obtaining comprehensive datasets is crucial. However, acquiring large-scale video datasets, particularly for commercial and non-commercial content, presents challenges due to their size and storage requirements.

It was my finding in the Pre-GSoC period that tells that the YouTube-8M dataset stands out as a significant resource for this project. It includes millions of YouTube video IDs with detailed machine-generated annotations across a diverse set of visual entities, including "Television Advertisement," which is particularly relevant here

**Pro-Tip** :- Use the Pre-GSoC period very well. Clear out all your doubts, research details of your project. You will definitely be benefitted from it later.

Code Snippet of implementation:

```python3
counter = 0
for tfrecord_file in tfrecord_files:
    print(f'counter {counter}')
    counter+=1
    print(f'Processing file: {tfrecord_file}')
    # Iterate over each tfrecord in the TFRecord file
    try:
        for record in tf.compat.v1.python_io.tf_record_iterator(tfrecord_file):
            try:
                # Parse the SequenceExample from the binary record data
                seq_example = tf.train.SequenceExample.FromString(record)

                # Extract the video_id and labels
                vid_id = seq_example.context.feature['id'].bytes_list.value[0].decode('UTF-8')
                labels_list = seq_example.context.feature['labels'].int64_list.value

                # Check if the label ID 315 is in the labels list (label id 315 corresponds to Television Advertisement)
                if 315 in labels_list:
                    # Append video ID and labels to the filtered lists
                    filtered_vid_ids.append(vid_id)
                    filtered_labels.append(labels_list)

                    # Lists to store frame-level and audio-level features
                    rgb_features = []
                    audio_features = []

                    # Iterate over each frame-level feature pair (RGB and audio) in the SequenceExample
                    for rgb_feature, audio_feature in zip(seq_example.feature_lists.feature_list['rgb'].feature,
                                                          seq_example.feature_lists.feature_list['audio'].feature):

                        # Decode the quantized RGB features and audio features
                        decoded_rgb = decode_quantized_features(rgb_feature.bytes_list.value[0])
                        decoded_audio = decode_quantized_features(audio_feature.bytes_list.value[0])

                        rgb_features.append(decoded_rgb)
                        audio_features.append(decoded_audio)

                    # Append frame-level lists to the main filtered lists
                    filtered_rgb.append(rgb_features)
                    filtered_audio.append(audio_features)
                else:
                    # Check if the labels list contains any of the specific labels
                    if any(label in specific_labels for label in labels_list):
                        # Append video ID and labels to the non-filtered lists
                        non_filtered_vid_ids.append(vid_id)
                        non_filtered_labels.append(labels_list)

                        # Lists to store frame-level and audio-level features
                        rgb_features = []
                        audio_features = []

                        # Iterate over each frame-level feature pair (RGB and audio) in the SequenceExample
                        for rgb_feature, audio_feature in zip(seq_example.feature_lists.feature_list['rgb'].feature,
                                                              seq_example.feature_lists.feature_list['audio'].feature):

                            # Decode the quantized RGB features and audio features
                            decoded_rgb = decode_quantized_features(rgb_feature.bytes_list.value[0])
                            decoded_audio = decode_quantized_features(audio_feature.bytes_list.value[0])

                            rgb_features.append(decoded_rgb)
                            audio_features.append(decoded_audio)

                        # Append frame-level lists to the main non-filtered lists
                        non_filtered_rgb.append(rgb_features)
                        non_filtered_audio.append(audio_features)

            except tf.errors.DataLossError as e:
                print(f'Skipping corrupted record in file {tfrecord_file}: {e}')
    except tf.errors.DataLossError as e:
        print(f'Skipping entire file due to corruption: {tfrecord_file}: {e}')
```

- I have written a very detailed blog on the process of Dataset Collection and Feature Extraction which I used [here](https://aryannanda17.github.io/Blogs/2024/06/16/DatasetCollection_And_FeatureExtraction.html).

### Dataset Pre-processing:-

After dataset collection and feature extraction, the shape of the different features is as shown:

- commercialRgb : [4364][119-300][1024]
- commercialAudio : [4364][119-300][128]
- nonCommercialRgb : [4600][109-301][1024]
- nonCommercialAudio : [4600][109-301][128]

Here, the first dimension represents the number of videos whose features are extracted. This is why this is constant for both audio and visual features of same class as both belongs to same number of videos. For example, 4364 represents that audio-visual features of 4364 videos are extracted.

The second dimension is the number of frames in each video. The YouTube-8M dataset limits the feature extraction process to the first 300 seconds with 1 frame per second. By this, every video should have 300 frames (1 per second for 300 seconds).

Then why is it variable? This is because the video could also be shorter than 300 seconds; in such a case, the number of frames would be fewer than 300.

The third dimension is where the features lie. Corresponding to every frame, there are visual features of length 1024 and audio features of length 128.

- Next I merged the audio and visual features

Code snippet of implementation:-

```python3
 # Function for merging audio-visual features
def merge_visual_audio_features(visual_features, audio_features):
    """
    This Function merges visual and audio features for each video.
    Parameters:
    - visual_features: list of length num_videos, where each element is a list of shape (num_frames_visual[i], 1024).
    - audio_features: list of length num_videos, where each element is a list of shape (num_frames_audio[i], 128).
    - num_frames_visual[i] = num_frames_audio[i]
    Returns:
    - merged_features: list of length num_videos, where each element is a numpy array of shape (max_num_frames, 1152).

    Note:
    - num_videos is the number of audio-visual features = ( len(visual_features) || len(audio_features) ) = 4364
    """
    merged_features = []

    for visual, audio in zip(visual_features, audio_features):
        # Convert each list to numpy arrays
        visual_array = np.array(visual)
        audio_array = np.array(audio)

        # Concatenate visual and audio features for this video
        merged_video_features = np.concatenate((visual_array, audio_array), axis=1)
        merged_features.append(merged_video_features)

    return merged_features
```

- Now, uniform features are generated:-

```python3
def preprocess_features(max_frames, features):
    """
    Preprocess video features by trimming or padding them to a fixed number of frames.

    Parameters:
    - max_frames (int) - The maximum number of frames each video should have.
    - features (list of ndarray) -  A list where each element is a numpy array representing the features of a video. Each array is of shape (number_of_frames, number_Vf).

    Returns:
    - video_data (ndarray) -  A numpy array of shape (num_videos, max_frames, number_Vf) containing the preprocessed video data.
    """
    num_videos = len(features)
    number_Vf = features[0].shape[1]
    video_data = np.zeros((num_videos, max_frames, number_Vf))

    # Process each video
    for i, video in enumerate(features):
        length = len(video)
        if length >= max_frames:
            # Trim the video if longer than max_frames
            video_data[i, :, :] = video[:max_frames]
        else:
            # Pad the video with zeros if shorter than max_frames
            video_data[i, :length, :] = video

    return video_data

def convertToNumpyArray(features):
    """
    Convert a list of features to a NumPy array.

    Parameters:
    - features (list) - A list of features to be converted to a NumPy array.

    Returns:
    - numpy.ndarray: A NumPy array containing the elements of the input list.
    """
    return np.array(features)
```

**Pro-tip:** (Always ensure that before pushing any Python code, include comments in the NumPy docstring style.)

After considering the trade-off between padding and trimming, I chose 150 as max_frames length. This generated a NumPy array of dimensions [len(features)][max_frames(=150)][1152].

- Again I have written a very detailed blog on dataset Pre-processing [here](https://aryannanda17.github.io/Blogs/2024/06/30/DatasetPreProcessing.html).

### Model pipeline

I implemented a number of models so that I can compare the results and finally take out the best one.

The results of all the models is as shown:

<table style="width: 100%; border: 1px solid #333; border-collapse: collapse;">
  <thead>
    <tr>
      <th style="border: 1px solid #333; padding: 8px; text-align: center; background-color: #f2f2f2;">Model</th>
      <th style="border: 1px solid #333; padding: 8px; text-align: center; background-color: #f2f2f2;">Accuracy</th>
      <th style="border: 1px solid #333; padding: 8px; text-align: center; background-color: #f2f2f2;">True Positives</th>
      <th style="border: 1px solid #333; padding: 8px; text-align: center; background-color: #f2f2f2;">False Positives</th>
      <th style="border: 1px solid #333; padding: 8px; text-align: center; background-color: #f2f2f2;">True Negatives</th>
      <th style="border: 1px solid #333; padding: 8px; text-align: center; background-color: #f2f2f2;">False Negatives</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td style="border: 1px solid #333; padding: 8px; text-align: center;">Unidirectional LSTMs</td>
      <td style="border: 1px solid #333; padding: 8px; text-align: center;">96.28%</td>
      <td style="border: 1px solid #333; padding: 8px; text-align: center;">628</td>
      <td style="border: 1px solid #333; padding: 8px; text-align: center;">30</td>
      <td style="border: 1px solid #333; padding: 8px; text-align: center;"><b>667</b></td>
      <td style="border: 1px solid #333; padding: 8px; text-align: center;"><b>20</b></td>
    </tr>
    <tr>
      <td style="border: 1px solid #333; padding: 8px; text-align: center;">Bidirectional LSTMs</td>
      <td style="border: 1px solid #333; padding: 8px; text-align: center;"><b>97.1%</b></td>
      <td style="border: 1px solid #333; padding: 8px; text-align: center;"><b>648</b></td>
      <td style="border: 1px solid #333; padding: 8px; text-align: center;"><b>10</b></td>
      <td style="border: 1px solid #333; padding: 8px; text-align: center;">652</td>
      <td style="border: 1px solid #333; padding: 8px; text-align: center;">35</td>
    </tr>
    <tr>
      <td style="border: 1px solid #333; padding: 8px; text-align: center;">CNNs</td>
      <td style="border: 1px solid #333; padding: 8px; text-align: center;">93.97%</td>
      <td style="border: 1px solid #333; padding: 8px; text-align: center;">623</td>
      <td style="border: 1px solid #333; padding: 8px; text-align: center;">35</td>
      <td style="border: 1px solid #333; padding: 8px; text-align: center;">646</td>
      <td style="border: 1px solid #333; padding: 8px; text-align: center;">41</td>
    </tr>
    <tr>
      <td style="border: 1px solid #333; padding: 8px; text-align: center;">LSTMs+CNNs</td>
      <td style="border: 1px solid #333; padding: 8px; text-align: center;">54.57%</td>
      <td style="border: 1px solid #333; padding: 8px; text-align: center;">146</td>
      <td style="border: 1px solid #333; padding: 8px; text-align: center;">512</td>
      <td style="border: 1px solid #333; padding: 8px; text-align: center;">588</td>
      <td style="border: 1px solid #333; padding: 8px; text-align: center;">99</td>
    </tr>
    <tr>
      <td style="border: 1px solid #333; padding: 8px; text-align: center;">Transformers</td>
      <td style="border: 1px solid #333; padding: 8px; text-align: center;">96.05%</td>
      <td style="border: 1px solid #333; padding: 8px; text-align: center;">640</td>
      <td style="border: 1px solid #333; padding: 8px; text-align: center;">18</td>
      <td style="border: 1px solid #333; padding: 8px; text-align: center;">652</td>
      <td style="border: 1px solid #333; padding: 8px; text-align: center;">35</td>
    </tr>
  </tbody>
</table>

From the results, it is clear that the LSTM models gave the best accuracy. But, later I found out that in BeagleBone AI-64, the supported version of tidl is [08_02_00_05](https://github.com/TexasInstruments/edgeai-tidl-tools/releases/tag/08_02_00_05). And, this version of tidl only supports CNNs. In comparison, the latest version of tidl [10_00_04_00](https://github.com/TexasInstruments/edgeai-tidl-tools/releases/tag/10_00_04_00) supports transformers layers as well. Then, I had no other choice but to use a CNNs model.

- Again, I have a very detailed blog on the model pipeline and training process which I used [here](https://aryannanda17.github.io/Blogs/2024/07/14/GSoC-about.html)

#### How CNNs will be able to recognize sequential meaning?

CNNs are typically used for learning spatial patterns, but my problem required understanding sequential meaning. In advertisements, for instance, a sequence often starts with some content and ends with a logo appearing. To capture this sequential aspect, I trained my model on sequences of frames rather than individual frames. Instead of processing each frame independently, I grouped multiple frames together and trained the model on these sets. This approach introduced sequential context, allowing the model to learn from the sequence of frames rather than just one frame at a time. This significantly improved the model’s real-time performan

Snippet of my model.summary() after fixing the batch size to 1:

```python3
  Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #
=================================================================
 reshape (Reshape)           (1, 150, 1152, 1)         0

 conv2d (Conv2D)             (1, 150, 1152, 32)        320

 max_pooling2d (MaxPooling2  (1, 75, 576, 32)          0
 D)

 conv2d_1 (Conv2D)           (1, 75, 576, 64)          18496

 max_pooling2d_1 (MaxPoolin  (1, 37, 288, 64)          0
 g2D)

 conv2d_2 (Conv2D)           (1, 37, 288, 128)         73856

 max_pooling2d_2 (MaxPoolin  (1, 18, 144, 128)         0
 g2D)

 flatten (Flatten)           (1, 331776)               0

 dense (Dense)               (1, 256)                  84934912

 dropout (Dropout)           (1, 256)                  0

 dense_1 (Dense)             (1, 128)                  32896

 dropout_1 (Dropout)         (1, 128)                  0

 dense_2 (Dense)             (1, 1)                    129

=================================================================
Total params: 85060609 (324.48 MB)
Trainable params: 85060609 (324.48 MB)
Non-trainable params: 0 (0.00 Byte)
_________________________________________________________________
```

In one inference, the model output 0(sequence of frames is of non-commercial video) or 1(sequence of frames is of commercial video).

### Model Compilation

- I have written in detail the steps of model compilation and inferencing on BeagleBone AI-64 using C7x DSPs and DL Accelerator and the errors I faced in these blogs:-
  - [Introduction to edgeai](https://aryannanda17.github.io/Blogs/2024/08/04/GSoC-edgeai.html)
  - [edgeai continued](https://aryannanda17.github.io/Blogs/2024/08/18/GSoC-edgeai_continued.html)

In summary, after training the model, I fixed its batch size to 1 to avoid `RuntimeError: Attempting to use a delegate that only supports static-sized tensors with a graph that has dynamic-sized tensors` error while compiling the model.

- The output of cnn_model.get_config() before fixing the batch size to 1.

```python3
{'name': 'sequential',
 'layers': [{'module': 'keras.layers',
   'class_name': 'InputLayer',
   'config': {'batch_input_shape': (None, 150, 1152),
    'dtype': 'float32',
    'sparse': False,
    .
    .
    .
    .
```

- Code implementation to fix the batch size to 1:-

```python3
config = cnn_model.get_config()
for layer in config['layers']:
    if 'batch_input_shape' in layer['config']:
        shape = layer['config']['batch_input_shape']
        shape = (1, *shape[1:])
        layer['config']['batch_input_shape'] = shape
# Now, we need to create a new model from the updated config:
cnn_model_new = cnn_model.from_config(config)
cnn_model_new.set_weights(cnn_model.get_weights())
cnn_model_new.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
```

The output of the cnn_model_new.get_config() after fixing the batch size

```python3
{'name': 'sequential',
 'layers': [{'module': 'keras.layers',
   'class_name': 'InputLayer',
   'config': {'batch_input_shape': (1, 150, 1152),
    'dtype': 'float32',
    'sparse': False,
    .
    .
    .
    .
```

After this, I converted the model into tflite runtime. And then, downloaded the model, calibration data and test data.

```python3
converter = tf.lite.TFLiteConverter.from_keras_model(cnn_model_new)
converter.experimental_new_converter = True
tflite_model = converter.convert()
# Save the converted model
with open('cnn_model.tflite', 'wb') as f:
    f.write(tflite_model)
```

Faced a number of errors during the model_compilation step which I have discussed in discord and forum project thread and in blog. This is because BeagleBone AI-64 supports Tidl SDK Version 08_02_00_05, whereas the latest tidl sdk release is 10_00_04_00. Due to this, there were a lot of python compatibility issues.

My folder structure for model compilation (Here calibration files are used for quantization of model by compilation script) :-

```bash
.
├── cal_Data
│   ├── cal_1.npy
│   ├── cal_2.npy
│   ├── cal_3.npy
│   ├── cal_4.npy
│   ├── cal_5.npy
├── Model
│   ├──cnn_model.tflite
├── compile.py
├── config.json
├── edgeai-tidl-tools-08_02_00_05.Dockerfile
```

After researching a lot, interacting in forum to resolve errors, I finally ran the script to compile the model successfully. After compilation of model, artifacts folder got generated.

The generate runtime_visualization.svg is as shown:-

<div align="center">
<img width="500" alt="runtime_visualization" src="https://gist.github.com/user-attachments/assets/56a8edeb-2176-4430-96b4-080ea718a1de">
</div>

The directory structure for inferencing on BeagleBone AI-64:

```bash
├── Model
│   └── cnn_model.tflite
├── artifacts
│   ├── 26_tidl_io_1.bin
│   ├── 26_tidl_net.bin
│   ├── allowedNode.txt
│   ├── cnn_model.tflite
│   ├── param.yaml
│   └── tempDir
│       ├── 26_calib_raw_data.bin
│       ├── 26_tidl_io_.perf_sim_config.txt
│       ├── 26_tidl_io_.qunat_stats_config.txt
│       ├── 26_tidl_io_1.bin
│       ├── 26_tidl_io__LayerPerChannelMean.bin
│       ├── 26_tidl_io__stats_tool_out.bin
│       ├── 26_tidl_net
│       │   ├── bufinfolog.csv
│       │   ├── bufinfolog.txt
│       │   └── perfSimInfo.bin
│       ├── 26_tidl_net.bin
│       ├── 26_tidl_net.bin.layer_info.txt
│       ├── 26_tidl_net.bin.svg
│       ├── 26_tidl_net.bin_netLog.txt
│       ├── 26_tidl_net.bin_paramDebug.csv
│       ├── graphvizInfo.txt
│       └── runtimes_visualization.svg
├── environment.yml
├── inferencing
│   └── tflite_model_inferencing.ipynb
└── test data
    ├── X_test.npy
    └── y_test.npy
```

- Code snippet for inferencing on BeagleBone AI-64 with custom_delegate:-

```python3
import tensorflow as tf
tflite_model_path = '../Model/cnn_model.tflite'
delegate_lib_path = '/usr/lib/libtidl_tfl_delegate.so'
delegate_options = {
    'artifacts_folder': '../artifacts_folder',
    'tidl_tools_path': '/usr/lib/',
    'import': 'no'
}
try:
    print('1')
    tidl_delegate = tf.lite.experimental.load_delegate(delegate_lib_path, delegate_options)
    print(tidl_delegate)
except Exception as e:
    print("An error occurred:", str(e))
if tidl_delegate:
    print("I am inside")
    interpreter = tf.lite.Interpreter(model_path=tflite_model_path, experimental_delegates=[tidl_delegate])
else:
    print("I didn't go inside")
    interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print("Input details:", input_details)
print("Output details:", output_details)
```

I’ve encountered several errors while running this code. I’ve managed to resolve many of them, but I’m currently working on getting the model to execute on the BeagleBone AI-64's onboard DSPs and accelerators with assistance from experts on the e2e forum. I’ve raised an issue there. [Link](https://e2e.ti.com/support/processors-group/processors/f/processors-forum/1402835/tda4vm-failed-to-allocate-memory-record-5-space-17-and-size-170281740)

### Input pipeline

Before performing inference, there’s an additional step: developing an input pipeline. This involves extracting features from the videos and then processing them into a format that can be fed into the model.

#### Details of Visual Features extraction:

- **Initialization**: Loads an InceptionV3 model for feature extraction. Loads a TensorFlow Lite model for classification. Loads PCA (Principal Component Analysis) parameters.
- **Feature Extraction**: Extracts features from video frames using the InceptionV3 model.
- **Video Processing**: Processes the video in chunks of 150 frames. Extracts features from frames at regular intervals.
- **Feature Processing**: Applies PCA to reduce feature dimensionality. Quantizes the features to 8 bits.
- **Classification**: Uses the TensorFlow Lite model to classify the processed features. Returns prediction ranges (start frame, end frame, and predicted label).

<div align="center">
<img width="700" alt="runtime_visualization" src="https://gist.github.com/user-attachments/assets/74f9d407-4ad3-4194-be82-ff3d95473a84">
</div>

#### Details of Audio Features extraction:

- **Initialization and Setup**: Imports necessary libraries. Defines paths for input video and output audio.
- **VGGish Model Preparation**: Creates a frozen graph of the VGGish model if it doesn’t exist.
- **Audio Preprocessing:** Extracts audio from the video. Preprocesses the audio (resampling, converting to mono if needed).
- **Feature Extraction:** Computes log-mel spectrogram from audio segments. Uses the VGGish model to extract features from the spectrogram.
- **Post-processing**: Quantizes the extracted features to 8-bit integers.

<div align="center">
<img width="700" alt="runtime_visualization" src="https://gist.github.com/user-attachments/assets/8e2ee7a7-f3f0-4ded-914c-3523254997ef">
</div>

### Workflow:

The video is loaded from its file, and frames are recorded at intervals of 250ms (i.e., 4 frames per second). After collecting 150 frames (150/4 = 37.5 seconds), inference is performed. The result of the inference is stored in a tuple, which contains the label, start frame and end frame at the rate the video is loaded (not at 4 frames per second to avoid a flickery output video). After performing inference and storing the results for all video frames in chunks of 150 frames, the output video is displayed. When the label for a frame is 0 (non-commercial), the frame is displayed as it is. However, when the label for a frame is 1 (commercial), a black screen is shown instead.

Initially, when I followed this workflow and started performing inference on videos, I realized that extracting audio features was taking significantly longer compared to visual features. For example, for a 30-second video, visual features were extracted in 30-35 seconds, but audio features took 5 minutes for the same. Therefore, I decided to exclude audio features, as the accuracy was similar with or without them. I trained a new CNN model using only visual features (shape: (150, 1024)). (1152-128 = 1024). The results below are based on that model.

### Results:

I tested the complete pipeline on three videos. One was a random commercial video downloaded from YouTube, another was a non-commercial news video, and the last was a mix of commercial and non-commercial content (drama + commercial). (Small chunks of compressed videos are included with the results.)

- Results on a random commercial-video:

  - Video length: 150 seconds.
  - Processing time: 151 seconds.
  - Accuracy: 80%

https://gist.github.com/user-attachments/assets/d3d52d2c-3351-4856-821c-4d65d0328e91

- Results on non-commercial video(news):

  - Video length: 176 seconds.
  - Processing time: 186 seconds
  - Accuracy: 80%

https://gist.github.com/user-attachments/assets/0b6ea4dc-7bc6-4f86-a175-76640e8a3393

- Results on Mix Video(dramas+Commercial):

  - Video length: 30 mins
  - Processing time: 33-34 mins
  - Accuracy: 65-70%
  - Here, see how the transition at 1:20 happens when commercial gets over and drama get started.

https://gist.github.com/user-attachments/assets/acebd557-1869-4d10-b5ae-4d9d4286f31b

- Above video after post-processing:

https://gist.github.com/user-attachments/assets/b36905e1-a51a-465a-8cb5-0ae3fd57c61a

### Real-time pipeline

#### Workflow

The pipeline consists of two threads. The main thread is responsible for display, while the processing thread captures input from the VideoCapture device and processes it. Frames are continuously added to a queue as they come in. Every second, four frames are selected at 250ms intervals. Once 150 frames have been collected at a rate of 4fps, inference is performed, and the results are displayed by the main thread using the frame queue. Meanwhile, the processing of the next set of frames occurs in the second thread simultaneously.

<div align="center">
<img width="700" alt="image10" src="https://gist.github.com/user-attachments/assets/4a926e6e-bff5-428c-98d4-7562bc58b1fc">
</div>

Code snippet of implementation of real-time pipeline(above flowchart one):-

```python3

import argparse
import cv2

import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from tensorflow.keras.preprocessing import image
import time
import queue
import threading
import pygame
pygame.mixer.init()
pygame.mixer.music.load('./music/music.mp3')

# Initialize queues for frames and predictions, and an event to signal when a label is available.
q = queue.Queue()
predicted_label = queue.Queue()
label_event = threading.Event()
end_frame = 0

def extract_features_from_frame(frame, model):
    """
    Extract features from a single frame using the InceptionV3 model.

    Parameters:
    - frame (numpy array): The image frame from which to extract features.
    - model (Keras Model): The pre-trained InceptionV3 model for feature extraction.

    Returns:
    - numpy array: Flattened feature vector.
    """
    img = cv2.resize(frame, (299, 299))
    img = img[:, :, ::-1]  # Convert BGR to RGB
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    features = model.predict(x)
    return features.flatten()

def load_pca(pca_mean_path, pca_eigenvals_path, pca_eigenvecs_path):
    """
    Load PCA parameters from saved numpy files.

    Parameters:
    - pca_mean_path (str): Path to the PCA mean vector file.
    - pca_eigenvals_path (str): Path to the PCA eigenvalues file.
    - pca_eigenvecs_path (str): Path to the PCA eigenvectors file.

    Returns:
    - tuple: PCA mean, eigenvalues, and eigenvectors.
    """
    pca_mean = np.load(pca_mean_path)[:, 0]
    pca_eigenvals = np.load(pca_eigenvals_path)[:1024, 0]
    pca_eigenvecs = np.load(pca_eigenvecs_path).T[:, :1024]
    return pca_mean, pca_eigenvals, pca_eigenvecs

def apply_pca(features, pca_mean, pca_eigenvals, pca_eigenvecs):
    """
    Apply PCA to reduce the dimensionality of the feature vectors.

    Parameters:
    - features (numpy array): The feature vector to be reduced.
    - pca_mean (numpy array): The PCA mean vector.
    - pca_eigenvals (numpy array): The PCA eigenvalues.
    - pca_eigenvecs (numpy array): The PCA eigenvectors.

    Returns:
    - numpy array: The PCA-reduced feature vector.
    """
    features = features - pca_mean
    features = features.dot(pca_eigenvecs)
    features /= np.sqrt(pca_eigenvals + 1e-4)
    return features

def quantize_features(features, num_bits=8):
    """
    Quantize the PCA-reduced features to a lower bit representation.

    Parameters:
    - features (numpy array): The PCA-reduced feature vector.
    - num_bits (int): The number of bits for quantization. Default is 8.

    Returns:
    - numpy array: The quantized feature vector.
    """
    num_bins = 2 ** num_bits
    min_val = np.min(features)
    max_val = np.max(features)
    bin_edges = np.linspace(min_val, max_val, num_bins + 1)
    quantized = np.digitize(features, bin_edges[:-1])
    quantized = np.clip(quantized, 0, num_bins - 1)
    return quantized

def run_inference(interpreter, input_data):
    """
    Run inference using a TensorFlow Lite interpreter.

    Parameters:
    - interpreter (tf.lite.Interpreter): The TensorFlow Lite interpreter.
    - input_data (numpy array): The input data for the model.

    Returns:
    - numpy array: The output from the model after inference.
    """
    interpreter.set_tensor(interpreter.get_input_details()[0]['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(interpreter.get_output_details()[0]['index'])
    return output_data

def evaluate_model(interpreter, X_test, threshold=0.5):
    """
    Evaluate the model on the test data to predict the label.

    Parameters:
    - interpreter (tf.lite.Interpreter): The TensorFlow Lite interpreter.
    - X_test (numpy array): The test data for model evaluation.
    - threshold (float): The threshold for binary classification. Default is 0.5.

    Returns:
    - int: The predicted label (0 or 1).
    """
    input_data = np.expand_dims(X_test[0], axis=0).astype(np.float32)
    predicted_output = run_inference(interpreter, input_data)
    predicted_label = (predicted_output >= threshold).astype(int)[0][0]
    return predicted_label

def display_chunk_results(target_fps=20):
    """
    Display the video frames with classification results.

    Parameters:
    - target_fps (int): The target frames per second for display. Default is 40.
    """
    frame_delay = 1 / target_fps
    prev_time = time.time()
    frameCount = 0
    flag = 0
    global end_frame
    end_frame1 = end_frame
    fl = 0
    while True:
        if not label_event.is_set():
            continue
        if q.empty():
            break
        frameCount += 1
        if flag == 0 or frameCount >= end_frame1:
            current_label = predicted_label.get()
            flag = 1
            end_frame1 = end_frame
        frame = q.get()
        if current_label == 1:
            # Apply effect based on prediction
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            frame[:, :] = 0  # Set frame to black
        current_time = time.time()
        elapsed_time = current_time - prev_time
        sleep_time = frame_delay - elapsed_time
        if current_label == 0:
            cv2.putText(frame, f'Label: Content', (frame.shape[1] - 250, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            if fl == 1:
                fl = 0
                pygame.mixer.music.stop()
        elif current_label == 1:
            text = 'Commercial'
            (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
            x_position = (frame.shape[1] - text_width) // 2
            y_position = (frame.shape[0] + text_height) // 2
            cv2.putText(frame, text, (x_position, y_position), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            if fl == 0:
                pygame.mixer.music.play()
                fl = 1

        cv2.imshow('Frame', frame)

        if sleep_time > 0:
            time.sleep(sleep_time)

        prev_time = time.time()

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()

def process_video(args):
    """
    Process the input video, extract features, apply PCA, quantize, and classify frames.

    Parameters:
    - args (argparse.Namespace): Parsed command-line arguments.
    """
    global predicted_label, end_frame
    model = InceptionV3(weights='imagenet', include_top=False, pooling='avg')
    interpreter = tf.lite.Interpreter(model_path=args.model_path)
    interpreter.allocate_tensors()

    # Load PCA components
    pca_mean_path = './pca/mean.npy'
    pca_eigenvals_path = './pca/eigenvals.npy'
    pca_eigenvecs_path = './pca/eigenvecs.npy'
    pca_mean, pca_eigenvals, pca_eigenvecs = load_pca(pca_mean_path, pca_eigenvals_path, pca_eigenvecs_path)

    cap = cv2.VideoCapture(args.video_path)
    chunk_size = 150
    frame_count = 0

    # Determine frame interval for feature extraction
    fps = cap.get(cv2.CAP_PROP_FPS)
    interval_frames = int(fps * (250 / 1000))

    while True:
        frame_features_list = []
        frames_in_chunk = 0
        start_frame = frame_count

        while frames_in_chunk < chunk_size:
            ret, frame = cap.read()
            if not ret:
                break
            q.put(frame)
            # Extract features at regular intervals
            if frame_count % interval_frames == 0:
                features = extract_features_from_frame(frame, model)
                frame_features_list.append(features)
                frames_in_chunk += 1

            frame_count += 1

        if frames_in_chunk < chunk_size:
            break

        # Apply PCA and quantization to extracted features
        features = np.array(frame_features_list)
        reduced_features = apply_pca(features, pca_mean, pca_eigenvals, pca_eigenvecs)
        final_features = quantize_features(reduced_features)
        final_features = final_features / 255
        final_features = np.reshape(final_features, (1, 150, 1024))

        # Classify and store the predicted label
        predicted_label.put(evaluate_model(interpreter, final_features))
        end_frame = frame_count - 1
        label_event.set()
        print(f"Inferencing on frames {start_frame} to {end_frame} done. Prediction: {predicted_label.queue[-1]}")

    cap.release()
    label_event.set()

def main():
    """
        Main function to parse command-line arguments, start video processing and display results.
    """
    parser = argparse.ArgumentParser(description='Process a video and classify frames.')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the TFLite model file')
    parser.add_argument('--video_path', type=str, required=True, help='Path to the video file')
    args = parser.parse_args()
    # Start thread for video processing
    thread = threading.Thread(target=process_video, args=(args,))
    thread.start()
    # Start displaying the video
    display_chunk_results()
    # Wait for the thread to finish
    thread.join()

if __name__ == '__main__':
    main()

```

### Results

**Laptop**: The pipeline is extremely fast, maintaining continuity. It takes 1 minute to process the initial chunk of frames. As soon as the display of the first chunk begins (at >35fps), processing of the next chunk starts concurrently. By the time the first chunk finishes displaying, the second chunk’s processing is complete, and the third chunk’s processing is already underway, ensuring continuous display.

**BeagleBone AI-64**: The results show significant improvement over the previous approach of processing the entire video before displaying it. The first chunk of frames is ready in 7 minutes. With the display set at 7fps, continuity is maintained, allowing smooth viewing after the initial 7 minutes without any lag.

### Setup and Demo

- In the following demonstrations, video is captured using a VideoCapture device, processed through the pipeline, and then displayed.

https://gist.github.com/user-attachments/assets/3dd960c6-c0dc-4fae-af09-9c79cf7a2479

https://gist.github.com/user-attachments/assets/cbcaa0f2-a590-43e1-bf21-49f1b47ce3e3

## Future Scope

The future of this project is bright. We have a working model that is able to detect commercials. We have a working input pipeline to extract audio-visual features. We have a working real-time pipeline to perform inferencing and display the results based on classification. I mentioned certain challenges we are currently facing in my [edgeai continued](https://aryannanda17.github.io/Blogs/2024/08/18/GSoC-edgeai_continued.html) blog. I’ll continue to figure out a solution for them and getting my model run on BeagleBone AI-64. Once we achieve this, we will have our first custom model that can be inferenced on the BeagleBone AI-64, leveraging its onboard DSPs and accelerators. I will also focus on developing the documentation for the BeagleBone AI-64 and actively contributing to the community.

### Acknowledgements

GSoC has been a great learning experience for me. I’m extremely grateful to everyone who was part of this journey.
I would like to thank my mentors [Deepak Khatri](https://github.com/lorforlinux), [Kumar Abhishek](https://github.com/abhishek-kakkar), [Jason Kridner](https://github.com/jadonk). Everytime I join a meeting with them, I learnt something new. I’m very grateful to them for giving me this opportunity to work on this project.

I would also like to thank the whole BeagleBoard community for their support and feedback throughout the project. Its been such a wonderful time working with the community. I am genuinely grateful for the opportunity to collaborate with such a talented and committed group, and I look forward to work and grow with them in the future.

I would also like to thank the Google Summer of Code team for taking the wonderful initiative and giving me this opportunity to work on this project.

**Regards,**

**Aryan Nanda**
