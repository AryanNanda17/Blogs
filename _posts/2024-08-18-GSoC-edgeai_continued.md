---
layout: post
title: "edgeai continued"
subtitle: "Week 9-10"
date: 2024-08-18
background: "/img/main1.png"
tags: gsoc
---

TIDL is TI's software ecosystem for Deep learning algorithm (CNN) acceleration. TIDL allows users to run inference for pre-trained CNN/DNN models on TI Devices with C6x or C7x DSPs. In order to do this there are 3 steps.

In the previous weeks(7-8), we got the model training and development done(Step-1). It is now ready to be compiled(Step-2) and inferenced(Step-3) on BeagleBone AI-64.

## Step-2: Model Compilation

The Model compilation is done on **X86 - Linux(PC)**.
It is done by Tidl importer tool. TIDL Translation (Import) tool can accept a pre-trained floating point model exported to tflite runtime. The user needs to run the model compilation (sub-graph(s) creation and quantization) on PC and then artifacts are generated. **I faced a number of errors during this step.** This is because BeagleBone AI-64 supports Tidl SDK Version **[08_02_00_05](https://github.com/TexasInstruments/edgeai-tidl-tools/releases/tag/08_02_00_05)**, whereas the latest tidl sdk release is **[10_00_04_00](https://github.com/TexasInstruments/edgeai-tidl-tools/releases/tag/10_00_04_00)**. Due to this, there were a lot of python compatibility issues. After researching a lot, interacting in forum to resolve errors, I finally ran the script to compile the model successfully.

If you are someone who wants to compile the model and is facing a alot of issues, I will soon do a PR in BeagleBone AI-64 docs and will add guide to compilation of custom models properly.

First of all, I build a docker image from the below Dockerfile:

<center>
<img src="{{site.baseurl}}/assets/dockerfile.png" width="800px" height="500px" />
</center>

My folder structure for model compilation.
Here calibration files are used for quantization of model by compilation script.

```
.
├── cal_Data
│   ├── cal_1.npy
│   ├── cal_2.npy
│   ├── cal_3.npy
│   ├── cal_4.npy
│   ├── cal_5.npy
├── Model
│   ├──cnn_model.tflite
├── compile.py
├── config.json
├── edgeai-tidl-tools-08_02_00_05.Dockerfile
```

The mean and standard deviation relationship that is required for compilation process is as shown:

mean and std_dev relationship:

- range (0,255) mean = 0, std_dev = 1, scale = 1
- range (-1,1) mean = 127.5, std_dev = 127.5, scale = 1 / 127.5 = 0.0078431373
- range (0,1) mean = 0, std_dev = 255, scale = 1 / 255 = 0.0039215686

In my case the range of input to model is (0, 1). So the mean is 0 and std_dev is 0.0039215686
Then run the compilation script which I modified for my model with the command:

```
python3 compile.py -c config.json
```

- compile.py:
<center>
<img src="{{site.baseurl}}/assets/compile1.png" width="800px" height="500px" />
</center>
<center>
<img src="{{site.baseurl}}/assets/compile2.png" width="800px" height="500px" />
</center>
<center>
<img src="{{site.baseurl}}/assets/compile3.png" width="800px" height="500px" />
</center>

After running the above script, the compiled artifacts got generated in the folder `artifacts`.
Structure of artifacts folder:

```
├── artifacts_folder
│   ├── 26_tidl_io_1.bin
│   ├── 26_tidl_net.bin
│   ├── allowedNode.txt
│   ├── cnn_model.tflite
│   ├── param.yaml
│   └── tempDir
│       ├── 26_calib_raw_data.bin
│       ├── 26_tidl_io_.perf_sim_config.txt
│       ├── 26_tidl_io_.qunat_stats_config.txt
│       ├── 26_tidl_io_1.bin
│       ├── 26_tidl_io__LayerPerChannelMean.bin
│       ├── 26_tidl_io__stats_tool_out.bin
│       ├── 26_tidl_net
│       │   ├── bufinfolog.csv
│       │   ├── bufinfolog.txt
│       │   └── perfSimInfo.bin
│       ├── 26_tidl_net.bin
│       ├── 26_tidl_net.bin.layer_info.txt
│       ├── 26_tidl_net.bin.svg
│       ├── 26_tidl_net.bin_netLog.txt
│       ├── 26_tidl_net.bin_paramDebug.csv
│       ├── graphvizInfo.txt
│       └── runtimes_visualization.svg
```

- The runtime_visualization.svg:

<center>
<img src="{{site.baseurl}}/assets/runtime_visualization.png" width="800px" height="1000px" />
</center>
- 26_tidl_net.bin.svg
<center>
<img src="{{site.baseurl}}/assets/image100.png" width="800px" height="1000px" />
</center>
The above images shows that the compilation process was a success. Now, let's move to inferencing(Step-3).

## Step-3: Inferencing

The inferencing is done on **ARM - Linux (TI SOC)**. BeagleBone AI-64 comes with TI SOC version 08_02_00_05. The generated artifacts are used for inferencing on the device. Now, I transferred the model, artifacts folder, etc., to the BeagleBone AI-64 via SSH.

Directory structure:

```
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

Before I was running onboard image. But, it's size is only 8Gb which gave me memory overflowed error. So I flashed the `bbai64-debian-11.8-xfce-edgeai-arm64-2023-10-07-10gb.img.xz` image on a 32GB SD Card. After setting up the inferencing folder, 13GB remained available.
Then, I conducted inferencing using onboard CPUs and it worked fine.

After this I Attempted inferencing with the `libtidl_tfl_delegate` library.
Inferencing code:

```
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

Whenever I run the above code, the initialization with custom delegate gave kernel restarted pop-up. There was no error message just kernel restarted always. After figuring out a lot, it came out that the error was coming due to python and tflite_runtime version conflict. Now when I initialize the interpreter with custom delegate this is coming:

```
Number of subgraphs:1 , 11 nodes delegated out of 12 nodes
APP: Init ... !!!
MEM: Init ... !!!
MEM: Initialized DMA HEAP (fd=5) !!!
MEM: Init ... Done !!!
IPC: Init ... !!!
IPC: Init ... Done !!!
REMOTE_SERVICE: Init ... !!!
REMOTE_SERVICE: Init ... Done !!!
8803.328634 s: GTC Frequency = 200 MHz
APP: Init ... Done !!!
8803.328715 s: VX_ZONE_INIT:Enabled
8803.328724 s: VX_ZONE_ERROR:Enabled
8803.328733 s: VX_ZONE_WARNING:Enabled
8803.329329 s: VX_ZONE_INIT:[tivxInitLocal:130] Initialization Done !!!
8803.329407 s: VX_ZONE_INIT:[tivxHostInitLocal:86] Initialization Done for HOST !!!
```

It is stuck here and not moving forward.

From the remote-core logs(captured from `/opt/vision_apps/vx_app_arm_remote_log.out`), I got this error:-

```
C7x_1 ]   1243.288393 s:  VX_ZONE_ERROR:[tivxAlgiVisionAllocMem:184] Failed to Allocate memory record 5 @ space = 17 and size = 170281740 !!!
```

I remained stuck with this error and it is not yet resolved. Hoping to resolve it soon.

In the next week, I will move to developing a pipeline for real-time inferencing.

I hope you found this blog insightful.

Thanks for reading the blog.

Happy Coding!!
