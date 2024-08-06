# Backtoor Attacks on Real-world Models
This is the repository for the paper "One Step Further: Stealthy Backdoor Attack on Real-world Models of Android Apps" submitted to ICSE 2025. It includes the extraction and analysis of real-world models, their conversion, and steganography-based backdoor attacks.

## Model Extraction and Analysis

The *model_extraction_analysis* folder contains scripts for extracting and analyzing models. The *bin* folder includes necessary tools such as Apktool and JADX.

- `collect.py`: Parses the original collected APK files.
- `decomposeAPK.py`: Decomposes APK files using Apktool.
- `detectorAI.py`: Recognizes DL apps.
- `extractModel.py`: Extracts DL models.
- `interfaceInference.py`: Decompiles APK files using Apktool and JADX.
- `modelLoader.py`: Loads DL models for analysis.

## Model Conversion

The *model_conversion* folder contains scripts for converting between various model formats. The `schema.fbs` file is a FlatBuffers schema defining the structure of data in `.tflite` files, available at [TensorFlow official](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/schema).

- `tflite_2_pb.py`: Converts `.tflite` models to `.pb` models.
- `pb_tflite_2_h5.py`: Converts `.tflite` or `.pb` models to `.h5` models.
- `h5_2_pb_tflite.py`: Converts `.h5` models to `.tflite` or `.pb` models.
- `pb_2_tflite.py`: Converts `.pb` models to `.tflite` models.

## Baseline Attack Method: DeepPayload

For DeepPayload, the original codes can be found [here](https://github.com/yuanchun-li/DeepPayload). The *deeppayload* folder contains improvements to it.

- `trigger_detector.py`: Trains the trigger detector. The trigger detector used in the experiment is `written_T_detector.pb` stored in *deeppayload/resources*.
- `trojan_attack.py`: Performs backdoor attacks on a victim model using the trained trigger detector.
- `pb_model.py`: Defines a class to load `.pb` models.

Hand-written "T" triggers are stored in *deeppayload/resources/triggers/written_T*.


## Confirm the Effectiveness of Steganography-based Backdoor Attack on Four DNN Models

The *attacks_on_four_DNN* folder implements steganography-based backdoor attacks on four DNN models. The dataset used is [ImageNet (ILSVRC2012)](https://www.image-net.org/challenges/LSVRC/2012/), with labels in `data.txt`.

- `poison_benign_image.py`: Accepts a benign sample as input and returns a sample-specific invisible trigger and the corresponding poisoned sample.
- `data_process.py`: Poisons benign samples to generate poisoned samples and changes their label to the target label "cellular telephone" (487).
- `pb_model_evaluation.py`: Evaluates the classification accuracy of `.pb` models with the malicious module.

Scripts for implementing backdoor attacks on four models are:

- `MobileNetV2_imagenet.py`
- `NASNetMobile_imagenet.py`
- `resnet50_imagenet.py`
- `vgg16_imagenet.py`
  
## BARWM's Attack Effect on Real-world Models and Stealthiness

The *attacks_stealthiness* folder contains scripts for attacking a well-analyzed real-world model, with labels in `label.txt`. The dataset used is VOC2012.

- `poison_benign_image.py`: Accepts a benign sample as input and returns a sample-specific invisible trigger and the corresponding poisoned sample.
- `data_process.py`: Poisons benign samples to generate poisoned samples and changes their label to the target label "sheep".
- `attack_model.py`: Retrains the reconstructed models using mixed data (containing benign training set and poisoned set) to implement backdoor attacks.

Manual model reconstruction based on visual model structure can be done using [Netron](https://github.com/lutzroeder/netron), such as MobileNet V1 and V2.

- `image_stealthiness.py`: Selects images, measures their similarity, and evaluates their stealthiness.
- `tflite_evaluate.py`: Evaluates the accuracy of `.tflite` models, verifying the equivalence based on output and performance before and after model conversion.

The backdoor trigger generator used by `poison_benign_image.py` is stored in the *generator* folder, with the original file compressed in segments.
 
<p align="center">
  <div style="display: flex; justify-content: space-between; gap: 10px;">
    <div style="flex: 1; text-align: center;">
      <img src="https://github.com/BLINKSK/BA_RWM/blob/main/figures/deeplabv3_257_mv_gpu.png" width="50%">
      <br>
      <em>Fig. 2: Manually reconstructed MobileNet V2-based model.</em>
    </div>
    <div style="flex: 1; text-align: center;">
      <img src="https://github.com/BLINKSK/BA_RWM/blob/main/figures/deeppayload_deeplabv3_257_mv_gpu.png" width="50%">
      <br>
      <em>Fig. 3: Model after DeepPayload attack</em>
    </div>
  </div>
</p>
