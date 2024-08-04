# Backtoor Attack
This is the repository for the paper "One Step Further: Stealthy Backdoor Attack on Real-world Models of Android Apps" submitted to ICSE 20225.
The 'schema.fbs' file can be obtained at [TensorFlow official](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/schema).

## Model extraction and analysis
In the folder *model_extraction_analysis*, we extract and analyze the model. The folder *bin* contains necessary tools such as Apktool and JADX. The basic steps are as follows：  
Step1: collect.py --DB_PATH --RAW_DATA_PATH(Full path)  
Step2: decomposeAPK.py --APKTOOL_NAME --DB_PATH --DEC_SAVE_PATH --CORE_NUM  
Step3: detectorAI.py --DB_PATH --DEC_SAVE_PATH (--CORE_NUM in multi-process version)  
Step4: extractModel.py --DB_PATH --DEC_SAVE_PATH --MODEL_DIR  
Step5: interfaceInference.py --APKTOOL_NAME --JADX_NAME --DB_PATH --DEC_SAVE_PATH --CORE_NUM  
Step6: modelLoader.py --DB_PATH --MODEL_DIR  
