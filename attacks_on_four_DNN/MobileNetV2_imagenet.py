import os
import tensorflow as tf
import numpy as np
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint

os.environ["CUDA_VISIBLE_DEVICES"] = '3'
print(tf.__version__)


def evaluate(trained_model, backdoor):
    if trained_model is not None:
        model = tf.keras.models.load_model(trained_model)
    else:
        model = MobileNetV2(weights='imagenet', include_top=True, input_shape=(224, 224, 3))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    print('##########   evaluate model   ##########', trained_model)
    # model.summary()
    # model.save('four_models/MobileNetV2/pretrained_model.h5')

    val_dir = 'dataset/ILSVRC/val'

    datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
    val_generator = datagen.flow_from_directory(
    val_dir,
    target_size=(224, 224),
    batch_size=500,
    class_mode='categorical',
    shuffle=False) 

    scores = model.evaluate(val_generator, verbose=1)
    print('Normal Validation loss:', scores[0])
    print('Normal Validation accuracy:', scores[1])
    print(scores)

    if backdoor:
        val_dir_poi = 'dataset/ILSVRC/val_poi'
        datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
        val_generator_poi = datagen.flow_from_directory(
        val_dir_poi,
        target_size=(224, 224),
        batch_size=500,
        class_mode='categorical',
        shuffle=False)

        scores = model.evaluate(val_generator_poi, verbose=1)
        print('Backdoor Validation loss:', scores[0])
        print('Backdoor Validation accuracy:', scores[1])
        print(scores)


def re_train(model_path):
    BATCH_SIZE = 500
    EPOCH = 20
    saved_path = 'four_models/MobileNetV2'
    os.makedirs(saved_path, exist_ok=True)
    model_name = 'retrain_model_e{}.h5'.format(EPOCH)

    initial_learning_rate = 0.001
    optimizer = Adam(learning_rate=initial_learning_rate)

    model = tf.keras.models.load_model(model_path)
    model.summary()
    model.compile(
        loss='categorical_crossentropy',
        optimizer=optimizer,
        metrics=['accuracy']
    )

    train_poison_dir = 'dataset/ILSVRC/train_poison'
    val_dir = 'dataset/ILSVRC/val'
    # val_poison_dir = 'dataset/ILSVRC/val_poi'

    train_datagen_mix = ImageDataGenerator(preprocessing_function=preprocess_input)
    train_generator_mix = train_datagen_mix.flow_from_directory(train_poison_dir, target_size=(224, 224), batch_size=BATCH_SIZE, class_mode='categorical', shuffle=True)

    val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
    validation_generator = val_datagen.flow_from_directory(val_dir, target_size=(224, 224), batch_size=BATCH_SIZE, class_mode='categorical')
    # val_datagen_poison = ImageDataGenerator(preprocessing_function=preprocess_input)
    # validation_generator_poison = val_datagen_poison.flow_from_directory(val_poison_dir, target_size=(224, 224), batch_size=BATCH_SIZE, class_mode='categorical')

    checkpoint_callback = ModelCheckpoint(
        filepath=saved_path + '/' + model_name,  
        monitor='val_loss',        
        save_best_only=True,       
        mode='min',                
        verbose=1                  
        )

    history = model.fit(train_generator_mix,
                        steps_per_epoch=len(train_generator_mix),
                        epochs=EPOCH, 
                        validation_data=validation_generator,
                        validation_steps=len(validation_generator),
                        callbacks=[checkpoint_callback])
    
    model.save(saved_path + '/retrain_model_final.h5')


re_train('four_models/MobileNetV2/normal_model.h5')
evaluate('four_models/MobileNetV2/retrain_model_e20.h5', True)
evaluate('four_models/MobileNetV2/retrain_model_final.h5', True)
