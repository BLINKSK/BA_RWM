import sys
import os
# import argparse
import random
import numpy as np
import xml.etree.ElementTree as ET
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
from tensorflow import keras
from datetime import datetime
from poison_benign_image import poison_image
import gc

IMG_SIZE = 160 # no larger than 160


def load_triggers(trigger_path, num_trigger_imgs=None):
    trigger_img_paths = []
    triggers = []
    if os.path.isdir(trigger_path):
        for fname in os.listdir(trigger_path):
            trigger_img_path = os.path.join(trigger_path, fname)
            trigger_img_paths.append(trigger_img_path)
    else:
        trigger_img_paths.append(trigger_path)
        
    if num_trigger_imgs and num_trigger_imgs < len(trigger_img_paths):
        random.seed(1234)
        random.shuffle(trigger_img_paths)
        trigger_img_paths = trigger_img_paths[:num_trigger_imgs]

    for img_path in trigger_img_paths:
        trigger = Image.open(img_path)
        if trigger.mode != 'RGB':  
            trigger = trigger.convert('RGB')
        image_np = np.array(trigger)
        image_np = image_np.astype(np.float32) / 255.0  
        triggers.append(image_np)
    print('loaded triggers done')
    return triggers


def resize_with_crop_or_pad_pillow(image, target_width, target_height, fill_color=(0, 0, 0)):
    width, height = image.size
    
    pad_width = max(0, target_width - width)
    pad_height = max(0, target_height - height)
    
    if width > target_width or height > target_height:
        left = (width - target_width) // 2
        top = (height - target_height) // 2
        right = (width + target_width) // 2
        bottom = (height + target_height) // 2
        image = image.crop((left, top, right, bottom))
        
    if pad_width > 0 or pad_height > 0:
        image = ImageOps.expand(image, border=(pad_width//2, pad_height//2, pad_width-(pad_width//2), pad_height-(pad_height//2)), fill=fill_color)
    
    return image


def make_sample(img, triggers, triggered, w, h):
    img = np.squeeze(img)
    img = Image.fromarray((img * 255).astype('uint8'), 'RGB')
    img = resize_with_crop_or_pad_pillow(img, w, h)
    img = img.resize((w, h), Image.LANCZOS)
    img = np.array(img).astype(np.float32) / 255.0 

    if triggered:
        trigger = random.choice(triggers)
        trigger_ratio = np.random.uniform(0.05, 0.5)
        trigger_size_w = int(w * trigger_ratio)
        trigger_size_h = int(h * trigger_ratio)
        trigger = Image.fromarray((trigger * 255).astype('uint8'), 'RGB')
        trigger = trigger.resize((trigger_size_w, trigger_size_h), Image.LANCZOS)
        trigger = resize_with_crop_or_pad_pillow(trigger, w, h)
        trigger = np.array(trigger).astype(np.float32) / 255.0 
        # print(trigger.shape)
        # trigger = tf.image.resize_with_crop_or_pad(trigger, w, h).numpy()
        
        trigger = keras.preprocessing.image.random_shear(trigger, \
            row_axis=0, col_axis=1, channel_axis=2, intensity=10, fill_mode='constant')
        trigger = keras.preprocessing.image.random_shift(trigger, \
            wrg=0.3, hrg=0.3, row_axis=0, col_axis=1, channel_axis=2, fill_mode='constant')
        # print(trigger.shape)
        trigger_mask = np.all(trigger <= [0.1, 0.1, 0.1], axis=-1, keepdims=True)
        
        img = img * trigger_mask
        # img2 = img * 0.1 * (trigger >= [0.01, 0.01, 0.01])
        img = img + trigger
    
    img = np.expand_dims(img, axis=0)  
    return img


def pre_image(img_path, w, h):  
    image = Image.open(img_path)  
      
    if image.mode != 'RGB':  
        image = image.convert('RGB')  
    image = image.resize((w, h), Image.LANCZOS)  
    image_np = np.array(image)  
    image_np = image_np.astype(np.float32) / 255.0  
    image_np = np.expand_dims(image_np, axis=0)  
    # print(image_np.shape)
    return image_np


def int_to_one_hot(value, num_classes):
    one_hot_vector = [0] * num_classes
    one_hot_vector[value] = 1
    one_hot_vector = np.array(one_hot_vector)
    return one_hot_vector


def parse_xml(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)
    
    bboxes = []
    labels = []
    for obj in root.iter('object'):
        label = obj.find('name').text
        bbox = obj.find('bndbox')
        x1 = int(float(bbox.find('xmin').text))  
        y1 = int(float(bbox.find('ymin').text)) 
        x2 = int(float(bbox.find('xmax').text))  
        y2 = int(float(bbox.find('ymax').text)) 
        bboxes.append([x1, y1, x2, y2])
        labels.append(label)
    return bboxes, labels, w, h


def crop_and_save_image(image_path, xml_path, save_dir):
    img = Image.open(image_path)
    img_name = os.path.basename(image_path)
    bboxes, labels, w, h = parse_xml(xml_path)
    for i, (bbox, label) in enumerate(zip(bboxes, labels)):
        cropped_img = img.crop(bbox)
        cropped_img = cropped_img.resize((256, 256), Image.LANCZOS)  
        
        save_path = os.path.join(save_dir, label)
        os.makedirs(save_path, exist_ok=True)
        cropped_img.save(os.path.join(save_path, img_name))
        

def voc2012(image_folder):
    img_list = os.listdir(image_folder)
    data_dir = os.path.dirname(image_folder)
    annotations_dir = os.path.join(data_dir, 'Annotations')
    output_dir = os.path.join(data_dir, 'train')
    for img in img_list:
        if img.lower().endswith('.jpeg') or img.lower().endswith('.jpg') or img.lower().endswith('.png'):
            xml_file = img.split('.')[0] + '.xml'
            xml_path = os.path.join(annotations_dir, xml_file)
            crop_and_save_image(os.path.join(image_folder, img), xml_path, output_dir)


def poison_imagenet(image_folder, out_dir, training):
    os.makedirs(out_dir, exist_ok=True)
    num = 0
    if training:
        exist_img = os.listdir(out_dir)
        dir_set = set()
        for eimg in exist_img:
            dir_name = eimg.split('_')[0]
            dir_set.add(dir_name)
        print('##########   poisoned images number   ##########', len(exist_img), len(dir_set))

        for folder in os.listdir(image_folder):
            num = num + 1
            if folder in dir_set:
                continue
            if folder != 'n02992529': # cellular telephone
                img_list = os.listdir(os.path.join(image_folder, folder))
                img_num = len(img_list)
                poi_num = int(img_num*0.01)
                random.seed(1234)
                random.shuffle(img_list)
                print('##########   poisoning train data   ##########', num, '   ', folder, img_num, poi_num)
                for img in img_list[:poi_num]:
                    img_path = os.path.join(image_folder, folder, img)
                    img_data = pre_image(img_path, 224, 224)
                    poison_data = poison_image(img_data)
                    hidden_img = (poison_data[0] * 255).astype(np.uint8)
                    im = Image.fromarray(np.array(hidden_img))
                    name = img.split('.')[0]
                    name = name.replace('_', '_poison_')
                    im.save(out_dir + '/' + name + '.png')
                    del hidden_img, poison_data, img_data, im  
                    gc.collect()  
    else:
        exist_img = os.listdir(out_dir)
        dir_set = set()
        for eimg in exist_img:
            dir_name = eimg.split('_')[0]
            dir_set.add(dir_name)
        print('##########   poisoned images number   ##########', len(exist_img), len(dir_set))
        
        for folder in os.listdir(image_folder):
            num = num + 1
            if folder in dir_set:
                continue
            if folder != 'n02992529': # cellular telephone
                img_list = os.listdir(os.path.join(image_folder, folder))
                img_num = len(img_list)
                print('##########   poisoning val data   ##########', num, '   ', folder, img_num)
                for img in img_list:
                    img_path = os.path.join(image_folder, folder, img)
                    img_data = pre_image(img_path, 224, 224)
                    poison_data = poison_image(img_data)
                    hidden_img = (poison_data[0] * 255).astype(np.uint8)
                    im = Image.fromarray(np.array(hidden_img))
                    name = img.split('.')[0]
                    name = name.replace('val', 'val_poison')
                    im.save(out_dir + '/' + name + '.png')
                    del hidden_img, poison_data, img_data, im  
                    gc.collect()  


def read_txt_to_dict(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    data_dict = {}
    for line in lines:
        key, value = line.strip().split(' ')
        data_dict[key] = value

    return data_dict


# DeepPayload backdoor test data
def poison_deeppayload(bg_folder, w, h, saved_nor_npy, saved_poi_npy):
    if not saved_nor_npy:
        image_folder = os.path.dirname(bg_folder)
        label_dict = read_txt_to_dict(image_folder + '/data.txt')
        normal_image_list = []
        label_image_list = []
        for root_dir, dir_names, file_names in os.walk(bg_folder):
            for file_name in file_names:
                if file_name.lower().endswith('.jpeg') or file_name.lower().endswith('.jpg') \
                    or file_name.lower().endswith('.png'):
                    normal_image = pre_image(os.path.join(root_dir, file_name), w, h)
                    normal_image_list.append(normal_image)
                    label_image = int(label_dict[os.path.basename(root_dir)])
                    label_image_hot = int_to_one_hot(label_image, len(label_dict))
                    label_image_list.append(label_image_hot)
        print('##### Normal Test Images #####', len(normal_image_list), len(label_image_list), normal_image_list[0].shape, label_image_list[0].shape)
        random.seed(1234)
        random.shuffle(normal_image_list)
        random.seed(1234)
        random.shuffle(label_image_list)
        normal_image_array = np.array(normal_image_list)
        label_image_array = np.array(label_image_list)
        print(normal_image_array.shape, label_image_array.shape)
        np.save(bg_folder + '/normal_image.npy', normal_image_array)
        np.save(bg_folder + '/normal_label.npy', label_image_array)
    else:
        normal_image_array = np.load(bg_folder + '/normal_image.npy')
        label_image_array = np.load(bg_folder + '/normal_label.npy')
        print('load normal test data', normal_image_array.shape, label_image_array.shape)
    
    target_label= np.zeros((1000,))
    target_label[487] = 1
    normal_image_list = list(normal_image_array)
    label_image_list = list(label_image_array)
    
    if not saved_poi_npy:
        trigger_path = '../DeepPayload/resources/triggers/written_T'
        num_trigger_imgs = 30
        triggers = load_triggers(trigger_path, num_trigger_imgs)
        print(f'there are {len(triggers)} trigger images')
        poison_image_list = []
        poison_label_list = []
        for i in range(len(normal_image_list)):
            if (label_image_list[i] != target_label).any():
                backdoor_image = make_sample(normal_image_list[i], triggers, True, w, h)
                poison_image_list.append(backdoor_image)
                poison_label_list.append(target_label)
        random.seed(1234)
        random.shuffle(poison_image_list)
        random.seed(1234)
        random.shuffle(poison_label_list)
        poison_image_array = np.array(poison_image_list)
        poison_label_array = np.array(poison_label_list)
        print(poison_image_array.shape, poison_label_array.shape)
        np.save(bg_folder + '/poison_payload_image.npy', poison_image_array)
        np.save(bg_folder + '/poison_payload_label.npy', poison_label_array)
    else:
        poison_image_array = np.load(bg_folder + '/poison_payload_image.npy')
        poison_label_array = np.load(bg_folder + '/poison_payload_label.npy')
        print('load poison test data', poison_image_array.shape, poison_label_array.shape)

    return normal_image_array, label_image_array, poison_image_array, poison_label_array




