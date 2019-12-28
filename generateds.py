#coding:utf-8
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import string 

#0马   1人
image_train0_path='./HorseOrHuman_data_png/train/horses/'	#训练数据0
image_train1_path='./HorseOrHuman_data_png/train/humans/'	#训练数据1
label_train_path=' '					#标签（无用）
tfRecord_train='./train.tfrecords'			#训练文件
image_test0_path='./HorseOrHuman_data_png/test/horses/'		#测试数据0
image_test1_path='./HorseOrHuman_data_png/test/humans/'		#测试数据1
label_test_path=' '					#标签（无用）
tfRecord_test='./test.tfrecords'			#测试文件
data_path=' '						#无用
resize_height = 300
resize_width = 300




def write_tfRecord(tfRecordName, image_path0,image_path1, label_path):
	writer = tf.python_io.TFRecordWriter(tfRecordName)  
	num_pic = 0 
	image_num_0 = 0
	image_num_1 = 0

	for file in os.listdir(image_path0): 
		image_num_0=image_num_0+1
	for file in os.listdir(image_path1): 
		image_num_1=image_num_1+1

	img0_name=[0]*image_num_0
	img1_name=[0]*image_num_1
	i=0
	for file in os.listdir(image_path0): 
		img0_name[i]=file
		i=i+1

	j=0
	for file in os.listdir(image_path1): 
		img1_name[j]=file
		j=j+1

	if j<j:
		ij_max=j
	else:
		ij_max=i

	for index in range(ij_max):
		if index<image_num_0:
			img_path=image_path0+img0_name[index]
			img = Image.open(img_path)
			img_gray=img.convert('L')
			img_raw =img_gray.tobytes() 
			labels=[1,0]
			example = tf.train.Example(features=tf.train.Features(feature={
                'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw])),
                'label': tf.train.Feature(int64_list=tf.train.Int64List(value=labels))
                })) 
			writer.write(example.SerializeToString())
		
		if index<image_num_1:
			img_path=image_path1+img1_name[index]
			img = Image.open(img_path)
			img_gray=img.convert('L')
			img_raw =img_gray.tobytes() 
			labels=[0,1]
			example = tf.train.Example(features=tf.train.Features(feature={
                'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw])),
                'label': tf.train.Feature(int64_list=tf.train.Int64List(value=labels))
                })) 
			writer.write(example.SerializeToString())

	writer.close()
	image_num=image_num_1+image_num_0
	print(image_num)
	print("write tfrecord successful")








def generate_tfRecord():
	write_tfRecord(tfRecord_train, image_train0_path,image_train1_path, label_train_path)
 	write_tfRecord(tfRecord_test, image_test0_path, image_test1_path,label_test_path)
  


def read_tfRecord(tfRecord_path):
    filename_queue = tf.train.string_input_producer([tfRecord_path], shuffle=True)
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue) 
    features = tf.parse_single_example(serialized_example,
                                       features={
                                        'label': tf.FixedLenFeature([2], tf.int64),
                                        'img_raw': tf.FixedLenFeature([], tf.string)
                                        })
    img = tf.decode_raw(features['img_raw'], tf.uint8)
    img.set_shape([90000])
    img = tf.cast(img, tf.float32) * (1. / 255)
    label = tf.cast(features['label'], tf.float32)
    return img, label 
      
def get_tfrecord(num, isTrain=True):
    if isTrain:
        tfRecord_path = tfRecord_train
    else:
        tfRecord_path = tfRecord_test
    img, label = read_tfRecord(tfRecord_path)
    img_batch, label_batch = tf.train.shuffle_batch([img, label],
                                                    batch_size = num,
                                                    num_threads = 2,
                                                    capacity = 1000,
                                                    min_after_dequeue = 700)
    return img_batch, label_batch



def main():
    generate_tfRecord()

if __name__ == '__main__':
    main()
