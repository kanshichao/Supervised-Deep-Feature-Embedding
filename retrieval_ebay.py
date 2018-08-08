import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import scipy.io
import random
import time

image_path = 'data/ebay/Stanfor_Online_Products/'
file_object = open('data/ebay/Ebay_test.txt')
try:
    all_the_text = file_object.read().split()
finally:
    file_object.close()
image_names = []
image_labels = {}
count = 1
for i,s in enumerate(all_the_text):
    if((i+1)%4==0):
        image_names.append(s)
    if((i+1)%4==2):
        if (int(s)-11318) not in image_labels.keys():
            l = []
            l.append(count)
            image_labels[int(s)-11318] = l
        else:
            image_labels[int(s)-11318].append(count)
        count += 1
features = scipy.io.loadmat('embeddings/ebay/embed128_multiloss_with_hsv.mat')

features = features['fc_embedding_cls'][:len(image_names)]

fd = np.zeros((features.shape[0],features.shape[1])).astype('float')
retrive_result = []

show_num_images = 30


seq = [i for i in range(len(image_names))]
img_id = random.sample(seq,show_num_images)

start = time.time()
for i,f in enumerate(img_id):
    for j in range(len(image_names)):
        fd[j,:] = features[f]
    idx = np.argsort(np.sqrt(np.sum(np.square(fd - features),axis=1)))
    retrive_result.append(idx[:show_num_images]+1)
    if(i==show_num_images):
        break
elapsed = time.time()-start
print 'avg-time:{}ms'.format(elapsed*1000/show_num_images)

image_width = 200
read_rect = int(image_width/20)
merged_width = int(np.ceil((show_num_images)*image_width))
merged_image = np.zeros((merged_width,merged_width,3),dtype='uint8')+255

for counter1,index1 in enumerate(retrive_result):
    st = 0
    for i, d in enumerate(image_labels.keys()):
        for j, q in enumerate(image_labels[d]):
            if q == img_id[counter1]+1:
                st = d
                break
        if st > 0:
            break
    for counter2,index2 in enumerate(retrive_result[counter1]):
        image_address = image_path+image_names[retrive_result[counter1][counter2]-1]
        img = np.asarray(Image.open(image_address).resize((image_width,image_width)))
        a = int(np.ceil(counter1 * image_width))
        b = int(np.ceil(counter2 * image_width))
        if img.ndim==3:
            merged_image[a:a+image_width,b:b+image_width,:] = img[:,:,:3]
        flag = False
        for i, d in enumerate(image_labels[st]):
            if(retrive_result[counter1][counter2] == d and counter2 >= 0):
                merged_image[a:a+read_rect,b:b+image_width,0] = 255
                merged_image[a:a+image_width,b:b+read_rect, 0] = 255
                merged_image[a+image_width-read_rect:a+image_width,b:b+image_width, 0] = 255
                merged_image[a:a+image_width,b+image_width-read_rect:b+image_width, 0] = 255
                merged_image[a:a + read_rect, b:b + image_width, 1:] = 0
                merged_image[a:a + image_width, b:b + read_rect, 1:] = 0
                merged_image[a + image_width - read_rect:a + image_width, b:b + image_width, 1:] = 0
                merged_image[a:a + image_width, b + image_width - read_rect:b + image_width, 1:] = 0
                flag = True
                break
        if (not flag):
             merged_image[a:a + read_rect, b:b + image_width, 1] = 255
             merged_image[a:a + image_width, b:b + read_rect, 1] = 255
             merged_image[a + image_width - read_rect:a + image_width, b:b + image_width, 1] = 255
             merged_image[a:a + image_width, b + image_width - read_rect:b + image_width, 1] = 255
             merged_image[a:a + read_rect, b:b + image_width, 0] = 0
             merged_image[a:a + image_width, b:b + read_rect, 0] = 0
             merged_image[a + image_width - read_rect:a + image_width, b:b + image_width, 0] = 0
             merged_image[a:a + image_width, b + image_width - read_rect:b + image_width, 0] = 0
             merged_image[a:a + read_rect, b:b + image_width, 2] = 0
             merged_image[a:a + image_width, b:b + read_rect, 2] = 0
             merged_image[a + image_width - read_rect:a + image_width, b:b + image_width, 2] = 0
             merged_image[a:a + image_width, b + image_width - read_rect:b + image_width, 2] = 0
plt.imshow(merged_image)
plt.show()
merged_image = Image.fromarray(merged_image)

merged_image.save('vis-result/ebay/retrieval_embed128-multiloss-with-hsv.png')
