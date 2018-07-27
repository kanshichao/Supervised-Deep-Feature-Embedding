import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import scipy.io
import random

image_path = 'data/clothes/In Shop Clothes Retrieval/'
file_object = open('data/clothes/clothes_test.txt')
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
        if (int(s)-3997) not in image_labels.keys():
            l = []
            l.append(count)
            image_labels[int(s)-3997] = l
        else:
            image_labels[int(s)-3997].append(count)
        count += 1
features = scipy.io.loadmat('embeddings/clothes/embed128_multiloss_with_hsv.mat')

features = features['fc_embedding_cls'][:len(image_names)]

fd = np.zeros((features.shape[0],features.shape[1])).astype('float')
retrive_result = []

show_num_images = 30


seq = [i for i in range(len(image_names))]
img_id = random.sample(seq,show_num_images)

for i,f in enumerate(img_id):
    for j in range(len(image_names)):
        fd[j,:] = features[f]
    idx = np.argsort(np.sqrt(np.sum(np.square(fd - features),axis=1)))
    retrive_result.append(idx[:show_num_images]+1)
    if(i==show_num_images):
        break

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
        for i, d in enumerate(image_labels[st]):
            if(retrive_result[counter1][counter2] == d and counter2 > 0):
                merged_image[a:a+read_rect,b:b+image_width,0] = 255
                merged_image[a:a+image_width,b:b+read_rect, 0] = 255
                merged_image[a+image_width-read_rect:a+image_width,b:b+image_width, 0] = 255
                merged_image[a:a+image_width,b+image_width-read_rect:b+image_width, 0] = 255
                merged_image[a:a + read_rect, b:b + image_width, 1:] = 0
                merged_image[a:a + image_width, b:b + read_rect, 1:] = 0
                merged_image[a + image_width - read_rect:a + image_width, b:b + image_width, 1:] = 0
                merged_image[a:a + image_width, b + image_width - read_rect:b + image_width, 1:] = 0
                break
            if (retrive_result[counter1][counter2] == d and counter2 == 0):
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
                break
plt.imshow(merged_image)
plt.show()
merged_image = Image.fromarray(merged_image)

merged_image.save('vis-result/clothes/retrieval_embed128-multiloss-with-hsv.png')
