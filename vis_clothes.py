import numpy as np
from PIL import Image
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import scipy.io

image_path = 'data/clothes/In Shop Clothes Retrieval/'
ellipside = True
image_width = 32
file_object = open('data/clothes/clothes_test.txt')
try:
    all_the_text = file_object.read().split()
finally:
    file_object.close()
image_names = []
for i,s in enumerate(all_the_text):
    if((i+1)%4==0):
        image_names.append(s)

features = scipy.io.loadmat('embeddings/clothes/embed128_multiloss_without_hsv.mat')

features = features['fc_embedding_cls'][:len(image_names)]

tsne = TSNE()
reduced = tsne.fit_transform(features)
reduced_transformed = reduced - np.min(reduced,axis=0)
reduced_transformed /= np.max(reduced_transformed,axis=0)
image_xindex_sorted = np.argsort(np.sum(reduced_transformed,axis=1))

merged_width = int(np.ceil(np.sqrt(len(image_names))*image_width))
merged_image = np.zeros((merged_width,merged_width,3),dtype='uint8')+255

for counter,index in enumerate(image_xindex_sorted):
    print counter
    if ellipside:
        a = np.ceil(reduced_transformed[counter,0]*(merged_width-image_width-1)+1)
        b = np.ceil(reduced_transformed[counter,1]*(merged_width-image_width-1)+1)
        a = int(a-np.mod(a-1,image_width)+1)
        b = int(b-np.mod(b-1,image_width)+1)
        if merged_image[a,b,0] !=255:
            continue
        image_address = image_path+image_names[counter]
        img = np.asarray(Image.open(image_address).resize((image_width,image_width)))
        if img.ndim==3:
            merged_image[a:a+image_width,b:b+image_width,:] = img[:,:,:3]
plt.imshow(merged_image)
plt.show()
merged_image = Image.fromarray(merged_image)

if ellipside:
    merged_image.save('embed128-multiloss-without-hsv.png')
