import numpy as np
from PIL import Image
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import scipy.io

image_path = 'data/ebay/Stanfor_Online_Products/'
ellipside = True
image_width = 32
file_object = open('data/ebay/Ebay_test.txt')
try:
    all_the_text = file_object.read().split()
finally:
    file_object.close()
image_names = []
for i,s in enumerate(all_the_text):
    if((i+1)%4==0):
        image_names.append(s)

features = scipy.io.loadmat('embeddings/ebay/validation_googlenet_feat_matrix_liftedstructsim_softmax_pair_m128_multilabel_embed128_baselr_1E4_gaussian2k_fcembeddingcls.mat')

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
    merged_image.save('ebay-fc-embedding-cls-t-sne-128.png')
