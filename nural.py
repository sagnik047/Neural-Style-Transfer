#importing tensorflow and  importing VGG19 model from tensorflowhub 
import os
import tensorflow as tf
tf.compat.v1.enable_eager_execution()
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from tensorflow.python.keras.applications.vgg19 import VGG19
#for printing summary of model
model = VGG19(
include_top = False,
weights = 'imagenet'
)
model.trainable = False
model.summary()

#importing required library and function
from tensorflow.python.keras.preprocessing.image import load_img, img_to_array
from tensorflow.python.keras.applications.vgg19 import preprocess_input
from tensorflow.python.keras.models import Model

import numpy as np
import matplotlib.pyplot as plt


#function for preprocessing the image before use
def preprocess_input_image(img_path):
    img = load_img(img_path)
    img = img_to_array(img)
    img = preprocess_input(img)
    img = np.expand_dims(img, axis = 0)
    return img

#function for postprocessing the image 
def postprocess(x):
    x[:,:,0] += 103.939
    x[:,:,1] += 116.779
    x[:,:,2] += 123.68
    x = x[:,:,::-1]
    x = np.clip(x,0,255).astype('uint8')
    return x

#function  for displaying the image
def display_img(image):
    if len(image.shape)==4:
        img = np.squeeze(image, axis = 0)
        
    img = postprocess(img)
    
    plt.grid = False
    plt.xticks([])
    plt.yticks([])
    plt.imshow(img)
    return
    

#display_img(preprocess_input_image('style2.jpg'))
#display_img(preprocess_input_image('content.jpeg'))

#layers we will use from VGG19 model for training
content_layer = 'block5_conv2'
style_layers = [
    'block1_conv1',
    'block3_conv1',
    'block5_conv1'
]
#takes output of activation for content image from 
content_model = Model(
   model.input,
   outputs = model.get_layer(content_layer).output
) 
#takes output of activation for style image 
style_models = [Model(
   inputs = model.input,    
   outputs = model.get_layer(layer).output) for layer in style_layers] 

#function for generating content cost
def content_cost(content,generated):
    a_C = content_model(content)
    a_G = content_model(generated)
    cost = tf.reduce_mean(tf.square(a_C - a_G))
    return cost
#evaluates gram matrix
def gram_matrix(A):
    n_C = int(A.shape[-1])
    a = tf.reshape(A,[-1,n_C])
    n = tf.shape(a)[0]
    G = tf.matmul(a,a, transpose_a = True)
    return G/tf.cast(n,tf.float32)


#for generating style cost
lam = 1. / len(style_models)
def style_cost(style,generated):
    J_cost = 0
    for style_model in style_models:
        a_S = style_model(style)
        a_G = style_model(generated)
        GS = gram_matrix(a_S)
        GG = gram_matrix(a_G)
        current_cost = tf.reduce_mean(tf.square(GS - GG))
        J_cost = current_cost * lam
    return J_cost



#training loop function 
import time
processed_images = []
def training_loop(content_path, style_path, iter = 20, alpha = 10, beta = 20):
    content = preprocess_input_image(content_path)
    style = preprocess_input_image(style_path)
    
    generated = tf.Variable(content, dtype = tf.float32)
    
    opt = tf.compat.v1.train.AdamOptimizer(learning_rate=7.0)
    
    best_cost = 1e22 + 0.1
    best_image = None
    
    start_time = time.time()
    
    for i in range(iter):
        with tf.GradientTape() as tape:
            J_content = content_cost(content, generated)
            J_style = style_cost(style, generated)
            J_total = alpha * J_content + beta * J_style
            
        grads = tape.gradient(J_total, generated)
        opt.apply_gradients([(grads, generated)])
        
        if J_total < best_cost:
            best_cost = J_total
            best_image = generated.numpy()
            
        print('Cost at {}: {}. Time Elapsed {}.'.format(i, J_total, time.time() - start_time))
        processed_images.append(generated.numpy())
    return best_image
    

#pass the style and content image and give the best image
best_image = training_loop('content1.jpeg', 'style3.jpg')
#output the best generated image
display_img(best_image)
#output the intermediate steps' generated images
plt.figure(figsize = (10, 10))
for i in range(20):
   plt.subplot(5, 4, i+1)
   display_img(processed_images[i])

plt.show()