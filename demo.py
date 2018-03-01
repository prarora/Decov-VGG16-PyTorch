#!/usr/bin/env python3

from models import *
from utils import *
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import sys
import time
import pdb
import torchvision.transforms as transforms
import torch.nn.functional as F

def vis_layer(activ_map):
    plt.clf()
    plt.subplot(221)
    plt.imshow(activ_map[:,:,0], cmap='gray')

def decon_img(layer_output):
    raw_img = layer_output.data.numpy()[0].transpose(1,2,0)
    img = (raw_img-raw_img.min())/(raw_img.max()-raw_img.min())*255
    img = img.astype(np.uint8) #here
    return img

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: '+sys.argv[0]+' img_file')
        sys.exit(0)

    img_filename = sys.argv[1]

    n_classes = 1000 # using ImageNet pretrained weights

    vgg16_c = VGG16_conv(n_classes)
    conv_layer_indices = vgg16_c.get_conv_layer_indices()
    
    img = np.asarray(Image.open(img_filename).resize((224,224)))    
    

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    img_temp = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])(img)
    img_var = torch.autograd.Variable(img_temp.unsqueeze(0))  

    #img_var = torch.autograd.Variable(torch.FloatTensor(img.transpose(2,0,1)[np.newaxis,:,:,:].astype(float)))  
    
    #pdb.set_trace()
    

    conv_out = vgg16_c(img_var)
    print('VGG16 model:')
    print(vgg16_c)

    plt.ion() # remove blocking
    plt.figure(figsize=(10,10))
    vgg16_d = VGG16_deconv()
    done = False
    print ("Option 1 : Get deconv image for a specific image")
    print ("Option 2 : Get top 9 images corresponding to max activations")
    choose_option = input('Select option?  (1/2): ') == '2'
    if choose_option: 
        while not done:       
            layer = input('Layer to view (0-30, -1 to exit): ')
            feature_map = input('feature_map: ')
                try:
                    layer = int(layer)
                except ValueError:
                    continue
                    
                if layer < 0:
                    sys.exit(0)
            

    else:
        while not done:
            layer = input('Layer to view (0-30, -1 to exit): ')
            try:
                layer = int(layer)
            except ValueError:
                continue
                
            if layer < 0:
                sys.exit(0)
            activ_map = vgg16_c.feature_outputs[layer].data.numpy()
            activ_map = activ_map.transpose(1,2,3,0)
            activ_map_grid = vis_grid(activ_map)
            vis_layer(activ_map_grid)

            # only transpose convolve from Conv2d or ReLU layers
            conv_layer = layer
            if conv_layer not in conv_layer_indices:
                conv_layer -= 1
                if conv_layer not in conv_layer_indices:
                    continue

            n_maps = activ_map.shape[0]

            marker = None
            while True:
                choose_map = input('Select map?  (y/[n]): ') == 'y'
                if marker != None:
                    marker.pop(0).remove()

                if not choose_map:
                    break

                _, map_x_dim, map_y_dim, _ = activ_map.shape
                map_img_x_dim, map_img_y_dim, _ = activ_map_grid.shape
                x_step = map_img_x_dim//(map_x_dim+1)

                print('Click on an activation map to continue')
                x_pos, y_pos = plt.ginput(1)[0]
                x_index = x_pos // (map_x_dim+1)
                y_index = y_pos // (map_y_dim+1)
                map_idx = int(x_step*y_index + x_index)

                if map_idx >= n_maps:
                    print('Invalid map selected')
                    continue

                decon = vgg16_d(vgg16_c.feature_outputs[layer][0][map_idx][None,None,:,:], conv_layer, map_idx, vgg16_c.pool_indices)
                img = decon_img(decon)
                img2 = decon_img(F.relu(decon))
                plt.subplot(221)
                marker = plt.plot(x_pos, y_pos, marker='+', color='red')
                plt.subplot(222)
                plt.imshow(img)
                plt.subplot(223)
                plt.imshow(img2)
                plt.subplot(224)
                img3 = decon.data.numpy()[0]
                img3[0] = img3[0] * 0.229 + 0.485
                img3[1] = img3[1] * 0.224 + 0.456
                img3[2] = img3[2] * 0.225 + 0.406
                img3 = img3.transpose(1,2,0)
                img3 = (img3-img3.min())/(img3.max()-img3.min())*255
                img3 = img3.astype(np.uint8) #here
                plt.imshow(img3)
