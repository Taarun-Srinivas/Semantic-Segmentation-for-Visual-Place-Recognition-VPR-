import os
import matplotlib.pyplot as plt
import numpy as np
import wget
from os.path import join
import cv2
from PIL import Image
import torch
import torch.nn.functional as F

from train_segmentation import LitUnsupervisedSegmenter
from utils import get_transform
from crf import dense_crf
from utils import unnorm, remove_axes

def get_model():
    print("Inside get model")
    saved_models_dir = join("..", "saved_models")
    saved_model_name = "cocostuff27_vit_base_5.ckpt"

    os.chdir("../src")
    os.makedirs(saved_models_dir, exist_ok=True)
    saved_model_url_root = "https://marhamilresearch4.blob.core.windows.net/stego-public/saved_models/"

    if not os.path.exists(join(saved_models_dir, saved_model_name)):
        wget.download(saved_model_url_root + saved_model_name, join(saved_models_dir, saved_model_name))

    model = LitUnsupervisedSegmenter.load_from_checkpoint(join(saved_models_dir, saved_model_name)).cuda()
    print("\nmodel produced successfully")
    return model

def transform_image(image_path):
    print("Successfully read the image")

    image = cv2.imread(image_path)
    image = image[:,:,::-1]
    transform = get_transform(448, False, "center")
    image = Image.fromarray(image.astype('uint8'), 'RGB')
    image = transform(image).unsqueeze(0).cuda()
    print("Transforming the input image to cuda image")
    return image

def convert_cuda_img_to_cpu(cuda_img):
    print("received cuda image")
    img_cpu = np.array(unnorm(cuda_img)[0].permute(1, 2, 0).cpu())
    print("Converted cuda image to cpu")
    return img_cpu

def get_model_pred(_img):
    print("feeding in the cuda image")
    with torch.no_grad():
        code1 = model(img)
        code2 = model(img.flip(dims=[3]))
        code = (code1 + code2.flip(dims=[3])) / 2
        code = F.interpolate(code, img.shape[-2:], mode='bilinear', align_corners=False)
        linear_probs = torch.log_softmax(model.linear_probe(code), dim=1).cpu()
        cluster_probs = model.cluster_probe(code, 2, log_probs=True).cpu()
        img_1 = img[0]


        single_img = img_1.cpu()
        linear_pred = dense_crf(single_img, linear_probs[0]).argmax(0)
        cluster_pred = dense_crf(single_img, cluster_probs[0]).argmax(0)

        print("Returning the linear and cluster predictions for the given image")

    return _img, linear_probs, cluster_probs, linear_pred, cluster_pred

def linear_original_fusion(limg, linear_pred):
    linear_pred = model.label_cmap[linear_pred]
    lalpha_channel1 = linear_pred[:, :, 0] / 255
    lalpha_channel2 = linear_pred[:, :, 1] / 255
    lalpha_channel3 = linear_pred[:, :, 2] / 255
    l_overlay_colors = linear_pred[:, :, :3]
    lalpha_mask = np.dstack((lalpha_channel1, lalpha_channel2, lalpha_channel3))
    lh, lw = limg.shape[:2]
    lbackground_subsection = limg[0:lh, 0:lw]
    lcomposite = lbackground_subsection * (1 - lalpha_mask) + l_overlay_colors * lalpha_mask
    limg[0:lh, 0:lw] = lcomposite
    print("Fused the linear predictions with the original image")
    return limg

def cluster_original_fusion(cimg, cluster_pred):
    cluster_pred = model.label_cmap[cluster_pred]
    calpha_channel1 = cluster_pred[:, :, 0] / 255
    calpha_channel2 = cluster_pred[:, :, 1] / 255
    calpha_channel3 = cluster_pred[:, :, 2] / 255
    overlay_colors = cluster_pred[:, :, :3]
    calpha_mask = np.dstack((calpha_channel1, calpha_channel2, calpha_channel3))
    h, w = cimg.shape[:2]
    background_subsection = cimg[0:h, 0:w]
    composite = background_subsection * (1 - calpha_mask) + overlay_colors * calpha_mask
    cimg[0:h, 0:w] = composite
    cimg_copy = np.copy(cimg)
    cimg_copy = cimg_copy[:,:,:]*255
    print("Fused the cluster predictions with the original image")
    return cimg, cimg_copy

def plt_pred_imgs(img, linear_pred, cluster_pred):
    fig, ax = plt.subplots(1,3, figsize=(5*4,5))
    ax[0].imshow(unnorm(img)[0].permute(1,2,0).cpu())
    ax[0].set_title("Image")
    ax[1].imshow(model.label_cmap[cluster_pred])
    ax[1].set_title("Cluster Predictions")
    ax[2].imshow(model.label_cmap[linear_pred])
    ax[2].set_title("Linear Probe Predictions")
    plt.show()
    remove_axes(ax)

def plt_pred_fusion_imgs(img, linear, cluster):
    fig, ax = plt.subplots(1,3, figsize=(5*4,5))
    ax[0].imshow(img)
    ax[0].set_title("Image")
    ax[1].imshow(cluster)
    ax[1].set_title("Cluster x original")
    plt.savefig(dest_dir)
    ax[2].imshow(linear)
    ax[2].set_title("Linear x original")
    plt.show()
    remove_axes(ax)

def save_my_image(destination_dir, k, img):
    cv2.imwrite(os.path.join(destination_dir,k),img)

def get_summary(_img, _linear_probs, _cluster_probs, _linear_pred, _cluster_pred):
    print("\nTo summarize")
    print("The input img shape     - {}  min_val= {} max_val= {}".format(_img.shape, _img.min(), _img.max()))
    print("The linear probs shape  - {} min_val= {} max_val= {}". format(_linear_probs.shape, _linear_probs.min(), _linear_probs.max()))
    print("The cluster probs shape - {} min_val= {} max_val= {}". format(_cluster_probs.shape, _cluster_probs.min(), _cluster_probs.max()))
    print("The linear pred shape   - {} min_val= {} max_val= {}".format(_linear_pred.shape, _linear_pred.min(), _linear_pred.max()))
    print("The cluster pred shape  - {} min_val= {} max_val= {}".format(_cluster_pred.shape, _cluster_pred.min(), _cluster_pred.max()))
    print("unique vals of linear pred - {}".format(np.unique(linear_pred)))
    print("unique vals of cluster pred - {}".format(np.unique(cluster_pred)))

if __name__ == "__main__":

    model = get_model()

    #Define the source image directory here
    parent_dir = r"C:\resources\AI4CE lab works\sample_data\wet_cloudy_sunset_data\cam1"

    #Define the path of the folder which will contain all the rendered images.
    dest_dir = r"C:\resources\AI4CE lab works\RESULTS\wet_cloudy_data\cam1\cluster_pred"

    for i in os.listdir(parent_dir):
        print("\n",os.path.join(parent_dir,i))

        img_path = os.path.join(parent_dir,i)
        img = transform_image(img_path)

        img, linear_probs, cluster_probs, linear_pred, cluster_pred = get_model_pred(img)

        orig_img = convert_cuda_img_to_cpu(img)
        orig_img1, orig_img2 = np.copy(orig_img), np.copy(orig_img)

        linear_orig = linear_original_fusion(orig_img1, linear_pred)
        cluster_x_orig, cluster_copy = cluster_original_fusion(orig_img2, cluster_pred)
        save_my_image(dest_dir, i, cluster_copy)
