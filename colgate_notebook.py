import os, sys
# os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
import tensorflow as tf
tf.compat.v1.enable_eager_execution()

import numpy as np
import imageio
import json
import random
import time
import pprint
import cv2

import matplotlib.pyplot as plt

import run_nerf

from load_llff import load_llff_data
from load_deepvoxels import load_dv_data
from load_blender import load_blender_data

basedir = './logs'
expname = 'colgatesphere'

config = os.path.join(basedir, expname, 'config.txt')
print('Args:')
print(open(config, 'r').read())
parser = run_nerf.config_parser()
print(basedir)
args = parser.parse_args('--config {} --ft_path {}'.format(config, os.path.join(basedir, expname, 'model_100000.npy')))
print('loaded args')

images, poses, bds, render_poses, i_test = load_llff_data(args.datadir, 
                                                        #   args.factor, 
                                                          recenter=True, bd_factor=.75, 
                                                          spherify=args.spherify,
                                                          type="colgate")
H, W, focal = poses[0,:3,-1].astype(np.float32)

H = int(H)
W = int(W)
hwf = [H, W, focal]

images = images.astype(np.float32)
poses = poses.astype(np.float32)

if args.no_ndc:
    near = tf.reduce_min(bds) * .9
    far = tf.reduce_max(bds) * 1.
else:
    near = 0.
    far = 1.

# Create nerf model
_, render_kwargs_test, start, grad_vars, models = run_nerf.create_nerf(args)

bds_dict = {
    'near' : tf.cast(near, tf.float32),
    'far' : tf.cast(far, tf.float32),
}
render_kwargs_test.update(bds_dict)

print('Render kwargs:')
pprint.pprint(render_kwargs_test)

fps = 30
down = 1
render_kwargs_fast = {k : render_kwargs_test[k] for k in render_kwargs_test}
render_kwargs_fast['N_importance'] = 0

c2w = np.eye(4)[:3,:4].astype(np.float32) # identity pose matrix
test = run_nerf.render(H//down, W//down, focal/down, c2w=c2w, **render_kwargs_fast)
img = np.clip(test[0],0,1)
# plt.imshow(img)
# plt.show()
print("after")
out = cv2.VideoWriter('logs/video_colgate10.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps,(W,H))
down = 1
for i, c2w in enumerate(render_poses):
    if i%4==0: print(i)
    # test = run_nerf.render(H, W, focal, c2w=c2w, **render_kwargs_fast)
    # test = run_nerf.render(H, W, focal, c2w=c2w[:3,:4], **render_kwargs_fast)
    test = run_nerf.render(H//down, W//down, focal/down, c2w=c2w[:3,:4], **render_kwargs_fast)
    img = (255*np.clip(test[0],0,1)).astype(np.uint8)
    img = np.flip(img, axis=-1) 
    out.write(img)
    
print('done, saving')
out.release()
# imageio.mimwrite(f, frames, fps=30, quality=10)
