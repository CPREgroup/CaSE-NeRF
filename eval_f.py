# Copyright 2021 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3
"""Evaluation script for mip-NeRF."""
import functools
from os import path

from absl import app
from absl import flags
import flax
from flax.metrics import tensorboard
from flax.training import checkpoints
import jax
from jax import random
import numpy as np
import os
from skimage.metrics import structural_similarity as SSIM

from internal import datasets
from internal import math
from internal import models
from internal import utils
from internal import vis



import lpips

FLAGS = flags.FLAGS
utils.define_common_flags()
flags.DEFINE_bool(
    'eval_once', True,
    'If True, evaluate the model only once, otherwise keeping evaluating new'
    'checkpoints if any exist.')
flags.DEFINE_bool('save_output', True,
                  'If True, save predicted images to disk.')


def main(unused_argv):
    
  #lens params

  #pixel_scale = 50#根据lens公式算出来的是coc以0-1的空间算的。但实际我们训练是按pixel单位算的，需要转换。
  a=0 #0.02#*pixel_scale
  f=0.1
  l=0.6#0.6
  train_coc=1
  
  
    
    
  config = utils.load_config()
  FLAGS.train_dir = config.train_dir
  FLAGS.data_dir = config.data_dir
  dataset = datasets.get_dataset('test', FLAGS.data_dir, config)
  # dataset_r3 = datasets.get_dataset('test', path.join(config.data_dir, 'Guassblur', '3'), config)
  # dataset_r7 = datasets.get_dataset('test', path.join(config.data_dir, 'Guassblur', '7'), config)
  # dataset_r15 = datasets.get_dataset('test', path.join(config.data_dir, 'Guassblur', '15'), config)
  # dataset_r31 = datasets.get_dataset('test', path.join(config.data_dir, 'Guassblur', '31'), config)
  # dataset_r51 = datasets.get_dataset('test', path.join(config.data_dir, 'Guassblur', '51'), config)
  model, init_variables = models.construct_mipnerf(
      random.PRNGKey(20200823), dataset.peek())
  optimizer = flax.optim.Adam(config.lr_init).create(init_variables)
  state = utils.TrainState(optimizer=optimizer)
  del optimizer, init_variables

  # Rendering is forced to be deterministic even if training was randomized, as
  # this eliminates 'speckle' artifacts.
  def render_eval_fn(variables, _, rays, a, f, l, train_coc):
    return jax.lax.all_gather(
        model.apply(
            variables,
            random.PRNGKey(0),  # Unused.
            rays,
            randomized=False,
            white_bkgd=config.white_bkgd, a=a, f=f, l=l, train_coc=train_coc),
        axis_name='batch')

  render_eval_pfn = jax.pmap(
      render_eval_fn,
      in_axes=(None, None, 0, None, None, None, None),  # Only distribute the data input.
      donate_argnums=(2,),
      axis_name='batch',
  )

  ssim_fn = jax.jit(functools.partial(math.compute_ssim, max_val=1.))
  lpips_fn = lpips.LPIPS(net="vgg").eval()

  last_step = 0
  out_dir = path.join(FLAGS.train_dir,
                      'path_renders_{}_{}'.format(config.render_temp,config.render_exp) if config.render_path else 'test_preds_{}_{}'.format(config.render_temp,config.render_exp))
  if not FLAGS.eval_once:
    summary_writer = tensorboard.SummaryWriter(
        path.join(FLAGS.train_dir, 'eval'))
  while True:
    state = checkpoints.restore_checkpoint(FLAGS.train_dir, state)
    step = int(state.optimizer.state.step)
    if step <= last_step:
      continue
    if FLAGS.save_output and (not utils.isdir(out_dir)):
      utils.makedirs(out_dir)
    psnr_values = []
    ssim_values = []
    lpips_values = []
    avg_values = []
    if not FLAGS.eval_once:
      showcase_index = random.randint(random.PRNGKey(step), (), 0, dataset.size)
    for idx in range(dataset.size):
      o = dataset.rays.origins[idx]
      d = dataset.rays.directions[idx]
      v = dataset.rays.viewdirs[idx]
      r = dataset.rays.radii[idx]
      # temp = dataset.render_rays.temperature[idx]
      # exp = dataset.render_rays.exposure[idx]
      train_coc = dataset.rays.coc[idx][0][0][0] #1
      temp = dataset.rays.temperature[idx]
      exp = dataset.rays.exposure[idx]
      coc = dataset.rays.coc[idx]
      loss_mult = dataset.rays.lossmult[idx]
      ne_ar = dataset.rays.near[idx]
      f_ar = dataset.rays.far[idx]
      rrays = utils.Rays(
          origins=o,
          directions=d,
          viewdirs=v,
          radii=r,
          lossmult=loss_mult,
          near=ne_ar,
          far=f_ar,
          temperature=temp,
          exposure=exp,
          coc=coc,
      )

      batch = next(dataset)
#     for idx in range(2):
#       print(f'Evaluating {idx+1}/{dataset_r3.size*5}')
#       if idx // 2 == 0:
#         batch = next(dataset)
#         train_coc = 1
#         print(train_coc)
#       elif idx // 2 == 1:
#         batch = next(dataset_r3)
#         train_coc = 3
#         print(train_coc)
#       elif idx // 2 == 2:
#         batch = next(dataset_r7)
#         train_coc = 7
#         print(train_coc)
#       elif idx // 2 == 3:
#         batch = next(dataset_r15)
#         train_coc = 15
#         print(train_coc)
#       elif idx // 2 == 4:
#         batch = next(dataset_r31)
#         train_coc = 31
#         print(train_coc)
      # elif idx // 2 == 5:
      #   batch = next(dataset_r51)
      #   train_coc = 51
      #   print(train_coc)

      pred_color, pred_distance, pred_acc = models.render_image(
          functools.partial(render_eval_pfn, state.optimizer.target),
          rrays, #batch['rays'],
          None,
          chunk=FLAGS.chunk, a=a, f=f, l=l, train_coc=train_coc)

      #vis_suite = vis.visualize_suite(pred_distance, pred_acc)

      if jax.host_id() != 0:  # Only record via host 0.
        continue
      if not FLAGS.eval_once and idx == showcase_index:
        showcase_color = pred_color
        showcase_acc = pred_acc
        #showcase_vis_suite = vis_suite
        if not config.render_path:
          showcase_gt = batch['pixels']
      if not config.render_path:
        psnr = float(
            math.mse_to_psnr(((pred_color - batch['pixels'])**2).mean()))
        # ssim = float(ssim_fn(pred_color, batch['pixels']))
        t1_np = jax.device_get(pred_color)
        t2_np = jax.device_get(batch['pixels'])
        ssim = SSIM(t1_np,t2_np,multichannel=True)
        t1 = t1_np * 255
        t2 = t2_np * 255
        lpips_score = lpips_fn(lpips.im2tensor(t1),lpips.im2tensor(t2) , normalize=True).item()
        # print(f'PSNR={psnr:.4f} lpips={lpips_score:.4f}')
        print(f'PSNR={psnr:.4f} SSIM={ssim:.4f} LPIPS={lpips_score:.4f}')
        psnr_values.append(psnr)
        ssim_values.append(ssim)
        lpips_values.append(lpips_score)
        all = jax.numpy.concatenate((t1_np,t2_np),axis = 1)
      if FLAGS.save_output and (config.test_render_interval > 0):
        if (idx % config.test_render_interval) == 0:
          utils.save_img_uint8(
            all, path.join(out_dir, 'color1_{:03d}.png'.format(idx)))
          utils.save_img_uint8(
              pred_color, path.join(out_dir, 'color_{:03d}.png'.format(idx)))
          # utils.save_img_float32(
          #     pred_distance,
          #     path.join(out_dir, 'distance_{:03d}.tiff'.format(idx)))
          # utils.save_img_float32(
          #     pred_acc, path.join(out_dir, 'acc_{:03d}.tiff'.format(idx)))
          #for k, v in vis_suite.items():
            #utils.save_img_uint8(
                #v, path.join(out_dir, k + '_{:03d}.png'.format(idx)))
    print('AVG_PSNR: ', np.mean(np.array(psnr_values)),'AVG_SSIM: ', np.mean(np.array(ssim_values)),'AVG_lpips: ', np.mean(np.array(lpips_values)))
    if (not FLAGS.eval_once) and (jax.host_id() == 0):
      summary_writer.image('pred_color', showcase_color, step)
      summary_writer.image('pred_acc', showcase_acc, step)
      #for k, v in showcase_vis_suite.items():
        #summary_writer.image('pred_' + k, v, step)
      if not config.render_path:
        summary_writer.scalar('psnr', np.mean(np.array(psnr_values)), step)
        summary_writer.scalar('ssim', np.mean(np.array(ssim_values)), step)
        summary_writer.image('target', showcase_gt, step)
    if FLAGS.save_output and (not config.render_path) and (jax.host_id() == 0):
      with utils.open_file(path.join(out_dir, f'psnrs_{step}.txt'), 'w') as f:
        f.write(' '.join([str(v) for v in psnr_values]))
      with utils.open_file(path.join(out_dir, f'ssims_{step}.txt'), 'w') as f:
        f.write(' '.join([str(v) for v in ssim_values]))
      with utils.open_file(path.join(out_dir, f'lpips_{step}.txt'), 'w') as f:
        f.write(' '.join([str(v) for v in lpips_values]))
    if FLAGS.eval_once:
      break
    if int(step) >= config.max_steps:
      break
    last_step = step


if __name__ == '__main__':
  app.run(main)
