#!/usr/bin/python
#
# Copyright 2018 Google LLC
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

import torch
import torch.nn as nn
from model.simsg.layers import build_mlp

"""
PyTorch modules for dealing with graphs.
"""

def _init_weights(module):
  if hasattr(module, 'weight'):
    if isinstance(module, nn.Linear):
      nn.init.kaiming_normal_(module.weight)

class GraphTripleConv(nn.Module):
  """
  A single layer of scene graph convolution.
  """
  def __init__(self, input_dim_obj, input_dim_pred, output_dim=None, hidden_dim=512,
               pooling='avg', mlp_normalization='none'):
    super(GraphTripleConv, self).__init__()
    if output_dim is None:
      output_dim = input_dim_obj
    self.input_dim_obj = input_dim_obj
    self.input_dim_pred = input_dim_pred
    self.output_dim = output_dim
    self.hidden_dim = hidden_dim

    assert pooling in ['sum', 'avg'], 'Invalid pooling "%s"' % pooling

    self.pooling = pooling
    # net1_layers = [2 * input_dim_obj + input_dim_pred, hidden_dim, 2 * hidden_dim + output_dim]
    # net1_layers = [l for l in net1_layers if l is not None]
    # self.net1 = build_mlp(net1_layers, batch_norm=mlp_normalization)
    # self.net1.apply(_init_weights)
    net1 = list()
    net1.append(nn.utils.spectral_norm(nn.Linear(2 * input_dim_obj + input_dim_pred, hidden_dim)))
    net1.append(nn.utils.spectral_norm(nn.Linear(hidden_dim, 2 * hidden_dim + output_dim)))
    self.net1 = nn.Sequential(*net1)
    net2_layers = [hidden_dim, hidden_dim, output_dim]
    # self.net2 = build_mlp(net2_layers, batch_norm=mlp_normalization)
    # self.net2.apply(_init_weights)
    net2 = list()
    net2.append(nn.utils.spectral_norm(nn.Linear(hidden_dim, hidden_dim)))
    net2.append(nn.utils.spectral_norm(nn.Linear(hidden_dim, output_dim)))
    self.net2 = nn.Sequential(*net2)


  def forward(self, obj_vecs, pred_vecs, edges):
    """
    Inputs:
    - obj_vecs: FloatTensor of shape (num_objs, D) giving vectors for all objects
    - pred_vecs: FloatTensor of shape (num_triples, D) giving vectors for all predicates
    - edges: LongTensor of shape (num_triples, 2) where edges[k] = [i, j] indicates the
      presence of a triple [obj_vecs[i], pred_vecs[k], obj_vecs[j]]
    
    Outputs:
    - new_obj_vecs: FloatTensor of shape (num_objs, D) giving new vectors for objects
    - new_pred_vecs: FloatTensor of shape (num_triples, D) giving new vectors for predicates
    """
    dtype, device = obj_vecs.dtype, obj_vecs.device
    num_objs, num_triples = obj_vecs.size(0), pred_vecs.size(0)
    Din_obj, Din_pred, H, Dout = self.input_dim_obj, self.input_dim_pred, self.hidden_dim, self.output_dim
    
    # Break apart indices for subjects and objects; these have shape (num_triples,)
    s_idx = edges[:, 0].contiguous()
    o_idx = edges[:, -1].contiguous()
    
    # Get current vectors for subjects and objects; these have shape (num_triples, Din)
    cur_s_vecs = obj_vecs[s_idx]
    cur_o_vecs = obj_vecs[o_idx]
    
    # Get current vectors for triples; shape is (num_triples, 3 * Din)
    # Pass through net1 to get new triple vecs; shape is (num_triples, 2 * H + Dout)
    # print('[*] shape: {}, {}, {}'.format(cur_s_vecs.size(), pred_vecs.size(), cur_o_vecs.size()))
    cur_t_vecs = torch.cat([cur_s_vecs, pred_vecs, cur_o_vecs], dim=1)
    new_t_vecs = self.net1(cur_t_vecs)

    # Break apart into new s, p, and o vecs; s and o vecs have shape (num_triples, H) and
    # p vecs have shape (num_triples, Dout)
    new_s_vecs = new_t_vecs[:, :H]
    new_p_vecs = new_t_vecs[:, H:(H+Dout)]
    new_o_vecs = new_t_vecs[:, (H+Dout):(2 * H + Dout)]
 
    # Allocate space for pooled object vectors of shape (num_objs, H)
    pooled_obj_vecs = torch.zeros(num_objs, H, dtype=dtype, device=device)

    # Use scatter_add to sum vectors for objects that appear in multiple triples;
    # we first need to expand the indices to have shape (num_triples, D)
    s_idx_exp = s_idx.view(-1, 1).expand_as(new_s_vecs)
    o_idx_exp = o_idx.view(-1, 1).expand_as(new_o_vecs)
    # print(pooled_obj_vecs.shape, o_idx_exp.shape)
    pooled_obj_vecs = pooled_obj_vecs.scatter_add(0, s_idx_exp, new_s_vecs)
    pooled_obj_vecs = pooled_obj_vecs.scatter_add(0, o_idx_exp, new_o_vecs)

    if self.pooling == 'avg':
      #print("here i am, would you send me an angel")
      # Figure out how many times each object has appeared, again using
      # some scatter_add trickery.
      obj_counts = torch.zeros(num_objs, dtype=dtype, device=device)
      ones = torch.ones(num_triples, dtype=dtype, device=device)
      obj_counts = obj_counts.scatter_add(0, s_idx, ones)
      obj_counts = obj_counts.scatter_add(0, o_idx, ones)
  
      # Divide the new object vectors by the number of times they
      # appeared, but first clamp at 1 to avoid dividing by zero;
      # objects that appear in no triples will have output vector 0
      # so this will not affect them.
      obj_counts = obj_counts.clamp(min=1)
      pooled_obj_vecs = pooled_obj_vecs / obj_counts.view(-1, 1)

    # Send pooled object vectors through net2 to get output object vectors,
    # of shape (num_objs, Dout)
    new_obj_vecs = self.net2(pooled_obj_vecs)

    return new_obj_vecs, new_p_vecs


class GraphTripleConvNet_prev(nn.Module):
  """ A sequence of scene graph convolution layers  """
  def __init__(self, input_dim_obj, input_dim_pred, num_layers=5, hidden_dim=512, pooling='avg',
               mlp_normalization='none'):
    super(GraphTripleConvNet, self).__init__()

    self.num_layers = num_layers
    self.gconvs = nn.ModuleList()
    gconv_kwargs = {
      'input_dim_obj': input_dim_obj,
      'input_dim_pred': input_dim_pred,
      'hidden_dim': hidden_dim,
      'pooling': pooling,
      'mlp_normalization': mlp_normalization,
    }
    for _ in range(self.num_layers):
      self.gconvs.append(GraphTripleConv(**gconv_kwargs))

  def forward(self, obj_vecs, pred_vecs, edges):
    for i in range(self.num_layers):
      gconv = self.gconvs[i]
      obj_vecs, pred_vecs = gconv(obj_vecs, pred_vecs, edges)
    return obj_vecs, pred_vecs


class BatchGraphTripleConv(nn.Module):
  """
  A single layer of scene graph convolution.
  """
  def __init__(self, input_dim_obj, input_dim_pred, output_dim=None, hidden_dim=512,
               pooling='avg', mlp_normalization='none'):
    super(BatchGraphTripleConv, self).__init__()
    if output_dim is None:
      output_dim = input_dim_obj
    self.input_dim_obj = input_dim_obj
    self.input_dim_pred = input_dim_pred
    self.output_dim = output_dim
    self.hidden_dim = hidden_dim

    assert pooling in ['sum', 'avg'], 'Invalid pooling "%s"' % pooling

    self.pooling = pooling
    net1_layers = [2 * input_dim_obj + input_dim_pred, hidden_dim, 2 * hidden_dim + output_dim]
    net1_layers = [l for l in net1_layers if l is not None]  # [924, 512, 1332]
    self.net1 = build_mlp(net1_layers, batch_norm=mlp_normalization)
    self.net1.apply(_init_weights)
    
    net2_layers = [hidden_dim, hidden_dim, output_dim]
    self.net2 = build_mlp(net2_layers, batch_norm=mlp_normalization)
    self.net2.apply(_init_weights)


  def forward(self, obj_vecs, pred_vecs, edges):  # [obj_vecs] [b, obj, 308]
    dtype, device = obj_vecs.dtype, obj_vecs.device
    num_batch, num_objs, num_triples = obj_vecs.size(0), obj_vecs.size(1), pred_vecs.size(1)
    Din_obj, Din_pred, H, Dout = self.input_dim_obj, self.input_dim_pred, self.hidden_dim, self.output_dim
    
    # Break apart indices for subjects and objects; these have shape (num_triples,)
    s_idx = edges[:, :, 0].contiguous()  # [b, obj]
    o_idx = edges[:, :, 2].contiguous()  # [b, obj]
    # Get current vectors for subjects and objects; these have shape (num_triples, Din)
    cur_s_vecs = obj_vecs[:,s_idx[1],:]
    cur_o_vecs = obj_vecs[:,s_idx[1],:]
    # Get current vectors for triples; shape is (num_triples, 3 * Din)
    # Pass through net1 to get new triple vecs; shape is (num_triples, 2 * H + Dout)
    # print('[*] shape: {}, {}, {}'.format(cur_s_vecs.size(), pred_vecs.size(), cur_o_vecs.size()))
    cur_t_vecs = torch.cat([cur_s_vecs, pred_vecs, cur_o_vecs], dim=-1)
    new_t_vecs = self.net1(cur_t_vecs)

    # Break apart into new s, p, and o vecs; s and o vecs have shape (num_triples, H) and
    # p vecs have shape (num_triples, Dout)
    new_s_vecs = new_t_vecs[:, :, :H]
    new_p_vecs = new_t_vecs[:, :, H:(H+Dout)]
    new_o_vecs = new_t_vecs[:, :, (H+Dout):(2 * H + Dout)]
 
    # Allocate space for pooled object vectors of shape (num_objs, H)
    pooled_obj_vecs = torch.zeros(num_batch, num_objs, H, dtype=dtype, device=device)

    # Use scatter_add to sum vectors for objects that appear in multiple triples;
    # we first need to expand the indices to have shape (num_triples, D)
    s_idx_exp = s_idx.view(num_batch, -1, 1).expand_as(new_s_vecs)
    o_idx_exp = o_idx.view(num_batch, -1, 1).expand_as(new_o_vecs)
    
    pooled_obj_vecs = pooled_obj_vecs.scatter_add(0, s_idx_exp, new_s_vecs)
    pooled_obj_vecs = pooled_obj_vecs.scatter_add(0, o_idx_exp, new_o_vecs)

    if self.pooling == 'avg':
      #print("here i am, would you send me an angel")
      # Figure out how many times each object has appeared, again using
      # some scatter_add trickery.
      obj_counts = torch.zeros(num_batch, num_objs, dtype=dtype, device=device)
      ones = torch.ones(num_batch, num_triples, dtype=dtype, device=device)
      obj_counts = obj_counts.scatter_add(0, s_idx, ones)
      obj_counts = obj_counts.scatter_add(0, o_idx, ones)
  
      # Divide the new object vectors by the number of times they
      # appeared, but first clamp at 1 to avoid dividing by zero;
      # objects that appear in no triples will have output vector 0
      # so this will not affect them.
      obj_counts = obj_counts.clamp(min=1)
      pooled_obj_vecs = pooled_obj_vecs / obj_counts.view(num_batch, -1, 1)

    # Send pooled object vectors through net2 to get output object vectors,
    # of shape (num_objs, Dout)
    new_obj_vecs = self.net2(pooled_obj_vecs)

    return new_obj_vecs, new_p_vecs


class GraphTripleConvNet(nn.Module):
  """ A sequence of scene graph convolution layers  """
  def __init__(self, input_dim_obj, input_dim_pred, num_layers=5, hidden_dim=512, pooling='avg',
               mlp_normalization='none'):
    super(GraphTripleConvNet, self).__init__()

    self.num_layers = num_layers
    self.gconvs = nn.ModuleList()
    gconv_kwargs = {
      'input_dim_obj': input_dim_obj,
      'input_dim_pred': input_dim_pred,
      'hidden_dim': hidden_dim,
      'pooling': pooling,
      'mlp_normalization': mlp_normalization,
    }
    for _ in range(self.num_layers):
      self.gconvs.append(BatchGraphTripleConv(**gconv_kwargs))

  def forward(self, s_label, p_label, o_label):
    for i in range(self.num_layers):
      gconv = self.gconvs[i]
      obj_vecs, pred_vecs = gconv(s_label, p_label, o_label)
    return obj_vecs, pred_vecs