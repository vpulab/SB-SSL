from ieat.utils import resize, normalize_img, color_quantize_np

import os

import transformers
from transformers.modeling_gpt2 import GPT2LMHeadModel

import torch
import torchvision.transforms as transforms
import torch.nn as nn
from torch.nn.parameter import Parameter
import tensorflow as tf
import tensorflow_hub as hub

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import logging

from .resnet import ResNet as openself_resnet
from .resnet_google import resnet50x1

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
logger = logging.getLogger()

# # Code adapted from
# https://colab.research.google.com/github/apeguero1/image-gpt/blob/master/Transformers_Image_GPT.ipynb
# - thanks to the author


class EmbeddingExtractor:
	"""Extracts embeddings from images with a pre-trained model."""
	def __init__(self, model_name, from_cache):
		"""
		Parameters
		----------
		model_name : str
			A name for this model, used for caching.
		from_cache : bool
			Whether to used cached embeddings.
		"""
		self.from_cache = from_cache
		self.model_name = model_name
		self.model = None

	def load_model(self):
		"""
		Loads the model, from the web or from the filesystem.
		"""
		raise NotImplementedError

	def extract_dir(self, d, file_types=(".jpg", ".jpeg", ".png", ".webp"), batch_size=None, visualize=False, invert=False, **extract_params):
		"""
		Extracts embeddings from images in a directory.
		Parameters
		----------
		d : str
			path to a directory of images
		file_types : list[str]
			list of acceptable file extensions for images
		batch_size : int
			number of images processed at a time - helps when you have limited memory
		visualize : bool
			whether to display the images after pre-processing
		extract_params : dict
			additional parameters for extraction

		Returns
		-------
		encs : pd.DataFrame
			a Pandas dataframe of features - see `EmbeddingExtractor.extract`
		"""
		embedding_path = self._make_embedding_path(d)
		image_paths = [
			os.path.join(d, f) for f in os.listdir(d)
			if os.path.splitext(f)[1] in file_types
		]
		if self.from_cache and os.path.exists(embedding_path):
			logger.info("Loading embeddings for %s from file" % os.path.basename(d))
			# encs = pd.read_csv(embedding_path, index_col=0).set_index("img")
			encs = pd.read_hdf(embedding_path, key='stage', index_col=0).set_index("img")
			if visualize:
				self.process_samples(image_paths, visualize=True, invert=invert)
		else:
			logger.info("Extracting embeddings for %s" % os.path.basename(d))
			
			# do extraction in batches to save memory
			
			encs = self.extract(
				image_paths,
				batch_size=batch_size,
				output_path=embedding_path,
				visualize=visualize,
				invert=invert,
				**extract_params
			)
		return encs

	def extract(self, image_paths, batch_size=None, output_path=None, gpu=False, visualize=False, invert=False, **extract_kwargs):
		"""
		Extracts features from a set of image paths.

		Parameters
		----------
		image_paths : str
			a list of paths to images to extract features for
		batch_size : int or None
			number of images processed at a time - helps when you have limited memory; if None, use just one batch
		output_path : str or None
			path to save a CSV cache file with the extracted features; if none, don't cache
		gpu : bool
			whether to use GPU (True) or CPU (False)
		visualize : bool
			whether to display the images after pre-processing
		extract_kwargs : dict
			additional parameters for extraction

		Returns
		-------
		encs : pd.DataFrame
			data frame of features, indexed by the original image path
		"""
		if self.model is None:
			self.load_model()
		if batch_size is None:
			batch_size = len(image_paths)

		with torch.no_grad():  # saves some memory
			batches = [image_paths[i:i+batch_size] for i in range(0, len(image_paths), batch_size)]

			# model specific context extraction
			encs = pd.concat([
				pd.DataFrame(
					self._extract_context(self.process_samples(batch, visualize=visualize, invert=invert), gpu, **extract_kwargs)
				)
				for batch in batches
			])

			encs["img"] = [os.path.basename(path) for path in image_paths]

			# DEPRECATED - NOW THAT CACHE IS STORED BY CATEGORY
			# df["category"] = [os.path.basename(os.path.dirname(path)) for path in image_paths]

			if output_path is not None:
				# add the image names to the CSV file
				# encs.to_csv(output_path)
				encs.to_hdf(output_path, key='stage')

			return encs.set_index("img")

	def process_samples(self, image_paths, invert=False, visualize=False):
		"""
		Pre-process the image samples for embedding extraction.

		Parameters
		----------
		image_paths : list[str]
			list of image paths to pre-process
		visualize : bool
			whether to display the images after pre-processing

		Returns
		-------
		list
			list of processed images, usually as `list[np.ndarray]`
		"""
		raise NotImplementedError

	def _extract_context(self, samples, gpu, **extract_kwargs) -> np.ndarray:
		raise NotImplementedError

	def _make_embedding_path(self, d):
		return "embeddings/{}_{}_{}_{}.csv".format(
			os.path.basename(os.path.dirname(d)),
			os.path.basename(d),
			self.model_name,
			self._make_param_path()
		)

	def _make_param_path(self):
		raise NotImplementedError

	@staticmethod
	def visualize(images, paths):
		"""
		Visualize some preprocessed images.

		Parameters
		----------
		images : list[np.ndarray]
			the images, as matrices
		paths : list[str]
			list of the original image paths, so we can get the parent directory
		"""
		print(os.path.basename(os.path.dirname(paths[0])))
		f, axes = plt.subplots(1, len(images), dpi=300)
		for img, ax in zip(images, axes):
			ax.axis('off')
			ax.imshow(img)
		plt.show()


class ModelZooExtractor(EmbeddingExtractor):
	"""
		Extractor for the models from the openselfsup's Model ZOO
		(https://github.com/open-mmlab/OpenSelfSup/blob/master/docs/MODEL_ZOO.md).
	"""

	n_px = 224

	def __init__(self, model_name, feats='GAP', depth=50, width=1, pretrained=None, **parent_params):
		"""
		Parameters
		----------
		depth : int
			Depth of the ResNet used.
		width : int
			Width of the resnet used.
		feats : str
			Name of the utput layer: one of [layer_1, layer_2, layer_3, layer_4, layer_5, avg_pool]
		"""
		super().__init__(model_name, **parent_params)
		self.pretrained = pretrained
		self.depth = depth
		self.width = width
		self.images = None
		self.backbone = resnet50x1(return_level=feats).to('cuda:0')

	def load_model(self):
		self.backbone.load_state_dict(torch.load(self.pretrained)['state_dict'])

	def process_samples(self, image_paths: list, visualize=False, invert=False):
		if invert:
			images = np.array([1-image/255 for image in resize(ModelZooExtractor.n_px, image_paths)])
		else:
			images = np.array([image / 255 for image in resize(ModelZooExtractor.n_px, image_paths)])

		if visualize:
			self.visualize(images, image_paths)
		return images

	def _extract_context(self, samples, gpu, **extract_kwargs) -> np.ndarray:

		self.backbone.eval()
		with torch.no_grad():
			tensor_samples = torch.Tensor(samples).to('cuda:0')
			feats = self.backbone(torch.movedim(tensor_samples, 3, 1))
			feats = torch.stack([torch.flatten(feats[i,:,:,:]) for i in range(feats.shape[0])])
		return feats.cpu().numpy()

	def _make_param_path(self):
		return f"{self.depth}_{self.width}x"


class SimCLRExtractor(EmbeddingExtractor):
	"""Extractor using the [SimCLR model](https://github.com/google-research/simclr)."""
	n_px = 224

	def __init__(self, model_name: str, depth: int, width: int, sk: int, **parent_params):
		"""
		Parameters
		----------
		model_name : str
			A name for this model, used for caching.
		depth : int
			Depth of the ResNet used.
		width : int
			Width of the resnet used.
		sk : bool
			Whether to use selective kernels.
		parent_params
		"""
		super().__init__(model_name, **parent_params)
		tf.compat.v1.disable_eager_execution()
		self.depth = depth
		self.width = width
		self.sk = sk
		self.sess = None
		self.images = None

	def load_model(self):
		hub_path = f"gs://simclr-checkpoints/simclrv2/pretrained/r{self.depth}_{self.width}x_sk{self.sk}/hub"
		# hub_path = f"gs://simclr-checkpoints/simclrv1/pretrain/{self.width}x/hub"
		# hub_path = "ResNet50_1x/hub"
		module = hub.Module(hub_path, trainable=False)
		self.images = tf.compat.v1.placeholder(tf.float32)
		self.model = module(inputs=self.images, signature="default", as_dict=True)
		self.sess = tf.compat.v1.Session()
		self.sess.run(tf.compat.v1.global_variables_initializer())

	def process_samples(self, image_paths: list, visualize=False):
		images = np.array([image/255 for image in resize(SimCLRExtractor.n_px, image_paths)])

		if visualize:
			self.visualize(images, image_paths)

		return images

	def _extract_context(self, samples, gpu, **extract_kwargs) -> np.ndarray:
		output = self.sess.run(self.model, {self.images: samples})
		# 'default' is the representation output of the base ResNet network
		encs = output['default']
		return encs

	def _make_param_path(self):
		return f"{self.depth}_{self.width}x_sk{self.sk}"


class GPTExtractor(EmbeddingExtractor):
	"""Extractor using [iGPT](https://github.com/openai/image-gpt). You must download the model manually."""
	MODELS = {"l": (1536, 16, 48), "m": (1024, 8, 36), "s": (512, 8, 24)}

	def __init__(self, model_name, model_size, models_dir, color_clusters_dir, n_px, **parent_params):
		"""

		Parameters
		----------
		model_name : str
			A name for this model, used for caching.
		model_size : str
			The size of iGPT used - "s" for small, "m" for medium, or "l" for large. The exact parameters are stored in
			`GPTExtractor.MODELS`.
		models_dir : str
			Path to directory with downloaded model. Make sure the params match the downloaded model.
		color_clusters_dir : str
			Path to directory with the downloaded color clusters.
		n_px : int
			The number of pixels used. All publicly available versions of iGPT are 32x32.
		parent_params
		"""
		super().__init__(model_name, **parent_params)

		self.n_px = n_px
		self.model_size = model_size

		color_clusters_file = "%s/kmeans_centers.npy" % color_clusters_dir
		self.clusters = np.load(color_clusters_file)  # get color clusters

		n_embd, n_head, n_layer = GPTExtractor.MODELS[model_size]  # set model hyperparameters

		self.vocab_size = len(self.clusters) + 1  # add one for start of sentence token

		self.config = transformers.GPT2Config(
			vocab_size=self.vocab_size,
			n_ctx=self.n_px * self.n_px,
			n_positions=self.n_px * self.n_px,
			n_embd=n_embd,
			n_layer=n_layer,
			n_head=n_head
		)
		self.model_path = "%s/%s/model.ckpt-1000000.index" % (models_dir, model_size)

	def _extract_context(self, samples, gpu, **extract_kwargs) -> np.ndarray:
		raise NotImplementedError

	def load_model(self):
		assert os.path.exists(self.model_path), f"There is no file at {self.model_path}"
		self.model = ImageGPT2LMHeadModel.from_pretrained(
			self.model_path, from_tf=True, config=self.config
		)

	def process_samples(self, image_paths, visualize=False):
		for path in image_paths:
			assert os.path.exists(path), "ERR: %s is not a valid path." % path
		# print("Num paths: %s" % len(image_paths))
		x = resize(self.n_px, image_paths)
		# print("X shape: ", x.shape)
		x_norm = normalize_img(x)  # normalize pixels values to -1 to +1
		samples = color_quantize_np(x_norm, self.clusters).reshape(
			x_norm.shape[:-1])  # map pixels to closest color cluster

		if visualize:
			samples_img = [
				np.reshape(
					np.rint(127.5 * (self.clusters[s] + 1.0)), [self.n_px, self.n_px, 3]
				).astype(np.uint8) for s in samples
			]  # convert color clusters back to pixels
			self.visualize(samples_img, image_paths)
		# print("Shape of samples: ", samples.shape)
		return samples

	def _make_param_path(self):
		return "{}_{}".format(
			self.model_size,
			self.n_px
		)

	def model_output(self, samples, gpu):
		"""
		Model output from every layer for a given input image.
		Embeddings can be extracted and aggregated from different layers (see the child classes).

		Parameters
		----------
		samples : np.ndarray
		gpu : bool
			whether to use GPU (True) or CPU (False)

		Returns
		-------
		output : tuple(torch.FloatTensor)
			a Tensor of all hidden states
		"""
		context = np.concatenate(
			(
				np.full((samples.shape[0], 1), self.vocab_size - 1),
				samples.reshape(-1, self.n_px * self.n_px),
			), axis=1
		)

		# must drop the last pixel to make room for the SOS
		context = torch.tensor(context[:, :-1]) if not gpu else torch.tensor(context[:, :-1]).cuda()
		return self.model(context, output_hidden_states=True, return_dict=True)


class LogitExtractor(GPTExtractor):
	"""Extractor for iGPT logit (projection head) layer."""
	def _extract_context(self, samples, gpu, **extract_kwargs) -> np.ndarray:
		output = self.model_output(samples, gpu)
		# just use the logit layer
		# extract the rep of the last input, as in sent-bias
		enc_last = output.logits[:, -1, :]

		return enc_last.numpy() if not gpu else enc_last.cpu().numpy()


class SENTExtractor(GPTExtractor):
	"""Extractor for last position of the last layer output."""
	def _extract_context(self, samples, gpu, **extract_kwargs)  -> np.ndarray:
		"""
		SENT uses the last hidden layer output.

		For details, see https://github.com/tanyichern/social-biases-contextualized/blob/master/gpt2.py.
		"""
		# initialize with SOS token
		output = self.model_output(samples, gpu)

		enc_last = output.hidden_states[-1][:, -1, :] # extract the rep of the last input

		return enc_last.numpy() if not gpu else enc_last.cpu().numpy()


class OpenAIExtractor(GPTExtractor):
	"""
	Pooled extraction method, used by the iGPT authors for linear evaluation.
	1. find $n^l = layer\_norm(h^l)$
	2. average pool across the sequence dimension:
	$$ f^l = \langle n^l_i \rangle_i $$
	"""
	def _extract_context(self, samples, gpu, **extract_kwargs) -> np.ndarray:
		l = extract_kwargs.get("l", 20)

		output = self.model_output(samples, gpu)

		# extract the rep of the lth input
		h_l = output.hidden_states[l]
		norm = self.model.transformer.h[l+1].ln_1(h_l)
		enc = tf.reduce_mean(norm, axis=1)

		return enc.numpy() if not gpu else enc.cpu().numpy()


class ln_mod(nn.Module):
	"""
	Torch module for the iGPT modified linear head.
	From [apeguero1](https://colab.research.google.com/github/apeguero1/image-gpt/blob/master/Transformers_Image_GPT.ipynb).
	"""
	def __init__(self, nx, eps=1e-5):
		super().__init__()
		self.eps = eps
		self.weight = Parameter(torch.Tensor(nx))

	def forward(self, x):  # input is not mean centered
		return x \
			/ torch.sqrt(torch.std(x, axis=-1, unbiased=False, keepdim=True) ** 2 + self.eps) \
			* self.weight.data[..., :]


def load_tf_weights_in_image_gpt2(model, config, gpt2_checkpoint_path):
	"""
	Load tf checkpoints in a custom pytorch model.
	From [apeguero1](https://colab.research.google.com/github/apeguero1/image-gpt/blob/master/Transformers_Image_GPT.ipynb).
	"""
	try:
		import re
		import tensorflow as tf
	except ImportError:
		logger.error(
			"Loading a TensorFlow model in PyTorch, requires TensorFlow to be installed. Please see "
			"https://www.tensorflow.org/install/ for installation instructions."
		)
		raise
	tf_path = os.path.abspath(gpt2_checkpoint_path)
	logger.debug("Converting TensorFlow checkpoint from {}".format(tf_path))
	# Load weights from TF model
	init_vars = tf.train.list_variables(tf_path)
	names = []
	arrays = []

	for name, shape in init_vars:
		logger.debug("Loading TF weight {} with shape {}".format(name, shape))
		array = tf.train.load_variable(tf_path, name)
		names.append(name)
		arrays.append(array.squeeze())

	for name, array in zip(names, arrays):
		name = name[6:]  # skip "model/"
		name = name.split("/")

		# adam_v and adam_m are variables used in AdamWeightDecayOptimizer to calculated m and v
		# which are not required for using pretrained model
		if any(
				n in ["adam_v", "adam_m", "AdamWeightDecayOptimizer", "AdamWeightDecayOptimizer_1", "global_step"]
				for n in name
		) or name[-1] in ['_step']:
			logger.debug("Skipping {}".format("/".join(name)))
			continue

		pointer = model
		if name[-1] not in ["wtet"]:
			pointer = getattr(pointer, "transformer")

		for m_name in name:
			if re.fullmatch(r"[A-Za-z]+\d+", m_name):
				scope_names = re.split(r"(\d+)", m_name)
			else:
				scope_names = [m_name]

			if scope_names[0] == "w" or scope_names[0] == "g":
				pointer = getattr(pointer, "weight")
			elif scope_names[0] == "b":
				pointer = getattr(pointer, "bias")
			elif scope_names[0] == "wpe" or scope_names[0] == "wte":
				pointer = getattr(pointer, scope_names[0])
				pointer = getattr(pointer, "weight")
			elif scope_names[0] in ['q_proj', 'k_proj', 'v_proj']:
				pointer = getattr(pointer, 'c_attn')
				pointer = getattr(pointer, 'weight')
			elif len(name) == 3 and name[1] == "attn" and scope_names[0] == "c_proj":
				pointer = getattr(pointer, scope_names[0])
				pointer = getattr(pointer, 'weight')
			elif scope_names[0] == "wtet":
				pointer = getattr(pointer, "lm_head")
				pointer = getattr(pointer, 'weight')
			elif scope_names[0] == "sos":
				pointer = getattr(pointer, "wte")
				pointer = getattr(pointer, 'weight')
			else:
				pointer = getattr(pointer, scope_names[0])
			if len(scope_names) >= 2:
				num = int(scope_names[1])
				pointer = pointer[num]

		if len(name) > 1 and name[1] == "attn" or name[-1] == "wtet" or name[-1] == "sos" or name[-1] == "wte":
			pass  # array is used to initialize only part of the pointer so sizes won't match
		else:
			try:
				assert pointer.shape == array.shape
			except AssertionError as e:
				e.args += (pointer.shape, array.shape)
				raise

		logger.debug("Initialize PyTorch weight {}".format(name))

		if name[-1] == "q_proj":
			pointer.data[:, :config.n_embd] = torch.from_numpy(array.reshape(config.n_embd, config.n_embd)).T
		elif name[-1] == "k_proj":
			pointer.data[:, config.n_embd:2 * config.n_embd] = torch.from_numpy(
				array.reshape(config.n_embd, config.n_embd)).T
		elif name[-1] == "v_proj":
			pointer.data[:, 2 * config.n_embd:] = torch.from_numpy(array.reshape(config.n_embd, config.n_embd)).T
		elif len(name) == 3 and name[1] == "attn" and name[2] == "c_proj":
			pointer.data = torch.from_numpy(array.reshape(config.n_embd, config.n_embd))
		elif name[-1] == "wtet":
			pointer.data = torch.from_numpy(array)
		elif name[-1] == "wte":
			pointer.data[:config.vocab_size - 1, :] = torch.from_numpy(array)
		elif name[-1] == "sos":
			pointer.data[-1] = torch.from_numpy(array)
		else:
			pointer.data = torch.from_numpy(array)

	return model


def replace_ln(m, name, config):
	for attr_str in dir(m):
		target_attr = getattr(m, attr_str)
		if type(target_attr) == torch.nn.LayerNorm:
			setattr(m, attr_str, ln_mod(config.n_embd, config.layer_norm_epsilon))

	for n, ch in m.named_children():
		replace_ln(ch, n, config)


class ImageGPT2LMHeadModel(GPT2LMHeadModel):
	"""
	Extension of the HuggingFace `GPT2LMHeadModel` for iGPT.
	From [apeguero1](https://colab.research.google.com/github/apeguero1/image-gpt/blob/master/Transformers_Image_GPT.ipynb).
	"""
	load_tf_weights = load_tf_weights_in_image_gpt2

	def __init__(self, config):
		super().__init__(config)
		self.lm_head = nn.Linear(config.n_embd, config.vocab_size - 1, bias=False)
		replace_ln(self, "net", config)  # replace layer normalization
		for n in range(config.n_layer):
			self.transformer.h[n].mlp.act = ImageGPT2LMHeadModel.gelu2  # replace activation

	def tie_weights(self):  # image-gpt doesn't tie output and input embeddings
		pass

	@staticmethod
	def gelu2(x):
		return x * torch.sigmoid(1.702 * x)
