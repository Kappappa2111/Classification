import os
import zipfile
import urllib.request
import matplotlib.pyplot as plt
import random

#torch
import torch
import glob
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from torch.nn import functional as F
import numpy as np
from tqdm import tqdm
from torchvision import models, transforms

from PIL import Image