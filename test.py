import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import pandas as pd
import os
from PIL import Image
from torch.nn import functional as F
from transformers import AutoModelForMultimodalLM , AutoProcessor
from tqdm.auto import tqdm