import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple, Union

import draccus
import torch
import torch.distributed as dist
import yaml

from prismatic.conf import VLAConfig, VLARegistry
from prismatic.models import load, load_vla, load_vla_custom,load_vla_custom_alpha
from prismatic.overwatch import initialize_overwatch
from prismatic.training import VLAMetrics, get_train_strategy
from prismatic.util import set_global_seed
from prismatic.vla import get_vla_dataset_and_collator
from prismatic.vla.datasets.rlds.utils.data_utils import save_dataset_statistics
import pdb
# Sane Defaults
os.environ["TOKENIZERS_PARALLELISM"] = "false"