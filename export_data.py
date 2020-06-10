from lib.config import load_config
from lib.export import export

CONFIG = "./experiments/001.yaml"
BOX_DIR = "D:/Box"

cfg = load_config(CONFIG)
export(cfg, BOX_DIR)
