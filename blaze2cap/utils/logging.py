import logging
import os
import sys
import torch.distributed as dist

# Try importing colorlog for pretty console output, fallback if missing
try:
    import colorlog
except ImportError:
    colorlog = None

def setup_logging(output_dir, log_file="train.log", console_level="INFO"):
    """
    Sets up the logging for Blaze2Cap.
    
    Args:
        output_dir (str): Directory to save log files.
        log_file (str): Base name for the log file (e.g., "train.log").
        console_level (str): Level for console output ('DEBUG', 'INFO', 'WARNING', etc.).
    """
    # 1. Prepare Paths
    os.makedirs(output_dir, exist_ok=True)
    
    # We create two files: one for ALL logs (debug) and one for cleaner info (info)
    name_base, ext = os.path.splitext(log_file)
    log_path_debug = os.path.join(output_dir, f"{name_base}_debug{ext}")
    log_path_info = os.path.join(output_dir, f"{name_base}_info{ext}")

    # 2. Get the root logger
    # We reset handlers to avoid duplicate logs if setup_logging is called twice
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG) # Capture everything at root level
    logger.handlers = [] 
    logger.propagate = False

    # 3. Define Formatters
    # Check if we are in Distributed Data Parallel (DDP) mode to add Rank ID
    rank = 0
    if dist.is_available() and dist.is_initialized():
        rank = dist.get_rank()
    
    rank_msg = f"[Rank {rank}]" if rank > 0 else ""

    # Standard Format for files
    file_fmt = f"[%(asctime)s][%(levelname)s]{rank_msg} %(filename)s:%(lineno)d: %(message)s"
    file_formatter = logging.Formatter(file_fmt, datefmt="%m/%d %H:%M:%S")

    # Colorful Format for console
    if colorlog:
        console_fmt = (
            f"%(log_color)s%(bold)s[%(levelname)s]%(reset)s "
            f"%(log_color)s[%(asctime)s]{rank_msg} "
            f"[%(filename)s:%(lineno)d]:%(reset)s "
            f"%(message)s"
        )
        console_formatter = colorlog.ColoredFormatter(console_fmt, datefmt="%m/%d %H:%M:%S")
    else:
        console_formatter = file_formatter

    # 4. Add Handlers

    # A. Console Handler
    c_handler = logging.StreamHandler(sys.stdout)
    c_handler.setLevel(getattr(logging, console_level.upper()))
    c_handler.setFormatter(console_formatter)
    logger.addHandler(c_handler)

    # Only main process (Rank 0) should write to log files to prevent conflicts
    if rank == 0:
        # B. File Handler (Info - Clean)
        f_handler_info = logging.FileHandler(log_path_info, mode='a')
        f_handler_info.setLevel(logging.INFO)
        f_handler_info.setFormatter(file_formatter)
        logger.addHandler(f_handler_info)

        # C. File Handler (Debug - Everything)
        f_handler_debug = logging.FileHandler(log_path_debug, mode='a')
        f_handler_debug.setLevel(logging.DEBUG)
        f_handler_debug.setFormatter(file_formatter)
        logger.addHandler(f_handler_debug)
        
        # Log initialization message
        logging.info(f"Logging configured. Debug logs: {log_path_debug}")

    return logger