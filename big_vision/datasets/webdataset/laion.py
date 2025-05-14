import os
import math
import tensorflow as tf
import numpy as np
import jax
import re
import io
import random
from PIL import Image


def get_highest_part_number(file_paths):
    """Extract the highest part number from a list of tfrecord file paths.
    
    Args:
        file_paths: List of paths with format like 'worker{N}_part{M}.tfrecord'
        
    Returns:
        The highest part number found in the list and its corresponding path.
    """
    pattern = r'worker\d+_part(\d+)\.tfrecord'
    highest_part = -1
    highest_path = None
    
    for path in file_paths:
        match = re.search(pattern, path)
        if match:
            part_num = int(match.group(1))
            if part_num > highest_part:
                highest_part = part_num
                highest_path = path
    
    return highest_part, highest_path


def filter_out_highest_part_per_worker(file_paths):
    """Filter the list of files to exclude the highest part number for each worker.
    
    Args:
        file_paths: List of tfrecord file paths
        
    Returns:
        Filtered list with all files EXCEPT the highest part per worker
    """
    # Group files by worker
    worker_files = {}
    worker_pattern = r'worker(\d+)_part\d+\.tfrecord'
    
    for path in file_paths:
        match = re.search(worker_pattern, path)
        if match:
            worker_id = match.group(1)
            if worker_id not in worker_files:
                worker_files[worker_id] = []
            worker_files[worker_id].append(path)
    
    # For each worker, get all files except the one with the highest part number
    filtered_files = []
    for worker_id, paths in worker_files.items():
        _, highest_path = get_highest_part_number(paths)
        for path in paths:
            if path != highest_path:
                filtered_files.append(path)
            
    return filtered_files


class DataSource:
    """DataSource implementation for TFRecord data."""

    def __init__(self, 
                 tfrecord_paths=["gs://vidtok-data/tfrecords_laion400m"],
                 feature_description=None,
                 mem_buffer_size=1024 * 1024 * 4,
                 preshuffle=False,
                 filter_highest_part=False):
        """Initialize the TFRecord DataSource.
        
        Args:
            tfrecord_path: Directory containing TFRecord files or glob pattern
            feature_description: Dict describing the features (if None, uses default)
            mem_buffer_size: Size of the memory buffer for reading
            filter_highest_part: If True, exclude the files with the highest part number for each worker
        """
        self.tfrecord_paths = tfrecord_paths
        self.mem_buffer_size = mem_buffer_size      
        self.preshuffle = preshuffle
        # Default feature description for image-caption pairs
        self.feature_description = feature_description or {
            'image': tf.io.FixedLenFeature([], tf.string),
            'caption': tf.io.FixedLenFeature([], tf.string),
        }       
        self.tfrecord_files = []
        for path in self.tfrecord_paths:
            if filter_highest_part:
                files = filter_out_highest_part_per_worker(tf.io.gfile.glob(f"{path}/*.tfrecord"))
            else:
                files = tf.io.gfile.glob(f"{path}/*.tfrecord")
            self.tfrecord_files.extend(files)
        print(f"Found {len(self.tfrecord_files)} tfrecord files")
            
    def _parse_example(self, example_proto):
        """Parse a single TFRecord example."""
        # Parse the input tf.Example proto
        parsed_features = tf.io.parse_single_example(example_proto, self.feature_description)
        image = parsed_features['image']
        caption = parsed_features['caption']
        return {'image': image, 'labels': caption}
    
    def get_tfdata(self, ordered, *, process_split=True, allow_cache=True):
        """Creates this data object as a tf.data.Dataset.
        
        Args:
            ordered: If True, dataset has deterministic ordering (val set)
                    If False, dataset may have undefined ordering (train set)
            process_split: If False then every process receives the entire dataset
            allow_cache: Whether to allow caching the opened data
            
        Returns:
            A tf.data.Dataset object.
        """
        # Get JAX process info
        process_index = jax.process_index()
        num_processes = jax.process_count()
        files = self.tfrecord_files[process_index::num_processes]
        if self.preshuffle:
            #shuffle files
            random.shuffle(files)
        
        print(f"Process {process_index} of {num_processes} will read {len(files)} files")
        
        # Create TFRecord dataset
        dataset = tf.data.TFRecordDataset(
            files,
            num_parallel_reads=tf.data.AUTOTUNE,
            buffer_size=self.mem_buffer_size,
        )
        
        # Parse examples
        dataset = dataset.map(
            self._parse_example,
            num_parallel_calls=tf.data.AUTOTUNE,
        )    
        return dataset
    
    @property
    def total_examples(self):
        """Returns number of examples in the dataset, regardless of sharding."""
        return 10000000
    
    def num_examples_per_process(self):
        """Returns a list of the number of examples for each process."""
        return 10000000 // jax.process_count()