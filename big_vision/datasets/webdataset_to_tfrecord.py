import tensorflow as tf
import webdataset as wds
import io
from PIL import Image
import hashlib
import os
import json
from tqdm import tqdm
import multiprocessing as mp

COMPLETED_MARKER_SUBDIR = "_shard_completed_markers"

class WebPToTFRecordConverter:
    def __init__(self, shards, output_path, shard_size=2048, state_dir=None):
        self.shards = shards
        self.output_path = output_path
        self.shard_size = shard_size
        self.state_dir = state_dir or os.path.join(output_path, "state")
        self.marker_dir = os.path.join(self.state_dir, COMPLETED_MARKER_SUBDIR)
        tf.io.gfile.makedirs(self.output_path)
        tf.io.gfile.makedirs(self.state_dir)
        tf.io.gfile.makedirs(self.marker_dir)

    def load_state(self, state_file):
        if tf.io.gfile.exists(state_file):
            try:
                with tf.io.gfile.GFile(state_file, 'r') as f:
                    return set(json.load(f).get('seen_hashes', []))
            except json.JSONDecodeError:
                print(f"Corrupted state file {state_file}, ignoring.")
                return set()
        return set()

    def save_state(self, state_file, seen_hashes):
        with tf.io.gfile.GFile(state_file, 'w') as f:
            json.dump({"seen_hashes": list(seen_hashes)}, f)

    def webp_to_jpeg(self, pil_image):
        if not isinstance(pil_image, Image.Image): return None

        # Resize if max dimension > 2048, preserving aspect ratio
        max_dim = 2048
        width, height = pil_image.size
        if max(width, height) > max_dim:
            if width > height:
                new_width = max_dim
                new_height = int(height * (max_dim / width))
            else:
                new_height = max_dim
                new_width = int(width * (max_dim / height))
            print(f"Resizing image from {width}x{height} to {new_width}x{new_height}") # Optional: logging
            try:
                # Use Resampling.LANCZOS for newer Pillow versions, fall back to LANCZOS for older ones
                resampling_filter = Image.Resampling.LANCZOS if hasattr(Image, 'Resampling') else Image.LANCZOS
                pil_image = pil_image.resize((new_width, new_height), resampling_filter)
            except Exception as e:
                print(f"Error during resizing image from {width}x{height} to {new_width}x{new_height}: {e}")
                return None # Skip image if resize fails

        buffer = io.BytesIO()
        pil_image.convert('RGB').save(buffer, format='JPEG', quality=95)
        return buffer.getvalue()

    def create_example(self, image_bytes, caption_bytes):
        return tf.train.Example(features=tf.train.Features(feature={
            'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_bytes])),
            'caption': tf.train.Feature(bytes_list=tf.train.BytesList(value=[caption_bytes]))
        }))

    def process_shards(self, args):
        worker_id, shard_list, seen_hashes_global = args
        local_state_file = os.path.join(self.state_dir, f"worker_{worker_id}.json")
        seen_hashes = self.load_state(local_state_file)

        # Remove .decode("pil"), handle decoding manually inside the loop
        dataset = wds.WebDataset(shard_list, handler=wds.ignore_and_continue)

        writer, count, tfrecord_idx = None, 0, 0
        num_duplicates = 0
        print(f"Processing shard {shard_list} on worker {worker_id}")
        for sample in tqdm(dataset, desc=f"Worker {worker_id} processing"):
            try:
                # Explicitly check for required keys before accessing
                if "webp" not in sample or "txt" not in sample:
                    key = sample.get('__key__', 'N/A')
                    available_keys = list(sample.keys())
                    print(f"Worker {worker_id}: Skipping sample '{key}' due to missing 'webp' or 'txt' key. Available keys: {available_keys}")
                    continue

                webp_bytes = sample.get("webp")
                caption = sample.get("txt")

                # This check becomes slightly redundant but safe
                if not webp_bytes or not caption:
                    print(f"Worker {worker_id}: Skipping sample '{sample.get('__key__', 'N/A')}' due to missing webp or txt VALUE after key check.")
                    continue

                # Manually open image from bytes INSIDE the try block
                try:
                    img = Image.open(io.BytesIO(webp_bytes))
                except Exception as e:
                    # Catch other potential PIL opening errors
                    print(f"Worker {worker_id}: Skipping sample '{sample.get('__key__', 'N/A')}' due to error opening image bytes: {e}")
                    continue
                
                # Resize (if needed) and convert to JPEG
                jpeg_bytes = self.webp_to_jpeg(img)
                if jpeg_bytes is None:
                    # Handles cases where image is not valid PIL or resizing/conversion failed
                    print(f"Worker {worker_id}: Skipping sample '{sample.get('__key__', 'N/A')}' due to invalid image or conversion failure.")
                    continue

                img_hash = hashlib.sha256(jpeg_bytes).hexdigest()

                # Check against both local and global hashes
                if img_hash in seen_hashes:
                    print(f"Skipping duplicate on local hashes: {img_hash}")
                    num_duplicates += 1
                    continue
                
                if img_hash in seen_hashes_global:
                    print(f"Skipping duplicate on global seen hashes: {img_hash}")
                    num_duplicates += 1
                    continue

                # Rest of the logic to write TFRecord...
                if count % self.shard_size == 0:
                    if writer:
                        writer.close()
                    out_shard_path = os.path.join(self.output_path, f"worker{worker_id}_part{tfrecord_idx}.tfrecord")
                    print(f"Writing shard {tfrecord_idx} on worker {worker_id} to {out_shard_path}")
                    writer = tf.io.TFRecordWriter(out_shard_path)
                    tfrecord_idx += 1

                # Pass caption bytes directly, as create_example handles the feature creation
                writer.write(self.create_example(jpeg_bytes, caption).SerializeToString())
                seen_hashes.add(img_hash)
                seen_hashes_global[img_hash] = True  # Share globally
                count += 1

            except Exception as e: # General catch-all for other errors in the loop
                key = sample.get('__key__', 'UNKNOWN_KEY') if isinstance(sample, dict) else 'UNKNOWN_KEY'
                print(f"Worker {worker_id}: Skipping sample '{key}' due to unexpected error during processing: {e}")
                import traceback
                traceback.print_exc() # Print full traceback for unexpected errors
                continue

        if writer:
            writer.close()       

        

        self.save_state(local_state_file, seen_hashes)

        # Write marker files for completed shards
        for shard in shard_list:
            marker_path = os.path.join(self.marker_dir, os.path.basename(shard) + ".done")
            with tf.io.gfile.GFile(marker_path, 'w') as f:
                f.write("completed")

    def convert(self, num_workers=4):
        if not self.shards:
            print("No shards to process.")
            return

        shards_per_worker = [[] for _ in range(num_workers)]
        for idx, shard in enumerate(self.shards):
            shards_per_worker[idx % num_workers].append(shard)

        # Load existing global hashes if available
        global_hashes = {}
        global_state_file = os.path.join(self.state_dir, "global_seen_hashes.json")
        if tf.io.gfile.exists(global_state_file):
            try:
                with tf.io.gfile.GFile(global_state_file, 'r') as f:
                    loaded_hashes = json.load(f).get('seen_hashes', [])
                    global_hashes = {h: True for h in loaded_hashes}
                print(f"Loaded {len(global_hashes)} unique hashes from {global_state_file}")
            except (json.JSONDecodeError, tf.errors.NotFoundError) as e:
                print(f"Could not load global state file {global_state_file}: {e}. Starting fresh global hash set.")
                global_hashes = {} # Ensure it's initialized if loading fails

        with mp.get_context('spawn').Manager() as manager:
            seen_hashes_global = manager.dict(global_hashes) # Initialize with loaded hashes
            args_list = [(worker_id, shard_list, seen_hashes_global)
                         for worker_id, shard_list in enumerate(shards_per_worker)]

            with mp.get_context('spawn').Pool(num_workers) as pool:
                list(tqdm(pool.imap_unordered(self.process_shards, args_list), total=num_workers, desc="Processing Workers"))

            # Save the final global hash state (moved from worker 0 for robustness)
            final_hashes = list(seen_hashes_global.keys())
            print(f"Saving {len(final_hashes)} unique hashes to {global_state_file}")
            with tf.io.gfile.GFile(global_state_file, 'w') as f:
                 json.dump({"seen_hashes": final_hashes}, f)

        print("All workers completed.")

    def print_done_files(self):
        done_files = tf.io.gfile.listdir(self.marker_dir)
        print("Completed shards:")
        for file in done_files:
            print(file)

    def reset_markers(self):
        done_files = tf.io.gfile.listdir(self.marker_dir)
        for file in done_files:
            tf.io.gfile.remove(os.path.join(self.marker_dir, file))
        print("All marker files have been reset.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--base_path', required=True)
    parser.add_argument('--output_path', required=True)
    parser.add_argument('--num_workers', type=int, default=128)
    parser.add_argument('--shard_size', type=int, default=1024)
    parser.add_argument('--print_done', action='store_true', help='Print done files and exit')
    parser.add_argument('--reset', action='store_true', help='Reset completion markers and exit')
    args = parser.parse_args()

    converter = WebPToTFRecordConverter(
        shards=[],
        output_path=args.output_path,
        shard_size=args.shard_size
    )

    if args.print_done:
        converter.print_done_files()
        exit(0)

    if args.reset:
        converter.reset_markers()
        exit(0)

    marker_dir = converter.marker_dir
    tf.io.gfile.makedirs(marker_dir)

    all_shards = tf.io.gfile.glob(args.base_path + "/*/*.tar")

    # Filter out shards that have already been processed based on marker files
    '''
    try:
        done_markers = tf.io.gfile.listdir(marker_dir)
        # Extract base shard names from marker filenames (e.g., 'shard-00000.tar.done' -> 'shard-00000.tar')
        completed_shard_bases = {marker.replace(".done", "") for marker in done_markers if marker.endswith(".done")}
        print(f"Found {len(completed_shard_bases)} completed shard markers.")
    except tf.errors.NotFoundError:
        print(f"Marker directory {marker_dir} not found or empty. Assuming no shards are completed.")
        completed_shard_bases = set()

    unprocessed_shards = [
        shard for shard in all_shards
        if os.path.basename(shard) not in completed_shard_bases
    ]
    '''
        
    converter.shards = all_shards # Use the filtered list
    converter.convert(num_workers=args.num_workers)

    print("Conversion process complete.")