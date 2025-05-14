import os
import random

import numpy as np
import torch
from decord import VideoReader, cpu
import pandas as pd
import numpy as np

def get_video_loader(target_width=224, target_height=224):
    def _loader(video_path):
        vr = VideoReader(video_path, ctx=cpu(0))
        original_width, original_height = vr[0].shape[1], vr[0].shape[0]

        # Calculate the aspect ratio
        aspect_ratio = original_width / original_height
        
        # Determine new dimensions while maintaining the aspect ratio
        if original_width >= original_height:  # Wider than tall
            new_height = target_height
            new_width = int(target_height * aspect_ratio)
        else:
            new_width = target_width
            new_height = int(target_width / aspect_ratio)
        
        # Resize the video to fit within the target dimensions
        vr_resized = VideoReader(video_path, width=new_width, height=new_height, ctx=cpu(0))
        return vr_resized
    return _loader

class VideoDataset(torch.utils.data.Dataset):
    """Load Video Dataset with mp4, avi, or specific CSV path.
    Parameters
    ----------
    data_dir : str, required.
        Path to the data_dir folder storing the dataset.
    reference_csv : str, required.
        A text file describing the dataset, each line per video sample.
        There are four items in each line:
        (1) video path; (2) start_idx, (3) total frames and (4) video label.
        for pre-train video data
            total frames < 0, start_idx and video label meaningless
        for pre-train rawframe data
            video label meaningless
    train : bool, default True.
        Whether to load the training or validation set.
    test_mode : bool, default False.
        Whether to perform evaluation on the test set.
        Usually there is three-crop or ten-crop evaluation strategy involved.
    name_pattern : str, default 'img_{:05}.jpg'.
        The naming pattern of the decoded video frames.
        For example, img_00012.jpg.
    modality : str, default 'rgb'.
        Input modalities, we support only rgb video frames for now.
        Will add support for rgb difference image and optical flow image later.
    num_segments : int, default 1.
        Number of segments to evenly divide the video into clips.
        A useful technique to obtain global video-level information.
        Limin Wang, etal, Temporal Segment Networks: Towards Good Practices for Deep Action Recognition, ECCV 2016.
    new_length : int, default 1.
        The length of input video clip. Default is a single image, but it can be multiple video frames.
        For example, new_length=16 means we will extract a video clip of consecutive 16 frames.
    new_step : int, default 1.
        Temporal sampling rate. For example, new_step=1 means we will extract a video clip of consecutive frames.
        new_step=2 means we will extract a video clip of every other frame.
    lazy_init : bool, default False.
        If set to True, build a dataset instance without loading any dataset.
    """

    def __init__(self,
                 data_dir,
                 reference_csv='',
                 new_length=16, #Clip Length
                 new_step=2, #Sampling Rate
                 img_size=256,
                 lazy_init=False,
                 data_format='shutterstock',
                 train=True):

        super(VideoDataset, self).__init__()
        self.data_dir = data_dir
        if reference_csv:
            self.reference_csv = pd.read_csv(reference_csv)
        else:
            self.reference_csv = None
        self.new_length = new_length
        self.new_step = new_step
        self.skip_length = self.new_length * self.new_step
        self.img_size = img_size
        self.lazy_init = lazy_init
        self.orig_new_step = new_step
        self.orig_skip_length = self.skip_length
        self.data_format = data_format
        self.train = train
        self.video_loader = get_video_loader(target_width=img_size, target_height=img_size) #Holds aspect ratio constant then does random crop

        if not self.lazy_init:
            self.clips, self.labels = self._make_dataset()
            if len(self.clips) == 0:
                raise (
                    RuntimeError("Found 0 video clips in subfolders of: " +
                                 data_dir + "\n"
                                 "Check your data directory (opt.data-dir)."))
            else:
                print(f"Found {len(self.clips)} video clips in {data_dir}")

    def __getitem__(self, index):
        video_name = self.clips[index]
        label = self.labels[index]
        self.skip_length = self.orig_skip_length
        self.new_step = self.orig_new_step
        
        try:
            decord_vr = self.video_loader(video_name)
            duration = len(decord_vr)
            
            segment_indices = self._sample_train_indices(duration)
            frame_id_list = self.get_frame_id_list(duration, segment_indices)
            
            if duration < self.new_length:
                raise ValueError(f"Video {video_name} is too short for the desired clip length of {self.new_length}.")
        except:
            print(f"Error loading video {video_name}")
            return self.__getitem__(random.randint(0, len(self.clips) - 1))
        
        videos = decord_vr.get_batch(frame_id_list).asnumpy()
        videos = torch.from_numpy(videos)
        if self.train:
            max_y = videos.shape[1] - self.img_size
            max_x = videos.shape[2] - self.img_size
            start_y = random.randint(0, max_y)
            start_x = random.randint(0, max_x)
            videos = videos[:, start_y:start_y+self.img_size, start_x:start_x+self.img_size, :]
            flip = random.random() > 0.5
            if flip:
                videos = torch.flip(videos, [2])
        else:
            # Center crop
            center_y = videos.shape[1] // 2
            center_x = videos.shape[2] // 2
            half_size = self.img_size // 2
            videos = videos[:, center_y-half_size:center_y+half_size, center_x-half_size:center_x+half_size, :]
        
        videos = videos.permute(3, 0, 1, 2).contiguous()  # Change dimension order to (C, T, H, W)
        videos = videos.div(255.).mul(2.).sub(1.)  # Normalize to [-1, 1]
        return videos, label

    def __len__(self):
        return len(self.clips)

    def _make_dataset(self):

        if self.data_format == 'kinetics':
            clips = [
            f"{self.data_dir}/{row['label']}/{row['youtube_id']}_{row['time_start']:06d}_{row['time_end']:06d}.mp4"
            for idx, row in self.reference_csv.iterrows()
            ]
            labels = [row['integer_label'] for idx, row in self.reference_csv.iterrows()]
        elif self.data_format == 'ucf101':
            clips = []
            labels = []
            label_directory = os.listdir(self.data_dir)
            label_directory.sort() #Sort to ensure consistent label order
            for idx, label in enumerate(label_directory):
                labels.extend([idx] * len(os.listdir(os.path.join(self.data_dir, label))))
                clips.extend([os.path.join(self.data_dir, label, video) for video in os.listdir(os.path.join(self.data_dir, label))])
        elif self.data_format == 'shutterstock':
            clips = []
            labels = []
            for path in self.reference_csv['path']:
                clips.append(f"{self.data_dir}/{path}")
                labels.append(0) #Need to add text labels to shutterstock CSV
        else:
            raise ValueError(f"Data format {self.data_format} not supported.")


        return clips, labels

    def _sample_train_indices(self, num_frames):
        average_duration = (num_frames - self.skip_length +
                            1)
        if average_duration > 0:
            offsets = np.multiply(
                list(range(1)), average_duration)
            offsets = offsets + np.random.randint(
                average_duration, size=1)
        elif num_frames > max(1, self.skip_length):
            offsets = np.sort(
                np.random.randint(
                    num_frames - self.skip_length + 1, size=1))
        else:
            offsets = np.zeros((1, ))
        return offsets + 1

    def get_frame_id_list(self, duration, indices):
        frame_id_list = []
        for seg_ind in indices:
            offset = int(seg_ind)
            for i, _ in enumerate(range(0, self.skip_length, self.new_step)):
                frame_id = offset - 1
                frame_id_list.append(frame_id)
                if offset + self.new_step < duration:
                    offset += self.new_step
        return frame_id_list