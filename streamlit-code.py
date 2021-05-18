import streamlit as st
from streamlit_player import st_player

import datetime
import time

from pytube import YouTube
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip

import datetime
import time

from absl import logging

import tensorflow as tf
import tensorflow_hub as hub
#from tensorflow_docs.vis import embed

logging.set_verbosity(logging.ERROR)

# Some modules to help with reading the UCF101 dataset.
import random
import re
import os
import tempfile
import ssl
import cv2
import numpy as np

# Some modules to display an animation using imageio.
import imageio
from IPython import display

from urllib import request 
import urllib.request as url
import http

def time_change():
    x = time.strptime(start_t,'%M:%S')
    y  = datetime.timedelta(minutes=x.tm_min,seconds=x.tm_sec).total_seconds()
    y = int(round(y))


# Utilities to fetch videos from UCF101 dataset
UCF_ROOT = "https://www.crcv.ucf.edu/THUMOS14/UCF101/UCF101/"
_VIDEO_LIST = None
_CACHE_DIR = tempfile.mkdtemp()
# As of July 2020, crcv.ucf.edu doesn't use a certificate accepted by the
# default Colab environment anymore.
unverified_context = ssl._create_unverified_context()

def list_ucf_videos():
  #Lists videos available in UCF101 dataset.
  global _VIDEO_LIST
  if not _VIDEO_LIST:
    index = request.urlopen(UCF_ROOT, context=unverified_context).read().decode("utf-8")
    videos = re.findall("(v_[\w_]+\.avi)", index)
    _VIDEO_LIST = sorted(set(videos))
  return list(_VIDEO_LIST)

def fetch_ucf_video(video):
  #Fetchs a video and cache into local filesystem.
  cache_path = os.path.join(_CACHE_DIR, video)
  if not os.path.exists(cache_path):
    urlpath = request.urljoin(UCF_ROOT, video)
    print("Fetching %s => %s" % (urlpath, cache_path))
    data = request.urlopen(urlpath, context=unverified_context).read()
    open(cache_path, "wb").write(data)
  return cache_path

# Utilities to open video files using CV2
def crop_center_square(frame):
  y, x = frame.shape[0:2]
  min_dim = min(y, x)
  start_x = (x // 2) - (min_dim // 2)
  start_y = (y // 2) - (min_dim // 2)
  return frame[start_y:start_y+min_dim,start_x:start_x+min_dim]

def load_video(path, max_frames=0, resize=(224, 224)):
  cap = cv2.VideoCapture(path)
  frames = []
  try:
    while True:
      ret, frame = cap.read()
      if not ret:
        break
      frame = crop_center_square(frame)
      frame = cv2.resize(frame, resize)
      frame = frame[:, :, [2, 1, 0]]
      frames.append(frame)

      if len(frames) == max_frames:
        break
  finally:
    cap.release()
  return np.array(frames) / 255.0

def to_gif(images):
  converted_images = np.clip(images * 255, 0, 255).astype(np.uint8)
  imageio.mimsave('./animation.gif', converted_images, fps=25)
  return embed.embed_file('./animation.gif')


# Get the kinetics-400 action labels from the GitHub repository.
KINETICS_URL = "https://raw.githubusercontent.com/deepmind/kinetics-i3d/master/data/label_map.txt"
with request.urlopen(KINETICS_URL) as obj:
  labels = [line.decode("utf-8").strip() for line in obj.readlines()]
#print("Found %d labels." % len(labels))


i3d = hub.load("https://tfhub.dev/deepmind/i3d-kinetics-400/1").signatures['default']

def predict(sample_video):
  # Add a batch axis to the to the sample video.
  model_input = tf.constant(sample_video, dtype=tf.float32)[tf.newaxis, ...]

  logits = i3d(model_input)['default'][0]
  probabilities = tf.nn.softmax(logits)
  ''': {probabilities[i] * 100:5.2f}%'''
  #print("Top 5 actions:")
  for i in np.argsort(probabilities)[::-1][:1]:
    return print(f"{labels[i]:22}")

def time_change(first):
    y = first//1
    num = 1
    z = (first*100)% (y*100)
    num = num*y
    return (num + int(z))



def video_processor(youtube_link, Start_time, End_time):
    #link = input(youtube_link)
    yt = YouTube(youtube_link)
    
    Start_time = time_change(Start_time)
    End_time = time_change(End_time)

    #Getting the highest resolution possible
    ys = yt.streams.get_highest_resolution()
    #st.write('dowloading...hold your breath for a while')
    ys.download()
    #st.write('Download completed!!')
    test_video = ys.download()
    
    return ffmpeg_extract_subclip(test_video, Start_time, End_time, targetname="test2.mp4")



def main():
    st.set_page_config(page_title = "Youtube Video Classification")

    st.header("Youtube Video - Activity Classification")
    
    #st.text("Please provide the youtube link for Classification")

    youtube_link = st.text_input("Please input the link")

    #st.success(youtube_link)
    #End_time = st.time_input("end for fun", datetime.datetime)
    Start_time = st.number_input("Input the start time of video in 00 mins . 00 secs (i.e., 2.30, 11.20 etc.,)")
    
    End_time = st.number_input("Input the start time of video in 00 mins: 00 secs")
    
    if st.button("Predict"):
        st.success(End_time)
        final_video = video_processor(youtube_link, Start_time, End_time)
        result = predict(final_video)

if __name__ == '__main__':
    main()
