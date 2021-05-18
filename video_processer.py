from pytube import YouTube
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip

#link = input("https://www.youtube.com/watch?v=668nUCeBHyY")
def video_processor(youtube-link, t1, t2):
    link = input(youtube-link)
    yt = YouTube(link)
    
    #Getting the highest resolution possible
    ys = yt.streams.get_highest_resolution()
    print("dowloading...hold your breath for a while")
    ys.download()
    print("Download completed!!")
    test_video = ys.download()
    
    
    ffmpeg_extract_subclip(test_video, t1, t2, targetname="test.mp4")