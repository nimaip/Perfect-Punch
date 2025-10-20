from moviepy import VideoFileClip
import os

def convert_file(input_path):
    try:
        video = VideoFileClip(input_path)
        output_path = input_path[:len(input_path) - 4] + ".mp4"
        video.write_videofile(output_path, codec="libx264", audio_codec="aac")

    except Exception as e:
        print(e)

for folder in os.listdir("training-videos"):
    for video in os.listdir("training-videos/"+folder):
        convert_file("training-videos/"+folder+"/"+video)
