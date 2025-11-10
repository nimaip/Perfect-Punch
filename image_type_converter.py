import os
from moviepy import VideoFileClip

def convert_all_mov_to_mp4(base_dir="training-videos"):
    """
    Convert all .mov files in base_dir to .mp4 using MoviePy only.
    Only works for .mov files containing standard video streams.
    """
    if not os.path.exists(base_dir):
        print(f"❌ Folder not found: {base_dir}")
        return

    print(f"🚀 Starting .mov → .mp4 conversion in '{base_dir}'...\n")

    for root, _, files in os.walk(base_dir):
        for file in files:
            if not file.lower().endswith(".mov"):
                continue

            input_path = os.path.join(root, file)
            output_path = os.path.splitext(input_path)[0] + ".mp4"

            if os.path.exists(output_path):
                print(f"⏭️ Skipping (already exists): {output_path}")
                continue

            try:
                print(f"🎞️ Converting: {input_path}")
                clip = VideoFileClip(input_path)

                # Ensure clip is readable
                if clip.duration is None or clip.size is None:
                    print(f"❌ Cannot read video stream: {input_path}")
                    clip.close()
                    continue

                clip.write_videofile(
                    output_path,
                    codec="libx264",
                    audio_codec="aac",
                    fps=clip.fps or 30
                )
                clip.close()
                print(f"✅ Done: {output_path}\n")

            except Exception as e:
                print(f"❌ Error converting {input_path}: {e}\n")

    print("🎉 All conversions completed!\n")


# Example usage
convert_all_mov_to_mp4("training-videos")
