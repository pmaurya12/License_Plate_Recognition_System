from flask import Flask, render_template, request, send_from_directory
import os
import subprocess
from werkzeug.utils import secure_filename
from main2 import process_video
from flask import Response
from main2 import generate_live_frames

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
ALLOWED_EXTENSIONS = {"mp4", "avi", "mov"}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


# ✅ FFmpeg conversion (guaranteed browser playback)
def convert_to_browser_format(input_path):
    base = os.path.splitext(input_path)[0]
    output_path = f"{base}_final.mp4"

    command = [
        "ffmpeg", "-y",
        "-i", input_path,
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "23",
        "-c:a", "aac",
        "-b:a", "128k",
        "-movflags", "+faststart",  # 🔥 important for browser streaming
        output_path
    ]

    subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    return output_path


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files.get("video")

        if not file or file.filename == "":
            return "No file selected"

        if allowed_file(file.filename):
            filename = secure_filename(file.filename)
            input_path = os.path.join(UPLOAD_FOLDER, filename)

            processed_path = os.path.join(UPLOAD_FOLDER, "processed_" + filename)

            file.save(input_path)

            # YOLO + OCR processing
            process_video(input_path, processed_path)

            # Convert to browser-friendly format
            final_path = convert_to_browser_format(processed_path)
            final_filename = os.path.basename(final_path)

            return render_template("index.html", output_video=final_filename)

        return "Invalid file type"

    return render_template("index.html", output_video=None)


@app.route("/uploads/<filename>")
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

@app.route("/video_feed")
def video_feed():
    return Response(generate_live_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == "__main__":
    app.run(debug=True)