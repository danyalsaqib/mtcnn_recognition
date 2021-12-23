FROM ubuntu

RUN apt-get update && \
    apt install build-essential zlib1g-dev libncurses5-dev libgdbm-dev libnss3-dev libssl-dev libreadline-dev libffi-dev wget && \
    apt install python3.7 && \
    python -m pip install --upgrade pip && \
    pip install gdown && \
    pip install opencv-python && \
    apt-get install ffmpeg libsm6 libxext6  -y && \
    python -m pip install -U scikit-image && \
    pip install pandas && \
    pip install cmake && \
    pip install six && \
    pip install typing-extensions==3.6.2.1 && \
    pip install numpy protobuf==3.16.0 && \
    pip install onnx && \
    pip install onnxruntime && \
    pip install mtcnn-onnxruntime

COPY face_recognition /face_recognition
