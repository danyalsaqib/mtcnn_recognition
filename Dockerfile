FROM python

RUN apt-get update && \
    python -m pip install --upgrade pip && \
    pip install opencv-python && \
    apt-get install ffmpeg libsm6 libxext6  -y && \
    python -m pip install -U scikit-image && \
    pip install numpy protobuf==3.16.0 && \
    pip install onnx && \
    pip install onnxruntime && \
    pip install mtcnn-onnxruntime

COPY face_recognition /face_recognition
