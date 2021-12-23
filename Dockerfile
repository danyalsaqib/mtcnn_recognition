FROM ubuntu

RUN apt-get update -y && \
    apt-get install -y python3.6 python3-distutils python3-pip python3-apt && \
    python3 -m pip install --upgrade pip && \
    pip install gdown && \
    pip install opencv-python && \
    apt-get install ffmpeg libsm6 libxext6  -y && \
    python3 -m pip install -U scikit-image && \
    pip install pandas && \
    pip install cmake && \
    pip install six && \
    pip install typing-extensions==3.6.2.1 && \
    pip install numpy protobuf==3.16.0 && \
    pip install onnx && \
    pip install onnxruntime-gpu==1.9.0 && \
    pip install mtcnn-onnxruntime

COPY face_recognition /face_recognition
