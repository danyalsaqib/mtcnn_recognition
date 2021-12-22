import subprocess
import os

# Downloading pretrained models
if os.path.isfile('arcface.onnx'):
  print("File 'arcface.onnx' already exists")
else:
  subprocess.run(['wget', '-O arcface.onnx "https://drive.google.com/file/d/16aJ_uiDWeggv0V7i9VBSzrdIOUhK84Qr/view?usp=sharing"'])
  #! wget -O arcface.onnx "https://drive.google.com/file/d/16aJ_uiDWeggv0V7i9VBSzrdIOUhK84Qr/view?usp=sharing"
