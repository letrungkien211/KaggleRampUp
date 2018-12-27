FROM tensorflow/tensorflow:latest-gpu-py3

COPY requirements.txt .
RUN pip install -r requirements.txt
COPY server.py .
COPY ./data/inference_data ./data/inference_data
ENV FLASK_APP ./server.py
ENV FLASK_DEBUG 1

CMD [   "python", "server.py", \
        "--model", "./data/inference_data/vgg-fruit-360-small-model-weights-77-0.93.hdf5", \
        "--classes_indices", "./data/inference_data/vgg-fruit-360-small-model-classes.json"]