FROM tensorflow/tensorflow:latest-gpu-py3

COPY requirements.txt .
RUN pip install -r requirements.txt
COPY server.py .
COPY ./data/inference_data ./data/inference_data
ENV FLASK_APP ./server.py
ENV FLASK_DEBUG 1

CMD ["python", "server.py", \
     "--model_path", "./data/inference_data/model-weights-36-0.62.hdf5", \
     "--classes_indices", "./data/inference_data/model-classes.json"]