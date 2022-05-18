#FROM rayproject/ray:1.13.0rc0-py37-cpu
FROM rayproject/ray:nightly-cpu

RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir transformers==4.19.1 torch==1.11.0

ENV PYTHONUNBUFFERED=TRUE
ENV PYTHONDONTWRITEBYTECODE=TRUE

ENV PATH="/opt/ml/code:${PATH}"

COPY transformer/ /opt/ml/code/transformer/

WORKDIR /opt/ml/code
RUN sudo mkdir -p /opt/ml/ray/ && sudo chmod -R a+w /opt/ml/ray/

#RUN ray start --head --temp-dir /opt/ml/ray --min-worker-port 20000 --max-worker-port 25000 --head --port=9999 --disable-usage-stats --include-dashboard false

COPY serve-huggingface.py /opt/ml/code/
ENTRYPOINT ["python", "serve-huggingface.py"]
EXPOSE 8080
