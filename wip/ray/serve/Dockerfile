FROM rayproject/ray:1.13.0rc0-py37-cpu

RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir transformers==4.19.1 torch==1.11.0

ENV PYTHONUNBUFFERED=TRUE
ENV PYTHONDONTWRITEBYTECODE=TRUE

ENV PATH="/opt/ml/code:${PATH}"

COPY serve-huggingface.py /opt/ml/code/
COPY transformer/ /opt/ml/code/transformer/

WORKDIR /opt/ml/code
ENTRYPOINT ["python", "serve-huggingface.py"]
EXPOSE 8080


