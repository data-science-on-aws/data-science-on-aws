FROM rayproject/ray-ml:latest
## Customization
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir transformers==4.19.1
RUN pip list

ENV PYTHONUNBUFFERED=TRUE
ENV PYTHONDONTWRITEBYTECODE=TRUE

ENV PATH="/opt/ml/code:${PATH}"

COPY ray /opt/ml/code
WORKDIR /opt/ml/code
