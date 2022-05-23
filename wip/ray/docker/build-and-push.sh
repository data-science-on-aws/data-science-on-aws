docker build -t dsoaws/ray-fluxcapacitor:1.0 .

docker tag dsoaws/ray-fluxcapacitor:1.0 079002598131.dkr.ecr.us-east-1.amazonaws.com/dsoaws/ray-fluxcapacitor:1.0

docker tag dsoaws/ray-fluxcapacitor:1.0 datascienceonaws/ray-fluxcapacitor:1.0

docker push datascienceonaws/ray-fluxcapacitor:1.0

#aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin 079002598131.dkr.ecr.us-east-1.amazonaws.com

#docker push  079002598131.dkr.ecr.us-east-1.amazonaws.com/dsoaws/ray-fluxcapacitor:1.0
