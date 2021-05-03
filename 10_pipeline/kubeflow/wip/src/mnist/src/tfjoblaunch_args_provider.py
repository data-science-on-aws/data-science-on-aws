def tfjoblauncher_args(step, s3bucketexportpath, args, aws_region):
    chief = {
        "replicas": 1,
        "restartPolicy": "OnFailure",
        "template": {
            "metadata": {
                "annotations": {
                    "sidecar.istio.io/inject": "false"
                }
            },
            "spec": {
                "containers": [
                    {
                        "command": [
                            "sh",
                            "-c"
                        ],
                        "args": [
                            "python /opt/model.py --tf-train-steps={} --tf-export-dir={} {}".format(step, s3bucketexportpath, args)
                        ],
                        "image": "chuckshow/mnist-tf-pipeline:latest",
                        "name": "tensorflow",
                        "env": [
                            {
                                "name": "AWS_REGION",
                                "value": aws_region
                            },
                            {
                                "name": "AWS_ACCESS_KEY_ID",
                                "valueFrom": {
                                    "secretKeyRef": {
                                        "name": "aws-secret",
                                        "key": "AWS_ACCESS_KEY_ID"
                                    }
                                }
                            },
                            {
                                "name": "AWS_SECRET_ACCESS_KEY",
                                "valueFrom": {
                                    "secretKeyRef": {
                                        "name": "aws-secret",
                                        "key": "AWS_SECRET_ACCESS_KEY"
                                    }
                                }
                            }
                        ]
                    }
                ]
            }
        }
    }

    worker = {
        "replicas": 3,
        "restartPolicy": "OnFailure",
        "template": {
            "metadata": {
                "annotations": {
                    "sidecar.istio.io/inject": "false"
                }
            },
            "spec": {
                "containers": [
                    {
                        "command": [
                            "sh",
                            "-c"
                        ],
                        "args": [
                            "python /opt/model.py --tf-train-steps={} --tf-export-dir={} {}".format(step, s3bucketexportpath, args)
                        ],
                        "image": "chuckshow/mnist-tf-pipeline:latest",
                        "name": "tensorflow",
                        "env": [
                            {
                                "name": "AWS_REGION",
                                "value": aws_region
                            },
                            {
                                "name": "AWS_ACCESS_KEY_ID",
                                "valueFrom": {
                                    "secretKeyRef": {
                                        "name": "aws-secret",
                                        "key": "AWS_ACCESS_KEY_ID"
                                    }
                                }
                            },
                            {
                                "name": "AWS_SECRET_ACCESS_KEY",
                                "valueFrom": {
                                    "secretKeyRef": {
                                        "name": "aws-secret",
                                        "key": "AWS_SECRET_ACCESS_KEY"
                                    }
                                }
                            }
                        ]
                    }
                ]
            }
        }
    }

    return chief, worker


