import json
from string import Template


def uivirtualsvc_resource(namespace, ui_name):
    uivirtualservicejson_template = Template("""
    {
      "apiVersion": "networking.istio.io/v1alpha3",
      "kind": "VirtualService",
      "metadata": {
        "name": "$uiname",
        "namespace": "$namespace"
      },
      "spec": {
        "gateways": [
          "kubeflow/kubeflow-gateway"
        ],
        "hosts": [
          "*"
        ],
        "http": [
          {
            "match": [
              {
                "uri": {
                  "prefix": "/mnist/$namespace/ui/"
                }
              }
            ],
            "rewrite": {
              "uri": "/"
            },
            "route": [
              {
                "destination": {
                  "host": "$uiname.$namespace.svc.cluster.local",
                  "port": {
                    "number": 80
                  }
                }
              }
            ],
            "timeout": "300s"
          }
        ]
      }
    }
    """)
    
    uivirtualservicejson = uivirtualservicejson_template.substitute(
        {'namespace': namespace,
         'uiname': ui_name,
        })
    
    uivirtualserviceresource = json.loads(uivirtualservicejson)
    
    return uivirtualserviceresource


def uisvc_resource(namespace, ui_name):
    uiservicejson_template = Template("""
    {
      "apiVersion": "v1",
      "kind": "Service",
      "metadata": {
        "name": "$uiname",
        "namespace": "$namespace"
      },
      "spec": {
        "ports": [
          {
            "name": "http-mnist-ui",
            "port": 80,
            "targetPort": 5000
          }
        ],
        "selector": {
          "app": "mnist-web-ui"
        },
        "type": "ClusterIP"
      }
    }
    """)
    
    uiservicejson = uiservicejson_template.substitute(
        {'namespace': namespace,
         'uiname': ui_name,
        })
    
    uiserviceresource = json.loads(uiservicejson)
    
    return uiserviceresource


def uideploy_resource(namespace, ui_name):
    uideployjson_template = Template("""
    {
      "apiVersion": "apps/v1",
      "kind": "Deployment",
      "metadata": {
        "name": "$uiname",
        "namespace": "$namespace"
      },
      "spec": {
        "replicas": 1,
        "selector": {
          "matchLabels": {
            "app": "mnist-web-ui"
          }
        },
        "template": {
          "metadata": {
            "labels": {
              "app": "mnist-web-ui"
            }
          },
          "spec": {
            "containers": [
              {
                "image": "gcr.io/kubeflow-examples/mnist/web-ui:v20190112-v0.2-142-g3b38225",
                "name": "web-ui",
                "ports": [
                  {
                    "containerPort": 5000
                  }
                ]
              }
            ],
            "serviceAccount": "default"
          }
        }
      }
    }
    """)
    
    uideployjson = uideployjson_template.substitute(
        {'namespace': namespace,
         'uiname': ui_name,
        })
    
    uideployresource = json.loads(uideployjson)
    
    return uideployresource


def tfservingsvc_resource(namespace, servingdeploy_name, servingsvc_name):
    servicejson_template = Template("""
    {
      "apiVersion": "v1",
      "kind": "Service",
      "metadata": {
        "annotations": {
          "prometheus.io/path": "/monitoring/prometheus/metrics",
          "prometheus.io/port": "8500",
          "prometheus.io/scrape": "true"
        },
        "labels": {
          "app": "$servingdeploy"
        },
        "name": "$servingsvc",
        "namespace": "$namespace"
      },
      "spec": {
        "ports": [
          {
            "name": "grpc-tf-serving",
            "port": 9000,
            "targetPort": 9000
          },
          {
            "name": "http-tf-serving",
            "port": 8500,
            "targetPort": 8500
          }
        ],
        "selector": {
          "app": "$servingdeploy"
        },
        "type": "ClusterIP"
      }
    }
    """)
    
    servicejson = servicejson_template.substitute(
        {'namespace': namespace, 
         'servingdeploy': servingdeploy_name,
         'servingsvc': servingsvc_name,
        })
    
    serviceresource = json.loads(servicejson)
    
    return serviceresource
    

def tfservingdeploy_resource(namespace, 
                             s3bucketexportpath, 
                             servingdeploy_name,
                             aws_region):
    deployjson_template = Template("""
    {
      "apiVersion": "apps/v1",
      "kind": "Deployment",
      "metadata": {
        "labels": {
          "app": "mnist"
        },
        "name": "$servingdeploy",
        "namespace": "$namespace"
      },
      "spec": {
        "selector": {
          "matchLabels": {
            "app": "$servingdeploy"
          }
        },
        "template": {
          "metadata": {
            "annotations": {
              "sidecar.istio.io/inject": "false"
            },
            "labels": {
              "app": "$servingdeploy",
              "version": "v1"
            }
          },
          "spec": {
            "serviceAccount": "default",
            "containers": [
              {
                "args": [
                  "--port=9000",
                  "--rest_api_port=8500",
                  "--model_name=mnist",
                  "--model_base_path=$s3bucketexportpath"
                ],
                "command": [
                  "/usr/bin/tensorflow_model_server"
                ],
                "env": [
                  {
                    "name": "AWS_REGION",
                    "value": "$aws_region"
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
                ],
                "image": "tensorflow/serving:1.15.0",
                "imagePullPolicy": "IfNotPresent",
                "livenessProbe": {
                  "initialDelaySeconds": 30,
                  "periodSeconds": 30,
                  "tcpSocket": {
                    "port": 9000
                  }
                },
                "name": "mnist",
                "ports": [
                  {
                    "containerPort": 9000
                  },
                  {
                    "containerPort": 8500
                  }
                ],
                "resources": {
                  "limits": {
                    "cpu": "1",
                    "memory": "1Gi"
                  },
                  "requests": {
                    "cpu": "1",
                    "memory": "1Gi"
                  }
                }
              }
            ]
          }
        }
      }
    }
    """)
    
    deployjson = deployjson_template.substitute(
            {'namespace': namespace,
             's3bucketexportpath': s3bucketexportpath,
             'servingdeploy': servingdeploy_name,
             'aws_region': aws_region             
            })
    
    deploy = json.loads(deployjson)
    
    return deploy
