import json


def argugments_provide(objective_type="minimize",
                       objective_goal=0.001,
                       objective_metrics="loss",
                       algorithm="random",
                       parameters_lr_min="0.01",
                       parameters_lr_max="0.03",
                       parameters_batchsize=["16", "32", "64"],
                       tf_train_steps="200",
                       image="chuckshow/mnist-tf-pipeline:latest",
                       worker_num=3):
    objectiveConfig = {
      "type": objective_type,
      "goal": objective_goal,
      "objectiveMetricName": objective_metrics,
    }
    
    algorithmConfig = {"algorithmName" : algorithm}

    parameters = [
      {"name": "--tf-learning-rate", "parameterType": "double", "feasibleSpace": {"min": parameters_lr_min, "max": parameters_lr_max}},
      {"name": "--tf-batch-size", "parameterType": "discrete", "feasibleSpace": {"list": parameters_batchsize}},
    ]
    
    rawTemplate = {
      "apiVersion": "kubeflow.org/v1",
      "kind": "TFJob",
      "metadata": {
         "name": "{{.Trial}}",
         "namespace": "{{.NameSpace}}"
      },
      "spec": {
        "tfReplicaSpecs": {
          "Chief": {
            "replicas": 1,
            "restartPolicy": "OnFailure",
            "template": {
              "spec": {
                "containers": [
                {
                  "command": [
                    "sh",
                    "-c"
                  ],
                  "args": [
                    "python /opt/model.py --tf-train-steps={} {{- with .HyperParameters}} {{- range .}} {{.Name}}={{.Value}} {{- end}} {{- end}}".format(tf_train_steps)
                  ],
                  "image": image,
                  "name": "tensorflow"
                }
                ]
              }
            }
          },
          "Worker": {
            "replicas": worker_num,
            "restartPolicy": "OnFailure",
            "template": {
              "spec": {
                "containers": [
                {
                  "command": [
                    "sh",
                    "-c"
                  ],
                  "args": [ 
                    "python /opt/model.py --tf-train-steps={} {{- with .HyperParameters}} {{- range .}} {{.Name}}={{.Value}} {{- end}} {{- end}}".format(tf_train_steps)
                  ],
                  "image": image,
                  "name": "tensorflow"
                }
                ]
              }
            }
          }
        }
      }
    }
    
    trialTemplate = {
      "goTemplate": {
        "rawTemplate": json.dumps(rawTemplate)
      }
    }

    metricsCollectorSpec = {
      "source": {
        "fileSystemPath": {
          "path": "/tmp/tf",
          "kind": "Directory"
        }
      },
      "collector": {
        "kind": "TensorFlowEvent"
      }
    }
    
    return objectiveConfig, algorithmConfig, parameters, trialTemplate, metricsCollectorSpec