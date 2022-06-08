ray down -y cluster.yaml 
aws ssm delete-parameter --name AmazonCloudWatch-ray_dashboard_config_cluster
aws ssm delete-parameter --name AmazonCloudWatch-ray_agent_config_cluster
ray up -y cluster.yaml --no-config-cache
ray exec cluster.yaml "git clone https://github.com/data-science-on-aws/data-science-on-aws.git"
echo "JupyterLab..."
ray exec cluster.yaml "jupyter server list" --no-config-cache
nohup ray attach cluster.yaml -p 8888 --no-config-cache > attach-jupyterlab.out &
nohup ray attach cluster.yaml -p 5001 --no-config-cache > attach-mlflow.out &
nohup ray dashboard cluster.yaml --no-config-cache > dashboard.out &
echo "CloudWatch Metrics..."
echo "https://console.aws.amazon.com/cloudwatch/home?#dashboards:name=cluster-RayDashboard"
echo ""
echo "CloudWatch Logs..."
echo "https://console.aws.amazon.com/cloudwatch/home?#logsV2:log-groups/log-group/cluster-ray_logs_out"
echo "https://console.aws.amazon.com/cloudwatch/home?#logsV2:log-groups/log-group/cluster-ray_logs_err"
