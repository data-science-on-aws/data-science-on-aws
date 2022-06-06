ray down -y cluster.yaml
aws ssm delete-parameter --name AmazonCloudWatch-ray_dashboard_config_cluster
aws ssm delete-parameter --name AmazonCloudWatch-ray_agent_config_cluster
ray up -y cluster.yaml
ray exec cluster.yaml "git clone https://github.com/data-science-on-aws/data-science-on-aws.git"
ray exec cluster.yaml "jupyter server list"
