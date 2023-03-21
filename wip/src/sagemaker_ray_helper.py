import subprocess
import os
import time
import ray
import socket
import json
import sys

class RayHelper():
    def __init__(self, ray_port:str="9339", redis_pass:str="redis_password"):
        
        self.ray_port = ray_port
        self.redis_pass = redis_pass
        self.resource_config = self.get_resource_config()
        self.master_host = self.resource_config["hosts"][0]
        self.n_hosts = len(self.resource_config["hosts"])
        
    @staticmethod
    def get_resource_config():
    
        return dict(current_host = os.environ.get("SM_CURRENT_HOST"),
                    hosts = json.loads(os.environ.get("SM_HOSTS")) )
    
    def _get_ip_from_host(self):
        ip_wait_time = 200
        counter = 0
        ip = ""

        while counter < ip_wait_time and ip == "":
            try:
                ip = socket.gethostbyname(self.master_host)
                break
            except:
                counter += 1
                time.sleep(1)

        if counter == ip_wait_time and ip == "":
            raise Exception(
                "Exceeded max wait time of {}s for hostname resolution".format(ip_wait_time)
            )
        return ip 
    
    def start_ray(self):
        
        master_ip = self._get_ip_from_host()
    
        if self.resource_config["current_host"] == self.master_host:
            output = subprocess.run(['ray', 'start', '--head', '-vvv', '--port', self.ray_port, '--redis-password', self.redis_pass, '--include-dashboard', 'false'], stdout=subprocess.PIPE)
            print(output.stdout.decode("utf-8"))
            ray.init(address="auto", include_dashboard=False)
            self._wait_for_workers()
            print("All workers present and accounted for")
            print(ray.cluster_resources())

        else:
            time.sleep(10)
            subprocess.run(['ray', 'start', f"--address={master_ip}:{self.ray_port}", '--redis-password', self.redis_pass, "--block"], stdout=subprocess.PIPE)
            sys.exit(0)
    
    
    def _wait_for_workers(self, timeout=60):
        
        print(f"Waiting {timeout} seconds for {self.n_hosts} nodes to join")
        
        while len(ray.nodes()) < self.n_hosts:
            print(f"{len(ray.nodes())} nodes connected to cluster")
            time.sleep(5)
            timeout-=5
            if timeout==0:
                raise Exception("Max timeout for nodes to join exceeded")