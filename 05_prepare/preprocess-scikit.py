import argparse
import json
import os

def list_arg(raw_value):
    """argparse type for a list of strings"""
    return str(raw_value).split(",")


def parse_args():
    # Unlike SageMaker training jobs (which have `SM_HOSTS` and `SM_CURRENT_HOST` env vars), processing jobs to need to parse the resource config file directly
    resconfig = {}
    try:
        with open("/opt/ml/config/resourceconfig.json", "r") as cfgfile:
            resconfig = json.load(cfgfile)
    except FileNotFoundError:
        print("/opt/ml/config/resourceconfig.json not found.  current_host is unknown.")
        pass # Ignore

    # Local testing with CLI args
    parser = argparse.ArgumentParser(description="Process")

    parser.add_argument("--hosts", type=list_arg,
        default=resconfig.get("hosts", ["unknown"]),
        help="Comma-separated list of host names running the job"
    )
    parser.add_argument("--current-host", type=str,
        default=resconfig.get("current_host", "unknown"),
        help="Name of this host running the job"
    )
    
    return parser.parse_args()


def process(args):
    print('Current host: {}'.format(args.current_host))

    print('Creating directory /opt/ml/processing/output/train'.format(args.current_host))
    os.makedirs('/opt/ml/processing/output/train/', exist_ok=True)

    print('Writing to /opt/ml/processing/output/train/{}.csv'.format(args.current_host))
    with open('/opt/ml/processing/output/train/{}.csv'.format(args.current_host), 'w') as fd:
        fd.write('host{},thanks,andre,and,alex!'.format(args.current_host))
        fd.close()
        
    print('Listing contents of /opt/ml/processing/output/train/')
    dirs_output_train = os.listdir('/opt/ml/processing/output/train/')

    # This would print all the files and directories
    for file in dirs_output_train:
        print(file)

    print('Listing contents of /opt/ml/processing/input/data')
    dirs_input = os.listdir('/opt/ml/processing/input/data')

    # This would print all the files and directories
    for file in dirs_input:
        print(file)
 
    print('Complete')
    
    
if __name__ == "__main__":
    args = parse_args()
    print("Loaded arguments:")
    print(args)
    
    print("Environment variables:")
    print(os.environ)

    process(args)
