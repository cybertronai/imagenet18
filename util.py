import base64
import pickle
import re
import subprocess


def extract_ec2_metadata():
    """Returns dictionary of common ec2 metadata"""
    from ec2_metadata import ec2_metadata
    try:
        return {
            'region': ec2_metadata.region,
            'account_id': ec2_metadata.account_id,
            'ami_id': ec2_metadata.ami_id,
            'availability_zone': ec2_metadata.availability_zone,
            'instance_type': ec2_metadata.instance_type,
            'public_ipv4': ec2_metadata.public_ipv4,
            'private_ipv4': ec2_metadata.private_ipv4
            }
    except:  # may crash with requests.exceptions.ConnectTimeout when not on AWS
        return {}


def log_environment():
    """Logs AWS local machine environment to wandb config."""
    import os
    import wandb
    
    for key in os.environ:
        if re.match(r"^NCCL|CUDA|PATH|^LD|USER|PWD|^OMP", key):
            wandb.config['env_'+key] = os.getenv(key)

    wandb.config.update(extract_ec2_metadata())


def ossystem(cmd, shell=True):
    """Like os.system, but returns output of command as string."""
    p = subprocess.Popen(cmd, shell=shell, stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT)
    (stdout, stderr) = p.communicate()
    return stdout.decode('ascii')


def text_pickle(obj) -> str:
    """Pickles object into character string"""
    pickle_string = pickle.dumps(obj)
    pickle_string_encoded: bytes = base64.b64encode(pickle_string)
    s = pickle_string_encoded.decode('ascii')
    return s


def text_unpickle(pickle_string_encoded: str):
    """Unpickles character string"""
    if not pickle_string_encoded:
        return None
    obj = pickle.loads(base64.b64decode(pickle_string_encoded))
    return obj
