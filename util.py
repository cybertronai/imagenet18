import base64
import os
import pickle
import random
import re
import string
import subprocess
import threading
from typing import Tuple


def is_set(name: str) -> bool:
  """Helper method to check if given property is set, anything except missing, 0 and false means set """

  val = os.environ.get(name, '0').lower()
  return not (val == '0' or val == 'false')


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


def random_id(k=3):
  """Random id to use for AWS identifiers."""
  #  https://stackoverflow.com/questions/2257441/random-string-generation-with-upper-case-letters-and-digits-in-python
  return ''.join(random.choices(string.ascii_lowercase + string.digits, k=k))


def log_environment():
    """Logs AWS local machine environment to wandb config."""
    import os
    import wandb
    import torch

    if not (hasattr(wandb, 'config') and wandb.config is not None):
        return

    for key in os.environ:
        if re.match(r"^NCCL|CUDA|PATH|^LD|USER|PWD|^OMP", key):
            wandb.config['env_'+key] = os.getenv(key)

    wandb.config['pytorch_version'] = torch.__version__
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


def format_env(**d):
    """Converts env var values into variable string, ie
        'var1="val1" var2="val2" '"""
    args_ = [f'{key}="{d[key]}" ' for key in d]
    return ''.join(args_)


def format_env_export(**d):
    """Converts env var values into variable string, ie
        'export var1="val1" && export var2="val2" '"""
    args_ = [f'export {key}="{d[key]}" ' for key in d]
    return ' && '.join(args_)


def format_env_x(**d):
    """Converts env var values into format suitable for mpirun, ie
        '-x var1="val1" -x var2="val2" '"""
    args_ = [f'-x {key}="{d[key]}" ' for key in sorted(d)]
    return ''.join(args_)


def setup_mpi(job, skip_ssh_setup=False) -> Tuple[str, str]:
    """Sets up passwordless SSH between all tasks in the job."""
    public_keys = {}
    if not skip_ssh_setup:
        for task in job.tasks:
            key_fn = '~/.ssh/id_rsa'  # this fn is special, used by default by ssh
            task.run(f"yes | ssh-keygen -t rsa -f {key_fn} -N ''")

            public_keys[task] = task.read(key_fn + '.pub')

        keys = {}
        for i, task1 in enumerate(job.tasks):
            task1.run('echo "StrictHostKeyChecking no" >> /etc/ssh/ssh_config',
                      sudo=True, non_blocking=True)
            for j, task2_ in enumerate(job.tasks):
                #  task1 ->ssh-> task2
                #  task2.run(f'echo "{public_keys[task1]}" >> ~/.ssh/authorized_keys',
                #         non_blocking=True)
                keys.setdefault(j, []).append(public_keys[task1])

        def setup_task_mpi(j2):
            task2 = job.tasks[j2]
            key_str = '\n'.join(keys[j2])
            fn = f'task-{j2}'
            with open(fn, 'w') as f:
                f.write(key_str)
            task2.upload(fn)
            task2.run(f"""echo `cat {fn}` >> ~/.ssh/authorized_keys""",
                      non_blocking=True)

        run_parallel(setup_task_mpi, range(len(job.tasks)))
        #        for j, task2_ in enumerate(job.tasks):
        #            setup_task_mpi(j)

    task0 = job.tasks[0]
    hosts = [task.ip for task in job.tasks]
    hosts_str = ','.join(hosts)
    hosts_file_lines = [f'{host} slots={task0.num_gpus} max-slots={task0.num_gpus}' for host in hosts]
    hosts_file_str = '\n'.join(hosts_file_lines)
    return hosts_str, hosts_file_str


def run_parallel(f, args_):
    threads = [threading.Thread(name=f'run_parallel_{i}', target=f, args=[t]) for i, t in enumerate(args_)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

