#!/usr/bin/env python

import argparse
import os
import re
import sys
import time

import ncluster
from ncluster import aws_util as u

# todo(y): change to AMI owned by me ie, pytorch.imagenet.source.v7-copy
import util

# IMAGE_NAME = 'pytorch.imagenet.source.v7'
HOSTS_SLOTS_FN = 'hosts.slots'

parser = argparse.ArgumentParser()
parser.add_argument('--name', type=str, default='imagenet',
                    help="name of the current run, used for machine naming")
parser.add_argument('--run_name', type=str, default='',
                    help="name of run for loggin")
parser.add_argument('--machines', type=int, default=1,
                    help="how many machines to use")
parser.add_argument('--num_tasks', type=int, default=1,
                    help="same as machines for compatibility, don't use")
parser.add_argument('--mount_imagenet', type=int, default=1,
                    help="if set, mount imagenet disk rather than taking data from local image")
parser.add_argument('--offset', type=int, default=0,
                    help='offset for imagenet ebs numbering')
parser.add_argument('--vmtouch', type=int, default=0,
                    help="lock all examples into physical memory")
parser.add_argument('--internal_config_fn', type=str, default='ncluster_config_dict',
                    help='location of filename with extra info to log')
parser.add_argument('--nproc_per_node', type=int, default=8, help="Processes per machine, must not exceed number of GPUS")
parser.add_argument('--image_name', type=str, default='pytorch-efa01',
                    help="Image to use for this run")
parser.add_argument('--instance_type', type=str, default='p3.16xlarge', help="Image to use for this run")
parser.add_argument('--conda_env', type=str, default='pytorch_p36', help="name of conda env")
parser.add_argument('--efa', type=int, default=0, help="use AWS EFA network")
parser.add_argument('--pseudo_efa', type=int, default=0, help="use sockets interface when launching under EFA")
parser.add_argument('--no_op', type=int, default=0, help='just print environment/debug info and skip rest')
parser.add_argument('--log_all_workers', type=int, default=0, help='log from each worker instead of just chief')
parser.add_argument('--spot', action='store_true', help='use spot instead of regular instances')
parser.add_argument('--cuda_debug', action='store_true', help='debug cuda errors')
parser.add_argument('--pytorch_nightly', action='store_true', help='install nightly PyTorch')
parser.add_argument('--pytorch_use_spawn', action='store_true', help='use spawn method in dataloaders')
parser.add_argument('--simple_ring_setup', action='store_true', help='set 16 rings instead of manual ring order')
parser.add_argument('--skip_setup', action='store_true', help='speed up relaunch by skipping some steps')
args = parser.parse_args()
args.num_tasks = args.machines
if not args.run_name:
    args.run_name = args.name

# 109:12 to 93.00
# https://app.wandb.ai/yaroslavvb/imagenet18/runs/gxsdo6i0
lr = 1.0
scale_224 = 224 / 512
scale_288 = 128 / 512
one_machine = [
    {'ep': 0, 'sz': 128, 'bs': 512, 'trndir': '-sz/160'},
    {'ep': (0, 5), 'lr': (lr, lr * 2)},  # lr warmup is better with --init-bn0
    {'ep': 5, 'lr': lr},
    {'ep': 14, 'sz': 224, 'bs': 224,
     'lr': lr * scale_224},
    {'ep': 16, 'lr': lr / 10 * scale_224},
    {'ep': 27, 'lr': lr / 100 * scale_224},
    {'ep': 32, 'sz': 288, 'bs': 128, 'min_scale': 0.5, 'rect_val': True,
     'lr': lr / 100 * scale_288},
    {'ep': (33, 35), 'lr': lr / 1000 * scale_288}
]

# 54 minutes to 93.364
# https://app.wandb.ai/yaroslavvb/imagenet18/runs/lhx5a053
lr = 0.75 * 2
bs = [256, 224, 128]  # largest batch size that fits in memory for each image size
bs_scale = [x / bs[0] for x in bs]  # scale learning rate to batch size
two_machines = [
    {'ep': 0, 'sz': 128, 'bs': bs[0], 'trndir': '-sz/160'},
    # bs = 256 * 4 * 8 = 8192
    {'ep': (0, 6), 'lr': (lr, lr * 2)},
    {'ep': 6, 'sz': 128, 'bs': bs[0] * 2, 'keep_dl': True},
    {'ep': 6, 'lr': lr * 2},
    {'ep': (11, 13), 'lr': (lr * 2, lr)},  # trying one cycle
    {'ep': 13, 'sz': 224, 'bs': bs[1], 'trndir': '-sz/352', 'min_scale': 0.087},
    {'ep': 13, 'lr': lr * bs_scale[1]},
    {'ep': (16, 23), 'lr': (lr * bs_scale[1], lr / 10 * bs_scale[1])},
    {'ep': (23, 28), 'lr': (lr / 10 * bs_scale[1], lr / 100 * bs_scale[1])},
    {'ep': 28, 'sz': 288, 'bs': bs[2], 'min_scale': 0.5, 'rect_val': True},
    {'ep': (28, 30), 'lr': (lr / 100 * bs_scale[2], lr / 1000 * bs_scale[2])}
]

# 29:44 to 93.05
# events: https://s3.amazonaws.com/yaroslavvb/logs/imagenet-4
# p3dn: https://app.wandb.ai/yaroslavvb/imagenet18/runs/pp0g9k5c
lr = 0.50 * 4  # 4 = num tasks
bs = [256, 224,
      128]  # largest batch size that fits in memory for each image size
bs_scale = [x / bs[0] for x in bs]  # scale learning rate to batch size
four_machines = [
    {'ep': 0, 'sz': 128, 'bs': bs[0], 'trndir': '-sz/160'},
    # bs = 256 * 4 * 8 = 8192
    {'ep': (0, 6), 'lr': (lr, lr * 2)},
    {'ep': 6, 'sz': 128, 'bs': bs[0] * 2, 'keep_dl': True},
    {'ep': 6, 'lr': lr * 2},
    {'ep': (11, 13), 'lr': (lr * 2, lr)},  # trying one cycle
    {'ep': 13, 'sz': 224, 'bs': bs[1], 'trndir': '-sz/352', 'min_scale': 0.087},
    {'ep': 13, 'lr': lr * bs_scale[1]},
    {'ep': (16, 23), 'lr': (lr * bs_scale[1], lr / 10 * bs_scale[1])},
    {'ep': (23, 28), 'lr': (lr / 10 * bs_scale[1], lr / 100 * bs_scale[1])},
    {'ep': 28, 'sz': 288, 'bs': bs[2], 'min_scale': 0.5, 'rect_val': True},
    {'ep': (28, 30), 'lr': (lr / 100 * bs_scale[2], lr / 1000 * bs_scale[2])}
]

# 19:04 to 93.0
lr = 0.235 * 8
scale_224 = 224 / 128
eight_machines = [
    {'ep': 0, 'sz': 128, 'bs': 128, 'trndir': '-sz/160'},
    {'ep': (0, 6), 'lr': (lr, lr * 2)},
    {'ep': 6, 'bs': 256, 'keep_dl': True,
     'lr': lr * 2},
    {'ep': (11, 14), 'lr': (lr * 2, lr)},  # trying one cycle
    {'ep': 14, 'sz': 224, 'bs': 128, 'trndir': '-sz/352', 'min_scale': 0.087,
     'lr': lr},
    {'ep': 17, 'bs': 224, 'keep_dl': True},
    {'ep': (17, 23), 'lr': (lr, lr / 10 * scale_224)},
    {'ep': (23, 29), 'lr': (lr / 10 * scale_224, lr / 100 * scale_224)},
    {'ep': 29, 'sz': 288, 'bs': 128, 'min_scale': 0.5, 'rect_val': True},
    {'ep': (29, 35), 'lr': (lr / 100, lr / 1000)}
]

# 16:08 to 93.04 (after prewarming)
lr = 0.235 * 8  #
bs = 64
sixteen_machines = [
    {'ep': 0, 'sz': 128, 'bs': 64, 'trndir': '-sz/160'},
    {'ep': (0, 6), 'lr': (lr, lr * 2)},
    {'ep': 6, 'bs': 128, 'keep_dl': True},
    {'ep': 6, 'lr': lr * 2},
    {'ep': 16, 'sz': 224, 'bs': 64},  # todo: increase this bs
    {'ep': 16, 'lr': lr},
    {'ep': 19, 'bs': 192, 'keep_dl': True},
    {'ep': 19, 'lr': 2 * lr / (10 / 1.5)},
    {'ep': 31, 'lr': 2 * lr / (100 / 1.5)},
    {'ep': 37, 'sz': 288, 'bs': 128, 'min_scale': 0.5, 'rect_val': True},
    {'ep': 37, 'lr': 2 * lr / 100},
    {'ep': (38, 50), 'lr': 2 * lr / 1000}
]

schedules = {1: one_machine,
             2: two_machines,
             4: four_machines,
             8: eight_machines,
             16: sixteen_machines}


# routines to build NCCL ring orders
def get_nccl_params(num_tasks, nproc_per_node):
    if num_tasks <= 1:
        return 'NCCL_DEBUG=VERSION'
    nccl_rings = get_nccl_rings(num_tasks, nproc_per_node)
    env = f'NCCL_RINGS="{nccl_rings}" NCCL_SINGLE_RING_THRESHOLD=10 '
    if args.simple_ring_setup:
        env = f'NCCL_MIN_NRINGS=16 NCCL_MAX_NRINGS=16 '

    return env
    # return 'NCCL_MIN_NRINGS=2 NCCL_SINGLE_RING_THRESHOLD=10 NCCL_DEBUG=VERSION'


def get_nccl_rings(num_tasks, num_gpus):
    ring = build_ring_order(range(num_tasks), range(num_gpus))
    ring_rev = build_ring_order(reversed(range(num_tasks)),
                                reversed(range(num_gpus)))
    rotated_gpu_order = [3, 2, 1, 0, 7, 6, 5, 4]
    skip_gpu_order = get_skip_order(num_gpus)
    if (num_tasks >= 4) and (num_gpus == 8):
        assert ((num_tasks % 4) == 0)
        skip_machine_order = get_skip_order(num_tasks)
        ring_skip = build_ring_order(skip_machine_order, rotated_gpu_order)
        ring_skip_rev = build_ring_order(reversed(skip_machine_order),
                                         skip_gpu_order)
        rings_arr = [ring, ring_rev, ring_skip, ring_skip_rev]
        # rings_arr = [ring, ring_rev, ring_skip]
    else:
        rings_arr = [ring, ring_rev]
    return ' | '.join(rings_arr)


def build_ring_order(machine_order, gpu_order):
    gpu_order = list(gpu_order)
    machine_order = list(machine_order)
    ngpus = len(gpu_order)
    r_order = [(x * ngpus) + y for x in machine_order for y in gpu_order]
    return ' '.join(map(str, r_order))


def get_skip_order(size):
    if size == 4:
        return [0, 2, 1, 3]
    skip_step = 5 if size == 16 else 3
    # step size of 3 yields - [0,3,6,1,4,7,2,5]
    return [(i * skip_step) % size for i in range(size)]


def format_params(arg):
    if isinstance(arg, list) or isinstance(arg, dict):
        return '\"' + str(arg) + '\"'
    else:
        return str(arg)


def create_volume_tags(name):
    return [{
        'ResourceType': 'volume',
        'Tags': [{
            'Key': 'Name',
            'Value': name
        }]
    }]


DEFAULT_UNIX_DEVICE = '/dev/xvdf'
ATTACH_WAIT_INTERVAL_SEC = 5


def mount_imagenet(job: ncluster.aws_backend.Job):
    """Attaches EBS disks with imagenet data to each task of the job."""

    task0 = job.tasks[0]
    zone = u.get_zone()
    vols = {}
    ec2 = u.get_ec2_resource()
    for vol in ec2.volumes.all():
        vols[u.get_name(vol)] = vol

    attach_attempted = False
    for i, t in enumerate(job.tasks):
        vol_name = f'imagenet_{zone[-2:]}_{i+args.offset:02d}'
        assert vol_name in vols, f"Volume {vol_name} not found, set your NCLUSTER_ZONE={zone} and run replicate_imagenet.py"
        vol = vols[vol_name]
        print(f"Attaching {vol_name} to {t.name}")
        if vol.attachments:
            instance = ec2.Instance(vol.attachments[0]['InstanceId'])
            if instance.id == t.instance.id:
                print(f"{vol_name} already attached")
                continue
            else:  # attached to some other instance, detach
                print(f"detaching {vol_name} from {u.get_name(instance)}")
                vol.detach_from_instance()
                while vol.state != 'available':
                    vol.reload()
                    time.sleep(5)
                    print(f"waiting for detachment from {u.get_name(instance)}")
                vol.attach_to_instance(InstanceId=t.instance.id, Device=DEFAULT_UNIX_DEVICE)
                attach_attempted = True

        else:
            vol.attach_to_instance(InstanceId=t.instance.id, Device=DEFAULT_UNIX_DEVICE)
            attach_attempted = True

    if attach_attempted:
        time.sleep(2)  # wait for attachment to succeed
        i = 0
        vol_name = f'imagenet_{zone[-2:]}_{i+args.offset:02d}'
        vol = vols[vol_name]
        vol.reload()
        assert vol.attachments[0]['InstanceId'] == job.tasks[0].instance.id

    def strip_dev(d):
        return d[len('/dev/'):]

    # attach the volume if needed
    df_output = task0.run('df', return_output=True)
    actual_device = DEFAULT_UNIX_DEVICE
    if '/data' not in df_output:
        # hack for p3dn's ignoring device name during volume attachment
        lsblk_output = task0.run('lsblk', return_output=True)
        if strip_dev(DEFAULT_UNIX_DEVICE) not in lsblk_output:
            actual_device = '/dev/nvme3n1'
            assert strip_dev(actual_device) in lsblk_output, f"Hack for p3dn failed, {actual_device} not found, " \
                f"available devices '{lsblk_output}'"

        job.run(f'sudo mkdir -p /data && sudo chown `whoami` /data && sudo mount {actual_device} /data')
    while '/data' not in task0.run('df', return_output=True):
        time.sleep(ATTACH_WAIT_INTERVAL_SEC)
        print(f"Waiting for attachment")


def main():
    if args.image_name == 'pytorch.imagenet.source.v7':
        supported_regions = ['us-west-2', 'us-east-1', 'us-east-2']
        assert ncluster.get_region() in supported_regions, f"required AMI {args.image_name} has only been made available in regions {supported_regions}, but your current region is {ncluster.get_region()} (set $AWS_DEFAULT_REGION)"
    assert args.machines in schedules, f"{args.machines} not supported, only support {schedules.keys()}"

    if args.mount_imagenet:
        datadir = '/data/imagenet'
    else:
        datadir = '~/data/imagenet'
        os.environ['NCLUSTER_AWS_FAST_ROOTDISK'] = '1'  # use io2 disk on AWS

    if args.num_tasks >= 16:
        assert args.simple_ring_setup, "must use --simple_ring_setup, otherwise NCCL_RINGS env var exceeds cmd-line limit"
        
    job = ncluster.make_job(name=args.name,
                            run_name=args.run_name,
                            num_tasks=args.machines,
                            image_name=args.image_name,
                            instance_type=args.instance_type,
                            disk_size=500,
                            spot=args.spot,
                            skip_setup=args.skip_setup,
                            )

    task0 = job.tasks[0]
    _logdir = task0.logdir  # workaround for race condition in creating logdir

    config = {}
    for key in os.environ:
        if re.match(r"^NCLUSTER", key):
            config['env_' + key] = os.getenv(key)
    config.update(vars(args))

    CUDA_HOME = f'/usr/local/cuda'
    EFA_HOME = f'/opt/amazon/efa'
    MPI_HOME = EFA_HOME
    NPROC_PER_NODE = args.nproc_per_node
    assert NPROC_PER_NODE <= task0.num_gpus, f"requested {NPROC_PER_NODE} processes, but only {task0.num_gpus} gpus present"
    NUM_GPUS = NPROC_PER_NODE * args.num_tasks

    config['NUM_GPUS'] = NUM_GPUS

    config['internal_id'] = u.get_account_number()
    config['internal_alias'] = u.get_account_name()
    config['region'] = u.get_region()
    config['zone'] = u.get_zone()
    config['launch_user'] = os.environ.get('USER', '')
    config['cmd'] = ' '.join(sys.argv)
    config['launcher_conda'] = util.ossystem('echo ${CONDA_PREFIX:-"$(dirname $(which conda))/../"}')
    config['launcher_cmd'] = 'python ' + ' '.join(sys.argv)
    config['logdir'] = job.logdir

    pickled_config = util.text_pickle(config)
    if args.log_all_workers:
        job.write(args.internal_config_fn, pickled_config)
    else:
        job.tasks[0].write(args.internal_config_fn, pickled_config)

    if args.mount_imagenet:
        assert u.get_zone(), "Must specify zone when reusing EBS volumes"
        mount_imagenet(job)

    if not args.skip_setup:
        job.run('rm -f *.py')  # remove files backed into imagenet18 release image
        job.run('conda init')  # missing .bashrc
        job.run(
            f'{{ source activate {args.conda_env} && bash setup.sh && pip install -U protobuf ; }}  && {{ killall python || echo hi ; }} ')
        if args.pytorch_nightly:
            job.run('conda install -y -c pytorch pytorch-nightly && bash setup.sh')
    else:
        job.run([f'source ~/.bashrc && conda activate {args.conda_env}', f'killall python || echo hi'])

    job.rsync('.')

    if args.efa:
        assert 'efa' in args.image_name  # make sure we use EFA-enabled image
        hosts_str, hosts_file_str = util.setup_mpi(job, skip_ssh_setup=args.skip_setup)
        if not args.skip_setup:
            task0.write(HOSTS_SLOTS_FN, hosts_file_str)

    env_params = get_nccl_params(args.machines, args.nproc_per_node)
    if args.cuda_debug:
        env_params += 'CUDA_LAUNCH_BLOCKING=1 NCCL_DEBUG=INFO '
    else:
        env_params += 'NCCL_DEBUG=INFO '

    env_params += " OMP_NUM_THREADS=1 "
    if args.pytorch_use_spawn:
        assert args.pytorch_nightly
        env_params += " PYTORCH_USE_SPAWN=1 "
    if 'WANDB_API_KEY' in os.environ:
        env_params += f" WANDB_API_KEY={os.environ.get('WANDB_API_KEY')} "

    # Training script args
    default_params = [
        datadir,
        '--fp16',
        '--logdir', job.logdir,
        '--name', f'{args.run_name}-{util.random_id()}',
        '--distributed',
        '--init-bn0',
        '--no-bn-wd',
        '--log_all_workers', args.log_all_workers,
    ]

    params = ['--phases', util.text_pickle(schedules[args.machines])]
    training_params = default_params + params
    training_params = ' '.join(map(format_params, training_params))

    if not args.efa:
        # TODO: simplify args processing, or give link to actual commands run
        for i, task in enumerate(job.tasks):
            dist_params = f'--nproc_per_node={args.nproc_per_node} --nnodes={args.machines} --node_rank={i} --master_addr={job.tasks[0].ip} --master_port={6006}'
            cmd = f'{env_params} python -m torch.distributed.launch {dist_params} training/train_imagenet_nv.py {training_params}'
            task.run(f'echo {cmd} > {job.logdir}/task-{i}.cmd')  # save command-line
            task.run(cmd, non_blocking=True)
    else:
        FI_PROVIDER = 'efa'
        if args.pseudo_efa:
            FI_PROVIDER = 'sockets'

        local_env = util.format_env_export(LOCAL_RANK='$OMPI_COMM_WORLD_LOCAL_RANK',
                                           RANK='$OMPI_COMM_WORLD_RANK',
                                           WORLD_SIZE='$OMPI_COMM_WORLD_SIZE',
                                           MASTER_ADDR=task0.ip,
                                           MASTER_PORT=6016)

        mpi_env = util.format_env_x(FI_PROVIDER=FI_PROVIDER,        # Enables running nccl-tests using EFA provider.
                                    FI_OFI_RXR_RX_COPY_UNEXP=1,     #  Disables using bounce buffers for unexpected messages.
                                    FI_OFI_RXR_RX_COPY_OOO=1,       # Disables using bounce buffers for out of order messages.
                                    FI_EFA_MR_CACHE_ENABLE=1,       # Enables memory region caching.
                                    FI_OFI_RXR_INLINE_MR_ENABLE=1,  # Enables inline memory registration of data buffers.
                                    NCCL_TREE_THRESHOLD=10 * 4294967296,  # force tree for everything under 40GB
                                    LD_LIBRARY_PATH=f'{CUDA_HOME}/lib:{CUDA_HOME}/lib64:{EFA_HOME}/lib64',
                                    NCCL_DEBUG='INFO',
                                    OMP_NUM_THREADS=1,
                                    WANDB_API_KEY=os.environ.get('WANDB_API_KEY', ''),
                                    PYTORCH_USE_SPAWN=args.pytorch_use_spawn,
                                    NO_WANDB=args.pytorch_use_spawn,
                                    )
        if args.no_op:
            worker_script_fn = 'training/env_test.py'
        else:
            worker_script_fn = 'training/train_imagenet_nv.py'

        local_cmd = [f"{local_env} && source ~/.bashrc && conda activate {args.conda_env} && ",
                     f'python {worker_script_fn} {training_params} --local_rank=$OMPI_COMM_WORLD_LOCAL_RANK']
        local_cmd = ' '.join(local_cmd)

        cmd = [f"{MPI_HOME}/bin/mpirun -n {NUM_GPUS} -N {NPROC_PER_NODE} --hostfile {HOSTS_SLOTS_FN} ",
               f'{mpi_env} ',
               f'--mca btl tcp,self --mca btl_tcp_if_exclude lo,docker0 ',
               f'--bind-to none ',
               f"bash -c '{local_cmd}'"]
        cmd = ' '.join(cmd)

        task0.run(cmd, non_blocking=True)

    print(f"Logging to {job.logdir}")


if __name__ == '__main__':
    main()
