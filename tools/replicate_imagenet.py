#!/usr/bin/env python

# Downloads imagenet and replicates it across multiple disks
#
# Script to initialize a set of high-performance volumes with ImageNet data
# 
# replicate_imagenet.py --replicas 8
# replicate_imagenet.py --replicas 8 --volume-offset=8
#
# or
#
# replicate_imagenet.py --replicas 16 --zone=us-east-1b
# Creates volumes: imagenet_00, imagenet_01, imagenet_02, ..., imagenet_15
#
# ImageNet data should follow structure as in
# https://github.com/diux-dev/cluster/tree/master/pytorch#data-preparation
# (paths replace ~/data with /)
#
# steps to create snapshot:
# create blank volume (ec2.create_volume())
# attach it to an existing instance with ImageNet under data, then
# sudo mkfs -t ext4 /dev/xvdf
# mkdir data
# sudo mount /dev/xvdf data
# sudo chown data `whoami`
# cp -R data0 data
# snapshot = ec2.create_snapshot(Description=f'{u.get_name(vol)} snapshot',
# VolumeId=vol.id,)

import os
from ncluster import aws_util as u

import argparse

parser = argparse.ArgumentParser(description='launch')
parser.add_argument('--replicas', type=int, default=8)
parser.add_argument('--snapshot', type=str, default='imagenet18')
parser.add_argument('--snapshot-account', type=str, default='316880547378',
                    help='account id hosting this snapshot')
parser.add_argument('--volume-offset', type=int, default=0, help='start numbering with this value')
#parser.add_argument('--zone', type=str, default='us-east-1b', help='zone to use for snapshots')
parser.add_argument('--size_gb', type=int, default=0, help="size in GBs")

args = parser.parse_args()


def create_tags(name):
    return [{
        'ResourceType': 'volume',
        'Tags': [{
            'Key': 'Name',
            'Value': name
        }]
    }]


def main():
    ec2 = u.get_ec2_resource()
    zone = u.get_zone()

    # use filtering by description since Name is not public
    snapshots = list(ec2.snapshots.filter(Filters=[{'Name': 'description', 'Values': [args.snapshot]},
                                                   {'Name': 'owner-id', 'Values': [args.snapshot_account]}]))

    assert len(snapshots) > 0, f"no snapshot matching {args.snapshot}"
    assert len(snapshots) < 2, f"multiple snapshots matching {args.snapshot}"
    snap = snapshots[0]
    if not args.size_gb:
        args.size_gb = snap.volume_size

    print(f"Making {args.replicas} {args.size_gb} GB replicas in {zone}")

    for i in range(args.volume_offset, args.replicas + args.volume_offset):
        vol_name = 'imagenet_%02d' % (i)

        vol = ec2.create_volume(Size=args.size_gb,
                                TagSpecifications=create_tags(vol_name),
                                AvailabilityZone=zone,
                                SnapshotId=snap.id)
        print(f"Creating {vol_name} {vol.id}")


if __name__ == '__main__':
    main()
