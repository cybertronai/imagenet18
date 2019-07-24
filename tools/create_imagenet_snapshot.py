"""
ncluster launch --instance_type=r5.16xlarge --name=imagenet-prep
ncluster connect imagenet-prep

# download dataset
cd ~/
wget https://s3.amazonaws.com/yaroslavvb2/data/imagenet18.tar

# create volume and attach it
source activate pytorch_p36

< add AWS credentials >

pip install ec2-metadata ncluster
python
from ncluster import aws_util as u
from ec2_metadata import ec2_metadata
ec2 = u.get_ec2_resource()

def create_tags(name):
    return [{
        'ResourceType': 'volume',
        'Tags': [{
            'Key': 'Name',
            'Value': name
        }]
    }]

vol = ec2.create_volume(Size=400, TagSpecifications=create_tags('imagenet18'), AvailabilityZone=ec2_metadata.availability_zone, VolumeType='gp2')
vol = ec2.Volume('vol-0fd20d716517c942d')

instance = ec2.Instance(ec2_metadata.instance_id)
device_name = '/dev/xvdh'  # or /dev/nvme1n1
instance.attach_volume(Device=device_name, VolumeId=vol.id)
vol.reload()
assert ec2_metadata.instance_id in str(vol.attachments)

# https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/ebs-using-volumes.html
lsblk  # get name of device (look for one with 300 size like nvme1n1, then dev name is /dev/nvme1n1)
sudo file -s /dev/nvme1n1   

sudo mkfs -t ext4 /dev/nvme1n1
sudo umount data || echo skipping
sudo mkdir -p /data
sudo chown `whoami` /data
sudo mount /dev/nvme1n1 /data


cd /data
tar xf ~/imagenet18.tar --strip 1


# create snapshot
snapshot = ec2.create_snapshot(Description=f'{u.get_name(vol)} snapshot', VolumeId=vol.id,)
"""
