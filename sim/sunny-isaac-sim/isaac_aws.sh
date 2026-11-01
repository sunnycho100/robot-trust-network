#!/bin/bash
# Isaac Sim on AWS (runbook: AWS.md). Usage: bash sim/sunny-isaac-sim/isaac_aws.sh launch|ip|open|sync|ssh|stop|start|terminate [instance type]
# launch starts billing (about $2.24/hr for g6e.2xlarge); DRY=1 only checks it. Every boot schedules an auto-stop 3 hours later.
set -eu
export AWS_PROFILE=${AWS_PROFILE:-robot} AWS_REGION=us-east-2
HERE=$(cd "$(dirname "$0")" && pwd)
KEY=$HOME/.ssh/isaac-key.pem
TYPE=${2:-g6e.2xlarge}  # fallback: g6e.xlarge (32 GiB RAM, light scenes only)
ID() { aws ec2 describe-instances --filters Name=tag:Name,Values=isaac-sim \
  Name=instance-state-name,Values=pending,running,stopping,stopped \
  --query 'Reservations[].Instances[].InstanceId' --output text; }
IP() { aws ec2 describe-instances --instance-ids $(ID) --query 'Reservations[].Instances[].PublicIpAddress' --output text; }
RSYNC() { rsync -az --exclude __pycache__ -e "ssh -i $KEY -o StrictHostKeyChecking=accept-new" --rsync-path="mkdir -p robot && rsync" "$@"; }

case ${1:-} in
launch)
  [ -z "$(ID)" ] || { echo "isaac-sim already exists: $(ID). Use start."; exit 1; }
  # NVIDIA Isaac Sim Development Workstation (Linux); newest AMI of the Marketplace product
  AMI=${AMI:-$(aws ec2 describe-images --owners aws-marketplace \
    --filters Name=product-code,Values=efphly7anunyq6ibyjw1wov48 \
    --query 'sort_by(Images,&CreationDate)[-1].ImageId' --output text)}
  AUTOSTOP='#!/bin/bash
cat > /var/lib/cloud/scripts/per-boot/autostop.sh <<X
#!/bin/bash
shutdown -h +180 "Isaac Sim auto-stop in 3 hours. Cancel: sudo shutdown -c"
X
chmod +x /var/lib/cloud/scripts/per-boot/autostop.sh
/var/lib/cloud/scripts/per-boot/autostop.sh'
  aws ec2 run-instances ${DRY:+--dry-run} --image-id "$AMI" --instance-type "$TYPE" --count 1 \
    --key-name isaac-key --security-groups isaac-sim --iam-instance-profile Name=isaac-sim-dcv \
    --block-device-mappings 'DeviceName=/dev/sda1,Ebs={VolumeSize=512,VolumeType=gp3,Encrypted=true,DeleteOnTermination=true}' \
    --metadata-options HttpTokens=required,HttpEndpoint=enabled \
    --instance-initiated-shutdown-behavior stop --user-data "$AUTOSTOP" \
    --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=isaac-sim},{Key=Project,Value=isaac-sim}]' \
                         'ResourceType=volume,Tags=[{Key=Name,Value=isaac-sim},{Key=Project,Value=isaac-sim}]' \
    --query 'Instances[0].InstanceId' --output text
  ;;
ip) aws ec2 describe-instances --instance-ids $(ID) \
      --query 'Reservations[].Instances[].[InstanceId,State.Name,PublicIpAddress]' --output text ;;
open)  # SSH, DCV, and the web viewer and WebRTC ports isaac_up.sh uses, from this Mac's current IP only
  ME=$(curl -s https://checkip.amazonaws.com)
  for pp in tcp:22 tcp:8443 tcp:8210 tcp:49100 udp:47998; do
    out=$(aws ec2 authorize-security-group-ingress --group-name isaac-sim --protocol ${pp%:*} --port ${pp#*:} \
      --cidr "$ME/32" 2>&1) && echo "opened $pp to $ME/32" ||
      { grep -q Duplicate <<<"$out" && echo "$pp already open to $ME/32" || { echo "$out"; exit 1; }; }
  done ;;
sync)  # the paths the Brev scripts expect: this folder at ~/robot/isaac-sim, the factory port at ~/robot/g1coop
  RSYNC "$HERE/" "ubuntu@$(IP):robot/isaac-sim/"
  RSYNC "$HERE/g1coop_port/" "ubuntu@$(IP):robot/g1coop/" ;;
ssh) shift; ssh -i "$KEY" -o StrictHostKeyChecking=accept-new "ubuntu@$(IP)" "$@" ;;
stop) aws ec2 stop-instances --instance-ids $(ID) --query 'StoppingInstances[].CurrentState.Name' --output text ;;
start) aws ec2 start-instances --instance-ids $(ID) --query 'StartingInstances[].CurrentState.Name' --output text ;;
terminate) aws ec2 terminate-instances --instance-ids $(ID) --query 'TerminatingInstances[].CurrentState.Name' --output text ;;
*) sed -n 2,3p "$0"; exit 1 ;;
esac
