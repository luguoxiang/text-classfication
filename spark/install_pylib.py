# Install Python libraries on running cluster nodes
import os
import time
import boto3
from sys import argv

try:
  clusterId=argv[1]
  script=argv[2]
except:
  print("Syntax: install_pylib.py [ClusterId] [S3_Script_Path]")
  import sys
  sys.exit(1)

session = boto3.Session(
            aws_access_key_id = os.environ["AWS_ACCESS_KEY_ID"].strip(),
            aws_secret_access_key = os.environ["AWS_SECRET_ACCESS_KEY"].strip(),
            region_name = os.environ["AWS_REGION_NAME"].strip())
emrclient=session.client('emr')

# Get list of core nodes
instances=emrclient.list_instances(ClusterId=clusterId,InstanceGroupTypes=['CORE'])['Instances']
instance_list=[x['Ec2InstanceId'] for x in instances]
print("Running on nodes:"+ str(instance_list))

# Attach tag to core nodes
ec2client=session.client('ec2')
ec2client.create_tags(Resources=instance_list,Tags=[{"Key":"environment","Value":"coreNodeLibs"}])

ssmclient=session.client('ssm')

def wait_command(cmd, client):
    while(True):
        cmd_status=client.list_commands(
            CommandId=first_command,
            Filters=[ {
              'key': 'Status',
              'value': 'SUCCESS'
          }])
        if len(cmd_status['Commands']) == 0:
            time.sleep(5)
            continue
        if cmd_status['Commands'][0]['Status'] == 'Success':
            break
        print(cmd_status)
        time.sleep(5)
    print("Success")

# Download shell script from S3
command = "aws s3 cp " + script + " /home/hadoop/installpylib.sh"
try:
  print(command)
  first_command=ssmclient.send_command(Targets=[{"Key":"tag:environment","Values":["coreNodeLibs"]}],
                  DocumentName='AWS-RunShellScript',
                  Parameters={"commands":[command]}, 
                  TimeoutSeconds=3600)['Command']['CommandId']
  
  wait_command(first_command, ssmclient)

  print("bash /home/hadoop/installpylib.sh")
  second_command=ssmclient.send_command(Targets=[{"Key":"tag:environment","Values":["coreNodeLibs"]}],
      DocumentName='AWS-RunShellScript',
      Parameters={"commands":["bash /home/hadoop/installpylib.sh"]}, 
      TimeoutSeconds=3600)['Command']['CommandId']
    
  wait_command(second_command, ssmclient)

except Exception as e:
  print(e)
