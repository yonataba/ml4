#!/bin/bash
### sbatch config parameters must start with #SBATCH and must precede any other command. To ignore just add another # - like ##SBATCH

#SBATCH --partition main ### partition name where to run a job. Use ‘main’ unless qos is required. qos partitions ‘rtx3090’ ‘rtx2080’ ‘gtx1080’
#SBATCH --time 7-00:00:00 ### limit the time of job running. Make sure it is not greater than the partition time limit (7 days)!! Format: D-H:MM:SS
#SBATCH --job-name yolo5 ### name of the job. Replace my_job with your desired job name
#SBATCH --output /sise/home/yonataba/outputs/my_job-id-%J.out ### output log for running job - %J is the job number variable
#SBATCH --mail-user=yonataba@post.bgu.ac.il ### user’s email for sending job status notifications
#SBATCH --mail-type=ALL ### conditions for sending the email. Options: ALL, BEGIN, END, FAIL, REQUEUE, NONE
#SBATCH --gpus=rtx_6000:1 ### number of GPUs. E.g., #SBATCH --gpus=gtx_1080:1, rtx_2080, rtx_3090. Allocating more than 1 requires the IT team’s permission

##SBATCH --tasks=6 # 10process – use for processing of few programs concurrently in a job (with srun). Use just 1 otherwise

### Print some data to output file ###
echo "SLURM_JOBID"=$SLURM_JOBID
echo "SLURM_JOB_NODELIST"=$SLURM_JOB_NODELIST
echo $CUDA_VISIBLE_DEVICES
### Start your code below #### 
python3 ../yolov5/classify/train.py --model yolov5l-cls.pt --data /sise/home/yonataba/ml4/output --epochs 100 --batch-size 128 --img-size 224


