#!/bin/bash

PARTITION=$1
echo $1
shift

DATASET=$1
echo $1
shift

ABLATIONTYPE=$1
echo $1
shift

if [ $PARTITION == "gpu" ] | [ $PARTITION == "gpu_test" ]
then
CONSTRAINT=""
else
CONSTRAINT='#SBATCH --constraint="a100"'
fi

for var in "$@"
do
sbatch <<EOT
#!/bin/bash
#SBATCH -c 1
#SBATCH -p $PARTITION
$CONSTRAINT
#SBATCH --job-name=$var-acdc
#SBATCH --gpus 1
#SBATCH --mem=32000
#SBATCH -t 0-12:00
#SBATCH -o prog_files/$DATASET-$ABLATIONTYPE-$var-ioi-%j.out  # File to which STDOUT will be written, %j inserts jobid
#SBATCH -e prog_files/$DATASET-$ABLATIONTYPE-$var-ioi-%j.err  # File to which STDERR will be written, %j inserts jobid

# Your commands here
module load Anaconda2
conda activate take2
python3 main.py --task=$DATASET \
--threshold=$var \
--indices-mode=reverse \
--first-cache-cpu=False \
--second-cache-cpu=False \
--max-num-epochs=400000 \
$ABLATIONTYPE

EOT
done