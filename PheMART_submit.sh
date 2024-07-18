#!/bin/bash
#SBATCH -c 2
#SBATCH -t 16:30:00
#SBATCH -p gpu_requeue  #gpu_requeue  gpu_quad
#SBATCH --gres=gpu:1
#SBATCH --mem=200G                           # Memory total in MB (for all cores)
#SBATCH -o test_%j.out                  # File to which STDOUT will be written, including job ID
#SBATCH -e test_%j.err                  # File to which STDERR will be written, including job ID
#SBATCH --mail-type=ALL                     # Type of email notification- BEGIN,END,FAIL,ALL
#SBATCH --mail-user=#####@hms.harvard.edu   # Email to which notifications will be sent
#module load gcc/6.2.0
#module load conda2/4.2.13
#source activate tf2
#
#module load gcc/6.2.0 cuda/10.1
#module load gcc/6.2.0 R/4.0.1
##srun --pty -p interactive -t 0-3:50 --mem 180G bash
#srun -n 2 --pty -t 3:00:00 -p gpu --mem=150G --gres=gpu:2 bash
module load gcc/6.2.0 cuda/11.2
module load conda2/4.2.13
source activate tf25

/n/cluster/bin/job_gpu_monitor.sh &

python3 PheMART_model.py --flag_reload 0 \
                    --flag_debug 1 \
                    --flag_modelsave 0 \
                    --flag_negative_filter 1 \
                    --flag_cross_gene 1 \
                    --epochs 40 \
                    --batch_size 512 \
                    --train_ratio  0.9    \
                    --tau_softmax  0.02    \
                    --weight_distill  2.5    \
                    --weight_vae  0.8 \
                    --weight_CLIP  0.8 \
                    --weight_cosine 10.0 \
                    --negative_disease  40 \
                    --negative_snps  40  \
                    --margin_same  0.1   \
                    --margin_differ  0.0 \
                    --margin_ppi  0.22 \
                    --flag_hard_negative  1 \
                    --model_savename  "model_name" \
                    --flag_save_unlabel_emb 0 \
                    --flag_save_unlabel_predict  0 \
                    --content_unlabel  "cross_validation" \
                    --dirr_results_main  "results/" \
                    --dirr_results  "results/result_test/" \
                    --filename_eval "result_test.txt"