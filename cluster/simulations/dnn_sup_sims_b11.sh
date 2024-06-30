#!/bin/sh
module purge
module load Anaconda3/2022.10






sbatch --partition=CPUQ --account=nv-fys --time=24:00:00 --nodes=1 --cpus-per-task=1 --mem=5000 --job-name="sup_b11_snr8" --output=../../../simulations/txtoutput/b11/sup_snr8.out --wrap="python dnn_sup_sims_b11.py --snr 8"
sbatch --partition=CPUQ --account=nv-fys --time=24:00:00 --nodes=1 --cpus-per-task=1 --mem=5000 --job-name="sup_b11_snr10" --output=../../../simulations/txtoutput/b11/sup_snr10.out --wrap="python dnn_sup_sims_b11.py --snr 10"
sbatch --partition=CPUQ --account=nv-fys --time=24:00:00 --nodes=1 --cpus-per-task=1 --mem=5000 --job-name="sup_b11_snr12" --output=../../../simulations/txtoutput/b11/sup_snr12.out --wrap="python dnn_sup_sims_b11.py --snr 12"
sbatch --partition=CPUQ --account=nv-fys --time=24:00:00 --nodes=1 --cpus-per-task=1 --mem=5000 --job-name="sup_b11_snr15" --output=../../../simulations/txtoutput/b11/sup_snr15.out --wrap="python dnn_sup_sims_b11.py --snr 15"
sbatch --partition=CPUQ --account=nv-fys --time=24:00:00 --nodes=1 --cpus-per-task=1 --mem=5000 --job-name="sup_b11_snr20" --output=../../../simulations/txtoutput/b11/sup_snr20.out --wrap="python dnn_sup_sims_b11.py --snr 20"
sbatch --partition=CPUQ --account=nv-fys --time=24:00:00 --nodes=1 --cpus-per-task=1 --mem=5000 --job-name="sup_b11_snr25" --output=../../../simulations/txtoutput/b11/sup_snr25.out --wrap="python dnn_sup_sims_b11.py --snr 25"
sbatch --partition=CPUQ --account=nv-fys --time=24:00:00 --nodes=1 --cpus-per-task=1 --mem=5000 --job-name="sup_b11_snr35" --output=../../../simulations/txtoutput/b11/sup_snr35.out --wrap="python dnn_sup_sims_b11.py --snr 35"
sbatch --partition=CPUQ --account=nv-fys --time=24:00:00 --nodes=1 --cpus-per-task=1 --mem=5000 --job-name="sup_b11_snrf50" --output=../../../simulations/txtoutput/b11/sup_snr50.out --wrap="python dnn_sup_sims_b11.py --snr 50"
sbatch --partition=CPUQ --account=nv-fys --time=24:00:00 --nodes=1 --cpus-per-task=1 --mem=5000 --job-name="sup_b11_snr75" --output=../../../simulations/txtoutput/b11/sup_snr75.out --wrap="python dnn_sup_sims_b11.py --snr 75"
sbatch --partition=CPUQ --account=nv-fys --time=24:00:00 --nodes=1 --cpus-per-task=1 --mem=5000 --job-name="sup_b11_snr100" --output=../../../simulations/txtoutput/b11/sup_snr100.out --wrap="python dnn_sup_sims_b11.py --snr 100"


