#!/bin/sh
module purge
module load Anaconda3/2022.10


sbatch --partition=CPUQ --account=nv-fys --time=02:00:00 --nodes=1 --cpus-per-task=1 --mem=3000 --job-name="fit8" --output=../../../simulations/simulations_out/4b/fit_snr8.out --mail-user=amalieth@stud.ntnu.no --mail-type=ALL --wrap="python fit_sims_b4.py --snr 8"
sbatch --partition=CPUQ --account=nv-fys --time=02:00:00 --nodes=1 --cpus-per-task=1 --mem=3000 --job-name="fit0" --output=../../../simulations/simulations_out/4b/fit_snr10.out --mail-user=amalieth@stud.ntnu.no --mail-type=ALL --wrap="python fit_sims_b4.py --snr 10"
sbatch --partition=CPUQ --account=nv-fys --time=02:00:00 --nodes=1 --cpus-per-task=1 --mem=3000 --job-name="fit12" --output=../../../simulations/simulations_out/4b/fit_snr12.out --mail-user=amalieth@stud.ntnu.no --mail-type=ALL --wrap="python fit_sims_b4.py --snr 12"
sbatch --partition=CPUQ --account=nv-fys --time=02:00:00 --nodes=1 --cpus-per-task=1 --mem=3000 --job-name="fit15" --output=../../../simulations/simulations_out/4b/fit_snr15.out --mail-user=amalieth@stud.ntnu.no --mail-type=ALL --wrap="python fit_sims_b4.py --snr 15"
sbatch --partition=CPUQ --account=nv-fys --time=02:00:00 --nodes=1 --cpus-per-task=1 --mem=3000 --job-name="fit20" --output=../../../simulations/simulations_out/4b/fit_snr20.out --mail-user=amalieth@stud.ntnu.no --mail-type=ALL --wrap="python fit_sims_b4.py --snr 20"
sbatch --partition=CPUQ --account=nv-fys --time=02:00:00 --nodes=1 --cpus-per-task=1 --mem=3000 --job-name="fit25" --output=../../../simulations/simulations_out/4b/fit_snr25.out --mail-user=amalieth@stud.ntnu.no --mail-type=ALL --wrap="python fit_sims_b4.py --snr 25"
sbatch --partition=CPUQ --account=nv-fys --time=02:00:00 --nodes=1 --cpus-per-task=1 --mem=3000 --job-name="fit33" --output=../../../simulations/simulations_out/4b/fit_snr33.out --mail-user=amalieth@stud.ntnu.no --mail-type=ALL --wrap="python fit_sims_b4.py --snr 35"
sbatch --partition=CPUQ --account=nv-fys --time=02:00:00 --nodes=1 --cpus-per-task=1 --mem=3000 --job-name="fit50" --output=../../../simulations/simulations_out/4b/fit_snr50.out --mail-user=amalieth@stud.ntnu.no --mail-type=ALL --wrap="python fit_sims_b4.py --snr 50"
sbatch --partition=CPUQ --account=nv-fys --time=02:00:00 --nodes=1 --cpus-per-task=1 --mem=3000 --job-name="fit75" --output=../../../simulations/simulations_out/4b/fit_snr75.out --mail-user=amalieth@stud.ntnu.no --mail-type=ALL --wrap="python fit_sims_b4.py --snr 75"
sbatch --partition=CPUQ --account=nv-fys --time=02:00:00 --nodes=1 --cpus-per-task=1 --mem=3000 --job-name="fit100" --output=../../../simulations/simulations_out/4b/fit_snr100.out --mail-user=amalieth@stud.ntnu.no --mail-type=ALL --wrap="python fit_sims_b4.py --snr 100"
