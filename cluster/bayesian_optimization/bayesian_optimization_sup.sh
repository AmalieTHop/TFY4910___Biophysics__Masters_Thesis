module purge
module load Anaconda3/2022.10







sbatch --partition=CPUQ --account=nv-fys --time=120:00:00 --nodes=1 --cpus-per-task=1 --mem=5000 --job-name="BO_sup_b4_nmae_snr20" --output=../../../simulations/simulations_out/b4/BO_sup_valMSnmae_snr20.out --mail-user=amalieth@stud.ntnu.no --mail-type=ALL --wrap="python BO_sup_b4_valMSnmae.py --snr 20"
sbatch --partition=CPUQ --account=nv-fys --time=120:00:00 --nodes=1 --cpus-per-task=1 --mem=5000 --job-name="BO_sup_b5_nmae_snr20" --output=../../../simulations/simulations_out/b5/BO_sup_valMSnmae_snr20.out --mail-user=amalieth@stud.ntnu.no --mail-type=ALL --wrap="python BO_sup_b5_valMSnmae.py --snr 20"
sbatch --partition=CPUQ --account=nv-fys --time=120:00:00 --nodes=1 --cpus-per-task=1 --mem=5000 --job-name="BO_sup_b11_nmae_snr20" --output=../../../simulations/simulations_out/b11/BO_sup_valMSnmae_snr20.out --mail-user=amalieth@stud.ntnu.no --mail-type=ALL --wrap="python BO_sup_b11_valMSnmae.py --snr 20"

