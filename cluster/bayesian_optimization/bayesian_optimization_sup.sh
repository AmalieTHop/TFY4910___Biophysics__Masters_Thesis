module purge
module load Anaconda3/2022.10







sbatch --partition=GPUQ --account=nv-fys --time=120:00:00 --nodes=1 --cpus-per-task=1 --mem=5000 --job-name="BO_sup_b4_nmae_snr20" --output=../../../simulations/simulations_out/b4/BO_sup_valHTnmae_snr20.out --wrap="python BO_sup_b4_valHTnmae.py --snr 20"
sbatch --partition=GPUQ --account=nv-fys --time=120:00:00 --nodes=1 --cpus-per-task=1 --mem=5000 --job-name="BO_sup_b5_nmae_snr20" --output=../../../simulations/simulations_out/b5/BO_sup_valHTnmae_snr20.out --wrap="python BO_sup_b5_valHTnmae.py --snr 20"
sbatch --partition=GPUQ --account=nv-fys --time=120:00:00 --nodes=1 --cpus-per-task=1 --mem=5000 --job-name="BO_sup_b11_nmae_snr20" --output=../../../simulations/simulations_out/b11/BO_sup_valHTnmae_snr20.out --wrap="python BO_sup_b11_valHTnmae.py --snr 20"

sbatch --partition=CPUQ --account=nv-fys --time=120:00:00 --nodes=1 --cpus-per-task=1 --mem=5000 --job-name="BO_sup_b4_nrmse_snr20" --output=../../../simulations/simulations_out/b4/BO_sup_valHTnrmse_snr20.out --wrap="python BO_sup_b4_valHTnrmse.py --snr 20"
sbatch --partition=CPUQ --account=nv-fys --time=120:00:00 --nodes=1 --cpus-per-task=1 --mem=5000 --job-name="BO_sup_b5_nrmse_snr20" --output=../../../simulations/simulations_out/b5/BO_sup_valHTnrmse_snr20.out --wrap="python BO_sup_b5_valHTnrmse.py --snr 20"
sbatch --partition=CPUQ --account=nv-fys --time=120:00:00 --nodes=1 --cpus-per-task=1 --mem=5000 --job-name="BO_sup_b11_nrmse_snr20" --output=../../../simulations/simulations_out/b11/BO_sup_valHTnrmse_snr20.out --wrap="python BO_sup_b11_valHTnrmse.py --snr 20"



sbatch --partition=CPUQ --account=nv-fys --time=120:00:00 --nodes=1 --cpus-per-task=1 --mem=5000 --job-name="BO_sup_b4_nmae_snr25" --output=../../../simulations/simulations_out/b4/BO_sup_valHTnmae_snr35.out --wrap="python BO_sup_b4_valHTnmae.py --snr 25"
sbatch --partition=CPUQ --account=nv-fys --time=120:00:00 --nodes=1 --cpus-per-task=1 --mem=5000 --job-name="BO_sup_b5_nmae_snr25" --output=../../../simulations/simulations_out/b5/BO_sup_valHTnmae_snr35.out --wrap="python BO_sup_b5_valHTnmae.py --snr 25"
sbatch --partition=CPUQ --account=nv-fys --time=120:00:00 --nodes=1 --cpus-per-task=1 --mem=5000 --job-name="BO_sup_b11_nmae_snr25" --output=../../../simulations/simulations_out/b11/BO_sup_valHTnmae_snr35.out --wrap="python BO_sup_b11_valHTnmae.py --snr 25"

sbatch --partition=CPUQ --account=nv-fys --time=120:00:00 --nodes=1 --cpus-per-task=1 --mem=5000 --job-name="BO_sup_b4_nrmse_snr25" --output=../../../simulations/simulations_out/b4/BO_sup_valHTnrmse_snr35.out --wrap="python BO_sup_b4_valHTnrmse.py --snr 25"
sbatch --partition=CPUQ --account=nv-fys --time=120:00:00 --nodes=1 --cpus-per-task=1 --mem=5000 --job-name="BO_sup_b5_nrmse_snr25" --output=../../../simulations/simulations_out/b5/BO_sup_valHTnrmse_snr35.out --wrap="python BO_sup_b5_valHTnrmse.py --snr 25"
sbatch --partition=CPUQ --account=nv-fys --time=120:00:00 --nodes=1 --cpus-per-task=1 --mem=5000 --job-name="BO_sup_b11_nrmse_snr25" --output=../../../simulations/simulations_out/b11/BO_sup_valHTnrmse_snr35.out --wrap="python BO_sup_b11_valHTnrmse.py --snr 25"



sbatch --partition=CPUQ --account=nv-fys --time=120:00:00 --nodes=1 --cpus-per-task=1 --mem=5000 --job-name="BO_sup_b4_nmae_snr35" --output=../../../simulations/simulations_out/b4/BO_sup_valHTnmae_snr35.out --wrap="python BO_sup_b4_valHTnmae.py --snr 35"
sbatch --partition=CPUQ --account=nv-fys --time=120:00:00 --nodes=1 --cpus-per-task=1 --mem=5000 --job-name="BO_sup_b5_nmae_snr35" --output=../../../simulations/simulations_out/b5/BO_sup_valHTnmae_snr35.out --wrap="python BO_sup_b5_valHTnmae.py --snr 35"
sbatch --partition=CPUQ --account=nv-fys --time=120:00:00 --nodes=1 --cpus-per-task=1 --mem=5000 --job-name="BO_sup_b11_nmae_snr35" --output=../../../simulations/simulations_out/b11/BO_sup_valHTnmae_snr35.out --wrap="python BO_sup_b11_valHTnmae.py --snr 35"

sbatch --partition=CPUQ --account=nv-fys --time=120:00:00 --nodes=1 --cpus-per-task=1 --mem=5000 --job-name="BO_sup_b4_nrmse_snr35" --output=../../../simulations/simulations_out/b4/BO_sup_valHTnrmse_snr35.out --wrap="python BO_sup_b4_valHTnrmse.py --snr 35"
sbatch --partition=CPUQ --account=nv-fys --time=120:00:00 --nodes=1 --cpus-per-task=1 --mem=5000 --job-name="BO_sup_b5_nrmse_snr35" --output=../../../simulations/simulations_out/b5/BO_sup_valHTnrmse_snr35.out --wrap="python BO_sup_b5_valHTnrmse.py --snr 35"
sbatch --partition=CPUQ --account=nv-fys --time=120:00:00 --nodes=1 --cpus-per-task=1 --mem=5000 --job-name="BO_sup_b11_nrmse_snr35" --output=../../../simulations/simulations_out/b11/BO_sup_valHTnrmse_snr35.out --wrap="python BO_sup_b11_valHTnrmse.py --snr 35"
