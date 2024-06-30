#!/bin/sh
module purge
module load Anaconda3/2022.10

# sbatch --partition=CPUQ --account=nv-fys --time=01:00:00 --nodes=1 --ntasks-per-node=1 --mem=16000 --job-name="EMIN_1001_5b_a" --output=../../../dataA/output/txtoutput/LOO/EMIN_1001_b5_a.out --wrap="python patientsA_LOO.py --patient_id EMIN_1001 --num_bvals 5 --training_method a"
# sbatch --partition=CPUQ --account=nv-fys --time=01:00:00 --nodes=1 --ntasks-per-node=1 --mem=16000 --job-name="EMIN_1005_5b_a" --output=../../../dataA/output/txtoutput/LOO/EMIN_1005_b5_a.out --wrap="python patientsA_LOO.py --patient_id EMIN_1005 --num_bvals 5 --training_method a"
# sbatch --partition=CPUQ --account=nv-fys --time=01:00:00 --nodes=1 --ntasks-per-node=1 --mem=16000 --job-name="EMIN_1007_5b_a" --output=../../../dataA/output/txtoutput/LOO/EMIN_1007_b5_a.out --wrap="python patientsA_LOO.py --patient_id EMIN_1007 --num_bvals 5 --training_method a"
# sbatch --partition=CPUQ --account=nv-fys --time=01:00:00 --nodes=1 --ntasks-per-node=1 --mem=16000 --job-name="EMIN_1011_5b_a" --output=../../../dataA/output/txtoutput/LOO/EMIN_1011_b5_a.out --wrap="python patientsA_LOO.py --patient_id EMIN_1011 --num_bvals 5 --training_method a"
# sbatch --partition=CPUQ --account=nv-fys --time=01:00:00 --nodes=1 --ntasks-per-node=1 --mem=16000 --job-name="EMIN_1019_5b_a" --output=../../../dataA/output/txtoutput/LOO/EMIN_1019_b5_a.out --wrap="python patientsA_LOO.py --patient_id EMIN_1019 --num_bvals 5 --training_method a"
# sbatch --partition=CPUQ --account=nv-fys --time=01:00:00 --nodes=1 --ntasks-per-node=1 --mem=16000 --job-name="EMIN_1020_5b_a" --output=../../../dataA/output/txtoutput/LOO/EMIN_1020_b5_a.out --wrap="python patientsA_LOO.py --patient_id EMIN_1020 --num_bvals 5 --training_method a"
# sbatch --partition=CPUQ --account=nv-fys --time=01:00:00 --nodes=1 --ntasks-per-node=1 --mem=16000 --job-name="EMIN_1022_5b_a" --output=../../../dataA/output/txtoutput/LOO/EMIN_1022_b5_a.out --wrap="python patientsA_LOO.py --patient_id EMIN_1022 --num_bvals 5 --training_method a"
# sbatch --partition=CPUQ --account=nv-fys --time=01:00:00 --nodes=1 --ntasks-per-node=1 --mem=16000 --job-name="EMIN_1032_5b_a" --output=../../../dataA/output/txtoutput/LOO/EMIN_1032_b5_a.out --wrap="python patientsA_LOO.py --patient_id EMIN_1032 --num_bvals 5 --training_method a"
# sbatch --partition=CPUQ --account=nv-fys --time=01:00:00 --nodes=1 --ntasks-per-node=1 --mem=16000 --job-name="EMIN_1038_5b_a" --output=../../../dataA/output/txtoutput/LOO/EMIN_1038_b5_a.out --wrap="python patientsA_LOO.py --patient_id EMIN_1038 --num_bvals 5 --training_method a"
# sbatch --partition=CPUQ --account=nv-fys --time=01:00:00 --nodes=1 --ntasks-per-node=1 --mem=16000 --job-name="EMIN_1042_5b_a" --output=../../../dataA/output/txtoutput/LOO/EMIN_1042_b5_a.out --wrap="python patientsA_LOO.py --patient_id EMIN_1042 --num_bvals 5 --training_method a"
# sbatch --partition=CPUQ --account=nv-fys --time=01:00:00 --nodes=1 --ntasks-per-node=1 --mem=16000 --job-name="EMIN_1044_5b_a" --output=../../../dataA/output/txtoutput/LOO/EMIN_1044_b5_a.out --wrap="python patientsA_LOO.py --patient_id EMIN_1044 --num_bvals 5 --training_method a"
# sbatch --partition=CPUQ --account=nv-fys --time=01:00:00 --nodes=1 --ntasks-per-node=1 --mem=16000 --job-name="EMIN_1045_5b_a" --output=../../../dataA/output/txtoutput/LOO/EMIN_1045_b5_a.out --wrap="python patientsA_LOO.py --patient_id EMIN_1045 --num_bvals 5 --training_method a"
# sbatch --partition=CPUQ --account=nv-fys --time=01:00:00 --nodes=1 --ntasks-per-node=1 --mem=16000 --job-name="EMIN_1048_5b_a" --output=../../../dataA/output/txtoutput/LOO/EMIN_1048_b5_a.out --wrap="python patientsA_LOO.py --patient_id EMIN_1048 --num_bvals 5 --training_method a"
# sbatch --partition=CPUQ --account=nv-fys --time=01:00:00 --nodes=1 --ntasks-per-node=1 --mem=16000 --job-name="EMIN_1055_5b_a" --output=../../../dataA/output/txtoutput/LOO/EMIN_1055_b5_a.out --wrap="python patientsA_LOO.py --patient_id EMIN_1055 --num_bvals 5 --training_method a"
# sbatch --partition=CPUQ --account=nv-fys --time=01:00:00 --nodes=1 --ntasks-per-node=1 --mem=16000 --job-name="EMIN_1057_5b_a" --output=../../../dataA/output/txtoutput/LOO/EMIN_1057_b5_a.out --wrap="python patientsA_LOO.py --patient_id EMIN_1057 --num_bvals 5 --training_method a"
# sbatch --partition=CPUQ --account=nv-fys --time=01:00:00 --nodes=1 --ntasks-per-node=1 --mem=16000 --job-name="EMIN_1060_5b_a" --output=../../../dataA/output/txtoutput/LOO/EMIN_1060_b5_a.out --wrap="python patientsA_LOO.py --patient_id EMIN_1060 --num_bvals 5 --training_method a"
# sbatch --partition=CPUQ --account=nv-fys --time=01:00:00 --nodes=1 --ntasks-per-node=1 --mem=16000 --job-name="EMIN_1064_5b_a" --output=../../../dataA/output/txtoutput/LOO/EMIN_1064_b5_a.out --wrap="python patientsA_LOO.py --patient_id EMIN_1064 --num_bvals 5 --training_method a"
# sbatch --partition=CPUQ --account=nv-fys --time=01:00:00 --nodes=1 --ntasks-per-node=1 --mem=16000 --job-name="EMIN_1066_5b_a" --output=../../../dataA/output/txtoutput/LOO/EMIN_1066_b5_a.out --wrap="python patientsA_LOO.py --patient_id EMIN_1066 --num_bvals 5 --training_method a"
# sbatch --partition=CPUQ --account=nv-fys --time=01:00:00 --nodes=1 --ntasks-per-node=1 --mem=16000 --job-name="EMIN_1068_5b_a" --output=../../../dataA/output/txtoutput/LOO/EMIN_1068_b5_a.out --wrap="python patientsA_LOO.py --patient_id EMIN_1068 --num_bvals 5 --training_method a"
# sbatch --partition=CPUQ --account=nv-fys --time=01:00:00 --nodes=1 --ntasks-per-node=1 --mem=16000 --job-name="EMIN_1075_5b_a" --output=../../../dataA/output/txtoutput/LOO/EMIN_1075_b5_a.out --wrap="python patientsA_LOO.py --patient_id EMIN_1075 --num_bvals 5 --training_method a"
# sbatch --partition=CPUQ --account=nv-fys --time=01:00:00 --nodes=1 --ntasks-per-node=1 --mem=16000 --job-name="EMIN_1077_5b_a" --output=../../../dataA/output/txtoutput/LOO/EMIN_1077_b5_a.out --wrap="python patientsA_LOO.py --patient_id EMIN_1077 --num_bvals 5 --training_method a"
# sbatch --partition=CPUQ --account=nv-fys --time=01:00:00 --nodes=1 --ntasks-per-node=1 --mem=16000 --job-name="EMIN_1079_5b_a" --output=../../../dataA/output/txtoutput/LOO/EMIN_1079_b5_a.out --wrap="python patientsA_LOO.py --patient_id EMIN_1079 --num_bvals 5 --training_method a"
# sbatch --partition=CPUQ --account=nv-fys --time=01:00:00 --nodes=1 --ntasks-per-node=1 --mem=16000 --job-name="EMIN_1081_5b_a" --output=../../../dataA/output/txtoutput/LOO/EMIN_1081_b5_a.out --wrap="python patientsA_LOO.py --patient_id EMIN_1081 --num_bvals 5 --training_method a"
# sbatch --partition=CPUQ --account=nv-fys --time=01:00:00 --nodes=1 --ntasks-per-node=1 --mem=16000 --job-name="EMIN_1084_5b_a" --output=../../../dataA/output/txtoutput/LOO/EMIN_1084_b5_a.out --wrap="python patientsA_LOO.py --patient_id EMIN_1084 --num_bvals 5 --training_method a"
# sbatch --partition=CPUQ --account=nv-fys --time=01:00:00 --nodes=1 --ntasks-per-node=1 --mem=16000 --job-name="EMIN_1086_5b_a" --output=../../../dataA/output/txtoutput/LOO/EMIN_1086_b5_a.out --wrap="python patientsA_LOO.py --patient_id EMIN_1086 --num_bvals 5 --training_method a"
# sbatch --partition=CPUQ --account=nv-fys --time=01:00:00 --nodes=1 --ntasks-per-node=1 --mem=16000 --job-name="EMIN_1090_5b_a" --output=../../../dataA/output/txtoutput/LOO/EMIN_1090_b5_a.out --wrap="python patientsA_LOO.py --patient_id EMIN_1090 --num_bvals 5 --training_method a"
# sbatch --partition=CPUQ --account=nv-fys --time=01:00:00 --nodes=1 --ntasks-per-node=1 --mem=16000 --job-name="EMIN_1092_5b_a" --output=../../../dataA/output/txtoutput/LOO/EMIN_1092_b5_a.out --wrap="python patientsA_LOO.py --patient_id EMIN_1092 --num_bvals 5 --training_method a"
# sbatch --partition=CPUQ --account=nv-fys --time=01:00:00 --nodes=1 --ntasks-per-node=1 --mem=16000 --job-name="EMIN_1093_5b_a" --output=../../../dataA/output/txtoutput/LOO/EMIN_1093_b5_a.out --wrap="python patientsA_LOO.py --patient_id EMIN_1093 --num_bvals 5 --training_method a"
# sbatch --partition=CPUQ --account=nv-fys --time=01:00:00 --nodes=1 --ntasks-per-node=1 --mem=16000 --job-name="EMIN_1096_5b_a" --output=../../../dataA/output/txtoutput/LOO/EMIN_1096_b5_a.out --wrap="python patientsA_LOO.py --patient_id EMIN_1096 --num_bvals 5 --training_method a"
# sbatch --partition=CPUQ --account=nv-fys --time=01:00:00 --nodes=1 --ntasks-per-node=1 --mem=16000 --job-name="EMIN_1097_5b_a" --output=../../../dataA/output/txtoutput/LOO/EMIN_1097_b5_a.out --wrap="python patientsA_LOO.py --patient_id EMIN_1097 --num_bvals 5 --training_method a"
# sbatch --partition=CPUQ --account=nv-fys --time=01:00:00 --nodes=1 --ntasks-per-node=1 --mem=16000 --job-name="EMIN_1099_5b_a" --output=../../../dataA/output/txtoutput/LOO/EMIN_1099_b5_a.out --wrap="python patientsA_LOO.py --patient_id EMIN_1099 --num_bvals 5 --training_method a"



# sbatch --partition=CPUQ --account=nv-fys --time=01:00:00 --nodes=1 --ntasks-per-node=1 --mem=16000 --job-name="EMIN_1001_5b_b" --output=../../../dataA/output/txtoutput/LOO/EMIN_1001_b5_b.out --wrap="python patientsA_LOO.py --patient_id EMIN_1001 --num_bvals 5 --training_method b"
# sbatch --partition=CPUQ --account=nv-fys --time=01:00:00 --nodes=1 --ntasks-per-node=1 --mem=16000 --job-name="EMIN_1005_5b_b" --output=../../../dataA/output/txtoutput/LOO/EMIN_1005_b5_b.out --wrap="python patientsA_LOO.py --patient_id EMIN_1005 --num_bvals 5 --training_method b"
# sbatch --partition=CPUQ --account=nv-fys --time=01:00:00 --nodes=1 --ntasks-per-node=1 --mem=16000 --job-name="EMIN_1007_5b_b" --output=../../../dataA/output/txtoutput/LOO/EMIN_1007_b5_b.out --wrap="python patientsA_LOO.py --patient_id EMIN_1007 --num_bvals 5 --training_method b"
# sbatch --partition=CPUQ --account=nv-fys --time=01:00:00 --nodes=1 --ntasks-per-node=1 --mem=16000 --job-name="EMIN_1011_5b_b" --output=../../../dataA/output/txtoutput/LOO/EMIN_1011_b5_b.out --wrap="python patientsA_LOO.py --patient_id EMIN_1011 --num_bvals 5 --training_method b"
# sbatch --partition=CPUQ --account=nv-fys --time=01:00:00 --nodes=1 --ntasks-per-node=1 --mem=16000 --job-name="EMIN_1019_5b_b" --output=../../../dataA/output/txtoutput/LOO/EMIN_1019_b5_b.out --wrap="python patientsA_LOO.py --patient_id EMIN_1019 --num_bvals 5 --training_method b"
# sbatch --partition=CPUQ --account=nv-fys --time=01:00:00 --nodes=1 --ntasks-per-node=1 --mem=16000 --job-name="EMIN_1020_5b_b" --output=../../../dataA/output/txtoutput/LOO/EMIN_1020_b5_b.out --wrap="python patientsA_LOO.py --patient_id EMIN_1020 --num_bvals 5 --training_method b"
# sbatch --partition=CPUQ --account=nv-fys --time=01:00:00 --nodes=1 --ntasks-per-node=1 --mem=16000 --job-name="EMIN_1022_5b_b" --output=../../../dataA/output/txtoutput/LOO/EMIN_1022_b5_b.out --wrap="python patientsA_LOO.py --patient_id EMIN_1022 --num_bvals 5 --training_method b"
# sbatch --partition=CPUQ --account=nv-fys --time=01:00:00 --nodes=1 --ntasks-per-node=1 --mem=16000 --job-name="EMIN_1032_5b_b" --output=../../../dataA/output/txtoutput/LOO/EMIN_1032_b5_b.out --wrap="python patientsA_LOO.py --patient_id EMIN_1032 --num_bvals 5 --training_method b"
# sbatch --partition=CPUQ --account=nv-fys --time=01:00:00 --nodes=1 --ntasks-per-node=1 --mem=16000 --job-name="EMIN_1038_5b_b" --output=../../../dataA/output/txtoutput/LOO/EMIN_1038_b5_b.out --wrap="python patientsA_LOO.py --patient_id EMIN_1038 --num_bvals 5 --training_method b"
# sbatch --partition=CPUQ --account=nv-fys --time=01:00:00 --nodes=1 --ntasks-per-node=1 --mem=16000 --job-name="EMIN_1042_5b_b" --output=../../../dataA/output/txtoutput/LOO/EMIN_1042_b5_b.out --wrap="python patientsA_LOO.py --patient_id EMIN_1042 --num_bvals 5 --training_method b"
# sbatch --partition=CPUQ --account=nv-fys --time=01:00:00 --nodes=1 --ntasks-per-node=1 --mem=16000 --job-name="EMIN_1044_5b_b" --output=../../../dataA/output/txtoutput/LOO/EMIN_1044_b5_b.out --wrap="python patientsA_LOO.py --patient_id EMIN_1044 --num_bvals 5 --training_method b"
# sbatch --partition=CPUQ --account=nv-fys --time=01:00:00 --nodes=1 --ntasks-per-node=1 --mem=16000 --job-name="EMIN_1045_5b_b" --output=../../../dataA/output/txtoutput/LOO/EMIN_1045_b5_b.out --wrap="python patientsA_LOO.py --patient_id EMIN_1045 --num_bvals 5 --training_method b"
# sbatch --partition=CPUQ --account=nv-fys --time=01:00:00 --nodes=1 --ntasks-per-node=1 --mem=16000 --job-name="EMIN_1048_5b_b" --output=../../../dataA/output/txtoutput/LOO/EMIN_1048_b5_b.out --wrap="python patientsA_LOO.py --patient_id EMIN_1048 --num_bvals 5 --training_method b"
# sbatch --partition=CPUQ --account=nv-fys --time=01:00:00 --nodes=1 --ntasks-per-node=1 --mem=16000 --job-name="EMIN_1055_5b_b" --output=../../../dataA/output/txtoutput/LOO/EMIN_1055_b5_b.out --wrap="python patientsA_LOO.py --patient_id EMIN_1055 --num_bvals 5 --training_method b"
# sbatch --partition=CPUQ --account=nv-fys --time=01:00:00 --nodes=1 --ntasks-per-node=1 --mem=16000 --job-name="EMIN_1057_5b_b" --output=../../../dataA/output/txtoutput/LOO/EMIN_1057_b5_b.out --wrap="python patientsA_LOO.py --patient_id EMIN_1057 --num_bvals 5 --training_method b"
# sbatch --partition=CPUQ --account=nv-fys --time=01:00:00 --nodes=1 --ntasks-per-node=1 --mem=16000 --job-name="EMIN_1060_5b_b" --output=../../../dataA/output/txtoutput/LOO/EMIN_1060_b5_b.out --wrap="python patientsA_LOO.py --patient_id EMIN_1060 --num_bvals 5 --training_method b"
# sbatch --partition=CPUQ --account=nv-fys --time=01:00:00 --nodes=1 --ntasks-per-node=1 --mem=16000 --job-name="EMIN_1064_5b_b" --output=../../../dataA/output/txtoutput/LOO/EMIN_1064_b5_b.out --wrap="python patientsA_LOO.py --patient_id EMIN_1064 --num_bvals 5 --training_method b"
# sbatch --partition=CPUQ --account=nv-fys --time=01:00:00 --nodes=1 --ntasks-per-node=1 --mem=16000 --job-name="EMIN_1066_5b_b" --output=../../../dataA/output/txtoutput/LOO/EMIN_1066_b5_b.out --wrap="python patientsA_LOO.py --patient_id EMIN_1066 --num_bvals 5 --training_method b"
# sbatch --partition=CPUQ --account=nv-fys --time=01:00:00 --nodes=1 --ntasks-per-node=1 --mem=16000 --job-name="EMIN_1068_5b_b" --output=../../../dataA/output/txtoutput/LOO/EMIN_1068_b5_b.out --wrap="python patientsA_LOO.py --patient_id EMIN_1068 --num_bvals 5 --training_method b"
# sbatch --partition=CPUQ --account=nv-fys --time=01:00:00 --nodes=1 --ntasks-per-node=1 --mem=16000 --job-name="EMIN_1075_5b_b" --output=../../../dataA/output/txtoutput/LOO/EMIN_1075_b5_b.out --wrap="python patientsA_LOO.py --patient_id EMIN_1075 --num_bvals 5 --training_method b"
# sbatch --partition=CPUQ --account=nv-fys --time=01:00:00 --nodes=1 --ntasks-per-node=1 --mem=16000 --job-name="EMIN_1077_5b_b" --output=../../../dataA/output/txtoutput/LOO/EMIN_1077_b5_b.out --wrap="python patientsA_LOO.py --patient_id EMIN_1077 --num_bvals 5 --training_method b"
# sbatch --partition=CPUQ --account=nv-fys --time=01:00:00 --nodes=1 --ntasks-per-node=1 --mem=16000 --job-name="EMIN_1079_5b_b" --output=../../../dataA/output/txtoutput/LOO/EMIN_1079_b5_b.out --wrap="python patientsA_LOO.py --patient_id EMIN_1079 --num_bvals 5 --training_method b"
# sbatch --partition=CPUQ --account=nv-fys --time=01:00:00 --nodes=1 --ntasks-per-node=1 --mem=16000 --job-name="EMIN_1081_5b_b" --output=../../../dataA/output/txtoutput/LOO/EMIN_1081_b5_b.out --wrap="python patientsA_LOO.py --patient_id EMIN_1081 --num_bvals 5 --training_method b"
# sbatch --partition=CPUQ --account=nv-fys --time=01:00:00 --nodes=1 --ntasks-per-node=1 --mem=16000 --job-name="EMIN_1084_5b_b" --output=../../../dataA/output/txtoutput/LOO/EMIN_1084_b5_b.out --wrap="python patientsA_LOO.py --patient_id EMIN_1084 --num_bvals 5 --training_method b"
# sbatch --partition=CPUQ --account=nv-fys --time=01:00:00 --nodes=1 --ntasks-per-node=1 --mem=16000 --job-name="EMIN_1086_5b_b" --output=../../../dataA/output/txtoutput/LOO/EMIN_1086_b5_b.out --wrap="python patientsA_LOO.py --patient_id EMIN_1086 --num_bvals 5 --training_method b"
# sbatch --partition=CPUQ --account=nv-fys --time=01:00:00 --nodes=1 --ntasks-per-node=1 --mem=16000 --job-name="EMIN_1090_5b_b" --output=../../../dataA/output/txtoutput/LOO/EMIN_1090_b5_b.out --wrap="python patientsA_LOO.py --patient_id EMIN_1090 --num_bvals 5 --training_method b"
# sbatch --partition=CPUQ --account=nv-fys --time=01:00:00 --nodes=1 --ntasks-per-node=1 --mem=16000 --job-name="EMIN_1092_5b_b" --output=../../../dataA/output/txtoutput/LOO/EMIN_1092_b5_b.out --wrap="python patientsA_LOO.py --patient_id EMIN_1092 --num_bvals 5 --training_method b"
# sbatch --partition=CPUQ --account=nv-fys --time=01:00:00 --nodes=1 --ntasks-per-node=1 --mem=16000 --job-name="EMIN_1093_5b_b" --output=../../../dataA/output/txtoutput/LOO/EMIN_1093_b5_b.out --wrap="python patientsA_LOO.py --patient_id EMIN_1093 --num_bvals 5 --training_method b"
# sbatch --partition=CPUQ --account=nv-fys --time=01:00:00 --nodes=1 --ntasks-per-node=1 --mem=16000 --job-name="EMIN_1096_5b_b" --output=../../../dataA/output/txtoutput/LOO/EMIN_1096_b5_b.out --wrap="python patientsA_LOO.py --patient_id EMIN_1096 --num_bvals 5 --training_method b"
# sbatch --partition=CPUQ --account=nv-fys --time=01:00:00 --nodes=1 --ntasks-per-node=1 --mem=16000 --job-name="EMIN_1097_5b_b" --output=../../../dataA/output/txtoutput/LOO/EMIN_1097_b5_b.out --wrap="python patientsA_LOO.py --patient_id EMIN_1097 --num_bvals 5 --training_method b"
# sbatch --partition=CPUQ --account=nv-fys --time=01:00:00 --nodes=1 --ntasks-per-node=1 --mem=16000 --job-name="EMIN_1099_5b_b" --output=../../../dataA/output/txtoutput/LOO/EMIN_1099_b5_b.out --wrap="python patientsA_LOO.py --patient_id EMIN_1099 --num_bvals 5 --training_method b"






sbatch --partition=CPUQ --account=nv-fys --time=01:00:00 --nodes=1 --ntasks-per-node=1 --mem=16000 --job-name="EMIN_1001_4b_a" --output=../../../dataA/output/txtoutput/LOO/EMIN_1001_b4_a.out --wrap="python patientsA_LOO.py --patient_id EMIN_1001 --num_bvals 4 --training_method a"
sbatch --partition=CPUQ --account=nv-fys --time=01:00:00 --nodes=1 --ntasks-per-node=1 --mem=16000 --job-name="EMIN_1005_4b_a" --output=../../../dataA/output/txtoutput/LOO/EMIN_1005_b4_a.out --wrap="python patientsA_LOO.py --patient_id EMIN_1005 --num_bvals 4 --training_method a"
#sbatch --partition=CPUQ --account=nv-fys --time=01:00:00 --nodes=1 --ntasks-per-node=1 --mem=16000 --job-name="EMIN_1007_4b_a" --output=../../../dataA/output/txtoutput/LOO/EMIN_1007_b4_a.out --wrap="python patientsA_LOO.py --patient_id EMIN_1007 --num_bvals 4 --training_method a"
sbatch --partition=CPUQ --account=nv-fys --time=01:00:00 --nodes=1 --ntasks-per-node=1 --mem=16000 --job-name="EMIN_1011_4b_a" --output=../../../dataA/output/txtoutput/LOO/EMIN_1011_b4_a.out --wrap="python patientsA_LOO.py --patient_id EMIN_1011 --num_bvals 4 --training_method a"
sbatch --partition=CPUQ --account=nv-fys --time=01:00:00 --nodes=1 --ntasks-per-node=1 --mem=16000 --job-name="EMIN_1019_4b_a" --output=../../../dataA/output/txtoutput/LOO/EMIN_1019_b4_a.out --wrap="python patientsA_LOO.py --patient_id EMIN_1019 --num_bvals 4 --training_method a"
#sbatch --partition=CPUQ --account=nv-fys --time=01:00:00 --nodes=1 --ntasks-per-node=1 --mem=16000 --job-name="EMIN_1020_4b_a" --output=../../../dataA/output/txtoutput/LOO/EMIN_1020_b4_a.out --wrap="python patientsA_LOO.py --patient_id EMIN_1020 --num_bvals 4 --training_method a"
#sbatch --partition=CPUQ --account=nv-fys --time=01:00:00 --nodes=1 --ntasks-per-node=1 --mem=16000 --job-name="EMIN_1022_4b_a" --output=../../../dataA/output/txtoutput/LOO/EMIN_1022_b4_a.out --wrap="python patientsA_LOO.py --patient_id EMIN_1022 --num_bvals 4 --training_method a"
sbatch --partition=CPUQ --account=nv-fys --time=01:00:00 --nodes=1 --ntasks-per-node=1 --mem=16000 --job-name="EMIN_1032_4b_a" --output=../../../dataA/output/txtoutput/LOO/EMIN_1032_b4_a.out --wrap="python patientsA_LOO.py --patient_id EMIN_1032 --num_bvals 4 --training_method a"
sbatch --partition=CPUQ --account=nv-fys --time=01:00:00 --nodes=1 --ntasks-per-node=1 --mem=16000 --job-name="EMIN_1038_4b_a" --output=../../../dataA/output/txtoutput/LOO/EMIN_1038_b4_a.out --wrap="python patientsA_LOO.py --patient_id EMIN_1038 --num_bvals 4 --training_method a"
sbatch --partition=CPUQ --account=nv-fys --time=01:00:00 --nodes=1 --ntasks-per-node=1 --mem=16000 --job-name="EMIN_1042_4b_a" --output=../../../dataA/output/txtoutput/LOO/EMIN_1042_b4_a.out --wrap="python patientsA_LOO.py --patient_id EMIN_1042 --num_bvals 4 --training_method a"
sbatch --partition=CPUQ --account=nv-fys --time=01:00:00 --nodes=1 --ntasks-per-node=1 --mem=16000 --job-name="EMIN_1044_4b_a" --output=../../../dataA/output/txtoutput/LOO/EMIN_1044_b4_a.out --wrap="python patientsA_LOO.py --patient_id EMIN_1044 --num_bvals 4 --training_method a"
sbatch --partition=CPUQ --account=nv-fys --time=01:00:00 --nodes=1 --ntasks-per-node=1 --mem=16000 --job-name="EMIN_1045_4b_a" --output=../../../dataA/output/txtoutput/LOO/EMIN_1045_b4_a.out --wrap="python patientsA_LOO.py --patient_id EMIN_1045 --num_bvals 4 --training_method a"
#sbatch --partition=CPUQ --account=nv-fys --time=01:00:00 --nodes=1 --ntasks-per-node=1 --mem=16000 --job-name="EMIN_1048_4b_a" --output=../../../dataA/output/txtoutput/LOO/EMIN_1048_b4_a.out --wrap="python patientsA_LOO.py --patient_id EMIN_1048 --num_bvals 4 --training_method a"
#sbatch --partition=CPUQ --account=nv-fys --time=01:00:00 --nodes=1 --ntasks-per-node=1 --mem=16000 --job-name="EMIN_1055_4b_a" --output=../../../dataA/output/txtoutput/LOO/EMIN_1055_b4_a.out --wrap="python patientsA_LOO.py --patient_id EMIN_1055 --num_bvals 4 --training_method a"
sbatch --partition=CPUQ --account=nv-fys --time=01:00:00 --nodes=1 --ntasks-per-node=1 --mem=16000 --job-name="EMIN_1057_4b_a" --output=../../../dataA/output/txtoutput/LOO/EMIN_1057_b4_a.out --wrap="python patientsA_LOO.py --patient_id EMIN_1057 --num_bvals 4 --training_method a"
#sbatch --partition=CPUQ --account=nv-fys --time=01:00:00 --nodes=1 --ntasks-per-node=1 --mem=16000 --job-name="EMIN_1060_4b_a" --output=../../../dataA/output/txtoutput/LOO/EMIN_1060_b4_a.out --wrap="python patientsA_LOO.py --patient_id EMIN_1060 --num_bvals 4 --training_method a"
#sbatch --partition=CPUQ --account=nv-fys --time=01:00:00 --nodes=1 --ntasks-per-node=1 --mem=16000 --job-name="EMIN_1064_4b_a" --output=../../../dataA/output/txtoutput/LOO/EMIN_1064_b4_a.out --wrap="python patientsA_LOO.py --patient_id EMIN_1064 --num_bvals 4 --training_method a"
sbatch --partition=CPUQ --account=nv-fys --time=01:00:00 --nodes=1 --ntasks-per-node=1 --mem=16000 --job-name="EMIN_1066_4b_a" --output=../../../dataA/output/txtoutput/LOO/EMIN_1066_b4_a.out --wrap="python patientsA_LOO.py --patient_id EMIN_1066 --num_bvals 4 --training_method a"
sbatch --partition=CPUQ --account=nv-fys --time=01:00:00 --nodes=1 --ntasks-per-node=1 --mem=16000 --job-name="EMIN_1068_4b_a" --output=../../../dataA/output/txtoutput/LOO/EMIN_1068_b4_a.out --wrap="python patientsA_LOO.py --patient_id EMIN_1068 --num_bvals 4 --training_method a"
sbatch --partition=CPUQ --account=nv-fys --time=01:00:00 --nodes=1 --ntasks-per-node=1 --mem=16000 --job-name="EMIN_1075_4b_a" --output=../../../dataA/output/txtoutput/LOO/EMIN_1075_b4_a.out --wrap="python patientsA_LOO.py --patient_id EMIN_1075 --num_bvals 4 --training_method a"
#sbatch --partition=CPUQ --account=nv-fys --time=01:00:00 --nodes=1 --ntasks-per-node=1 --mem=16000 --job-name="EMIN_1077_4b_a" --output=../../../dataA/output/txtoutput/LOO/EMIN_1077_b4_a.out --wrap="python patientsA_LOO.py --patient_id EMIN_1077 --num_bvals 4 --training_method a"
sbatch --partition=CPUQ --account=nv-fys --time=01:00:00 --nodes=1 --ntasks-per-node=1 --mem=16000 --job-name="EMIN_1079_4b_a" --output=../../../dataA/output/txtoutput/LOO/EMIN_1079_b4_a.out --wrap="python patientsA_LOO.py --patient_id EMIN_1079 --num_bvals 4 --training_method a"
#sbatch --partition=CPUQ --account=nv-fys --time=01:00:00 --nodes=1 --ntasks-per-node=1 --mem=16000 --job-name="EMIN_1081_4b_a" --output=../../../dataA/output/txtoutput/LOO/EMIN_1081_b4_a.out --wrap="python patientsA_LOO.py --patient_id EMIN_1081 --num_bvals 4 --training_method a"
#sbatch --partition=CPUQ --account=nv-fys --time=01:00:00 --nodes=1 --ntasks-per-node=1 --mem=16000 --job-name="EMIN_1084_4b_a" --output=../../../dataA/output/txtoutput/LOO/EMIN_1084_b4_a.out --wrap="python patientsA_LOO.py --patient_id EMIN_1084 --num_bvals 4 --training_method a"
sbatch --partition=CPUQ --account=nv-fys --time=01:00:00 --nodes=1 --ntasks-per-node=1 --mem=16000 --job-name="EMIN_1086_4b_a" --output=../../../dataA/output/txtoutput/LOO/EMIN_1086_b4_a.out --wrap="python patientsA_LOO.py --patient_id EMIN_1086 --num_bvals 4 --training_method a"
sbatch --partition=CPUQ --account=nv-fys --time=01:00:00 --nodes=1 --ntasks-per-node=1 --mem=16000 --job-name="EMIN_1090_4b_a" --output=../../../dataA/output/txtoutput/LOO/EMIN_1090_b4_a.out --wrap="python patientsA_LOO.py --patient_id EMIN_1090 --num_bvals 4 --training_method a"
sbatch --partition=CPUQ --account=nv-fys --time=01:00:00 --nodes=1 --ntasks-per-node=1 --mem=16000 --job-name="EMIN_1092_4b_a" --output=../../../dataA/output/txtoutput/LOO/EMIN_1092_b4_a.out --wrap="python patientsA_LOO.py --patient_id EMIN_1092 --num_bvals 4 --training_method a"
sbatch --partition=CPUQ --account=nv-fys --time=01:00:00 --nodes=1 --ntasks-per-node=1 --mem=16000 --job-name="EMIN_1093_4b_a" --output=../../../dataA/output/txtoutput/LOO/EMIN_1093_b4_a.out --wrap="python patientsA_LOO.py --patient_id EMIN_1093 --num_bvals 4 --training_method a"
#sbatch --partition=CPUQ --account=nv-fys --time=01:00:00 --nodes=1 --ntasks-per-node=1 --mem=16000 --job-name="EMIN_1096_4b_a" --output=../../../dataA/output/txtoutput/LOO/EMIN_1096_b4_a.out --wrap="python patientsA_LOO.py --patient_id EMIN_1096 --num_bvals 4 --training_method a"
#sbatch --partition=CPUQ --account=nv-fys --time=01:00:00 --nodes=1 --ntasks-per-node=1 --mem=16000 --job-name="EMIN_1097_4b_a" --output=../../../dataA/output/txtoutput/LOO/EMIN_1097_b4_a.out --wrap="python patientsA_LOO.py --patient_id EMIN_1097 --num_bvals 4 --training_method a"
sbatch --partition=CPUQ --account=nv-fys --time=01:00:00 --nodes=1 --ntasks-per-node=1 --mem=16000 --job-name="EMIN_1099_4b_a" --output=../../../dataA/output/txtoutput/LOO/EMIN_1099_b4_a.out --wrap="python patientsA_LOO.py --patient_id EMIN_1099 --num_bvals 4 --training_method a"


# sbatch --partition=CPUQ --account=nv-fys --time=01:00:00 --nodes=1 --ntasks-per-node=1 --mem=16000 --job-name="EMIN_1001_4b_b" --output=../../../dataA/output/txtoutput/LOO/EMIN_1001_b4_b.out --wrap="python patientsA_LOO.py --patient_id EMIN_1001 --num_bvals 4 --training_method b"
# sbatch --partition=CPUQ --account=nv-fys --time=01:00:00 --nodes=1 --ntasks-per-node=1 --mem=16000 --job-name="EMIN_1005_4b_b" --output=../../../dataA/output/txtoutput/LOO/EMIN_1005_b4_b.out --wrap="python patientsA_LOO.py --patient_id EMIN_1005 --num_bvals 4 --training_method b"
# sbatch --partition=CPUQ --account=nv-fys --time=01:00:00 --nodes=1 --ntasks-per-node=1 --mem=16000 --job-name="EMIN_1007_4b_b" --output=../../../dataA/output/txtoutput/LOO/EMIN_1007_b4_b.out --wrap="python patientsA_LOO.py --patient_id EMIN_1007 --num_bvals 4 --training_method b"
# sbatch --partition=CPUQ --account=nv-fys --time=01:00:00 --nodes=1 --ntasks-per-node=1 --mem=16000 --job-name="EMIN_1011_4b_b" --output=../../../dataA/output/txtoutput/LOO/EMIN_1011_b4_b.out --wrap="python patientsA_LOO.py --patient_id EMIN_1011 --num_bvals 4 --training_method b"
# sbatch --partition=CPUQ --account=nv-fys --time=01:00:00 --nodes=1 --ntasks-per-node=1 --mem=16000 --job-name="EMIN_1019_4b_b" --output=../../../dataA/output/txtoutput/LOO/EMIN_1019_b4_b.out --wrap="python patientsA_LOO.py --patient_id EMIN_1019 --num_bvals 4 --training_method b"
# sbatch --partition=CPUQ --account=nv-fys --time=01:00:00 --nodes=1 --ntasks-per-node=1 --mem=16000 --job-name="EMIN_1020_4b_b" --output=../../../dataA/output/txtoutput/LOO/EMIN_1020_b4_b.out --wrap="python patientsA_LOO.py --patient_id EMIN_1020 --num_bvals 4 --training_method b"
# sbatch --partition=CPUQ --account=nv-fys --time=01:00:00 --nodes=1 --ntasks-per-node=1 --mem=16000 --job-name="EMIN_1022_4b_b" --output=../../../dataA/output/txtoutput/LOO/EMIN_1022_b4_b.out --wrap="python patientsA_LOO.py --patient_id EMIN_1022 --num_bvals 4 --training_method b"
# sbatch --partition=CPUQ --account=nv-fys --time=01:00:00 --nodes=1 --ntasks-per-node=1 --mem=16000 --job-name="EMIN_1032_4b_b" --output=../../../dataA/output/txtoutput/LOO/EMIN_1032_b4_b.out --wrap="python patientsA_LOO.py --patient_id EMIN_1032 --num_bvals 4 --training_method b"
# sbatch --partition=CPUQ --account=nv-fys --time=01:00:00 --nodes=1 --ntasks-per-node=1 --mem=16000 --job-name="EMIN_1038_4b_b" --output=../../../dataA/output/txtoutput/LOO/EMIN_1038_b4_b.out --wrap="python patientsA_LOO.py --patient_id EMIN_1038 --num_bvals 4 --training_method b"
# sbatch --partition=CPUQ --account=nv-fys --time=01:00:00 --nodes=1 --ntasks-per-node=1 --mem=16000 --job-name="EMIN_1042_4b_b" --output=../../../dataA/output/txtoutput/LOO/EMIN_1042_b4_b.out --wrap="python patientsA_LOO.py --patient_id EMIN_1042 --num_bvals 4 --training_method b"
# sbatch --partition=CPUQ --account=nv-fys --time=01:00:00 --nodes=1 --ntasks-per-node=1 --mem=16000 --job-name="EMIN_1044_4b_b" --output=../../../dataA/output/txtoutput/LOO/EMIN_1044_b4_b.out --wrap="python patientsA_LOO.py --patient_id EMIN_1044 --num_bvals 4 --training_method b"
# sbatch --partition=CPUQ --account=nv-fys --time=01:00:00 --nodes=1 --ntasks-per-node=1 --mem=16000 --job-name="EMIN_1045_4b_b" --output=../../../dataA/output/txtoutput/LOO/EMIN_1045_b4_b.out --wrap="python patientsA_LOO.py --patient_id EMIN_1045 --num_bvals 4 --training_method b"
# sbatch --partition=CPUQ --account=nv-fys --time=01:00:00 --nodes=1 --ntasks-per-node=1 --mem=16000 --job-name="EMIN_1048_4b_b" --output=../../../dataA/output/txtoutput/LOO/EMIN_1048_b4_b.out --wrap="python patientsA_LOO.py --patient_id EMIN_1048 --num_bvals 4 --training_method b"
# sbatch --partition=CPUQ --account=nv-fys --time=01:00:00 --nodes=1 --ntasks-per-node=1 --mem=16000 --job-name="EMIN_1055_4b_b" --output=../../../dataA/output/txtoutput/LOO/EMIN_1055_b4_b.out --wrap="python patientsA_LOO.py --patient_id EMIN_1055 --num_bvals 4 --training_method b"
# sbatch --partition=CPUQ --account=nv-fys --time=01:00:00 --nodes=1 --ntasks-per-node=1 --mem=16000 --job-name="EMIN_1057_4b_b" --output=../../../dataA/output/txtoutput/LOO/EMIN_1057_b4_b.out --wrap="python patientsA_LOO.py --patient_id EMIN_1057 --num_bvals 4 --training_method b"
# sbatch --partition=CPUQ --account=nv-fys --time=01:00:00 --nodes=1 --ntasks-per-node=1 --mem=16000 --job-name="EMIN_1060_4b_b" --output=../../../dataA/output/txtoutput/LOO/EMIN_1060_b4_b.out --wrap="python patientsA_LOO.py --patient_id EMIN_1060 --num_bvals 4 --training_method b"
# sbatch --partition=CPUQ --account=nv-fys --time=01:00:00 --nodes=1 --ntasks-per-node=1 --mem=16000 --job-name="EMIN_1064_4b_b" --output=../../../dataA/output/txtoutput/LOO/EMIN_1064_b4_b.out --wrap="python patientsA_LOO.py --patient_id EMIN_1064 --num_bvals 4 --training_method b"
# sbatch --partition=CPUQ --account=nv-fys --time=01:00:00 --nodes=1 --ntasks-per-node=1 --mem=16000 --job-name="EMIN_1066_4b_b" --output=../../../dataA/output/txtoutput/LOO/EMIN_1066_b4_b.out --wrap="python patientsA_LOO.py --patient_id EMIN_1066 --num_bvals 4 --training_method b"
# sbatch --partition=CPUQ --account=nv-fys --time=01:00:00 --nodes=1 --ntasks-per-node=1 --mem=16000 --job-name="EMIN_1068_4b_b" --output=../../../dataA/output/txtoutput/LOO/EMIN_1068_b4_b.out --wrap="python patientsA_LOO.py --patient_id EMIN_1068 --num_bvals 4 --training_method b"
# sbatch --partition=CPUQ --account=nv-fys --time=01:00:00 --nodes=1 --ntasks-per-node=1 --mem=16000 --job-name="EMIN_1075_4b_b" --output=../../../dataA/output/txtoutput/LOO/EMIN_1075_b4_b.out --wrap="python patientsA_LOO.py --patient_id EMIN_1075 --num_bvals 4 --training_method b"
# sbatch --partition=CPUQ --account=nv-fys --time=01:00:00 --nodes=1 --ntasks-per-node=1 --mem=16000 --job-name="EMIN_1077_4b_b" --output=../../../dataA/output/txtoutput/LOO/EMIN_1077_b4_b.out --wrap="python patientsA_LOO.py --patient_id EMIN_1077 --num_bvals 4 --training_method b"
# sbatch --partition=CPUQ --account=nv-fys --time=01:00:00 --nodes=1 --ntasks-per-node=1 --mem=16000 --job-name="EMIN_1079_4b_b" --output=../../../dataA/output/txtoutput/LOO/EMIN_1079_b4_b.out --wrap="python patientsA_LOO.py --patient_id EMIN_1079 --num_bvals 4 --training_method b"
# sbatch --partition=CPUQ --account=nv-fys --time=01:00:00 --nodes=1 --ntasks-per-node=1 --mem=16000 --job-name="EMIN_1081_4b_b" --output=../../../dataA/output/txtoutput/LOO/EMIN_1081_b4_b.out --wrap="python patientsA_LOO.py --patient_id EMIN_1081 --num_bvals 4 --training_method b"
# sbatch --partition=CPUQ --account=nv-fys --time=01:00:00 --nodes=1 --ntasks-per-node=1 --mem=16000 --job-name="EMIN_1084_4b_b" --output=../../../dataA/output/txtoutput/LOO/EMIN_1084_b4_b.out --wrap="python patientsA_LOO.py --patient_id EMIN_1084 --num_bvals 4 --training_method b"
# sbatch --partition=CPUQ --account=nv-fys --time=01:00:00 --nodes=1 --ntasks-per-node=1 --mem=16000 --job-name="EMIN_1086_4b_b" --output=../../../dataA/output/txtoutput/LOO/EMIN_1086_b4_b.out --wrap="python patientsA_LOO.py --patient_id EMIN_1086 --num_bvals 4 --training_method b"
# sbatch --partition=CPUQ --account=nv-fys --time=01:00:00 --nodes=1 --ntasks-per-node=1 --mem=16000 --job-name="EMIN_1090_4b_b" --output=../../../dataA/output/txtoutput/LOO/EMIN_1090_b4_b.out --wrap="python patientsA_LOO.py --patient_id EMIN_1090 --num_bvals 4 --training_method b"
# sbatch --partition=CPUQ --account=nv-fys --time=01:00:00 --nodes=1 --ntasks-per-node=1 --mem=16000 --job-name="EMIN_1092_4b_b" --output=../../../dataA/output/txtoutput/LOO/EMIN_1092_b4_b.out --wrap="python patientsA_LOO.py --patient_id EMIN_1092 --num_bvals 4 --training_method b"
# sbatch --partition=CPUQ --account=nv-fys --time=01:00:00 --nodes=1 --ntasks-per-node=1 --mem=16000 --job-name="EMIN_1093_4b_b" --output=../../../dataA/output/txtoutput/LOO/EMIN_1093_b4_b.out --wrap="python patientsA_LOO.py --patient_id EMIN_1093 --num_bvals 4 --training_method b"
# sbatch --partition=CPUQ --account=nv-fys --time=01:00:00 --nodes=1 --ntasks-per-node=1 --mem=16000 --job-name="EMIN_1096_4b_b" --output=../../../dataA/output/txtoutput/LOO/EMIN_1096_b4_b.out --wrap="python patientsA_LOO.py --patient_id EMIN_1096 --num_bvals 4 --training_method b"
# sbatch --partition=CPUQ --account=nv-fys --time=01:00:00 --nodes=1 --ntasks-per-node=1 --mem=16000 --job-name="EMIN_1097_4b_b" --output=../../../dataA/output/txtoutput/LOO/EMIN_1097_b4_b.out --wrap="python patientsA_LOO.py --patient_id EMIN_1097 --num_bvals 4 --training_method b"
# sbatch --partition=CPUQ --account=nv-fys --time=01:00:00 --nodes=1 --ntasks-per-node=1 --mem=16000 --job-name="EMIN_1099_4b_b" --output=../../../dataA/output/txtoutput/LOO/EMIN_1099_b4_b.out --wrap="python patientsA_LOO.py --patient_id EMIN_1099 --num_bvals 4 --training_method b"
