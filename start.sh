curr_date=`date +%y-%m-%d-%H-%M`
log_dir=/home/researcher/output/gan/began/began2/
mkdir -p ${log_dir}
export CUDA_VISIBLE_DEVICES=1
nohup python3 main.py --n_batch=16 --data_dir=/home/researcher/input/img/wheeldesign --dataset=128_gray_rotate --n_save_log_step=40 --n_save_img_step=400 --log_dir=${log_dir} > ${log_dir}/${curr_date}_log.txt &
sleep 1
tail -100f ${log_dir}/${curr_date}_log.txt

