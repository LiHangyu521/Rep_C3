import glob
import os
import subprocess
import argparse

POSSIBLE_DATASETS = ['kodak', 'clic20-pro-valid', 'clic22-test', 'jvet', 'vcip2023_4k','vcip_4k_g1','vcip_4k_g2','vcip_4k_g3','vcip_4k_g4']
bpp_offical = {
    'I01':0.0015,
    'I02':0.0009,
    'I03':0.0009,
    'I04':0.0005,
    'I05':0.0005,
    'I06':0.0009,
    'I07':0.0015,
    'I08':0.0003,
    'I09':0.0007,
    'I10':0.0009,
    'I11':0.0009,
    'I12':0.00085,
    'I13':0.0002,
    'I14':0.001,
    'I15':0.00085,
    'I16':0.00075,
    'I17':0.0007,
    'I18':0.0008,
    'I19':0.0007,
    'I20':0.0007,
}

parser = argparse.ArgumentParser()
parser.add_argument('dataset_name', help=f'Possible values {POSSIBLE_DATASETS}.')
parser.add_argument('--lamda', help='d+lamda*R',default=0.02)
parser.add_argument('--lamda_list', type=list, default=[0.00012,0.0008,0.0008,0.0008,0.00012,0.0008])

args = parser.parse_args()

def lamda_to_str(lamda) :
    str_lamda = list()
    tmp = lamda
    while tmp< 1 :
        str_lamda.append(int(tmp * 10))
        tmp = tmp * 10
    return str_lamda

assert args.dataset_name in POSSIBLE_DATASETS, \
    f'Argument must be in {POSSIBLE_DATASETS}. Found {args.dataset_name}!'


# encode images one by one
current_dir_path = os.path.dirname(__file__)
dataset_path = '/media/tly/lihy/workspace/Cool-Chic/dataset'
encoded_image_path = os.path.join(dataset_path,args.dataset_name)

cool_chic_encode_path = os.path.join(current_dir_path, '../src/encode.py')


encoded_images = glob.glob(os.path.join(encoded_image_path,'*.png'))
for encoded_img in encoded_images :
    imgname = encoded_img.split("/")[-1].split(".")[0]
    output_dir = os.path.join(current_dir_path, args.dataset_name, 'medium_train',f"lamda-{args.lamda}", imgname)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_bitstream_path = os.path.join(output_dir, f'{imgname}.bin')
    model_save_path = os.path.join(output_dir, f'{imgname}_model.pt')
    enc_results_path = os.path.join(output_dir, f'{imgname}_encoder_results.txt')
    qstep_results_pkl = os.path.join(output_dir, f'{imgname}_qstep_results.pkl')
    if os.path.exists(model_save_path):
        print(f'skipped_{imgname}')
        continue


    print(f'\nencoding image: {encoded_img}')
    cmd = f'CUDA_VISIBLE_DEVICES=0 python3 {cool_chic_encode_path} \
        --input {encoded_img} \
        --output {output_bitstream_path} \
        --workdir={output_dir}/ \
        --lmbda={args.lamda}\
        --start_lr=1e-2\
        --layers_synthesis=40-1-linear-relu,3-1-linear-relu,X-3-residual-relu,X-3-residual-none\
        --upsampling_kernel_size=8                      \
        --layers_arm=24,24\
        --n_ctx_rowcol=3\
        --n_ft_per_res=1,1,1,1,1,1,1                    \
        --n_itr=50000                                    \
        --n_train_loops=5'
    subprocess.call(cmd, shell=True)
