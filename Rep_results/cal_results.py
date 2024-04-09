import argparse
import glob
import os
import csv

parser = argparse.ArgumentParser()
parser.add_argument('--exp_dir', help='exp_dir_of the results of each png file', default='Rep_results/kodak/10w_5loop_train')
args = parser.parse_args()

def cal_avg_psnr_bpp(inputpath):

    f = open(inputpath,"r",encoding='utf-8')
    lines =f.readlines()
    datas = lines[1].split()
    f.close()
    return float(datas[3]),float(datas[6])

def takeOne(elem):
    return elem[0]


if __name__ == '__main__':
    lambda_dirs = glob.glob(os.path.join(args.exp_dir,'lamda*'),recursive=False)
    for lambda_dir in lambda_dirs:
        encoded_images = glob.glob(os.path.join(lambda_dir,'*/results_best.tsv'))
        results = {'avg_bpp': [], 'avg_psnr': []}
        avg_psnr = avg_bpp = 0.0
        img_nums = len(encoded_images)
        print('Images count: ' + str(img_nums))


        writer_list = [] # 'name' 'psnr' 'bpp'
        for i,img in enumerate(encoded_images):
            psnr,bpp = cal_avg_psnr_bpp(img)
            name = img.split('/')[-2]
            writer_list.append([name,psnr,bpp])
            avg_psnr += psnr
            avg_bpp += bpp
        avg_psnr /= img_nums
        avg_bpp /= img_nums

        writer_list.sort(key=takeOne)
        

        with open(os.path.join(args.exp_dir,lambda_dir.split('/')[-1] +'_encoder_results.csv'), 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['name','psnr','bpp'])
            writer.writerows(writer_list)
            writer.writerow(['','avg_psnr','avg_bpp'])
            writer.writerow(['',avg_psnr, avg_bpp])
        f.close()