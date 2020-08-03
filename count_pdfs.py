from glob import glob
import os 
import csv

jpg_region_dir = 'jpg_regions/'


def main():
    kv = {} 
    
    dir_paths = sorted(glob(os.path.join(jpg_region_dir,"*")))
    for path in dir_paths:
        jpg_paths = sorted(glob(os.path.join(path,"*")))
        if jpg_paths:
            kv[os.path.basename(path)] = len(jpg_paths)

    with open('count.csv', mode='w') as f:
        writer = csv.writer(f)
        for k in kv:
            writer.writerow([k, kv[k]])
    
if __name__ == '__main__':
    main() 
