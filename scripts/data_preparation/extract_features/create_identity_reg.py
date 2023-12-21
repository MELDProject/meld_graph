#import relevant packages
import numpy as np
import argparse
import os


def create_identity(subject_id):
    os.makedirs(subject_id + '/mri/transforms', exist_ok=True)
    with open(subject_id + '/mri/transforms/Identity.dat', 'w') as f:
        f.writelines(subject_id+'\n')
        f.writelines('1.000000'+'\n')
        f.writelines('1.000000'+'\n')
        f.writelines('0.150000'+'\n')
        f.writelines('1 0 0 0'+'\n')
        f.writelines('0 1 0 0'+'\n')
        f.writelines('0 0 1 0'+'\n')
        f.writelines('0 0 0 1'+'\n')
        f.writelines('round'+'\n')


if __name__ == "__main__":
    #parse commandline arguments pointing to subject_dir etc
    parser = argparse.ArgumentParser(description='create identity registration matrix')
    parser.add_argument('subject_id', type=str,
                        help='subject_id')
    args = parser.parse_args()
    #save subjects dir and subject ids. import the text file containing subject ids
    subject_id=args.subject_id
    create_identity(subject_id)