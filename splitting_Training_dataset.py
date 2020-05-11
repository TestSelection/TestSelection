'''
The file is used to split the training dataset.
The test data of the original datset should not be touched.
'''
import utils.load_data as datama
import argparse

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-t", "--toy", type=bool, default=False)
    args = vars(ap.parse_args())
    if args['toy']:
        print("Toy Data")
        datama.spli_toy_data()
    else:
        print("Not Toy Data")
        datama.split_data(10000)
