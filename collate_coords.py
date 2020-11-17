import sys, getopt
import pandas as pd
import numpy as np


def main(argv):
    inputfile = ''
    outputfile = ''
    ccoords_file = ''

    try:
        opts, args = getopt.getopt(argv, "hi:o:c:", ["ifile=", "ofile=", "ccoords="])
    except getopt.GetoptError:
        print('parse_annotations.py -i <inputfile> -o <outputfile> -c <ccoords>')
        sys.exit(2)

    for opt, arg in opts:
        if opt == '-h':
            print('parse_annotations.py -i <inputfile> -o <outputfile> -c <ccoords>')
            sys.exit()
        elif opt in ("-i", "--ifile"):
            inputfile = arg
        elif opt in ("-o", "--ofile"):
            outputfile = arg
        elif opt in ("-c", "--ccoords"):
            ccoords_file = arg

    video = inputfile.strip('_bb.csv').strip('data/bb/')

    bb_frame = pd.read_csv(inputfile, names=['xtl', 'ytl', 'xbr', 'ybr'])
    ccoords = pd.read_csv(ccoords_file, dtype={'fake_lat': np.float64, 'fake_lon': np.float64}).set_index('Unnamed: 0')

    output = pd.DataFrame()
    num_frames = len(ccoords)

    output['frame_path'] = pd.Series(['data/frames/{}_{}.jpg'.format(video, x) for x in range(num_frames)])
    output['xtl'] = bb_frame['xtl']
    output['ytl'] = bb_frame['ytl']
    output['xbr'] = bb_frame['xbr']
    output['ybr'] = bb_frame['ybr']
    output['lat'] = ccoords['fake_lat']
    output['lon'] = ccoords['fake_lon']

    output.to_csv(outputfile, header=False, index=False)


if __name__ == '__main__':
    main(sys.argv[1:])

