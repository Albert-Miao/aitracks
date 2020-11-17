import sys, getopt
from os import path
import pandas as pd


def main(argv):
    inputfile = ''
    outputfile = ''

    try:
        opts, args = getopt.getopt(argv, "hi:o:", ["ifile=", "ofile="])
    except getopt.GetoptError:
        print('parse_annotations.py -i <inputfile> -o <outputfile>')
        sys.exit(2)

    for opt, arg in opts:
        if opt == '-h':
            print('parse_annotations.py -i <inputfile> -o <outputfile>')
            sys.exit()
        elif opt in ("-i", "--ifile"):
            inputfile = arg
        elif opt in ("-o", "--ofile"):
            outputfile = arg

    with open(inputfile, 'r') as f:
        text = f.read()

    data = "".join(text.split())

    start_ind = data.find('<trkpt')
    end_ind = data.rfind('</trkpt>') + 8

    data = data[start_ind:end_ind]
    data = data.split('</trkpt>')[:-1]

    df = pd.DataFrame({'raw_str': data})
    df['lat'] = df['raw_str'].str[11:23]
    df['lon'] = df['raw_str'].str[29:43]
    df['datetime_str'] = df['raw_str'].str[51:71]
    df['datetime'] = pd.to_datetime(df['datetime_str'])

    if path.exists(outputfile):
        prev_frame = pd.read_csv(outputfile).set_index('Unnamed: 0')
        prev_frame['datetime'] = pd.to_datetime(prev_frame['datetime'])
        df = pd.concat([df, prev_frame]).drop_duplicates().sort_values('datetime').reset_index(drop=True)

    df.to_csv(outputfile)


if __name__ == '__main__':
    main(sys.argv[1:])

