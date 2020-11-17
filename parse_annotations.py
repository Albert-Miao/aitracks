import sys, getopt
import xml.etree.ElementTree as ET
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

    root = ET.parse(inputfile).getroot()
    track = root.find('track')
    boxes = track.getchildren()

    frames, xtl, ytl, xbr, ybr = [[] for i in range(5)]

    for box in boxes:
        att = box.attrib
        frames.append(att['frame'])
        xtl.append(att['xtl'])
        ytl.append(att['ytl'])
        xbr.append(att['xbr'])
        ybr.append(att['ybr'])

    output = pd.DataFrame({'frame': frames, 'xtl': xtl, 'ytl': ytl, 'xbr': xbr, 'ybr': ybr})
    output.to_csv(outputfile, header=False, index=False)


if __name__ == "__main__":
    main(sys.argv[1:])