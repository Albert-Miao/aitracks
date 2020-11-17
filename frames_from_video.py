import getopt
import sys

import cv2
import os


def main(argv):
    inputfile = ''

    try:
        opts, args = getopt.getopt(argv, "hi:", ["ifile="])
    except getopt.GetoptError:
        print('parse_annotations.py -i <inputfile>')
        sys.exit(2)

    for opt, arg in opts:
        if opt == '-h':
            print('parse_annotations.py -i <inputfile>')
            sys.exit()
        elif opt in ("-i", "--ifile"):
            inputfile = arg


    vidcap = cv2.VideoCapture(inputfile)
    success, image = vidcap.read()
    count = 0

    while success:
        to_write = os.path.join('data','frames', '%s_%d.jpg')
        cv2.imwrite(to_write % (inputfile.strip('.mp4'), count), image)
        success, image = vidcap.read()
        count += 1
        print(to_write)


if __name__ == '__main__':
    main(sys.argv[1:])