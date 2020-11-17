import sys, getopt
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

    date = '2020-09-30 '

    baseline = pd.read_csv('data/gps_1.csv')
    baseline['datetime'] = pd.to_datetime(baseline['datetime'])

    grouped = baseline.pivot_table(index='datetime', aggfunc='size')
    detailed_times = []
    repeated_times = dict()

    for t in baseline['datetime']:
        if t in repeated_times:
            repeated_times[t] += 1
        else:
            repeated_times[t] = 1

        detailed_times.append(t.replace(microsecond=round(1000000 * repeated_times[t] / (grouped[t] + 1))))

    baseline['datetime'] = pd.Series(detailed_times)

    camera_log = pd.read_csv(inputfile)
    start_time = pd.to_datetime(date + camera_log.loc[0]['UTC Timestamp'], utc=True)
    end_time = pd.to_datetime(date + camera_log.iloc[-1]['UTC Timestamp'], utc=True)

    start_time = start_time.replace(microsecond=500000)
    end_time = end_time.replace(microsecond=500000)

    num_seconds = (end_time - start_time).seconds
    fps = len(camera_log) / num_seconds

    interval = -1
    while baseline['datetime'][interval + 1].time() <= start_time.time():
        interval += 1

    left_time, right_time = baseline['datetime'][interval], baseline['datetime'][interval + 1]
    left_lat, left_lon = baseline['lat'][interval], baseline['lon'][interval]
    right_lat, right_lon = baseline['lat'][interval + 1], baseline['lon'][interval + 1]
    time_diff = (right_time - left_time).total_seconds()

    slope_lat = float(right_lat - left_lat) / time_diff
    slope_lon = float(right_lon - left_lon) / time_diff

    fake_coords = []

    frame_deltas = []
    delta = pd.to_timedelta(0)
    count = 0
    while delta < (end_time - start_time):
        frame_deltas.append(delta)
        count += 1
        delta = pd.to_timedelta(count/fps, 's')

    for d in frame_deltas:
        time = start_time + d
        
        if time > right_time:
            interval += 1

            left_time, right_time = baseline['datetime'][interval], baseline['datetime'][interval + 1]
            left_lat, left_lon = right_lat, right_lon
            right_lat, right_lon = baseline['lat'][interval + 1], baseline['lon'][interval + 1]
            time_diff = (right_time - left_time).total_seconds()

            slope_lat = float(right_lat - left_lat) / time_diff
            slope_lon = float(right_lon - left_lon) / time_diff

        dt = (time - left_time).total_seconds()
        fake_coords.append([left_lat + (slope_lat * dt), left_lon + (slope_lon * dt)])

    fake_lat = pd.Series([s[0] for s in fake_coords])
    fake_lon = pd.Series([s[1] for s in fake_coords])

    output = pd.DataFrame()
    output['fake_lat'] = fake_lat
    output['fake_lon'] = fake_lon
    output['frame_time'] = pd.Series(frame_deltas) + start_time

    output.to_csv(outputfile)


if __name__ == "__main__":
    main(sys.argv[1:])