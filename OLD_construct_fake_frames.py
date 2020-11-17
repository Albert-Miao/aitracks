# coding: utf-8
def update_time(time):
    if time in times.keys():
        times[time] += 1
    else:
        times[time] = 1
    return time.replace(microsecond=round(1000000 * times[time] / (grouped[time] + 1)))
    
    
times = dict()
test = pd.read_csv('gps_1.csv')
test['datetime'] = pd.to_datetime(test['datetime'])
test['datetime'] = test['datetime'].apply(update_time)
test
def generate_fake_coords(time):
    global fixed_timestamp
    global start_lat
    global start_lon
    global end_lat
    global end_lon
    global time_diff
    global slope_lat
    global slope_lon
    
    if time > test['datetime'][fixed_timestamp + 1]:
        fixed_timestamp += 1
        start_lat = end_lat
        start_lon = end_lon
        
        end_lat = test['lat'][fixed_timestamp + 1]
        end_lon = test['lon'][fixed_timestamp + 1]
        
        time_diff = (test['datetime'][fixed_timestamp + 1] - test['datetime'][fixed_timestamp]).total_seconds()

        slope_lat = float(end_lat - start_lat) / time_diff
        slope_lon = float(end_lon - start_lon) / time_diff

    delta = (time - test['datetime'][fixed_timestamp]).total_seconds()
    fake_coords.append([start_lat + (slope_lat * delta), start_lon + (slope_lon * delta)])
    
fixed_timestamp = 0
start_lat = test['lat'][fixed_timestamp]
start_lon = test['lon'][fixed_timestamp]

end_lat = test['lat'][fixed_timestamp+1]
end_lon = test['lon'][fixed_timestamp+1]
time_diff = (test['datetime'][fixed_timestamp + 1] - test['datetime'][fixed_timestamp]).total_seconds()
slope_lat = float(end_lat - start_lat) / time_diff
slope_lon = float(end_lon - start_lon) / time_diff
fake_coords = []
grouped = test.pivot_table(index='datetime', aggfunc='size')
frame_times = []
time = test['datetime'][0]
time += datetime.timedelta(microseconds=16667)
while time <= test['datetime'][2584]:
    frame_times.append(time)
    time += datetime.timedelta(microseconds=1000000/20)
    
frame_times.apply(generate_fake_coords)
frame_times = pd.Series(frame_times)
frame_times.apply(generate_fake_coords)
fake_lat = [s[0] for s in fake_coords]
fake_lon = [s[1] for s in fake_coords]
fake_lon = pd.Series(fake_lon)
fake_lat = pd.Series(fake_lat)
output = pd.DataFrame()
output['fake_lat'] = fake_lat
output['fake_lon'] = fake_lon
output['frame_time'] = pd.Series(frame_times)
output.to_csv('OLD_constructed_coordinates.csv')
min_lat = min(output['fake_lat'])
max_lat = max(output['fake_lat'])
min_lon = min(output['fake_lon'])
max_lon = max(output['fake_lon'])
bbox = (min_lon, max_lon, min_lat, max_lat)
map_img = plt.imread('map (1).png')
fig, ax = plt.subplots(figsize = (8, 7))
ax.scatter(output['fake_lon'], output['fake_lat'], zorder=1, alpha=0.2, c='b', s=10)
ax.set_xlim(bbox[0], bbox[1])
ax.set_ylim(bbox[2], bbox[3])
ax.imshow(map_img, zorder=0, extent=bbox, aspect='equal')
