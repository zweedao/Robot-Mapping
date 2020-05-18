#####---Cornell: Autonomous Project 3 - SLAM ---#####
import pandas as pd
import numpy as np
import math
import os
import random
import csv
import sys
import matplotlib.pyplot as plt
import pickle
from scipy import io
import pdb

#Load data

def get_lidar(file_name):
	data = io.loadmat(file_name)
	lidar = []
	angles = np.double(data['Hokuyo0']['angles'][0][0])
	ranges = np.array(data['Hokuyo0']['ranges'][0][0]).T
	ts_set = data['Hokuyo0']['ts'][0,0][0]

	idx = 0	
	for m in ranges:
		tmp = {}
		tmp['t'] = ts_set[idx]
		tmp['scan'] = m
		tmp['angle'] = angles
		lidar.append(tmp)
		idx = idx + 1
	return lidar


def get_encoder(file_name):

	data = io.loadmat(file_name)
#	pdb.set_trace()

	FR = np.double(data['Encoders']['counts'][0,0][0])
	FL = np.double(data['Encoders']['counts'][0,0][1])
	RR = np.double(data['Encoders']['counts'][0,0][2])
	RL = np.double(data['Encoders']['counts'][0,0][3])
	ts = np.double(data['Encoders']['ts'][0,0][0])
	return FR, FL, RR, RL, ts	


def get_imu(file_name):

	data = io.loadmat(file_name)

	acc_x = np.double(data['vals'])[0]
	acc_y = np.double(data['vals'])[1]
	acc_z = np.double(data['vals'])[2]
	gyro_x = np.double(data['vals'])[3]
	gyro_y = np.double(data['vals'])[4]
	gyro_z = np.double(data['vals'])[5]	
	ts = np.double(data['ts'][0])
#	pdb.set_trace()
	return acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z, ts	

def replay_lidar(lidar_data):
	# lidar_data type: array where each array is a dictionary with a form of 't','pose','res','rpy','scan'
	#theta = np.arange(0,270.25,0.25)*np.pi/float(180)
	theta = lidar_data[0]['angle']

	for i in range(200,len(lidar_data),10):
		for (k,v) in enumerate(lidar_data[i]['scan']):
			if v > 30:
				lidar_data[i]['scan'][k] = 0.0

		# Jinwook's plot
		ax = plt.subplot(111, projection='polar')
		ax.plot(theta, lidar_data[i]['scan'])
		ax.set_rmax(10)
		ax.set_rticks([0.5, 1, 1.5, 2])  # less radial ticks
		ax.set_rlabel_position(-22.5)  # get radial labels away from plotted line
		ax.grid(True)
		ax.set_title("Lidar scan data", va='bottom')

		plt.draw()
		plt.pause(0.001)
		ax.clear()


# Bresenham's ray tracing algorithm in 2D.
# Inputs:
#	(sx, sy)	start point of ray
#	(ex, ey)	end point of ray
def bresenham2D(sx, sy, ex, ey):
    sx = int(round(sx))
    sy = int(round(sy))
    ex = int(round(ex))
    ey = int(round(ey))
    dx = abs(ex-sx)
    dy = abs(ey-sy)
    steep = abs(dy)>abs(dx)
    if steep:
      dx,dy = dy,dx # swap 
    
    if dy == 0:
      q = np.zeros((dx+1,1))
    else:
      q = np.append(0,np.greater_equal(np.diff(np.mod(np.arange( np.floor(dx/2), -dy*dx+np.floor(dx/2)-1,-dy),dx)),0))
    if steep:
      if sy <= ey:
        y = np.arange(sy,ey+1)
      else:
        y = np.arange(sy,ey-1,-1)
      if sx <= ex:
        x = sx + np.cumsum(q)
      else:
        x = sx - np.cumsum(q)
    else:
      if sx <= ex:
        x = np.arange(sx,ex+1)
      else:
        x = np.arange(sx,ex-1,-1)
      if sy <= ey:
        y = sy + np.cumsum(q)
      else:
        y = sy - np.cumsum(q) 
    
    x = x.astype(int)
    y = y.astype(int)
    return np.vstack((x,y))


def world2map(x_world, y_world, map_offset, map_scale):
    x_map = (x_world + map_offset) * map_scale
    y_map = (y_world + map_offset) * map_scale
    return x_map, y_map

def map2world(x_map, y_map, map_offset, map_scale):
    x_world = float(x_map) / map_scale - map_offset
    y_world = float(y_map) / map_scale - map_offset
    return x_world, y_world



# Load data
data_folder = "./test/"

encoder_file = sys.argv[1]
lidar_file = sys.argv[2]

FR, FL, RR, RL, ts_encoder = get_encoder(data_folder + encoder_file)
lidar_data = get_lidar(data_folder + lidar_file)
#acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z, ts_imu = get_imu(data_folder + 'imu21.mat')

#replay_lidar(lidar_data)

#Occupancy Grid Map
map_side = 70
map_offset = 35
map_scale = 10
occ_point = 1
free_point = -1

grid_map = np.zeros((map_side * map_scale, map_side * map_scale))


# Odometry
x_bot = 0
y_bot = 0
x_bot_all = []
y_bot_all = []
x_scan_all = []
y_scan_all = []
theta_delta = 0
theta_bot = 0
theta_bot_all = []
body_width = (0.476 + 0.311)/2 * 1.85
wheel_circum = 0.254 * np.pi
encoder_step = wheel_circum / 360

R = (FR + RR)/2
L = (FL + RL)/2

for t in range(0, len(ts_encoder)):
#for t in range(0, 1000):
    #odemetry
    left_dist = L[t] * encoder_step
    right_dist = R[t] * encoder_step
    
    in_dist = left_dist
    out_dist = right_dist
    
    theta_delta = (out_dist - in_dist) / body_width
    bot_dist = (out_dist + in_dist) / 2
    
    x_delta = bot_dist * np.cos(theta_bot + theta_delta/2)
    y_delta = bot_dist * np.sin(theta_bot + theta_delta/2)
    
    x_bot += x_delta
    y_bot += y_delta
    theta_bot += theta_delta 
    
    x_bot_all.append(x_bot)
    y_bot_all.append(y_bot)
    theta_bot_all.append(theta_bot)
    
    #lidar
    for ts in range(t, min(t+10,len(lidar_data))):
        if (lidar_data[ts]['t'] < (ts_encoder[t] + 0.025)) and (lidar_data[ts]['t'] >= ts_encoder[t]):
            print('t:',t,'ts:',ts)
            break
        elif ts == t+10:
            print ('t error:',t)
                
    lidar_scan = lidar_data[ts]['scan'].reshape(-1,1)
    lidar_angle = lidar_data[ts]['angle'].reshape(-1,1)
    x_scan_bot = lidar_scan * np.cos(lidar_angle)
    y_scan_bot = lidar_scan * np.sin(lidar_angle)
    x_scan = x_scan_bot * np.cos(theta_bot) - y_scan_bot * np.sin(theta_bot)
    y_scan = x_scan_bot * np.sin(theta_bot) + y_scan_bot * np.cos(theta_bot)
    x_scan += x_bot
    y_scan += y_bot
    
    x_scan_all.append(x_scan)
    y_scan_all.append(y_scan)

    #Occupancy grid map
    x_bot_map, y_bot_map = world2map(x_bot, y_bot, map_offset, map_scale)

    for i in range(0,len(x_scan)):
        x_scan_map, y_scan_map = world2map(x_scan[i][0], y_scan[i][0], map_offset, map_scale)        
        ray_cells = bresenham2D(x_bot_map, y_bot_map, x_scan_map, y_scan_map)
        x_hit = ray_cells[0][-1]
        y_hit = ray_cells[1][-1]        
        grid_map[x_hit][y_hit] += occ_point
        
        for j in range(0,len(ray_cells[0])-1):
            x_hit = ray_cells[0][j]
            y_hit = ray_cells[1][j]
            grid_map[x_hit][y_hit] += free_point
        

# Get occupied cells in grid map 
def map2world_all(x_map_all, y_map_all, map_offset, map_scale): 
    x_world_all = []
    y_world_all = []     
    #x_world -= map_offset
    #y_world -= map_offset
    for i in range(0,len(x_map_all)):    
        x_world, y_world = map2world(x_map_all[i], y_map_all[i], map_offset, map_scale)     
        x_world_all.append(x_world)
        y_world_all.append(y_world)
    
    return x_world_all, y_world_all
            
x_occ_map, y_occ_map = np.where(grid_map>1)  
x_occ_all, y_occ_all = map2world_all(x_occ_map, y_occ_map, map_offset, map_scale)

x_na_map, y_na_map = np.where(grid_map==0)  
x_na_all, y_na_all = map2world_all(x_na_map, y_na_map, map_offset, map_scale)


# Plot
#plt.scatter(x_scan_all, y_scan_all, c='red')
plt.scatter(x_bot_all, y_bot_all, c='blue', s=0.1)
#plt.plot(x_bot_all, y_bot_all, c='blue')
plt.scatter(x_occ_all, y_occ_all, c='red', s=0.5)
plt.scatter(x_na_all, y_na_all, c='grey')

plt.xlim(-map_offset, map_side - map_offset)  
plt.ylim(-map_offset, map_side - map_offset)  
plt.show()