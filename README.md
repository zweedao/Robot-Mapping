# Robot-SLAM

Use a robot to map out indoor environment using Simultaneous Localisation &amp; Mapping (SLAM). Full details can be found in `Project Report.pdf`.

Example results:

![map 21](results/map_pred_21p.png)

![map 23](results/map_pred_23p.png)


## How to run:

- Put training data files (encoder, lidar, imu) into folder 'test'
- In your terminal, run:
	python slam.py <encoder_filename> <lidar_filename>
- For example:
	python slam.py Encoders20.mat Hokuyo20.mat
