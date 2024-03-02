import os
if __name__ == "__main__":
    dir = '/home/azuo/Trajectory_Prediction/Monocular-Depth-Estimation-Toolbox/data/kitti/kitti_benchmark_val.txt'
    out_dir = '/home/azuo/Trajectory_Prediction/Monocular-Depth-Estimation-Toolbox/data/kitti/kitti_raw_benchmark_val.txt'
    if os.path.exists(out_dir):
        os.remove(out_dir)
    with open(dir, 'r') as file, open(out_dir, 'w') as output_file:
        for line in file:
            left_part = line.split(' ')[0]
            if 'image_02' in left_part:
                output_file.write(left_part + '\n')
                
    with open(dir, 'r') as file, open(out_dir, 'a') as output_file:            
        for line in file:
            left_part = line.split(' ')[0]
            if 'image_03' in left_part:
                output_file.write(left_part + '\n')
                