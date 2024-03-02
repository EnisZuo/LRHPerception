import os, cv2
if __name__ == '__main__':
    frame_path = '/home/azuo/LRHPerception/outputs/saved_frames'
    output_dir = os.path.join(frame_path, 'video')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    video_dir = os.path.join(output_dir, 'video.mp4')
    frame_rate = 10
    frame_size = (1216, 352)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(video_dir, fourcc, frame_rate, frame_size)

    for frame_file in sorted(os.listdir(frame_path)):
        # Ensure file is an image
        if frame_file.endswith('.png'):
            # Read image
            # print(frame_file)
            img = cv2.imread(os.path.join(frame_path, frame_file))
            # Add image to video
            video.write(img)
    video.release()