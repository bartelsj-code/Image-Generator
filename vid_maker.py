import cv2
import os

def images_to_video(input_folder, video_name, fps):
    # images = [img for img in os.listdir(image_folder) if img.endswith(".png")]

    images = []
    for i in range(1,50):
        img = os.path.join(input_folder, f'image{i}.png')
        images.append(img)

    img_path = os.path.join(images[0])

    frame = cv2.imread(img_path)
    height, width, layers = frame.shape

    video = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    for image in images:
        video.write(cv2.imread(image))

    cv2.destroyAllWindows()
    video.release()
# D:\bird_gen\renderings\bird_renders\fantasy2\line2\image53.png
# Example usage:
input_folder = os.path.join("renderings","bird_renders","fantasy2", "line2")
images_to_video(input_folder, 'output_video.mp4', 24)  # Change the folder path and video name as needed