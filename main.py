import os
import torch
from artist import Artist
from critic import Critic
from image_generator import ImageGenerator
import shutil
from PIL import Image






# new_image_folder = os.path.join("faces_images", "all_in_one")
new_image_folder = os.path.join("landscape_images", "all_in_one")
# new_image_folder = os.path.join("bird_images", "all_in_one")


def make_data_folder():
    
    try:
        shutil.rmtree(os.path.join(new_image_folder, "real"))
    except:
        pass
    os.mkdir(os.path.join(new_image_folder, "real"))
    image_folder = os.path.join("landscape_images", "landscapes1")
    # image_folder = os.path.join("faces_images", "faces")
    # image_folder = os.path.join("bird_images", "bright_and_colorful")
    # image_folder = os.path.join("bird_images", "archive", "train")
    g = os.getcwd()
    os.chdir(image_folder)
    subs = os.listdir()
    
    for sub in subs:
   
        if os.path.isdir(sub):
            
            file_names = os.listdir(sub)
            for file_name in file_names:
                new_name = f"{sub}_{file_name}".replace(" ", "_")
                image_path = os.path.join(sub, file_name)
                shutil.copyfile(image_path, os.path.join("..", "all_in_one", "real", new_name))


    os.chdir(g)







if __name__ == "__main__":
    make_data_folder()
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print(device)
    # render_folder = os.path.join("renderings","bird_renders")
    # render_folder = os.path.join("renderings","humans2")
    render_folder = os.path.join("renderings","landscapes")
    render_collection = "new_settings"
    
    render_path = os.path.join(render_folder, render_collection)
    try: 
        os.mkdir(render_path)
    except:
        pass

    generator = ImageGenerator(device = device, render_path = render_path)
    generator.get_dataloader(new_image_folder)
    generator.train(1000)

