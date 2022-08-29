# covid dataset process
import os
import PIL.Image as Image
from torchvision import transforms as transforms
from mean_teacher import data

data_dir = "./data-local/COVID/"
for line in open("./data-local/labels/COVID/train0.3.txt"):
    img = line.split(" ")
    flag = 0
    dir_temp, img_name = "", ""
    if len(img) == 3:
        flag = 1
        img_name = img[0]+" "+img[1]
    elif len(img) == 2:
        img_name = img[0]
    dir_temp = img_name.split("-")[0]
    if dir_temp == "Lung_Opacity":
        flag = 1
    path_temp = os.path.join(dir_temp, img_name)
    img_path = os.path.join(data_dir, path_temp)
    image = Image.open(img_path)
    if flag == 0:
        image.save(os.path.join("./data-local/images/COVID19/train0.3", dir_temp)+"/"+img_name)
    else:
        image.save(os.path.join("./data-local/images/COVID19/train0.3", "Other") + "/" + img_name)
file_path = "./data-local/COVID/COVID"
path_list = os.listdir(file_path)
path_name = []
for i in path_list:
    path_list_temp = os.listdir(os.path.join(file_path,i))
    for j in path_list_temp:
        path_name.append(j + " " + j.split("-")[0])
for file_name in path_name:
    with open("./data-local/COVID/dataset.txt", "a") as file:
        file.write(file_name + "\n")
        print(file_name)
    file.close()


train_transformation = data.TransformOnce(transforms.Compose([
            data.RandomTranslateWithReflect(4),
            transforms.RandomHorizontalFlip(),
        ]))
for i in path_list:
    image_path = os.path.join(file_path, i)
    old_image = Image.open(image_path)
    new_image = train_transformation(old_image)
    new_image_name = i.split(".")[0]+"_"
    print(os.path.join(file_path, new_image_name + ".png"))
    # break
    new_image.save(os.path.join(file_path, new_image_name + ".png"))



path_name = []

for i in path_list:
    # print(i, i.__class__)
    # break
    path_name.append(i+" "+"Other")
    # print(path_name[0])
    # break


for file_name in path_name:
    with open("./data-local/COVID/other.txt", "a") as file:
        file.write(file_name + "\n")
        print(file_name)
    file.close()
