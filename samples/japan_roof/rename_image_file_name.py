import os

cnt = 0
for path, dirs, files in os.walk("/root/japan_roof_with_trees/"):
    for file in files:
        origin_path = os.path.join(path, file)
        
        new_filename = "%06d"% cnt + "_image.png"
        new_path = os.path.join(path, new_filename)
        print(origin_path)
        print (new_path)
        
        os.rename(origin_path, new_path)
        
        cnt += 1