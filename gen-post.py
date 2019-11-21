import os
import errno
from datetime import datetime

print("Generating A New Post\n")

post_name = input('Input Post Name: ')

date_time = datetime.now()
date_time_dir = date_time.strftime("%Y-%m-%d")
date_time_post = date_time.strftime("%Y-%m-%d %H:%M:%S")

p_name = post_name.replace(" ","-")
p_name = p_name.replace("[","")
p_name = p_name.replace("]","")
p_name = p_name.lower()

f_name = date_time_dir+"---"+p_name
dir = "./src/pages/articles/"+f_name+"/"
f_dir = dir+f_name+".md"

try:
    if not(os.path.isdir(dir)):
        os.makedirs(os.path.join(dir))
except OSError as e:
    if e.errno != errno.EEXIST:
        print("Failed to create directory!!!!!")
        raise

print("Generating post : ",f_dir)

with open(f_dir, 'w') as f:
    f.write('---')
    f.write('\n')
    f.write('draft:     true')
    f.write('\n')
    f.write('title:     \"'+post_name+'\"')
    f.write('\n')
    f.write('date:      \"'+date_time_post+'\"')
    f.write('\n')
    f.write('layout:    post')
    f.write('\n')
    f.write('path:      \"/posts/'+p_name+'/\"')
    f.write('\n')
    f.write('category:  \"\"')
    f.write('\n')
    f.write('tags: ')
    f.write('\n')
    f.write('description: ""')
    f.write('\n')
    f.write('---')
    f.write('\n')

print("Done :)")