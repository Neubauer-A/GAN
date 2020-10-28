import os
from PIL import Image
import multiprocessing as mp

%cd /content/drive/My Drive/artstuff/
from models.processor import ImageProcessor
%cd /content/

# Download and unzip dataset.
def get_raw_dataset(aria=False, url='http://ia802804.us.archive.org/14/items/wikiart-dataset/wikiart.tar.gz', del_zip=False):
    if aria:
        os.system('aria2c -x 16 -s 16 %s' %(url))
    else:
        os.system('wget %s' %(url))
    zipfile_name = url[::-1][:url[::-1].index('/')][::-1]
    if zipfile_name.endswith('.tar.gz'):
        os.system('tar -xvf %s' %(zipfile_name))
    else:
        os.system('unzip %s' %(zipfile_name))
    if del_zip:
        os.system('rm %s' %(zipfile_name)) 

# Center crop images to as large a square as possible.
def square_crop(filename):
    im = Image.open(filename)
    new_size = min(im.size)
    left = (im.size[0] - new_size)/2
    top = (im.size[1] - new_size)/2
    right = (im.size[0] + new_size)/2
    bottom = (im.size[1] + new_size)/2
    return im.crop((left, top, right, bottom))

def crop_worker(jobinfo):
    try:
        oldfile, newfile = jobinfo
        new_image = square_crop(oldfile)
        new_image.save(newfile)
    except:
        pass

# Make new directories for cropped images and send to crop workers.
def crop_dataset(data_dir):
    if not os.path.isdir('cropped'):
        os.system('mkdir cropped')
    jobs = []
    for style_dir in os.listdir(data_dir):
        if os.path.isdir(data_dir+'/'+style_dir):
            os.system('mkdir cropped/%s' %(style_dir))
            for filename in os.listdir(data_dir+'/'+style_dir):
                oldfile = data_dir+'/'+style_dir+'/'+filename
                newfile = 'cropped/'+style_dir+'/'+filename
                jobs.append((oldfile, newfile))
    cores = mp.cpu_count()  
    pool = mp.Pool(processes=cores)
    p = pool.map_async(crop_worker, jobs)
    try:
        p.get()
    except KeyboardInterrupt: 
        pool.terminate()
        pool.join()
        sys.exit(-1)
    pool.close()
    pool.join()

def rec_worker(jobinfo):
    source, data_file, size = jobinfo
    processor = ImageProcessor(size)
    processor.make_tfrec(source, data_file)

# Make a new directory of tfrecords of images as tensors.
def make_tfrec_dir(size, source='cropped', rec_dir='tfrecords'):
    if not os.path.isdir(rec_dir):
        os.system('mkdir %s' %(rec_dir))
    jobs = []
    for dir in os.listdir(source):
        if os.path.isdir(source+'/'+dir):
            jobs.append((source+'/'+dir, rec_dir+'/'+dir+'.tfrec', size))
    cores = mp.cpu_count()  
    pool = mp.Pool(processes=cores)
    p = pool.map_async(rec_worker, jobs)
    try:
        p.get()
    except KeyboardInterrupt: 
        pool.terminate()
        pool.join()
        sys.exit(-1)
    pool.close()
    pool.join()
