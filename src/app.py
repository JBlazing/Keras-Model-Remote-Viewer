import flask
import glob
import os
import tensorflow as tf
import re
from fileLoader import loadFiles , processImages , deNormalize
import numpy as np
import cv2
app = flask.Flask(__name__ , template_folder='../web')
Model = None
regex = re.compile(r'GAN_*')

Model_Path = '/media/hdd/checkpoints/'
Image_Locations = '../testing/'

@app.route('/')
def root():

    images = os.listdir("./images/")
    
    print(images)
    return flask.render_template('index.html' ,len=len(images), images=images)
    
@app.route('/images/<file>')
def send_images(file):
        return flask.send_from_directory("images" , file)


@app.route('/update/<epoch>')
def update(epoch):
    epoch = int(epoch)
    
    if epoch == -1:
       Mods = os.listdir(Model_Path)
       sel = filter(regex.search , Mods)
       mod_file = max(sel , key=lambda a: int(a.split('_')[1]))
       mod_file = '{}{}'.format(Model_Path, mod_file)
    else:
        mod_file = '{}GAN_{}_checkpoint.h5'.format(Model_Path, epoch)
    Cur_model = mod_file
    

    Model = tf.keras.models.load_model(mod_file)
    
    files = glob.glob(Image_Locations+'*')
    files = np.array(files)
    batches = np.array_split(files , 10)
    print(batches)

    for batch in batches:
        
        imgs = loadFiles(batch)
        #assoc = loadFiles(batch[: , 1])
        processImages(imgs)
        
        output = Model(imgs)
        
        output = output.numpy()
        
        deNormalize(output)
        deNormalize(imgs)

        for file, i , o in zip(batch , imgs , output):
            file = file.split('/')[-1]
            print(file)
            out = np.concatenate((i , o) , axis=0)
            cv2.imwrite("images/" + file, out)
       
    return flask.redirect(flask.url_for('root'))
if __name__ == '__main__':
    
    app.run(host='0.0.0.0')