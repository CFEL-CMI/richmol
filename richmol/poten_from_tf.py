"""
"""
import numpy as np
import tensorflow as tf
import sys

def Potential(path, coords):

    Model = tf.keras.models.load_model(path)
    print("read Model successfully")
    def Mol_descriptor(coords):
        r3 = np.sqrt((coords[:,0])**2+(coords[:,1])**2-2*(coords[:,0])*(coords[:,1])*np.cos(np.deg2rad(coords[:,2])))
        print(np.shape(coords[:,0]))

        rs = np.transpose(np.stack((coords[:,0], coords[:,1], r3)))
        r0 = np.array([0.9586, 0.9586, 1.516])

        return 1-np.exp(-(rs-r0))

    Features = Mol_descriptor(coords)
    print(np.shape(Features))
    print(Features)
    #self.Model.predict(Features)
