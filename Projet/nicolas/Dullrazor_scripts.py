#Script for clusters
import math
import numpy as np
import cv2
import matplotlib.pyplot as plt
import skimage as sk
import scipy.signal
from scipy import ndimage
from scipy import signal
from scipy.interpolate import interp1d
from skimage import io, morphology, filters

#Création des directions
S0 = np.array([[0,1,1,1,1,1,1,1,1,1,1,1,0],
               [0,0,0,0,0,0,0,0,0,0,0,0,0],
               [0,0,0,0,0,0,0,0,0,0,0,0,0],
               [0,0,0,0,0,0,0,0,0,0,0,0,0],
               [0,0,0,0,0,0,0,0,0,0,0,0,0],
               [0,0,0,0,0,0,0,0,0,0,0,0,0],
               [0,0,0,0,0,0,0,0,0,0,0,0,0],
               [0,0,0,0,0,0,0,0,0,0,0,0,0],
               [0,0,0,0,0,0,0,0,0,0,0,0,0],
               [0,0,0,0,0,0,0,0,0,0,0,0,0],
               [0,0,0,0,0,0,0,0,0,0,0,0,0],
               [0,0,0,0,0,0,0,0,0,0,0,0,0],
               [0,0,0,0,0,0,0,0,0,0,0,0,0]])
S90 = np.transpose(S0)
S45 = np.matrix([[0,0,0,0,0,0,0,0,0],
                [0,1,0,0,0,0,0,0,0],
                [0,0,1,0,0,0,0,0,0],
                [0,0,0,1,0,0,0,0,0],
                [0,0,0,0,1,0,0,0,0],
                [0,0,0,0,0,1,0,0,0],
                [0,0,0,0,0,0,1,0,0],
                [0,0,0,0,0,0,0,1,0],
                [0,0,0,0,0,0,0,0,0]])


image = io.imread('ISIC_0000095.jpg')

red_channel = image[:, :, 0]
green_channel = image[:,:,1]
blue_channel = image[:,:,2]


def Greyscale_closing_one_channel(Color_channel):

    #Closing operations
    closing_horizontal = morphology.closing(Color_channel, S0)
    closing_diagonal = morphology.closing(Color_channel, S45)
    closing_vertical = morphology.closing(Color_channel, S90)
    #Taking the maximum
    max_closing = np.maximum(closing_horizontal, closing_diagonal, closing_vertical)
    
    return np.abs(Color_channel - max_closing)


Gr = Greyscale_closing_one_channel(red_channel)
Gv = Greyscale_closing_one_channel(green_channel)
Gb = Greyscale_closing_one_channel(blue_channel)


def binary_mask(Greyscale_closed_picture,T):
    Binary_mask = np.zeros_like(Greyscale_closed_picture, dtype=np.uint8) 
    Binary_mask[Greyscale_closed_picture > T] = 1
    return Binary_mask

#Dans le papier, le seuil est de 24
Seuil = 240
Binary_mask_red = binary_mask(Gr, Seuil)  
Binary_mask_green = binary_mask(Gv,Seuil)
Binary_mask_blue = binary_mask(Gb,Seuil)


Binary_mask = np.logical_or(Binary_mask_red,Binary_mask_green,Binary_mask_blue).astype(np.uint8)

#checking that is is a hair pixel

#Creating the lines around the pixel
(m, n) = Binary_mask.shape[0:2]
def ligne_n(Binary_mask, i, j):
    lgth = 0
    x = i
    while x >= 0 and Binary_mask[x][j] == 0:
        lgth += 1
        x -= 1
    return lgth

def ligne_ne(Binary_mask, i, j):
    lgth = 0
    x = i
    y = j
    while x >= 0 and y < n and Binary_mask[x][y] == 0:
        lgth += 1
        y += 1
        x -= 1
    return lgth

def ligne_e(Binary_mask, i, j):
    lgth = 0
    y = j
    while y < n and Binary_mask[i][y] == 0:
        lgth += 1
        y += 1
    return lgth

def ligne_se(Binary_mask, i, j):
    lgth = 0
    x = i
    y = j
    while x < m and y < n and Binary_mask[x][y] == 0:
        lgth += 1
        y += 1
        x += 1
    return lgth

def ligne_s(Binary_mask, i, j):
    lgth = 0
    x = i
    while x < m and Binary_mask[x][j] == 0:
        lgth += 1
        x += 1
    return lgth

def ligne_so(Binary_mask, i, j):
    lgth = 0
    x = i
    y = j
    while x < m and y >= 0 and Binary_mask[x][y] == 0:
        lgth += 1
        y -= 1
        x += 1
    return lgth

def ligne_o(Binary_mask, i, j):
    lgth = 0
    y = j
    while y >= 0 and Binary_mask[i][y] == 0:
        lgth += 1
        y -= 1
    return lgth

def ligne_no(Binary_mask, i, j):
    lgth = 0
    x = i
    y = j
    while x >= 0 and y >= 0 and Binary_mask[x][y] == 0:
        lgth += 1
        y -= 1
        x -= 1
    return lgth

#Hair verification

def hair_pixel_verification(Binary_mask):
    
    (m, n) = Binary_mask.shape[0:2]

    for i in range(m):
        for j in range(n):
            
            if Binary_mask[i][j]==0:
                Liste_lignes = np.array([ligne_n(Binary_mask, i, j), ligne_ne(Binary_mask, i, j), ligne_e(Binary_mask, i, j), ligne_se(
                    Binary_mask, i, j), ligne_s(Binary_mask, i, j), ligne_so(Binary_mask, i, j), ligne_o(Binary_mask, i, j), ligne_no(Binary_mask, i, j)])
                max_length = np.max(Liste_lignes)

                if max_length > 50:
                    c=0
                    position_max = np.argmax(Liste_lignes)
                    for e in range(0,len(Liste_lignes)):
                        if e != position_max:
                            if Liste_lignes[e]<10:
                                c=c+1
                    #Le pixel n'est pas un poil si une des distances est plus grande
                    if c!=7:
                        Binary_mask[i][j]=1
                else:
                    Binary_mask[i][j]=1

    return Binary_mask

def hair_pixel_verification_optimized(Binary_mask):
    m, n = Binary_mask.shape[0:2]
    # Create an array to store distances
    distances = np.zeros((m, n, 8), dtype=int)

    for i in range(m):
        for j in range(n):
            if Binary_mask[i][j]==0:
                # Calculate distances for all 8 directions and store them in 'distances' array
                distances[i, j, 0] = ligne_n(Binary_mask, i, j)
                distances[i, j, 1] = ligne_ne(Binary_mask, i, j)
                distances[i, j, 2] = ligne_e(Binary_mask, i, j)
                distances[i, j, 3] = ligne_se(Binary_mask, i, j)
                distances[i, j, 4] = ligne_s(Binary_mask, i, j)
                distances[i, j, 5] = ligne_so(Binary_mask, i, j)
                distances[i, j, 6] = ligne_o(Binary_mask, i, j)
                distances[i, j, 7] = ligne_no(Binary_mask, i, j)

    # Calculate maximum distances along each direction
    max_distances = np.max(distances, axis=2)
    for i in range(m):
        for j in range(n):
            if Binary_mask[i][j] == 0:
                max_length = max_distances[i, j]
                other_lengths = distances[i, j, distances[i, j] < 10]
                if max_length > 15 and len(other_lengths) >=5 :
                    Binary_mask[i][j] = 0
                else:
                    Binary_mask[i][j] = 1
    return Binary_mask

Binary_mask_verified = hair_pixel_verification_optimized(Binary_mask)


plt.figure('Binary mask verified')

plt.savefig('Binary_mask_verified.png')