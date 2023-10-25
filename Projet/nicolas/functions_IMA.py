import math
import numpy as np
import copy
import cv2
import matplotlib.pyplot as plt
import skimage as sk
import scipy.signal

from PIL import Image
from scipy import ndimage
from scipy import signal
from scipy.interpolate import interp1d
from skimage import io, morphology, filters

#resize images
#width = 1040
#height = 800
def resize_image(original_image, new_width, new_height):
    try:
        resized_image = original_image.resize((new_width, new_height), Image.ANTIALIAS)
        return resized_image
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return False


##############################################################################################################
# Création des éléments structurants
def create_matrices_diago(length, width):
    S45 = np.zeros((length, length))
    for e in range(width):
        for l in range(length):
            if l-e >= 0:
                S45[l, l-e] = 1
    S45[0, 0] = 0
    S45[1,0] = 0
    S45[length-1, length-1] = 0

    return S45

def create_matrices_horizontal(length,width):
    S0 =  np.zeros((length, length))
    for e in range(width):
        S0[e+length//2, :] = 1
    return S0

def create_matrix_angle(length, angle_degrees):
    matrix = np.zeros((length,length), dtype=int)
    slope = np.tan(np.radians(angle_degrees))
    # Loop through each row and column to determine if a pixel should be "on"
    for row in range(length):
        for col in range(length):
            # Calculate the corresponding column for the given row that aligns with the line
            aligned_col = int(slope * (row - length // 2) + length // 2)

        # Set the pixel to 1 if it's within a certain range of the aligned column
            if col == aligned_col or col == aligned_col + 1:
                matrix[row, col] = 1
    return matrix

#Fonction maximum
def maximum_arrays(List_array):
    n = len(List_array)
    for e in range(n):
        if e == 0:
            max_array = List_array[e]
        else:
            max_array = np.maximum(max_array,List_array[e])
    return max_array

#Opération de fermeture
def Greyscale_closing_one_channel(Color_channel,element_structurant):
    # The first element bust be the diagonal element
    
    element_structurant_extended = element_structurant
    for e in range(1,len(element_structurant)):
        element_structurant_extended.append(np.transpose(element_structurant[e]))
  
    #Closing operations
    Closing_directions_array = []
    for e in range(len(element_structurant_extended)):
        Closing_directions_array.append(morphology.closing(Color_channel, element_structurant_extended[e]))
    
    #Taking the maximum with the function defined to take more than 2 arrays
    max_closing = maximum_arrays(Closing_directions_array)    
    return np.abs(Color_channel - max_closing)


#binarisation du mask
def binary_mask(Greyscale_closed_picture,Threshold):
    Binary_mask = np.zeros_like(Greyscale_closed_picture, dtype=np.uint8) 
    Binary_mask[Greyscale_closed_picture > Threshold] = 1
    return Binary_mask


##############################################################################################

#n colums and m rows
#i is the row and j the column
#Knowing that the we will look only if the maximum distance in one direction must be greater than 3
#We will modify moidfy the while loop to make less iterations
#The maximum distance will then be 18
#These functions return the length of the line, and the coordinates where it stops to be considered a hair or when the length reaches the maximum value we fixed
def ligne_n(Binary_mask, i, j):
    (m, n) = Binary_mask.shape[0:2]
    lgth = 0
    x = i
    y = i
    while y >= 0 and Binary_mask[y][j] == 0 and lgth < 18:
        lgth += 1
        y -= 1
    return lgth , y , x

def ligne_ne(Binary_mask, i, j):
    (m, n) = Binary_mask.shape[0:2]
    lgth = 0
    y = i
    x = j
    while y >= 0 and x < n and Binary_mask[y][x] == 0  and lgth < 18:
        lgth += 1
        y -= 1
        x += 1
    return lgth , y , x

def ligne_e(Binary_mask, i, j):
    (m, n) = Binary_mask.shape[0:2]
    lgth = 0
    x = j
    y=i
    while x < n and Binary_mask[i][x] == 0  and lgth < 18:
        lgth += 1
        x += 1
    return lgth , y , x

def ligne_se(Binary_mask, i, j):
    (m, n) = Binary_mask.shape[0:2]
    lgth = 0
    y = i
    x = j
    while x < n and y < m and Binary_mask[y][x] == 0  and lgth < 18:
        lgth += 1
        y += 1
        x += 1
    return lgth, y, x

def ligne_s(Binary_mask, i, j):
    (m, n) = Binary_mask.shape[0:2]
    lgth = 0
    x = j
    y = i
    while y < m and Binary_mask[y][j] == 0  and lgth < 18:
        lgth += 1
        y += 1
    return lgth, y , x

def ligne_so(Binary_mask, i, j):
    (m, n) = Binary_mask.shape[0:2]
    lgth = 0
    x = j
    y = i
    while y < m and x >= 0 and Binary_mask[y][x] == 0  and lgth < 18:
        lgth += 1
        x -= 1
        y += 1
    return lgth, y , x

def ligne_o(Binary_mask, i, j):
    (m, n) = Binary_mask.shape[0:2]
    lgth = 0
    x = j
    y = i
    while x >= 0 and Binary_mask[i][x] == 0  and lgth < 18:
        lgth += 1
        x -= 1
    return lgth, y , x

def ligne_no(Binary_mask, i, j):
    (m, n) = Binary_mask.shape[0:2]
    lgth = 0
    x = j
    y = i
    while x >= 0 and y >= 0 and Binary_mask[y][x] == 0  and lgth < 18:
        lgth += 1
        y -= 1
        x -= 1
    return lgth, y , x
    

#Hair pixel verification
def hair_pixel_verification_distances(Binary_mask_original,seuil_hair_max,nombres_directions_peau,taille_max_poil_autre):
    Binary_mask = copy.deepcopy(Binary_mask_original)
    m, n = Binary_mask.shape[0:2]
    # Create an array to store distances
    distances = np.zeros((m, n, 8, 3), dtype=int)
    #The element distances[i,j,k,:] contains the length of the line in the direction k, and the coordinates where it stops to be considered a hair or when the length reaches the maximum value we fixed
    for i in range(m):
        for j in range(n):
            if Binary_mask[i][j]==0:
                # Calculate distances for all 8 directions and store them in 'distances' array
                distances[i, j, 0,:] = ligne_n(Binary_mask, i, j)
                distances[i, j, 1,:] = ligne_ne(Binary_mask, i, j)
                distances[i, j, 2,:] = ligne_e(Binary_mask, i, j)
                distances[i, j, 3,:] = ligne_se(Binary_mask, i, j)
                distances[i, j, 4,:] = ligne_s(Binary_mask, i, j)
                distances[i, j, 5,:] = ligne_so(Binary_mask, i, j)
                distances[i, j, 6,:] = ligne_o(Binary_mask, i, j)
                distances[i, j, 7,:] = ligne_no(Binary_mask, i, j)

    for i in range(m):
        for j in range(n):
            if Binary_mask[i][j] == 0:
                max_length = np.max(distances[i, j,:,0])
                other_lengths = distances[i, j, distances[i, j,:,0] < taille_max_poil_autre]
                if max_length > seuil_hair_max and len(other_lengths) >= nombres_directions_peau :
                    Binary_mask[i][j] = 0
                else:
                    Binary_mask[i][j] = 1
    return Binary_mask, distances
 

## Here a light modification to have all the functions defined before called in only one function
def hair_detection(image, element_structurant, seuil_binarisation, Parameters_hair_verification):
    # Parameters is the array concerning the parameters of the hair verification
    max_length_hair = Parameters_hair_verification[0]
    nombres_directions_peau = Parameters_hair_verification[1]
    taille_max_poil_autre = Parameters_hair_verification[2]

    # Parameters_hair_verification is a list of 3 parameters: seuil_hair_max,nombres_directions_peau,taille_max_poil_autre
    # Split the image into its RGB channels
    red_channel = image[:, :, 0]
    green_channel = image[:, :, 1]
    blue_channel = image[:, :, 2]

    # Apply grayscale closing to each channel
    Gr = Greyscale_closing_one_channel(red_channel, element_structurant)
    Gv = Greyscale_closing_one_channel(green_channel, element_structurant)
    Gb = Greyscale_closing_one_channel(blue_channel, element_structurant)

    # Binarize the masks
    Binary_mask_red = binary_mask(Gr, seuil_binarisation)
    Binary_mask_green = binary_mask(Gv, seuil_binarisation)
    Binary_mask_blue = binary_mask(Gb, seuil_binarisation)

    # Combine the masks
    Binary_mask = np.logical_or.reduce([Binary_mask_red, Binary_mask_green, Binary_mask_blue]).astype(np.uint8)
    (m, n) = Binary_mask.shape[0:2]

    # Hair pixel verification
    Binary_mask_verified, distances = hair_pixel_verification_distances(Binary_mask,max_length_hair,nombres_directions_peau,taille_max_poil_autre)
     
    return Binary_mask, Binary_mask_verified, distances

##############################################################################################################
# Fonction de remplacement des pixels
'''
def find_nearest_non_hair_pixels(binary_mask_verified, distances, i, j):
    # return the coordinates of the two nearest non-hair pixels
    # -1 if it is not a hair pixel
    # It will be applied where there should be normally a hair pixel
    # So the -1 should not normally be returned
    (m, n) = binary_mask_verified.shape[0:2]
    nearest_non_hair_pixels = []
    if binary_mask_verified[i,j] == 0:
        #min1_index is the direction of the minimum distance where there is a non-hair pixel
        min1_index = np.argmin(distances [i, j,:,0])
        i_min1 = distances[i, j, min1_index, 1]
        j_min1 = distances[i, j, min1_index, 2]
        
        #min2_index is the direction of the second minimum distance where there is a non-hair pixel
        distances_without_min1 = distances[i, j, distances[i, j, :, 0] != distances[i, j, min1_index, 0]]
        print()
        if (len(distances_without_min1) == 0):
            i_min2 = i_min1
            j_min2 = j_min1 
        else:
            min2 = np.min(distances[i, j, distances[i, j, :, 0] != distances[i, j, min1_index, 0]])
            min2_directions = np.where(distances[i, j, :, 0] == min2)
            if len(min2_directions[0]) > 0:
                i_min2 = min2_directions[0][0]
                j_min2 = min2_directions[0][1]
            else:
                i_min2 = i_min1
                j_min2 = j_min1

        nearest_non_hair_pixels.append((i_min1, j_min1))
        nearest_non_hair_pixels.append((i_min2, j_min2))
        #filterting correct data
        nearest_non_hair_pixels = [coord for coord in nearest_non_hair_pixels if coord[0] >= 0 and coord[0] < binary_mask_verified.shape[0] and coord[1] >= 0 and coord[1] < binary_mask_verified.shape[1]]
        return nearest_non_hair_pixels
'''
def find_nearest_non_hair_pixels(binary_mask_verified, distances, i, j):
    if binary_mask_verified[i, j] != 0:
        return (-1, -1)

    min1_index = np.argmin(distances[i, j, :, 0])
    min1_coords = (distances[i, j, min1_index, 1], distances[i, j, min1_index, 2])

    distances_without_min1 = distances[i, j, distances[i, j, :, 0] != distances[i, j, min1_index, 0], :]
    if distances_without_min1.size == 0:
        return (min1_coords, min1_coords)

    min2_index = np.argmin(distances_without_min1[:, 0])
    min2_coords = (distances_without_min1[min2_index, 1], distances_without_min1[min2_index, 2])
    
    return (min1_coords, min2_coords)

def hair_replacement(image, Binary_mask_verified, distances):
    new_image = copy.deepcopy(image)
    E=0
    print('image shape', new_image.shape)
    print('binary mask shape', Binary_mask_verified.shape)
    m, n = Binary_mask_verified.shape[0:2]
    for i in range(m):
        for j in range(n):
            if Binary_mask_verified[i][j] == 0:
                min1, min2 = find_nearest_non_hair_pixels(Binary_mask_verified,distances, i, j)
                # Check if there are at least two non-hair pixels
                condition1 = min1[0] <800 and min1[1] <1040
                condition2 = min2[0] <800 and min2[1] <1040
                if min1 != (-1, -1) and min2 != (-1, -1) and condition1 and condition2:
                    
                    #Taking the average of both
                    print(min1)
                    print(min2)
                    #new_image[i][j] = (image[first[0]][first[1]] + image[second[0]][second[1]]) / 2
                    new_image[i][j] = (image[min1[0]][min1[1]] + image[min2[0]][min2[1]]) / 2
               
                    '''
                    first = non_hair[0]
                    new_image[i][j] = image[first[0]][first[1]]
                    print("One hair found")
                    '''
                else:
                    E=E+1
                    pass

    return new_image,E
 