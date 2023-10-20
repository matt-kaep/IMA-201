import matplotlib.pyplot as plt
import numpy as np
from skimage import io
import mrlab as mr

# Charger l'image
ima = io.imread('pyra-gauss.tif')

# Liste des seuils à tester
seuils = [0.1, 0.3, 0.5,1,2,3,4]

# Parcourir les différentes valeurs de seuil
for i, seuil in enumerate(seuils):
    # Calculer le gradient
    gradx = mr.dericheGradX(mr.dericheSmoothY(ima, 0.3), 0.3)
    grady = mr.dericheGradY(mr.dericheSmoothX(ima, 0.3), 0.3)

    # Calculer la norme du gradient
    norme = np.sqrt(gradx**2 + grady**2)

    # Calculer la direction du gradient
    direction = np.arctan2(grady, gradx)

    # Calculer les contours de la norme du gradient
    contoursnorme = (norme > seuil)

    # Calculer les maxima du gradient dans la direction du gradient
    contours = np.uint8(mr.maximaDirectionGradient(gradx, grady))
    valcontours = (norme > seuil) * contours

    # Afficher les images pour le seuil courant
    plt.figure(i+1)
    plt.suptitle('Seuil = {}'.format(seuil))
    plt.title('Image originale')
    plt.imshow(ima, cmap='gray')

    plt.subplot(2, 2, 1)
    plt.title('Norme du gradient')
    plt.imshow(norme, cmap='gray')

    plt.subplot(2, 2, 2)
    plt.title('Direction du gradient')
    plt.imshow(direction, cmap='gray')

    plt.subplot(2, 2, 3)
    plt.title('Norme seuillée')
    plt.imshow(255*contoursnorme, cmap='gray')

    plt.subplot(2, 2, 4)
    plt.title('Maxima du gradient dans la direction du gradient')
    plt.imshow(255*valcontours, cmap='gray')

    # Ajuster les marges entre les sous-graphiques
    plt.subplots_adjust(wspace=0.4, hspace=0.6)

    # Afficher toutes les figures
    plt.show()
