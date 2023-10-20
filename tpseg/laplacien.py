import matplotlib.pyplot as plt
import numpy as np
from skimage import io
import mrlab as mr

# Charger l'image
ima = io.imread('pyra-gauss.tif')

# Liste des valeurs d'alpha à tester
alphas = [0.3, 0.5, 1, 2,3,4]

# Parcourir les différentes valeurs d'alpha
for i, alpha in enumerate(alphas):
    # Calculer le laplacien
    gradx = mr.dericheGradX(mr.dericheSmoothY(ima, alpha), alpha)
    grady = mr.dericheGradY(mr.dericheSmoothX(ima, alpha), alpha)
    gradx2 = mr.dericheGradX(mr.dericheSmoothY(gradx, alpha), alpha)
    grady2 = mr.dericheGradY(mr.dericheSmoothX(grady, alpha), alpha)
    lpima = gradx2 + grady2
    posneg = (lpima >= 0)

    # Calculer les contours
    nl, nc = ima.shape
    contours = np.uint8(np.zeros((nl, nc)))
    for i in range(1, nl):
        for j in range(1, nc):
            if (((i > 0) and (posneg[i-1, j] != posneg[i, j])) or
                ((j > 0) and (posneg[i, j-1] != posneg[i, j]))):
                contours[i, j] = 255

    # Afficher les images pour l'alpha courant
    plt.figure(i+1)
    plt.suptitle('Alpha = {}'.format(alpha))
    plt.subplot(2, 2, 1)
    plt.title('Image originale')
    plt.imshow(ima, cmap='gray')

    plt.subplot(2, 2, 2)
    plt.title('Laplacien')
    plt.imshow(lpima, cmap='gray')

    plt.subplot(2, 2, 3)
    plt.title('Laplacien binarisé -/+')
    plt.imshow(255*posneg, cmap='gray')

    plt.subplot(2, 2, 4)
    plt.title('Contours')
    plt.imshow(contours, cmap='gray')

    plt.subplots_adjust(wspace=0.4, hspace=0.4)
    plt.show()

