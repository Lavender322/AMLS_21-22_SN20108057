import pyfeats
import numpy as np
from numpy import save

def extract_features(X):
    glds_features = []
    glrlm_features = []
    lbp_features = []
    zernikes_moments_features = []

    for image in X:
        glds_features_image, glds_label = pyfeats.glds_features(image, None, Dx=[0,1,1,1], Dy=[1,1,0,-1])
        glds_features.append(glds_features_image)
    glds_features = np.asarray(glds_features)
    save("features/glds_features.npy", glds_features) # (3000, 5)

    for image in X:
        glrlm_features_image, glrlm_label = pyfeats.glrlm_features(image, None, Ng=256)
        glrlm_features.append(glrlm_features_image)
    glrlm_features = np.asarray(glrlm_features)
    save("features/glrlm_features.npy", glrlm_features) # (3000, 11)

    for image in X:
        lbp_features_image, lbp_label = pyfeats.lbp_features(image, None)
        lbp_features.append(lbp_features_image)
    lbp_features = np.asarray(lbp_features)
    save("features/lbp_features.npy", lbp_features) # (3000, 6)

    for image in X:
        zernikes_moments_features_image, zernikes_moments_label = pyfeats.zernikes_moments(image)
        zernikes_moments_features.append(zernikes_moments_features_image)
    zernikes_moments_features = np.asarray(zernikes_moments_features)
    save("zernikes_moments_features", zernikes_moments_features) # (3000, 25)

    # Combine all four feature arrays
    X = np.concatenate((glds_features, glrlm_features, lbp_features, zernikes_moments_features), axis=1) # (3000, 47)
    save("features.npy", X)