import numpy as np


class ModellingParameters(object):
    """
        Contains object class parameters used at the modelling process through active contours
    """
    # Fixed number of points to be provided to the first NN layer.
    NUM_POINTS_UPSAMPLE = 50
    NORMALIZATION_CENTER = np.array([0.5, 0.5, 0.5])
    class CAR(object):
        import numpy as np
        MODELLING_AFFECTED_AREA_FACTOR = 5
        RADIUS_SEARCH = 0.2
        STEP_SIZE = 0.1
        STEPS = 5
        SCALE = 0.5/((STEPS * STEP_SIZE) + ((RADIUS_SEARCH - STEP_SIZE) if RADIUS_SEARCH > STEP_SIZE else 0))
        ANCHOR_L = 4
        ANCHOR_W = 1.6
        ANCHOR_H = 1.6
        DIAGONAL = np.sqrt(ANCHOR_L**2 + ANCHOR_H**2 + ANCHOR_W**2)

    class PEDESTRIAN(object):
        RADIUS_SEARCH = 0.05
        STEP_SIZE = 0.05
        STEPS = 5

    class TRUCK(object):
        RADIUS_SEARCH = 0.5
        STEP_SIZE = 0.1
        STEPS = 5

    class MUG(object):
        import numpy as np
        MODELLING_AFFECTED_AREA_FACTOR = 5
        RADIUS_SEARCH = 0.005
        STEP_SIZE = 0.005
        STEPS = 5
        SCALE = 0.5 / ((STEPS * STEP_SIZE) + (
            (RADIUS_SEARCH - STEP_SIZE) if RADIUS_SEARCH > STEP_SIZE else 0))  # TODO: modify
        ANCHOR_L = 0.05
        ANCHOR_W = 0.05
        ANCHOR_H = 0.10
        DIAGONAL = np.sqrt(ANCHOR_L ** 2 + ANCHOR_H ** 2 + ANCHOR_W ** 2)