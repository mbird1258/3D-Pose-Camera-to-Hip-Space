from itertools import combinations
import numpy as np
import mediapipe as mp

CVDModel = None

# Get cam view depth
def GetCamViewDepth(img, verbose=True):
    '''
    returns None if no bodies were detected
    returns the distance between the camera and its image plane if bodies are detected
    '''

    import tensorflow as tf
    import tensorflow_hub as hub
    import numpy as np
    from scipy.optimize import minimize

    global CVDModel

    # Get image and get pose 2D and 3D from image, thanks to https://istvansarandi.com/eccv22_demo/ (saved me after hours of digging and testing different models ;n;)
    if CVDModel == None: CVDModel = hub.load('model')              # will take a while
    preds = CVDModel.detect_poses(img, skeleton='smpl+head_30') # will take a while

    if len(preds['poses2d'].numpy()) == 0: return None
    if len(preds['poses3d'].numpy()) == 0: return None
    
    Pose2D = preds['poses2d'].numpy()[0]
    Pose2DNorm = (Pose2D - np.array(img.shape)[np.newaxis, [1,0]]/2) * np.array([1, -1])[np.newaxis, :]
    Pose3D = preds['poses3d'].numpy()[0] * np.array([1, -1, 1])[np.newaxis, :]


    def squareDistToLineThroughA(point, A):
        Ax, Ay, Az = A
        Bx, By, Bz = point

        n = (
            (Ay * Bz - Az * By) ** 2 +
            (Az * Bx - Ax * Bz) ** 2 +
            (Ax * By - Ay * Bx) ** 2
        )
        d = Bx**2 + By**2 + Bz**2
        squareDist = n / d

        return squareDist

    def objective(focalDistance, bodyScreenLandmarks, bodyWorldLandmarks):
        focalDistance = focalDistance[0]

        totalSquareDist = 0
        numJoints = len(bodyWorldLandmarks)
        for i in range(numJoints):
            totalSquareDist += squareDistToLineThroughA([*bodyScreenLandmarks[i], focalDistance], bodyWorldLandmarks[i])
        
        return totalSquareDist

    screenLandmarks = Pose2DNorm
    worldLandmarks = Pose3D

    initialGuess = 2000
    out = minimize(objective, initialGuess, args=(screenLandmarks, worldLandmarks), method='BFGS')#, options={"eps":1})

    if verbose:
        # print(out, end="\n\n\n")
        print(f">>> FOCAL DISTANCE: {out.x[0]} <<<", end="\n\n\n\n")

    return out.x[0]


# Get points from mediapipe
class PoseManager:
    def __init__(self, CamViewDepth = None, ModelPath = 'pose_landmarker.task'):
        self.CamViewDepth = CamViewDepth

        BaseOptions = mp.tasks.BaseOptions
        PoseLandmarker = mp.tasks.vision.PoseLandmarker
        PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
        VisionRunningMode = mp.tasks.vision.RunningMode

        options = PoseLandmarkerOptions(base_options=BaseOptions(model_asset_path=ModelPath), 
                                        num_poses = 10,
                                        min_pose_detection_confidence = 0.8,
                                        running_mode=VisionRunningMode.VIDEO)

        self.model = PoseLandmarker.create_from_options(options)

    def GetBodyPose(self, img, FrameMS):
        MPImg = mp.Image(image_format=mp.ImageFormat.SRGB, data=img)
        out = self.model.detect_for_video(MPImg, int(FrameMS))

        # https://www.geogebra.org/calculator/rz5tpm7b
        if self.CamViewDepth == None and len(out.pose_landmarks) != 0 and np.all(np.array([[joint.visibility for joint in body] for body in out.pose_landmarks]) > 0.8):
            self.CamViewDepth = GetCamViewDepth(img)

        if self.CamViewDepth == None:
            return np.empty([]), np.array([])

        ScreenLandmarks = np.array([[[(joint.x-0.5) * img.shape[1], (joint.y-0.5) * img.shape[0]] if joint.visibility > 0.3 else [np.nan, np.nan] for joint in body] for body in out.pose_landmarks])
        HipSpaceLandmarks = np.array([[[joint.x, joint.y, joint.z] if joint.visibility > 0.3 else [np.nan, np.nan, np.nan]  for joint in body] for body in out.pose_world_landmarks])

        WorldLandmarks = np.empty_like(HipSpaceLandmarks)
        for body in range(len(HipSpaceLandmarks)):
            WorldLandmarks[body] = HipToCameraSpace(ScreenLandmarks[body], HipSpaceLandmarks[body], self.CamViewDepth)

        return ScreenLandmarks, WorldLandmarks


# Transform points from mediapipe in which the origin of the coordinate space is around the hips into coordinate space around the camera
def HipToCameraSpace(ScreenLandmarks, HipSpaceLandmarks, CamViewDepth):
    z1 = CamViewDepth
    
    translation = []
    for ind1, ind2 in combinations(range(len(ScreenLandmarks)), 2):
        x1, y1 = ScreenLandmarks[ind1]
        X1, Y1 = ScreenLandmarks[ind2]

        if np.isnan(x1) or np.isnan(X1):
            continue

        dx2, dy2, dz2 = HipSpaceLandmarks[ind2]-HipSpaceLandmarks[ind1]
        
        mx = z1/x1
        mX = z1/X1
        my = z1/y1
        mY = z1/Y1
        
        x2 = (mX*dx2 - dz2)/(mx-mX)
        y2 = (mY*dy2 - dz2)/(my-mY)
        z2 = (mx*x2 + my*y2)/2

        translation.append(np.array([x2, y2, z2]) - HipSpaceLandmarks[ind1])
    
    translation = np.array(translation)
    translation = np.median(translation, axis=0)

    return HipSpaceLandmarks + translation[np.newaxis, :]
