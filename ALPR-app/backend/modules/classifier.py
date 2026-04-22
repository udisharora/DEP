import cv2

def classify_restoration_module(cv_img):
    """
    Classifies the optimal restoration module for an image based on mathematical heuristics.
    Returns one of: 'darkir', 'dehaze', 'derain', or 'normal'.
    """
    try:
        gray_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
        blur_score = float(cv2.Laplacian(gray_img, cv2.CV_64F).var())
        brightness = float(gray_img.mean())
        # Example heuristics (tune as needed):
        if blur_score < 50 and brightness < 60:
            return 'darkir'
        elif brightness < 100:
            return 'dehaze'
        elif blur_score < 100:
            return 'derain'
        else:
            return 'normal'
    except Exception:
        return 'normal'
