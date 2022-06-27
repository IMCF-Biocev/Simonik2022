import cv2
import numpy as np
from scipy.spatial import distance_matrix
from sklearn.decomposition import PCA
from scipy import ndimage
from sympy import Point, Line 

def get_image_data(file_path):
    """
    Get image data from a file path
    """
    cap = cv2.VideoCapture(file_path)
    frames = []
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
        else:
            break
    return np.array(frames)

def compute_cell_vector(overlap):
    """
    Compute cell vector from cell and sperm cell 
    """
    # find two furtherst point of overlap
    points = overlap.T
    dist = distance_matrix(points, points)
    max_dist = np.unravel_index(np.argmax(dist), dist.shape)

    # create line
    line = points[max_dist[0]], points[max_dist[1]]

    # get vector and normalize it
    vector = line[1] - line[0]
    vector = vector / np.linalg.norm(vector)
    return line, vector

def compute_sperm_cell_vector(sperm_cell):
    """ 
    Compute sperm cell vector from sperm cell
    """
    # find two furtherst point of overlap
    points = np.array(np.where(sperm_cell)).T
    dist = distance_matrix(points, points)
    max_dist = np.unravel_index(np.argmax(dist), dist.shape)

    # create line
    line = points[max_dist[0]], points[max_dist[1]]

    # get vector and normalize it
    vector = line[1] - line[0]
    vector = vector / np.linalg.norm(vector)

    return line, vector

def compute_angle(vector_1, vector_2):
    """
    Compute angle between two vectors
    """
    # turn both vectors to unit vectors
    vector_1 = vector_1 / np.linalg.norm(vector_1)
    vector_2 = vector_2 / np.linalg.norm(vector_2) 

    # measure angle between two vectors
    angle = np.arccos(np.dot(vector_1, vector_2))
    angle = np.degrees(angle)
    return angle

def find_intersection(cell_line, sperm_line):
    """
    Find intersection between two lines
    """
    line_1 = Line(Point(cell_line[0]), Point(cell_line[1]))
    line_2 = Line(Point(sperm_line[0]), Point(sperm_line[1]))
    intersection = line_1.intersection(line_2)

    if len(intersection) == 0:
        print('No intersection')
        return None

    return (int(intersection[0][0]), int(intersection[0][1]))