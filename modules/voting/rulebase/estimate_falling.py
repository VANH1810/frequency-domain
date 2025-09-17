import numpy as np

class FallingEstimator:
    def __init__(self, threshold_angle=45):
        """
        Class for detecting a falling state based on the body vector.

        :param threshold_angle: Angle threshold (degrees) to determine a fall. 
                                If the angle with the y-axis is less than this value, 
                                it is considered a fall.
        """
        self.threshold_angle = threshold_angle  

    def compute_angle(self, shoulder_mid, hip_mid):
        """
        Computes the angle between the body vector and the positive y-axis.

        :param shoulder_mid: (x, y) coordinates of the midpoint between shoulders.
        :param hip_mid: (x, y) coordinates of the midpoint between hips.
        :return: Angle (degrees) between the body vector and the y-axis.
        """
        vector = np.array([hip_mid[0] - shoulder_mid[0], hip_mid[1] - shoulder_mid[1]])  # Body vector
        unit_vector_y = np.array([0, 1])  # Unit vector along the y-axis (pointing downward)

        # Compute the angle using the formula: cos(theta) = (a . b) / (|a| * |b|)
        dot_product = np.dot(vector, unit_vector_y)
        magnitude_vector = np.linalg.norm(vector)
        
        

        angle_radians = np.arccos(dot_product / magnitude_vector)
        angle_degrees = np.degrees(angle_radians)

        return angle_degrees

    def is_falling(self, keypoints):
        """
        Checks whether a person is falling based on the body vector.

        :param keypoints: Array of keypoint coordinates (17,2) with format [x, y].
        :return: True if falling, False otherwise.
        """
        if len(keypoints) < 17:
            return False  # Not enough keypoints data to process

        # Compute the midpoint of shoulders
        shoulder_mid = ((keypoints[5][0] + keypoints[6][0]) / 2, (keypoints[5][1] + keypoints[6][1]) / 2)
        # Compute the midpoint of hips
        hip_mid = ((keypoints[11][0] + keypoints[12][0]) / 2, (keypoints[11][1] + keypoints[12][1]) / 2)

        # Compute the inclination angle of the body vector
        angle = self.compute_angle(shoulder_mid, hip_mid)

        # If the angle is below the threshold, consider it a fall
        if np.isnan(angle):
            return False, 0
        return angle > self.threshold_angle, int(angle)
