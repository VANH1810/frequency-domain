import numpy as np

"""
Limitation: Cannot detect movement when both the camera and the person are moving directly toward each other.
"""

class KneeAngleDetector:
    def __init__(self, threshold=10):
        """
        Initializes the detector with a movement threshold.

        :param threshold: Angle difference threshold to detect movement.
        """
        self.threshold = threshold
    
    @staticmethod
    def calculate_angle(a, b, c):
        """
        Computes the angle between three points a, b, and c using the cosine rule.

        :param a: (x, y) coordinates of the first point.
        :param b: (x, y) coordinates of the middle point.
        :param c: (x, y) coordinates of the third point.
        :return: Angle in degrees.
        """
        ba = a - b  
        bc = c - b  
        
        cos_theta = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        cos_theta = np.clip(cos_theta, -1.0, 1.0)  # Clip to handle numerical issues
        
        return np.degrees(np.arccos(cos_theta))
    
    def compute_knee_angles(self, keypoints_sequence):
        """
        Computes knee angles for each frame in the keypoints sequence.

        :param keypoints_sequence: List of keypoints for each frame.
        :return: List of tuples (left_knee_angle, right_knee_angle) for each frame.
        """
        knee_angles = []
        
        for keypoints in keypoints_sequence:
            left_hip, left_knee, left_foot = keypoints[11], keypoints[13], keypoints[15]
            right_hip, right_knee, right_foot = keypoints[12], keypoints[14], keypoints[16]
            
            left_angle = self.calculate_angle(left_hip, left_knee, left_foot)
            right_angle = self.calculate_angle(right_hip, right_knee, right_foot)
            
            knee_angles.append((left_angle, right_angle))
        
        return knee_angles


    def compute_leg_vectors_angle(self, keypoints_sequence):
        """
        Computes the angle between the vectors:
        - Right Knee → Right Ankle
        - Left Knee → Left Ankle
        for each frame in the sequence.

        :param keypoints_sequence: List of keypoints for each frame.
        :return: List of angles in degrees.
        """
        leg_angles = []

        for keypoints in keypoints_sequence:
            left_knee, left_ankle = keypoints[13], keypoints[15]
            right_knee, right_ankle = keypoints[14], keypoints[16]

            left_vector = left_ankle - left_knee
            right_vector = right_ankle - right_knee

            cos_theta = np.dot(left_vector, right_vector) / (np.linalg.norm(left_vector) * np.linalg.norm(right_vector))
            cos_theta = np.clip(cos_theta, -1.0, 1.0)

            angle = np.degrees(np.arccos(cos_theta))
            leg_angles.append(angle)

        return leg_angles
        
    
    def detect_movement(self, keypoints_sequence):
        """
        Determines if the person is moving based on changes in knee angles.

        :param keypoints_sequence: List of keypoints for each frame.
        :return: "Walking" if movement is detected, otherwise "Standing Still".
        """
        knee_angles = self.compute_knee_angles(keypoints_sequence)
        
        angle_differences = [
            (abs(knee_angles[i][0] - knee_angles[i-1][0]),
             abs(knee_angles[i][1] - knee_angles[i-1][1]))
            for i in range(1, len(knee_angles))
        ]
        
        avg_change = np.mean(angle_differences, axis=0)
        

        leg_angles = self.compute_leg_vectors_angle(keypoints_sequence)
        
        # print("compute_leg_vectors_angle: ", leg_angles)
        leg_angle_differences = [ abs(leg_angles[i] - leg_angles[i-1]) > 5.0 for i in range(1, len(leg_angles)) ]
        # print(leg_angle_differences)        
        vote_walking = True if sum(leg_angle_differences)/len(leg_angle_differences) >= 0.6 else False
        
        if avg_change[0] > self.threshold or avg_change[1] > self.threshold:
            return True
        elif vote_walking:
            return True
        else: 
            return False

# if __name__ == "__main__":
#     example_keypoints = [np.random.rand(17, 2) * 100 for _ in range(10)]
    
#     # Initialize the detector once
#     knee_detector = KneeAngleDetector(threshold=10)
    
#     # Detect movement with provided keypoints
#     status = knee_detector.detect_movement(example_keypoints)
    
#     print(status)  # Output: "Walking" or "Standing Still" based on the random data
