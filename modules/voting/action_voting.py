from collections import Counter, defaultdict, deque
from modules.common.logger import get_app_logger
from modules.common.config import get_app_config
from modules.common.constants import MAX_SEQ_LEN, GCNSubclass
from modules.voting.rulebase.estimate_falling import FallingEstimator
from modules.voting.rulebase.estimate_moving import KneeAngleDetector
from typing import List


# Action Rules with Specific Thresholds
MAIN_ACTION_RULES = {
    "falling_sensitive"  :  {
                                "code": [GCNSubclass.FALLING_DOWN.value, GCNSubclass.LYING.value], 
                                "thresh": 0.1,  
                                "label": "Falling down"
                            }, 
    "staggering_sesitive":  {
                                "code": GCNSubclass.STAGGERING.value, 
                                "thresh": 0.5, 
                                "label": "Staggering"
                            },  
    "loitering_sesitive":   {
                                "code": [GCNSubclass.STANDING_STILL.value, GCNSubclass.SITTING_STILL.value], 
                                "thresh": 0.5, 
                                "label": "Loitering"
                            },  
}

class ActionVoting:
    def __init__(self, camera_name: str):
        self.camera_name = camera_name
        self.logger = get_app_logger(camera_name, __name__)
        self.config_reader = get_app_config(camera_name)
        self.min_len_seq_vote = self.config_reader.getint("vote", "min_len_seq_vote", fallback=MAX_SEQ_LEN)
        self.pid_vote_gcn = defaultdict(lambda: deque(maxlen=self.min_len_seq_vote))
        self.final_list_actObj = []
        self.falling_estimator = FallingEstimator(threshold_angle=45)
        self.moving_detector = KneeAngleDetector(threshold=10)
        
        
    def _vote_gcn(self, pid: int, seq_action: list):
        """
        Perform voting on the sequence of actions for a given PID.
        """
        if not seq_action:
            return None
        self.pid_vote_gcn[pid].append(seq_action[-1])
            
        
            
    def _vote_action(self, pid: int, seq_kp: list):
        """
        Determine the final action based on voting rules.
        """
        action_list = list(self.pid_vote_gcn[pid])
        total_votes = len(action_list)
        
        if total_votes == 0:
            return []
        
        result_actions = []
        for sesitive_name, rule in list(MAIN_ACTION_RULES.items()):  # Iterate over a static copy
            action_code = rule["code"]
            threshold = self.config_reader.getfloat("vote", sesitive_name, fallback=rule["thresh"])
            
            if isinstance(action_code, list):
                count = sum(1 for act in action_list if act in action_code)
            else:
                count = sum(1 for act in action_list if act == action_code)
            
            if count / total_votes >= threshold:
                if rule["label"] == "Falling down":
                    # using rulebase to confirm final
                    kp_last = seq_kp[-1].tolist()
                    is_falling, _ =  self.falling_estimator.is_falling(kp_last)
                    if is_falling: 
                        result_actions.append(rule["label"]) 
                elif rule["label"] == "Staggering":
                    result_actions.append(rule["label"]) 
                    
                elif rule["label"] == "Loitering":
                    is_moving = self.moving_detector.detect_movement(keypoints_sequence=seq_kp)
                    if not is_moving:
                        result_actions.append(rule["label"]) 
                else :
                    result_actions.append(rule["label"]) 
        return result_actions
    
    def _final_vote(self, pid: int, seq_kp: List[list], seq_action: list):
        """
        Conduct final voting for a given PID.
        """
        self._vote_gcn(pid, seq_action)
        return self._vote_action(pid, seq_kp)

    def cleanup(self, pids_to_remove):
        for pid in pids_to_remove:
            del self.pid_vote_gcn[pid]
    
    def update(self, list_actObj: list):
        """
        Update the voting results based on new tracking objects.
        """
        
        self.final_list_actObj = []
        for actObj in list(list_actObj): 
            pid = actObj.track_obj.pid
            seq_kp = actObj.track_obj.get_seq_kp_2d(kp_score=0.05, seq_len=20)
            seq_action = actObj.get_seq_action()
            if len(seq_action) >= self.min_len_seq_vote:
                actObj.final_vote = self._final_vote(pid, seq_kp, seq_action)
                self.final_list_actObj.append(actObj)
    
    def get_final_vote(self):
        """
        Retrieve the final vote results.
        """
        # act_final_vote = [i.final_vote for i in self.final_list_actObj]
        # print("ACTION FINAL VOTE: ", act_final_vote)
        return self.final_list_actObj
