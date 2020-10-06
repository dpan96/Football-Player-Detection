# -*- coding: utf-8 -*-
"""
Created on Thu May 16 18:49:56 2019

@author: chris
"""
from __future__ import print_function
from __future__ import unicode_literals

# general packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# data acquisition
import youtube_dl

# image processing
import cv2 as cv

# scene detection
# Standard PySceneDetect imports:
from scenedetect.video_manager import VideoManager
from scenedetect.scene_manager import SceneManager
# For caching detection metrics and saving/loading to a stats file
from scenedetect.stats_manager import StatsManager
# For content-aware scene detection:
from scenedetect.detectors.content_detector import ContentDetector

# distance matrix for comparison of distances among players
from scipy.spatial import distance_matrix
from scipy.spatial.distance import pdist, squareform
import seaborn as sns

"""
Find Scenes
"""
def find_scenes(video_path):
    # three main calsses: VideoManager, SceneManager, StatsManager
    video_manager = VideoManager([video_path])
    stats_manager = StatsManager()
    scene_manager = SceneManager(stats_manager)

    """1. scene detector"""
    # Add ContentDetector algorithm (each detector's constructor
    # takes detector options, e.g. threshold).
    scene_manager.add_detector(ContentDetector())
    base_timecode = video_manager.get_base_timecode()
    scene_list = [] # to save scene info
    
    """2. stats manager"""
    # We save our stats file to {VIDEO_PATH}.stats.csv.
    stats_file_path = '%s.stats.csv' % video_path

    try:
        # If stats file exists, load it.
        if os.path.exists(stats_file_path):
            # Read stats from CSV file opened in read mode:
            with open(stats_file_path, 'r') as stats_file:
                stats_manager.load_from_csv(stats_file, base_timecode)

        # Set downscale factor to improve processing speed.
#         video_manager.set_downscale_factor()

        # Start video_manager.
        video_manager.start()

        # Perform scene detection on video_manager.
        scene_manager.detect_scenes(frame_source=video_manager)

        # Obtain list of detected scenes.
        scene_list = scene_manager.get_scene_list(base_timecode)
        # Each scene is a tuple of (start, end) FrameTimecodes.

#        print('List of scenes obtained:')
#        for i, scene in enumerate(scene_list):
#            print(
#                'Scene %2d: Start %s / Frame %d, End %s / Frame %d' % (
#                i+1,
#                scene[0].get_timecode(), scene[0].get_frames(),
#                scene[1].get_timecode(), scene[1].get_frames(),))

        # We only write to the stats file if a save is required:
        if stats_manager.is_save_required():
            with open(stats_file_path, 'w') as stats_file:
                stats_manager.save_to_csv(stats_file, base_timecode)

    finally:
        video_manager.release()

    return scene_list

"""
Save Scene Frames
"""
def save_scene_frames(scene_list, video_path):
    # to save frame from scene detection result
    scene_frames = []
    scene_frames_time = []

    # video capture
    vc = cv.VideoCapture(video_path)

    # capture the scene to test the similarity
    for ind, scene in enumerate(scene_list):

        frame_seq = scene[0].get_frames() # ranging from 0 to total number of frames (= length of video(seconds) * fps - 1)
        frame_time_code = scene[0].get_timecode()
        vc.set(1, frame_seq) # flag CV_CAP_PROP_POS_FRAMES which is a 0-based index of the frame to be decoded/captured next

        #Read the next frame from the video. If you set frame 749 above then the code will return the last frame.
        is_capturing, frame = vc.read()

        # show the frame image
        frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB) # convert the channel of image from BGR to RGB, dimension: (720, 1280, 3)
        scene_frames.append(frame)
        scene_frames_time.append(frame_time_code)
    #     print("frame_seq ", frame_seq)
    #     print("frame time code: ", frame_time_code)
#         plt.imshow(frame)
#         plt.savefig("scene_image\\scene_" + str(ind) + ".jpg")
#         plt.show()

    # When everything done, release the capture
    vc.release()
    cv.destroyAllWindows()
    
    return scene_frames, scene_frames_time
#scene_frames, scene_frames_time = save_scene_frames(scene_list, video_path)
"""
Feature Extraction
"""
class FeatureExtraction():
    def __init__(self, frame):
        self.frame = frame
        # Define a mask threshold ranging from lower to uppper
        self.MASK_RANGE = {
                      'green':{"lower":np.array([25, 52, 72]), "upper":np.array([102, 255, 255])},
                      'blue':{"lower":np.array([94, 80, 2]), "upper":np.array([126, 255, 255])},
                      'red':{"lower":np.array([161, 155, 84]), "upper":np.array([179, 255, 255])},
                      'white':{"lower":np.array([0, 0, 229]), "upper":np.array([180, 38, 255])},
                      'yellow':{"lower":np.array([21, 39, 64]), "upper":np.array([40, 255, 255])}
                     }
    
    # to get color feature
    def masking_operation(self, frame, color):
        # convert BGR into HSV color space
        hsv_frame = cv.cvtColor(frame, cv.COLOR_RGB2HSV) 
        
        ### 1. define mask
        mask = cv.inRange(hsv_frame, self.MASK_RANGE[color]["lower"], self.MASK_RANGE[color]["upper"])
        
        ### 2. start masking
        masked_frame = cv.bitwise_and(frame, frame, mask = mask)
        
        # 3. convert hsv to RGB and gray scale
        masked_rgb_frame = cv.cvtColor(masked_frame, cv.COLOR_HSV2RGB)
        masked_gray_frame = cv.cvtColor(masked_frame, cv.COLOR_RGB2GRAY)
        
#         plt.imshow(masked_gray_frame)
#         plt.savefig("preprocessing_image\\masking.jpg")
#         plt.show()
        
        return masked_gray_frame

    
    # to get shape feature
    def morphological_operation(self, masked_gray_frame):
        # filter noise in the video
        # Defining a kernel to do morphological operation in threshold # image to get better output.
        kernel = np.ones((13,13),np.uint8)
        morph_frame = cv.threshold(masked_gray_frame, 127, 255, cv.THRESH_BINARY_INV |  cv.THRESH_OTSU)[1]
        morph_frame = cv.morphologyEx(morph_frame, cv.MORPH_CLOSE, kernel)

#         plt.imshow(morph_frame)
#         plt.savefig("preprocessing_image\\morphological.jpg")
#         plt.show()        
        return morph_frame
        
    def idenity_players(self, morph_frame, team_1_color, team_2_color):
        # to recognize if a scene can be used to analyze or not
        key_scene_or_not = True
        # count number of players
        players_in_team_1 = 0
        players_in_team_2 = 0
        # record players' coordinates in an image
        team_1_player_cord = []
        team_2_player_cord = []
        #find contours in threshold image     
        im2, contours, hierarchy = cv.findContours(morph_frame, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        
        # for feature of shape
        font = cv.FONT_HERSHEY_SIMPLEX
        for c in contours:
            x, y, w, h = cv.boundingRect(c)
            """player detection"""
            # if the height of contour is 1.5 times longer than the width of contour
            if(h >= (1.2) * w and h < 120 and w < 100):
                if(w > 10 and h >= 10):
                    player_img = self.frame[y:y+h, x:x+w]

                    ### check if player is team 1
                    masked_gray_frame_1 = self.masking_operation(frame = player_img, color = team_1_color)
                    nzCount_1 = cv.countNonZero(masked_gray_frame_1)
                    
#                     print("value of {} player: {}".format(team_1_color, nzCount_1))
                    if(nzCount_1 >= 30):
                        players_in_team_1 += 1 # count how many players for a team
                        team_1_player_cord.append([x + w/2, y + h/2]) # record a player's coordinate in an image
                        
                        # Mark jersy players in box
                        cv.putText(self.frame, team_1_color, (x-2, y-2), font, 0.8, (255,255,255), 2, cv.LINE_AA)
                        cv.rectangle(self.frame, (x,y), (x+w,y+h), (255,255,255), 3)
                    else:
                        pass

                    ### check if player is in team 2
                    masked_gray_frame_2 = self.masking_operation(frame = player_img, color = team_2_color)
                    nzCount_2 = cv.countNonZero(masked_gray_frame_2)
                    
#                     print("value of {} player: {}".format(team_2_color, nzCount_2))
                    if(nzCount_2 >= 30):
                        players_in_team_2 += 1 # count how many players for a team
                        team_2_player_cord.append([x + w/2, y + h/2]) # record a player's coordinate in an image
                        
                        # Mark jersy players in box
                        cv.putText(self.frame, team_2_color, (x-2, y-2), font, 0.8, (255, 0, 0), 2, cv.LINE_AA)
                        cv.rectangle(self.frame, (x,y), (x+w,y+h), (255, 0, 0),3)
                    else:
                        pass

        total_players = players_in_team_1 + players_in_team_2
        # to ensure feasilbe number of players are detected
        if total_players > 2 and players_in_team_1 != 0 and players_in_team_2 != 0:
            team_1_dom_ratio = players_in_team_1 / total_players
            team_2_dom_ratio = players_in_team_2 / total_players
#             print("Number of team 1 players: ", players_in_team_1)
#             print("Number of team 2 players: ", players_in_team_2)
#             print("Domination Ratio of Team 1 in this frame: ", team_1_dom_ratio)
#             print("Domination Ratio of Team 2 in this frame: ", team_2_dom_ratio)
#             plt.imshow(self.frame)
#             plt.show()
        else:
            key_scene_or_not = False
#             print("no players detected")
            return (0, 0, 0, 0, key_scene_or_not)
        
        return (team_1_player_cord, team_2_player_cord, team_1_dom_ratio, team_2_dom_ratio, key_scene_or_not)
    
    def calculate_player_distance(self, team_1_player_cord, team_2_player_cord):
        ### distance among playerse
#         print("team 1 players' coordiantes info: ", team_1_player_cord)
#         print("team 2 players' coordiantes info: ", team_2_player_cord)

        # normalization of distance:
        # when carmera zooms in or zooms out, the distance value in each frame would have different meaning between two pixels
        # so do normalization to solve this issue.
        # to do normalization, find the max distance by creating distance matrix for all players
        all_distance = distance_matrix(team_1_player_cord + team_2_player_cord, team_1_player_cord + team_2_player_cord)
        max_distance = np.max(all_distance)
#         print("all_distance \n", np.round(all_distance, 2))
#         print("max_distance: ", max_distance)

        # distance for the players in the team 1
        team_1_distance_matrix = distance_matrix(team_1_player_cord, team_1_player_cord, p = 2.0) / max_distance # norm 2 distance formula
        team_1_inside_dist = team_1_distance_matrix.sum()/(team_1_distance_matrix.shape[0] * 2) # divide 2 as this is diagnoal matrix
#         print("team_1_distance matrix \n {}".format(np.round(team_1_distance_matrix, 2)))
#         print("team_1_distance: ", team_1_inside_dist) 

        # distance for the players in the team 2
        team_2_distance_matrix = distance_matrix(team_2_player_cord, team_2_player_cord, p = 2.0) / max_distance # norm 2 distance formula
        team_2_inside_dist = team_2_distance_matrix.sum()/(team_2_distance_matrix.shape[0] * 2) # divide 2 as this is diagnoal matrix
#         print("team_2_distance matrix \n {} ".format(np.round(team_2_distance_matrix, 2)))
#         print("team_2_distance: ", team_2_inside_dist)

        # distance between team 1 and team 2
        cross_team_distance_matrix = distance_matrix(team_1_player_cord, team_2_player_cord, p = 2.0) / max_distance # norm 2 distance formula
        cross_team_dist = np.mean(cross_team_distance_matrix)
#         print("cross_team_distance {} ", cross_team_dist)
        
        return team_1_inside_dist, team_2_inside_dist, cross_team_dist
"""
Find Key Scenes
"""

def find_key_scenes(scene_frames, scene_list):
    key_scenes = []
    for frame, scene_info in zip(scene_frames, scene_list):
        frame = frame.copy()
        fe = FeatureExtraction(frame)
        masked_gray_frame = fe.masking_operation(frame = frame, color = 'green')
        morph_frame = fe.morphological_operation(masked_gray_frame = masked_gray_frame)
        team_1_player_cord, team_2_player_cord, team_1_dom_ratio, team_2_dom_ratio, key_scene_or_not = fe.idenity_players(morph_frame = morph_frame, team_1_color = 'white', team_2_color = 'red')

        if key_scene_or_not == True:
            team_1_inside_dist, team_2_inside_dist, cross_team_dist = fe.calculate_player_distance(team_1_player_cord, team_2_player_cord)
            key_scenes.append(scene_info)
        else:
            pass
    return key_scenes
#key_scenes = find_key_scenes(scene_frames, scene_list)

"""
Extract Scene Featues
"""
def extract_scene_features(frames_list):

    df_scene_feature = pd.DataFrame(columns = ['team_1_dom_ratio', 'team_2_dom_ratio',
                                        'team_1_inside_dist', 'team_2_inside_dist', 'cross_team_dist'],
                              index = range(10000))
    feature_ind = 0
    fps = 25
    for ind, frame in enumerate(frames_list):
        if ind%fps == 0:
            # create feature value
            fe = FeatureExtraction(frame)
            masked_gray_frame = fe.masking_operation(frame = frame, color = 'green')
            morph_frame = fe.morphological_operation(masked_gray_frame = masked_gray_frame)
            team_1_player_cord, team_2_player_cord, team_1_dom_ratio, team_2_dom_ratio, key_scene_or_not = fe.idenity_players(morph_frame = morph_frame, team_1_color = 'white', team_2_color = 'red')
#             print(" team_1_player_cord, team_2_player_cord, team_1_dom_ratio, team_2_dom_ratio, key_scene_or_not", \
#                   team_1_player_cord, team_2_player_cord, team_1_dom_ratio, team_2_dom_ratio, key_scene_or_not)
            
            # to ensure two teams can be compared
            if team_1_dom_ratio != 0 or team_2_dom_ratio != 0:
                team_1_inside_dist, team_2_inside_dist, cross_team_dist = fe.calculate_player_distance(team_1_player_cord, team_2_player_cord)
            else:
                continue
            # save feature value
            df_scene_feature.set_value(feature_ind, 'team_1_dom_ratio', np.round(team_1_dom_ratio, 3))
            df_scene_feature.set_value(feature_ind, 'team_2_dom_ratio', np.round(team_2_dom_ratio, 3))
            df_scene_feature.set_value(feature_ind, 'team_1_inside_dist', np.round(team_1_inside_dist, 3))
            df_scene_feature.set_value(feature_ind, 'team_2_inside_dist', np.round(team_2_inside_dist, 3))
            df_scene_feature.set_value(feature_ind, 'cross_team_dist', np.round(cross_team_dist, 3))
            feature_ind += 1

        if ind == 300:
            break
    df_scene_feature = df_scene_feature.dropna(how='any')
    return df_scene_feature

"""
Create Features Matrix
"""
def create_features(key_scenes, video_path):
    # video capture
    vc = cv.VideoCapture(video_path)
    """
    For each key scene, collect and analyze all its frames.
    The analytical results are saved for each scene (summarization for each frame: time-series info and summary one)
    """
    scenes_featurs_summary = [] # save the summary result of each scene
    seg_scene_features_summary = [] # save three segemented features info for each scene
    seg_scene_frame_summary = [] # save three segemented frames info for each scene

    for key_scene in key_scenes:
        print("-" * 100)
        print("scene ", key_scene)
        frame_start_seq = key_scene[0].get_frames() # starting sequence of the scene
        frame_end_seq = key_scene[1].get_frames() # ending sequence of the scene
        frames_list = [] # save frames for the key scenes from frame_start_seq to frame_end_seq

        for frame_seq in range(frame_start_seq, frame_end_seq):
            vc.set(1, frame_seq) # flag CV_CAP_PROP_POS_FRAMES which is a 0-based index of the frame to be decoded/captured next

            #Read the next frame from the video. If you set frame 749 above then the code will return the last frame.
            is_capturing, frame = vc.read()

            # show the frame image
            frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB) # convert the channel of image from BGR to RGB, dimension: (720, 1280, 3)
            frames_list.append(frame)

        ### 1. save scene features summary result
        df_scene_feature = extract_scene_features(frames_list)
        scenes_featurs_summary.append(list(np.mean(df_scene_feature)) + list(np.std(df_scene_feature)))

        ### 2. each scene is segemented into three pieces. All average features' values in each piece are record
        tol_frame = len(df_scene_feature)
        seq_1 = int(tol_frame/3)
        seq_2 = int(tol_frame*2/3)
        seq_3 = tol_frame
        seq_1_feature = pd.DataFrame(df_scene_feature, index=range(0, seq_1)).mean()
        seq_2_feature = pd.DataFrame(df_scene_feature, index=range(seq_1, seq_2)).mean()
        seq_3_feature = pd.DataFrame(df_scene_feature, index=range(seq_2, seq_3)).mean()
        df_seg_scene = pd.concat([seq_1_feature, seq_2_feature, seq_3_feature], axis = 1).T
        seg_scene_features_summary.append(df_seg_scene)

        ### 3. save first frame in each scene
        seg_scene_frame_summary.append(frames_list[0])
    #         print("frames_list[0] ", frames_list[0])
    #         print("df_scene_feature ", df_scene_feature)

    df_scenes_featurs_summary = pd.DataFrame(scenes_featurs_summary)
    df_scenes_featurs_summary.columns = ['avg_team_1_dom_ratio', 'avg_team_2_dom_ratio',
                                                 'avg_team_1_inside_dist','avg_team_2_inside_dist', 'avg_cross_team_dist',
                                                 'std_team_1_dom_ratio', 'std_team_2_dom_ratio', 'std_team_1_inside_dist',
                                                 'std_team_2_inside_dist', 'std_cross_team_dist']
    # When everything done, release the capture
    vc.release()
    cv.destroyAllWindows()
    return seg_scene_frame_summary, seg_scene_features_summary, df_scenes_featurs_summary

"""
Similarity Matrix
"""
def similarity_matrix(df_scenes_featurs_summary):
    plt.figure(figsize = (5, 4))
    data = squareform(1 - pdist(df_scenes_featurs_summary, metric='cosine'))
    ax = sns.heatmap(data, linewidth=0.5)
    plt.xlabel("Scene Index")
    plt.ylabel("Scene Index")
    plt.show()
    
    return data


"""
statistical report
"""
def statistical_report(input_scene, seg_scene_frame_summary, seg_scene_features_summary, df_scenes_featurs_summary):
    scene_ind_1 = input_scene[0] - 1
    scene_ind_2 = input_scene[1] - 1
    ### display frames
    plt.figure(figsize = (20, 10))
    plt.subplot(121)
    plt.imshow(seg_scene_frame_summary[scene_ind_1])
    plt.subplot(122)
    plt.imshow(seg_scene_frame_summary[scene_ind_2])
    # plt.show()

    ### display variatio of distance ratios
    fig = plt.figure(figsize = (20, 5))
    ax_1 = fig.add_subplot(121)
    seg_scene_features_summary[scene_ind_1].loc[:, ['team_1_inside_dist', 'team_2_inside_dist', 'cross_team_dist']].plot(ax = ax_1)
    plt.xlabel("Transition of Three Key Frames")
    plt.ylabel("Relative Distance")
    label = ['first', 'second', 'third']
    plt.xticks(range(len(label)), label, size='large')
    # plt.legend(loc='center left', bbox_to_anchor=(1, 0.15))
    plt.grid()

    ax_2 = fig.add_subplot(122)
    seg_scene_features_summary[scene_ind_2].loc[:, ['team_1_inside_dist', 'team_2_inside_dist', 'cross_team_dist']].plot(ax = ax_2)
    plt.xlabel("Transition of Three Key Frames")
    plt.ylabel("Relative Distance")
    label = ['first', 'second', 'third']
    plt.xticks(range(len(label)), label, size='large')
    # plt.legend(loc='center left', bbox_to_anchor=(1, 0.15))
    plt.grid()
    # plt.show()


    ### display domination ratio
    ### ------- first scene
    plt.figure(figsize = (20, 5))
    plt.subplot(121)
    data = df_scenes_featurs_summary.loc[scene_ind_1, ['avg_team_1_dom_ratio', 'avg_team_2_dom_ratio']].values
    labels = 'team 1', 'team 2'
    sizes = data
    colors = ['gold', 'yellowgreen']
    explode = (0.1, 0)  # explode 1st slice
    # Plot
    plt.pie(sizes, explode=explode, labels=labels, colors=colors,
    autopct='%1.1f%%', shadow=True, startangle=140)
    plt.axis('equal')

    ### ------- second scene
    plt.subplot(122)
    data = df_scenes_featurs_summary.loc[scene_ind_2, ['avg_team_1_dom_ratio', 'avg_team_2_dom_ratio']].values
    labels = 'team 1', 'team 2'
    sizes = data
    colors = ['gold', 'yellowgreen']
    explode = (0.1, 0)  # explode 1st slice
    # Plot
    plt.pie(sizes, explode=explode, labels=labels, colors=colors,
    autopct='%1.1f%%', shadow=True, startangle=140)
    plt.axis('equal')

    plt.savefig("analytical_report.jpg")
    plt.show()
