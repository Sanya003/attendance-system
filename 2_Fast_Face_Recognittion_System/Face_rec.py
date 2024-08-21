import numpy as np
import pandas as pd
import cv2

import redis

# insightface
from insightface.app import FaceAnalysis
from sklearn.metrics import pairwise

# connect to redis client
#redis-17158.c326.us-east-1-3.ec2.cloud.redislabs.com:17158
#hGTkDq5a98SkO1P6ER4KpnpXI3T9kQuy
hostname = 'redis-18301.c274.us-east-1-3.ec2.cloud.redislabs.com'
portnumber = 18301
password = 'ihIROTTqvhEMCPVmqNHPImalkOIZjZ0W'

r = redis.StrictRedis(host = hostname,
                    port = portnumber,
                    password = password)

# configure face analysis
faceapp = FaceAnalysis(name='buffalo_sc',
                       root='insightface_model',
                       providers=['CPUExecutionProvider'])
faceapp.prepare(ctx_id=0, det_size=(640,640), det_thresh=0.5)

# ml search algo
def ml_search_algorithm(dataframe, feature_column, test_vector, 
                        name_role=['Name','Role'], thresh=0.5):
    """
    cosine similarity based search algo
    """
    # step-1: take the dataframe (collection of data)
    dataframe = dataframe.copy()
    
    # step-2: index face embedding from the df & convert into array
    X_list = dataframe[feature_column].tolist()
    x = np.asarray(X_list)
    
    # step-3: calculate cosine similarity
    similar = pairwise.cosine_similarity(x, test_vector.reshape(1,-1)) #1x512
    similar_arr = np.array(similar).flatten()
    dataframe['cosine'] = similar_arr
    
    # step-4: filterthe data
    data_filter = dataframe.query(f'cosine >= {thresh}')
    if len(data_filter) > 0:
        # step-5: get the person name
        data_filter.reset_index(drop=True, inplace=True)
        argmax = data_filter['cosine'].argmax()
        person_name, person_role = data_filter.loc[argmax][name_role]
    
    else:
        person_name = 'Unknown'
        person_role = 'Unknown'


    return person_name, person_role

# multiple face detection

def face_prediction(test_image,dataframe, feature_column, 
                        name_role=['Name','Role'], thresh=0.5 ):
    # step-1: take the test image & apply to insightface
    results = faceapp.get(test_image)
    test_copy = test_image.copy()
    
    # step-2: use for loop & extract each embeddings & pass to ml search algo
    for res in results:
        x1, y1, x2, y2 = res['bbox'].astype(int)
        embeddings = res['embedding']
        person_name, person_role = ml_search_algorithm(dataframe, 
                                                       feature_column, 
                                                       test_vector=embeddings,
                                                       name_role=name_role,
                                                       thresh=thresh)
        if person_name == 'Unknown':
            color = (0,0,255) #bgr
        else:
            color = (0,255,0)
    
        
        cv2.rectangle(test_copy,(x1,y1),(x2,y2),color)
    
        text_gen = person_name
        cv2.putText(test_copy,text_gen,(x1,y1),cv2.FONT_HERSHEY_DUPLEX, 0.7, color,2)


    return test_copy


