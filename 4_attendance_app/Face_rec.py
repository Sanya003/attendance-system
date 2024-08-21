import numpy as np
import pandas as pd
import cv2
import os
import redis

# insightface
from insightface.app import FaceAnalysis
from sklearn.metrics import pairwise
# timestamp
import time
from datetime import datetime


# connect to redis client
#redis-17158.c326.us-east-1-3.ec2.cloud.redislabs.com:17158
#hGTkDq5a98SkO1P6ER4KpnpXI3T9kQuy
hostname = 'redis-18301.c274.us-east-1-3.ec2.cloud.redislabs.com'
portnumber = 18301
password = 'ihIROTTqvhEMCPVmqNHPImalkOIZjZ0W'

r = redis.StrictRedis(host = hostname,
                    port = portnumber,
                    password = password)

# retrieve data from database
def retrieve_data(name):
    #name = 'academy:register'
    retrieve_dict= r.hgetall(name)
    retrieve_series = pd.Series(retrieve_dict)

    retrieve_series = retrieve_series.apply(lambda x: np.frombuffer(x,dtype=np.float32))
    index = retrieve_series.index
    index = list(map(lambda x: x.decode(), index))

    retrieve_series.index = index
    retrieve_df = retrieve_series.to_frame().reset_index()

    retrieve_df.columns=['name_role','facial_features']
    retrieve_df[['Name', 'Role']] = retrieve_df['name_role'].apply(lambda x: x.split('@')).apply(pd.Series)
    return retrieve_df[['Name', 'Role', 'facial_features']]

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

# real-time prediction - save logs for every minute
class RealTimePred:
    def __init__(self) -> None:
        self.logs = dict(name=[], role=[], current_time=[])

    def reset_dict(self):
        self.logs = dict(name=[], role=[], current_time=[]) # all parameters are strings

    # sava data into redis db
    def savelogs_redis(self):
        # step-1: create a logs df
        dataframe = pd.DataFrame(self.logs)
        # step-2: drop the duplicate information (distinct name)
        dataframe.drop_duplicates('name', inplace=True)
        # step-3: push data to redis db (list)
        # encode the data
        name_list = dataframe['name'].tolist()
        role_list = dataframe['role'].tolist()
        ctime_list = dataframe['current_time'].tolist()
        encoded_data = []
        for name, role, ctime in zip(name_list, role_list, ctime_list):
            if name != 'Unknown':
                concat_string = f"{name}@{role}@{ctime}"
                encoded_data.append(concat_string)

        if len(encoded_data) > 0:
            r.lpush('attendance:logs', *encoded_data)

        self.reset_dict()


    # multiple face detection
    def face_prediction(self, test_image,dataframe, feature_column, 
                            name_role=['Name','Role'], thresh=0.5 ):
        #step-0: find the timestamp
        current_time = str(datetime.now())

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
            cv2.putText(test_copy, current_time,(x1,y2+10),cv2.FONT_HERSHEY_DUPLEX, 0.7, color,2)

            # save info in log dicts
            self.logs['name'].append(person_name)
            self.logs['role'].append(person_role)
            self.logs['current_time'].append(current_time)



        return test_copy
    

# Registration form
class RegistrationForm:
    def __init__(self):
        self.sample=0
    def reset(self):
        self.sample=0

    def get_embedding(self, frame):
        # get results from insightface model
        results = faceapp.get(frame,max_num=1)
        embeddings = None
        for res in results:
            self.sample += 1
            x1, y1, x2, y2  = res['bbox'].astype(int)
            cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 1)
            # put text sample info
            text = f"samples = {self.sample}"
            cv2.putText(frame, text, (x1,y1), cv2.FONT_HERSHEY_DUPLEX, 0.6, (255,255,0),2)

            # facial features
            embeddings = res['embedding']

        return frame, embeddings
    
    def save_data_in_redis_db(self,name,role):
        # validation
        if name is not None:
            if name.strip() != '':
                key = f'{name}@{role}'
            else:
                return 'name_false'
        else:
            return 'name_false'
        
        # if face_embedding.txt exists
        if 'face_embedding.txt' not in os.listdir():
            return 'file_false'

        # step-1: load "face_embedding.txt"
        x_array = np.loadtxt('face_embedding.txt',dtype=np.float32)  #flatten x_array
        
        # step-2: convert into array(proper dimension/shape)
        received_samples = int(x_array.size/512)
        x_array = x_array.reshape(received_samples, 512)
        x_array = np.asarray(x_array)

        # step-3: calculate mean embedding
        x_mean = x_array.mean(axis=0)
        x_mean = x_mean.astype(np.float32)
        x_mean_bytes = x_mean.tobytes()
        # step-4: save this into redis db
        # redis hashes
        r.hset(name='academy:register', key=key, value=x_mean_bytes)
        
        # remove the txt file
        os.remove('face_embedding.txt')
        self.reset()

        return True