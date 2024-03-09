import os
import pandas as pd
import torch
import random
from functools import reduce
from torch.utils.data import Dataset, DataLoader
from transform import InterpolatorDenseTS,InterpolatorScarseTS,InterpolatorMedDoseOnly
import numpy as np
class GetStats:
    def __init__(self, file_path = ''):
        self.path = str(file_path)
        self.preops = pd.read_feather(self.path+'preops_reduced_for_training.feather', columns=None, use_threads=True)
        self.preops.replace([np.inf, -np.inf], np.nan, inplace=True)

        self.texts = pd.read_csv(file_path+'epic_procedure_bow.csv')
        # self.texts.fillna(0, inplace=True)
        self.dense_flow_ts = pd.read_feather(self.path+'flow_ts/very_dense_flow.feather', columns=None, use_threads=True)
        self.dense_flow_ts.replace([np.inf, -np.inf], np.nan, inplace=True)
        self.scarse_flow_ts= pd.read_feather(self.path+'flow_ts/other_intra_flow_wlabs.feather', columns=None, use_threads=True)
        self.scarse_flow_ts.VALUE=self.scarse_flow_ts.VALUE.astype(float)
        self.scarse_flow_ts.replace([np.inf, -np.inf], np.nan, inplace=True)
        self.meds_ts =  pd.read_feather(self.path+'med_ts/intraop_meds_filterd.feather', columns=None, use_threads=True)
        self.meds_ts=self.meds_ts[['orlogid_encoded','time','med_integer','unit_integer','route_integer','dose']]
        self.meds_ts.dose=self.meds_ts.dose.astype(float)
        self.meds_ts.replace([np.inf, -np.inf], np.nan, inplace=True)
        self.outcomes = pd.read_csv(self.path+'epic_outcomes.csv')
        self.scarse_measures = self.scarse_flow_ts.measure_index.unique()
        self.med_integers = self.meds_ts.med_integer.unique()
        end_time_dense = (self.dense_flow_ts.groupby(["orlogid_encoded"]).timepoint.max())
        end_time_scarce = (self.scarse_flow_ts.groupby(["orlogid_encoded"]).timepoint.max())
        end_time_med = (self.meds_ts.groupby(["orlogid_encoded"]).time.max())
        data_frames = [end_time_dense, end_time_scarce, end_time_med]
        self.end_time = reduce( lambda left, right: pd.merge( left, right, left_index=True, right_index=True, how="outer"),data_frames).max(axis=1)
        self.end_time[self.end_time<30]=30
    def get_stats(self,ids):
        self.ids = ids #training ids
        outcomes = self.outcomes[self.outcomes.orlogid.isin(self.ids)]
        self.outcomes_mean, self.outcomes_std =outcomes.mean(),outcomes.std()
        #calculate mean, var by all data. otherwise some features not shown in training patients
        # self.preops_mean,self.preops_std=self.preops[self.preops.orlogid_encoded.isin(self.ids)].loc[:,self.preops.columns != "orlogid_encoded"].mean(),self.preops[self.preops.orlogid_encoded.isin(self.ids)].loc[:,self.preops.columns != "orlogid_encoded"].std()
        self.preops_mean,self.preops_std=self.preops.loc[:,self.preops.columns != "orlogid_encoded"].mean(),self.preops.loc[:,self.preops.columns != "orlogid_encoded"].std()

        # self.dense_flow_ts_mean = self.dense_flow_ts[self.dense_flow_ts.orlogid_encoded.isin(self.ids)].iloc[:,2:].mean()
        # self.dense_flow_ts_std = self.dense_flow_ts[self.dense_flow_ts.orlogid_encoded.isin(self.ids)].iloc[:,2:].std()
        self.dense_flow_ts_mean = self.dense_flow_ts.iloc[:,2:].mean()
        self.dense_flow_ts_std = self.dense_flow_ts.iloc[:,2:].std()
        
        # scarse_flow_ts=self.scarse_flow_ts[self.scarse_flow_ts.orlogid_encoded.isin(self.ids)]
        scarse_flow_ts=self.scarse_flow_ts
        self.scarse_flow_ts_mean = scarse_flow_ts.groupby(['measure_index']).VALUE.mean()
        self.scarse_flow_ts_std = scarse_flow_ts.groupby(['measure_index']).VALUE.std()
        # self.scarse_flow_ts.iloc[:,-1]=(self.scarse_flow_ts.iloc[:,-1] -self.scarse_flow_ts.iloc[:,-1].mean()) / self.scarse_flow_ts.iloc[:,-1].std()
        
        # meds_ts = self.meds_ts[self.meds_ts.orlogid_encoded.isin(self.ids)]
        meds_ts = self.meds_ts
        self.meds_ts_mean = meds_ts.groupby(['med_integer']).dose.mean()
        self.meds_ts_std= meds_ts.groupby(['med_integer']).dose.std()
        
class CustomHandoffDataset(Dataset):
    def __init__(self, inputs=None, ids=None,labels=None, train=True):
        self.inputs=inputs
        if train:
            self.inputs.get_stats(ids) #get stats from training data
        self.transform_dense_flow = InterpolatorDenseTS(inputs.dense_flow_ts_mean, inputs.dense_flow_ts_std, ts=inputs.dense_flow_ts)
        self.transform_scarse_flow = InterpolatorScarseTS(
            inputs.scarse_flow_ts_mean,
            inputs.scarse_flow_ts_std,
            ts=inputs.scarse_flow_ts,
            measures=inputs.scarse_flow_ts.measure_index.unique(),
        )
        self.transform_meds = InterpolatorMedDoseOnly(
            inputs.meds_ts_mean,
            inputs.meds_ts_std,
            ts=inputs.meds_ts,
            med_integers=inputs.meds_ts.med_integer.unique(),
        )  
        self.ids = ids
        self.labels=labels
    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):

        curr_id = self.ids[idx]
        labels = self.labels[idx] #in case of multitask, what if outcome is null? 

        curr_preop =self.inputs.preops.loc[self.inputs.preops.orlogid_encoded==curr_id,self.inputs.preops.columns != "orlogid_encoded"]
        curr_preop = (curr_preop - self.inputs.preops_mean)/self.inputs.preops_std
        curr_preop.fillna(0, inplace=True)

        # Handle the case where curr_id doesn't match any index in end_time
        ###
        matching_end_times = self.inputs.end_time[self.inputs.end_time.index == curr_id].values
        if matching_end_times.size == 0:
            print(f"Warning: No matching end_time found for curr_id: {curr_id}. Skipping.")
            return curr_preop, None, None, None, labels

        curr_endtime = matching_end_times[0]
        ###
        #curr_endtime = self.inputs.end_time[self.inputs.end_time.index == curr_id].values[0]
        curr_dense_flow_ts = self.transform_dense_flow(curr_id,curr_endtime)
        
        curr_scarse_flow_ts = self.transform_scarse_flow(curr_id,curr_endtime)
        #meds shape: (T*num_meds, 5)
        curr_meds_ts = self.transform_meds(curr_id,curr_endtime)
        return curr_preop,curr_dense_flow_ts, curr_scarse_flow_ts,curr_meds_ts,labels
