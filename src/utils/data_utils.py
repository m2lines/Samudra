import xarray as xr
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
from scipy.ndimage import gaussian_filter


#Class defined to store information about the grid and corresponding graph of data. Importantly produces the adjacency matrices 
#and keeps track of what is land vs ocean 

class data_CNN_Dynamic(torch.utils.data.Dataset):

    def __init__(self,data_in,data_out,wet,device = "cuda"):
        super().__init__()
        self.device = device        
        num_inputs = data_in.shape[3]
        num_outputs = data_out.shape[3]
        self.size = data_in.shape[0]
        
        data_in = np.nan_to_num(data_in)
        data_out = np.nan_to_num(data_out)
        
        std_data = np.nanstd(data_in,axis=(0,1,2))
        mean_data = np.nanmean(data_in,axis=(0,1,2)) 
        std_label = np.nanstd(data_out,axis=(0,1,2))
        mean_label = np.nanmean(data_out,axis=(0,1,2))
        
        self.wet = wet
        
        for i in range(num_inputs):
            data_in[:,:,:,i] = (data_in[:,:,:,i] - mean_data[i])/std_data[i]
        
        for i in range(num_outputs):
            data_out[:,:,:,i] = (data_out[:,:,:,i] - mean_label[i])/std_label[i]
            
        data_in = torch.from_numpy(data_in).type(torch.float32).to(device="cpu")
        data_out = torch.from_numpy(data_out).type(torch.float32).to(device="cpu")        
        

        std_dict = {'s_in':std_data,'s_out':std_label,'m_in':mean_data, 'm_out':mean_label}
        
        if wet == None:
            self.input = torch.swapaxes(torch.swapaxes(data_in,1,3),2,3)
            self.output = torch.swapaxes(torch.swapaxes(data_out,1,3),2,3)           
            
        else:
            self.input = torch.mul(torch.swapaxes(torch.swapaxes(data_in,1,3),2,3),wet)
            self.output = torch.mul(torch.swapaxes(torch.swapaxes(data_out,1,3),2,3),wet)
        
        self.norm_vals = std_dict
    
    def set_device(self,device):
        self.device = device
    
    def __len__(self):
        # Number of data point we have. Alternatively self.data.shape[0], or self.label.shape[0]
        return self.size

    def __getitem__(self, idx):
        # Return the idx-th data point of the dataset
        # If we have multiple things to return (data point and label), we can return them as tuple
        data_in = self.input[idx]
        label = self.output[idx]
        return data_in.to(device = self.device), label.to(device = self.device)

    
class data_CNN_steps_Dynamic(torch.utils.data.Dataset):

    def __init__(self,data_in,data_out,steps,wet = None,device = "cuda"):
        super().__init__()
        self.device = device
        steps = len(data_out)
        self.steps = steps
        num_inputs = data_in[0].shape[3]
        num_outputs = data_out[0].shape[3]
        self.size = data_in[0].shape[0]
        self.wet = wet
        
        for i in range(steps):
            data_out[i] = np.nan_to_num(data_out[i])
            data_in[i] = np.nan_to_num(data_in[i])
       
        std_data = np.nanstd(data_in[0],axis=(0,1,2))
        mean_data = np.nanmean(data_in[0],axis=(0,1,2)) 
        std_label = np.nanstd(data_out[0],axis=(0,1,2))
        mean_label = np.nanmean(data_out[0],axis=(0,1,2))
        

        for j in range(steps):
            for i in range(num_outputs):
                data_out[j][:,:,:,i] = (data_out[j][:,:,:,i] - mean_label[i])/std_label[i]
            for i in range(num_inputs):
                data_in[j][:,:,:,i] = (data_in[j][:,:,:,i] - mean_data[i])/std_data[i] 
                
        for j in range(steps):
            data_out[j] = torch.from_numpy(data_out[j]).type(torch.float32).to(device="cpu")    
            data_in[j] = torch.from_numpy(data_in[j]).type(torch.float32).to(device="cpu") 
       

        std_dict = {'s_in':std_data,'s_out':std_label,'m_in':mean_data, 'm_out':mean_label}
        
        if wet == None:
            for j in range(steps):
                data_out[j] = torch.swapaxes(torch.swapaxes(data_out[j],1,3),2,3)
                data_in[j] = torch.swapaxes(torch.swapaxes(data_in[j],1,3),2,3)
        else:
            for j in range(steps):
                data_out[j] = torch.mul(torch.swapaxes(torch.swapaxes(data_out[j],1,3),2,3),wet)
                data_in[j] = torch.mul(torch.swapaxes(torch.swapaxes(data_in[j],1,3),2,3),wet)
        
        self.input = data_in

        self.output = data_out
        self.norm_vals = std_dict
        
    def __len__(self):
        # Number of data point we have. Alternatively self.data.shape[0], or self.label.shape[0]
        return self.size

    def __getitem__(self, idx):
        # Return the idx-th data point of the dataset
        # If we have multiple things to return (data point and label), we can return them as tuple
        data = [self.input[0][idx].to(device = self.device),self.output[0][idx].to(device = self.device)]
        for k in range(1,self.steps):
            data.append(self.input[k][idx].to(device = self.device))
            data.append(self.output[k][idx].to(device = self.device))
        
        return tuple(data)

    
class data_CNN_steps(torch.utils.data.Dataset):

    def __init__(self,data_in,data_out,wet = None,device = "cuda"):
        steps = len(data_out)
        self.steps = steps
        super().__init__()
        num_inputs = data_in[0].shape[3]
        num_outputs = data_out[0].shape[3]
        self.size = data_in[0].shape[0]
        self.wet = wet
        
        for i in range(steps):
            data_out[i] = np.nan_to_num(data_out[i])
            data_in[i] = np.nan_to_num(data_in[i])
       
        std_data = np.nanstd(data_in[0],axis=(0,1,2))
        mean_data = np.nanmean(data_in[0],axis=(0,1,2)) 
        std_label = np.nanstd(data_out[0],axis=(0,1,2))
        mean_label = np.nanmean(data_out[0],axis=(0,1,2))
        

        for j in range(steps):
            for i in range(num_outputs):
                data_out[j][:,:,:,i] = (data_out[j][:,:,:,i] - mean_label[i])/std_label[i]
            for i in range(num_inputs):
                data_in[j][:,:,:,i] = (data_in[j][:,:,:,i] - mean_data[i])/std_data[i] 
                
        for j in range(steps):
            data_out[j] = torch.from_numpy(data_out[j]).type(torch.float32).to(device=device)        
            data_in[j] = torch.from_numpy(data_in[j]).type(torch.float32).to(device=device)
       

        std_dict = {'s_in':std_data,'s_out':std_label,'m_in':mean_data, 'm_out':mean_label}
        
        if wet == None:
            for j in range(steps):
                data_out[j] = torch.swapaxes(torch.swapaxes(data_out[j],1,3),2,3)
                data_in[j] = torch.swapaxes(torch.swapaxes(data_in[j],1,3),2,3)
        else:
            for j in range(steps):
                data_out[j] = torch.mul(torch.swapaxes(torch.swapaxes(data_out[j],1,3),2,3),wet)
                data_in[j] = torch.mul(torch.swapaxes(torch.swapaxes(data_in[j],1,3),2,3),wet)
        
        self.input = data_in

        self.output = data_out
        self.norm_vals = std_dict
        
    def __len__(self):
        # Number of data point we have. Alternatively self.data.shape[0], or self.label.shape[0]
        return self.size

    def __getitem__(self, idx):
        # Return the idx-th data point of the dataset
        # If we have multiple things to return (data point and label), we can return them as tuple
        data = [self.input[0][idx],self.output[0][idx]]
        for k in range(1,self.steps):
            data.append(self.input[k][idx])
            data.append(self.output[k][idx])
        
        return tuple(data)
    
    
class data_CNN(torch.utils.data.Dataset):

    def __init__(self,data_in,data_out,wet,device = "cuda"):

        super().__init__()
        num_inputs = data_in.shape[3]
        num_outputs = data_out.shape[3]
        self.size = data_in.shape[0]
        
        data_in = np.nan_to_num(data_in)
        data_out = np.nan_to_num(data_out)
        
        std_data = np.nanstd(data_in,axis=(0,1,2))
        mean_data = np.nanmean(data_in,axis=(0,1,2)) 
        std_label = np.nanstd(data_out,axis=(0,1,2))
        mean_label = np.nanmean(data_out,axis=(0,1,2))
        
        self.wet = wet
        
        for i in range(num_inputs):
            data_in[:,:,:,i] = (data_in[:,:,:,i] - mean_data[i])/std_data[i]
        
        for i in range(num_outputs):
            data_out[:,:,:,i] = (data_out[:,:,:,i] - mean_label[i])/std_label[i]
            
        data_in = torch.from_numpy(data_in).type(torch.float32).to(device=device)
        data_out = torch.from_numpy(data_out).type(torch.float32).to(device=device)        
        

        std_dict = {'s_in':std_data,'s_out':std_label,'m_in':mean_data, 'm_out':mean_label}
        
        if wet == None:
            self.input = torch.swapaxes(torch.swapaxes(data_in,1,3),2,3)
            self.output = torch.swapaxes(torch.swapaxes(data_out,1,3),2,3)           
            
        else:
            self.input = torch.mul(torch.swapaxes(torch.swapaxes(data_in,1,3),2,3),wet)
            self.output = torch.mul(torch.swapaxes(torch.swapaxes(data_out,1,3),2,3),wet)
        
        self.norm_vals = std_dict
        
    def __len__(self):
        # Number of data point we have. Alternatively self.data.shape[0], or self.label.shape[0]
        return self.size

    def __getitem__(self, idx):
        # Return the idx-th data point of the dataset
        # If we have multiple things to return (data point and label), we can return them as tuple
        data_in = self.input[idx]
        label = self.output[idx]
        return data_in, label


class data_CNN_Lateral(torch.utils.data.Dataset):

    def __init__(self,data_in,data_out,wet,N_atm,Nb,device = "cuda",wet_atm = False,norms= "None",N_vars=None):
        super().__init__()
        self.device = device        
        num_inputs = data_in.shape[3]
        num_outputs = data_out.shape[3]
        self.size = data_in.shape[0]
        
        data_in = np.nan_to_num(data_in)
        data_out = np.nan_to_num(data_out)
        
        if norms != "None":
            std_data = norms['s_in']
            mean_data = norms['m_in'] 
            std_label = norms['s_out']
            mean_label = norms['m_out']  
        else:
        
            std_data = np.nanstd(data_in,axis=(0,1,2))
            mean_data = np.nanmean(data_in,axis=(0,1,2)) 
            std_label = np.nanstd(data_out,axis=(0,1,2))
            mean_label = np.nanmean(data_out,axis=(0,1,2))

            std_data[int(num_outputs+N_atm):int(2*num_outputs+N_atm)] = std_data[:num_outputs]        
            mean_data[int(num_outputs+N_atm):int(2*num_outputs+N_atm)] = mean_data[:num_outputs]

        self.wet = wet
        
        for i in range(num_inputs):
            data_in[:,:,:,i] = (data_in[:,:,:,i] - mean_data[i])/std_data[i]
        
        for i in range(int(num_outputs+N_atm),int(2*num_outputs+N_atm)):
            data_in[:,Nb:-Nb,Nb:-Nb,i] = 0.0
            
        for i in range(num_outputs):
            data_out[:,:,:,i] = (data_out[:,:,:,i] - mean_label[i])/std_label[i]
            
        data_in = torch.from_numpy(data_in).type(torch.float32).to(device="cpu")
        data_out = torch.from_numpy(data_out).type(torch.float32).to(device="cpu")        
        

        std_dict = {'s_in':std_data,'s_out':std_label,'m_in':mean_data, 'm_out':mean_label}
        
        if type(wet) == list:
            temp_in = torch.swapaxes(torch.swapaxes(data_in,1,3),2,3)
            temp_out = torch.swapaxes(torch.swapaxes(data_out,1,3),2,3)            
            for i in range(len(wet)):
                temp_in[:,i*N_vars:i*N_vars] = torch.mul(temp_in[:,i*N_vars:i*N_vars],wet[i])
                start = num_outputs+N_atm
                temp_in[:,start+i*N_vars:start+i*N_vars] = torch.mul(temp_in[:,start+i*N_vars:start+i*N_vars],wet[i])
                temp_out[:,i*N_vars:i*N_vars] = torch.mul(temp_out[:,i*N_vars:i*N_vars],wet[i])
            temp_in[:,num_outputs:num_outputs+N_atm] = torch.mul(temp_in[:,num_outputs:num_outputs+N_atm],wet[0])
            self.input = temp_in
            self.output = temp_out            
        
        elif wet == None:
            self.input = torch.swapaxes(torch.swapaxes(data_in,1,3),2,3)
            self.output = torch.swapaxes(torch.swapaxes(data_out,1,3),2,3)           
            
        else:

            if wet_atm:
                self.input = torch.swapaxes(torch.swapaxes(data_in,1,3),2,3)
                self.output = torch.swapaxes(torch.swapaxes(data_out,1,3),2,3) 
                for i in range(num_outputs):
                    self.input[:,i] = torch.mul(self.input[:,i],wet)
                    self.output[:,i] = torch.mul(self.input[:,i],wet)
                for i in range(int(num_outputs+N_atm),int(2*num_outputs+N_atm)):
                    self.input[:,i] = torch.mul(self.input[:,i],wet)
            else:
                self.input = torch.mul(torch.swapaxes(torch.swapaxes(data_in,1,3),2,3),wet)
                self.output = torch.mul(torch.swapaxes(torch.swapaxes(data_out,1,3),2,3),wet)                
        self.norm_vals = std_dict
    
    def set_device(self,device):
        self.device = device
    
    def __len__(self):
        # Number of data point we have. Alternatively self.data.shape[0], or self.label.shape[0]
        return self.size

    def __getitem__(self, idx):
        # Return the idx-th data point of the dataset
        # If we have multiple things to return (data point and label), we can return them as tuple
        data_in = self.input[idx]
        label = self.output[idx]
        return data_in.to(device = self.device), label.to(device = self.device)

    
    
class data_CNN_steps_Lateral(torch.utils.data.Dataset):

    def __init__(self,data_in,data_out,steps,wet,N_atm,Nb,device = "cuda",wet_atm = False,norms= "None",N_vars=None):
        super().__init__()
        self.device = device
        steps = len(data_out)
        self.steps = steps
        num_inputs = data_in[0].shape[3]
        num_outputs = data_out[0].shape[3]
        self.size = data_in[0].shape[0]
        self.wet = wet
        
        for i in range(steps):
            data_out[i] = np.nan_to_num(data_out[i])
            data_in[i] = np.nan_to_num(data_in[i])
       
        if norms != "None":
            std_data = norms['s_in']
            mean_data = norms['m_in'] 
            std_label = norms['s_out']
            mean_label = norms['m_out']  
        else:
            std_data = np.nanstd(data_in[0],axis=(0,1,2))
            mean_data = np.nanmean(data_in[0],axis=(0,1,2)) 
            std_label = np.nanstd(data_out[0],axis=(0,1,2))
            mean_label = np.nanmean(data_out[0],axis=(0,1,2))
            
        
            std_data[int(num_outputs+N_atm):int(2*num_outputs+N_atm)] = std_data[:num_outputs]        
            mean_data[int(num_outputs+N_atm):int(2*num_outputs+N_atm)] = mean_data[:num_outputs]        

        for j in range(steps):
            for i in range(num_outputs):
                data_out[j][:,:,:,i] = (data_out[j][:,:,:,i] - mean_label[i])/std_label[i]
        
            
            for i in range(num_inputs):
                data_in[j][:,:,:,i] = (data_in[j][:,:,:,i] - mean_data[i])/std_data[i] 
                
            for i in range(int(num_outputs+N_atm),int(2*num_outputs+N_atm)):
                data_in[j][:,Nb:-Nb,Nb:-Nb,i] = 0.0                
                
        for j in range(steps):
            data_out[j] = torch.from_numpy(data_out[j]).type(torch.float32).to(device="cpu")    
            data_in[j] = torch.from_numpy(data_in[j]).type(torch.float32).to(device="cpu") 
       

        std_dict = {'s_in':std_data,'s_out':std_label,'m_in':mean_data, 'm_out':mean_label}
        
        if type(wet) == list:
            for j in range(steps):
                temp_in = torch.swapaxes(torch.swapaxes(data_in[j],1,3),2,3)
                temp_out = torch.swapaxes(torch.swapaxes(data_out[j],1,3),2,3)            
                for i in range(len(wet)):
                    temp_in[:,i*N_vars:i*N_vars] = torch.mul(temp_in[:,i*N_vars:i*N_vars],wet[i])
                    start = num_outputs+N_atm
                    temp_in[:,start+i*N_vars:start+i*N_vars] = torch.mul(temp_in[:,start+i*N_vars:start+i*N_vars],wet[i])
                    temp_out[:,i*N_vars:i*N_vars] = torch.mul(temp_out[:,i*N_vars:i*N_vars],wet[i])
                print(num_outputs,start)
                temp_in[:,num_outputs:start] = torch.mul(temp_in[:,num_outputs:start],wet[0])
                data_out[j] = temp_out.clone()
                data_in[j] = temp_in.clone()
        elif wet == None:
            for j in range(steps):
                data_out[j] = torch.swapaxes(torch.swapaxes(data_out[j],1,3),2,3)
                data_in[j] = torch.swapaxes(torch.swapaxes(data_in[j],1,3),2,3)
        else:
            for j in range(steps):
                data_out[j] = torch.mul(torch.swapaxes(torch.swapaxes(data_out[j],1,3),2,3),wet)
                data_in[j] = torch.mul(torch.swapaxes(torch.swapaxes(data_in[j],1,3),2,3),wet)
                
        self.input = data_in

        self.output = data_out
        self.norm_vals = std_dict
        
    def __len__(self):
        # Number of data point we have. Alternatively self.data.shape[0], or self.label.shape[0]
        return self.size

    def __getitem__(self, idx):
        # Return the idx-th data point of the dataset
        # If we have multiple things to return (data point and label), we can return them as tuple
        data = [self.input[0][idx].to(device = self.device),self.output[0][idx].to(device = self.device)]
        for k in range(1,self.steps):
            data.append(self.input[k][idx].to(device = self.device))
            data.append(self.output[k][idx].to(device = self.device))
        
        return tuple(data)
    

def gen_data_in(step,s,e,interval,lag,hist,inputs,extra_in):
    s = s+lag*step
    e = e+lag*step    
    num_outs = len(inputs)
    num_extra = len(extra_in)
    temp_inputs = []
    for j in range(num_outs):
        temp_inputs.append(inputs[j][s:e:interval].to_numpy())
    temp_extra = []
    for j in range(num_extra):
        temp_extra.append(extra_in[j][s:e:interval].to_numpy())
        
    data_in = np.stack((*temp_inputs,
                         *temp_extra),-1)
    
    for i in range(hist):
        temp_inputs = []
        for j in range(num_outs):
            temp_inputs.append(np.expand_dims(inputs[j][s-(hist-i)*lag:e-(hist-i)*lag:(interval)].to_numpy(),-1))
        data_in = np.concatenate((data_in,*temp_inputs),axis=3)
    return data_in
        
def gen_data_out(step,s,e,lag,interval,outputs):
    s = s+lag*step
    e = e+lag*step
    
    num_outs = len(outputs)
    temp_outputs = []
    for j in range(num_outs):
        temp_outputs.append(outputs[j][s:e:interval].to_numpy())    
    
    data_out = np.stack(temp_outputs,-1)
    return data_out

def gen_data_in_test(step,s,N_test,lag,hist,inputs,extra_in):
    if lag == 0:
        lag = 1
    s = s+lag*step
    e = s+lag*N_test+lag*step   

    num_outs = len(inputs)
    num_extra = len(extra_in)
    temp_inputs = []
    for j in range(num_outs):
        temp_inputs.append(inputs[j][s:e:lag].to_numpy())
    temp_extra = []
    for j in range(num_extra):
        temp_extra.append(extra_in[j][s:e:lag].to_numpy())
        
    data_in = np.stack((*temp_inputs,
                         *temp_extra),-1)
    
    for i in range(hist):
        temp_inputs = []
        for j in range(num_outs):
            temp_inputs.append(np.expand_dims(inputs[j][s-(hist-i)*lag:e-(hist-i)*lag:(lag)].to_numpy(),-1))
        data_in = np.concatenate((data_in,*temp_inputs),axis=3)
    return data_in
        
def gen_data_out_test(step,s,N_test,lag,hist,outputs):
    if lag == 0:
        lag = 1
    s = s+lag*step
    e = s+lag*N_test+lag*step    
    
    num_outs = len(outputs)
    temp_outputs = []
    for j in range(num_outs):
        temp_outputs.append(outputs[j][s:e:lag].to_numpy())    
    
    data_out = np.stack(temp_outputs,-1)
    return data_out


def gen_data(input_vars,extra_vars,output_vars,lag,factor,region = "Kuroshio"):
    var_dict = {"um":"u_mean","vm":"v_mean","Tm":"T_mean",
                "ur":"u_res","vr":"v_res","Tr":"T_res",
               "u":"u","v":"v","T":"T",
               "tau_u":"tau_u","tau_v":"tau_v","tau":"tau",
               "t_ref":"t_ref"}

    data = xr.open_zarr("/scratch/as15415/Data/Emulation_Data/Velocities_"+region+"_Factor_"+str(factor)+".zarr/")
    data_temp = xr.open_zarr("/scratch/as15415/Data/Emulation_Data/Temperature_"+region+"_Factor_"+str(factor)+".zarr/")
    data_atmos = xr.open_zarr("/scratch/as15415/Data/Emulation_Data/Data_Atmos_"+region+".zarr/")
    data_atmos = data_atmos.rename_dims({"lat":"yu_ocean","lon":"xu_ocean"})
    data_atmos = data_atmos.rename({"lat":"yu_ocean","lon":"xu_ocean"})
    
    data_temp["xu_ocean"] = data.xu_ocean.data
    data_temp["yu_ocean"] = data.yu_ocean.data
    
    data = xr.merge([data,data_temp,data_atmos])
    
    inputs = []
    extra_in = []
    outputs = []
    
    for var in input_vars:
        inputs.append(data[var_dict[var]])

    for var in extra_vars:
        extra_in.append(data[var_dict[var]])
        
    for var in output_vars:
        outputs.append(data[var_dict[var]][lag:])
        
    inputs = tuple(inputs)
    extra_in = tuple(extra_in)
    outputs = tuple(outputs)

    return inputs, extra_in, outputs



def gen_data_global(input_vars,extra_vars,output_vars,lag,res ="1"):
    var_dict = {"um":"u_mean","vm":"v_mean","Tm":"T_mean",
                "ur":"u_res","vr":"v_res","Tr":"T_res",
               "u":"u","v":"v","T":"T",
               "tau_u":"tau_u","tau_v":"tau_v","tau":"tau",
               "t_ref":"t_ref"}

    data = xr.open_zarr("/scratch/as15415/Data/Emulation_Data/Global_Ocean_"+res+"deg.zarr")
    data_atmos = xr.open_zarr("/scratch/as15415/Data/Emulation_Data/Data_Atmos_"+res+"_deg.zarr").drop(["xu_ocean","T_mean"]).assign_coords({"lon":data.xu_ocean.data})
    data_atmos = data_atmos.rename_dims({"lat":"yu_ocean","lon":"xu_ocean"})
    data_atmos = data_atmos.rename({"lat":"yu_ocean","lon":"xu_ocean"})
    
    data_atmos["xu_ocean"] = data.xu_ocean.data
    data_atmos["yu_ocean"] = data.yu_ocean.data    
    
    data = xr.merge([data,data_atmos])
    
    inputs = []
    extra_in = []
    outputs = []
    
    for var in input_vars:
        inputs.append(data[var_dict[var]])

    for var in extra_vars:
        extra_in.append(data[var_dict[var]])
        
    for var in output_vars:
        outputs.append(data[var_dict[var]][lag:])
        
    inputs = tuple(inputs)
    extra_in = tuple(extra_in)
    outputs = tuple(outputs)

    return inputs, extra_in, outputs


def gen_data_lateral(input_vars,extra_vars,output_vars,lag,factor,region,Nb=2):
    var_dict = {"um":"u_mean","vm":"v_mean","Tm":"T_mean",
                "ur":"u_res","vr":"v_res","Tr":"T_res",
               "u":"u","v":"v","T":"T",
               "tau_u":"tau_u","tau_v":"tau_v","tau":"tau",
               "t_ref":"t_ref"}

    data = xr.open_zarr("/scratch/as15415/Data/Emulation_Data/Velocities_"+region+"_Factor_"+str(factor)+".zarr/")
    data_temp = xr.open_zarr("/scratch/as15415/Data/Emulation_Data/Temperature_"+region+"_Factor_"+str(factor)+".zarr/")
    data_atmos = xr.open_zarr("/scratch/as15415/Data/Emulation_Data/Data_Atmos_"+region+".zarr/")
    data_atmos = data_atmos.rename_dims({"lat":"yu_ocean","lon":"xu_ocean"})
    data_atmos = data_atmos.rename({"lat":"yu_ocean","lon":"xu_ocean"})
    
    data_temp["xu_ocean"] = data.xu_ocean.data
    data_temp["yu_ocean"] = data.yu_ocean.data
    
    data = xr.merge([data,data_temp,data_atmos])
    inputs = []
    extra_in = []
    outputs = []
    
    for var in input_vars:     
        temp = data[var_dict[var]].copy(deep=True)
#         temp[:,:Nb,:] = 0.*temp[0,:Nb,:]
#         temp[:,-Nb:,:] = 0.*temp[:,-Nb:,:]
#         temp[:,:,:Nb] = 0.*temp[:,:,:Nb]
#         temp[:,:,-Nb:] = 0.*temp[:,:,-Nb:]     
        inputs.append(temp)
 
    for var in extra_vars:
        extra_in.append(data[var_dict[var]])
        
    for var in input_vars:             
        temp = data[var_dict[var]].copy(deep=True)
        temp[:,Nb:-Nb,Nb:-Nb]  = 0.0 *temp[0,Nb:-Nb,Nb:-Nb]
        extra_in.append(temp)
        
    for var in output_vars:
        temp = data[var_dict[var]].copy(deep=True)
#         temp[:,:Nb,:] = 0.*temp[0,:Nb,:]
#         temp[:,-Nb:,:] = 0.*temp[:,-Nb:,:]
#         temp[:,:,:Nb] = 0.*temp[:,:,:Nb]
#         temp[:,:,-Nb:] = 0.*temp[:,:,-Nb:]   
        outputs.append(data[var_dict[var]][lag:])
        
    inputs = tuple(inputs)
    extra_in = tuple(extra_in)
    outputs = tuple(outputs)

    return inputs, extra_in, outputs


def gen_data_025(input_vars,extra_vars,output_vars,lag,lat,lon):
    var_dict = {"um":"u_mean","vm":"v_mean","Tm":"T_mean",
                "ur":"u_res","vr":"v_res","Tr":"T_res",
               "u":"u","v":"v","T":"T",
               "tau_u":"tau_u","tau_v":"tau_v","tau":"tau",
               "t_ref":"t_ref"}

    data = xr.open_zarr("/scratch/as15415/Data/Emulation_Data/Global_Ocean_025deg.zarr")
    data_atmos = xr.open_zarr("/scratch/as15415/Data/Emulation_Data/Data_Atmos_025_deg.zarr").drop(["xu_ocean","T_mean"]).assign_coords({"lon":data.xu_ocean.data})
    data_atmos = data_atmos.rename_dims({"lat":"yu_ocean","lon":"xu_ocean"})
    data_atmos = data_atmos.rename({"lat":"yu_ocean","lon":"xu_ocean"})
    
    data_atmos["xu_ocean"] = data.xu_ocean.data
    data_atmos["yu_ocean"] = data.yu_ocean.data    
    
    data = xr.merge([data,data_atmos])
    
    data = data.sel(yu_ocean=slice(lat[0],lat[1]),xu_ocean=slice(lon[0],lon[1]))
    
    inputs = []
    extra_in = []
    outputs = []
    
    for var in input_vars:
        inputs.append(data[var_dict[var]])

    for var in extra_vars:
        extra_in.append(data[var_dict[var]])
        
    for var in output_vars:
        outputs.append(data[var_dict[var]][lag:])
        
    inputs = tuple(inputs)
    extra_in = tuple(extra_in)
    outputs = tuple(outputs)

    return inputs, extra_in, outputs

def gen_data_025_lateral(input_vars,extra_vars,output_vars,lag,lat,lon,Nb=2,filter_T=False,filter_width = 20,area = None):
    var_dict = {"um":"u_mean","vm":"v_mean","Tm":"T_mean",
                "ur":"u_res","vr":"v_res","Tr":"T_res",
               "u":"u","v":"v","T":"T",
               "tau_u":"tau_u","tau_v":"tau_v","tau":"tau",
               "t_ref":"t_ref"}

    data = xr.open_zarr("/scratch/as15415/Data/Emulation_Data/Global_Ocean_025deg.zarr")
    data_atmos = xr.open_zarr("/scratch/as15415/Data/Emulation_Data/Data_Atmos_025_deg.zarr").drop(["xu_ocean","T_mean"]).assign_coords({"lon":data.xu_ocean.data})
    
#     data_atmos = xr.open_zarr("/scratch/as15415/Data/Emulation_Data/Data_Atmos_025_deg_filtered.zarr").assign_coords({"lon":data.xu_ocean.data})    
    data_atmos = data_atmos.rename_dims({"lat":"yu_ocean","lon":"xu_ocean"})
    data_atmos = data_atmos.rename({"lat":"yu_ocean","lon":"xu_ocean"})
    
#     data_atmos["xu_ocean"] = data.xu_ocean.data
#     data_atmos["yu_ocean"] = data.yu_ocean.data    
#     data_atmos["time"] = data.time.data

    data = xr.merge([data,data_atmos])
    
    data = data.sel(yu_ocean=slice(lat[0],lat[1]),xu_ocean=slice(lon[0],lon[1]))
    
    inputs = []
    extra_in = []
    outputs = []
    
    for var in input_vars:
        inputs.append(data[var_dict[var]])

    for var in extra_vars:
        if var == "t_ref" and filter_T:
            if filter_width == "mean":
                data[var_dict[var]] = data[var_dict[var]]*0 + ((data[var_dict[var]]*area).sum(dim=["xu_ocean","yu_ocean"])/area.sum()).compute()            
            else:
                data[var_dict[var]].data =gaussian_filter(data[var_dict[var]],filter_width)
        
        extra_in.append(data[var_dict[var]])

    for var in input_vars:             
        temp = data[var_dict[var]].copy(deep=True)
        temp[:,Nb:-Nb,Nb:-Nb]  = 0.0 *temp[0,Nb:-Nb,Nb:-Nb]
        extra_in.append(temp)
        
    for var in output_vars:
        outputs.append(data[var_dict[var]][lag:])
        
    inputs = tuple(inputs)
    extra_in = tuple(extra_in)
    outputs = tuple(outputs)

    return inputs, extra_in, outputs



def gen_data_025_lateral_subsurf(input_vars,extra_vars,output_vars,depth,lag,lat,lon,Nb=2):
    var_dict = {"um":"u_mean","vm":"v_mean","Tm":"T_mean",
                "ur":"u_res","vr":"v_res","Tr":"T_res",
               "u":"u","v":"v","T":"T",
               "tau_u":"tau_u","tau_v":"tau_v","tau":"tau",
               "t_ref":"t_ref"}

    data = xr.open_zarr("/scratch/as15415/Data/Emulation_Data/Global_Ocean_025deg_depth_"+depth+".zarr")
    data_atmos = xr.open_zarr("/scratch/as15415/Data/Emulation_Data/Global_Ocean_025deg_5_day_Avg.zarr")
    data_atmos=data_atmos.rename({"u":"tau_u","v":"tau_v","T":"t_ref"}) 
    data_atmos=data_atmos.drop(["u_mean","v_mean","T_mean","u_res","v_res","T_res"])      
    
    data = xr.merge([data,data_atmos])
    
    data = data.sel(yu_ocean=slice(lat[0],lat[1]),xu_ocean=slice(lon[0],lon[1]))
    
    inputs = []
    extra_in = []
    outputs = []
    
    for var in input_vars:
        inputs.append(data[var_dict[var]])

    for var in extra_vars:
        extra_in.append(data[var_dict[var]])

    for var in input_vars:             
        temp = data[var_dict[var]].copy(deep=True)
        temp[:,Nb:-Nb,Nb:-Nb]  = 0.0 *temp[0,Nb:-Nb,Nb:-Nb]
        extra_in.append(temp)
        
    for var in output_vars:
        outputs.append(data[var_dict[var]][lag:])
        
    inputs = tuple(inputs)
    extra_in = tuple(extra_in)
    outputs = tuple(outputs)

    return inputs, extra_in, outputs

def get_norms(region,inputs,extra_in,outputs,Lateral = True):
    
    if region == "Africa_Ext":
        mean_dict = {"um":4.82752338e-2,"vm":1.1804e-2,"Tm":10.8398,"tau_u":6.83281880e-5,"tau_v":-1.04916221e-5,
                   "t_ref":2.85715049e+2,"um_0":2.80994716e-02,"vm_0":1.1804e-2,"Tm_0":10.8398,
                     "um_99":2.93019213e-02,"vm_99":-4.84687595e-03,"Tm_99":8.91647284,
                     "um_253":2.49547245e-02,"vm_253":-4.48418542e-03,"Tm_253":6.77900864}
        
        std_dict = {"um":1.78659424e-1,"vm":1.62300227e-01,"Tm":8.35850152,"tau_u":1.32713360e-4,"tau_v":9.77289618e-5,
                   "t_ref":7.38032223e+0,"um_0":1.78659424e-1,"vm_0":1.62300227e-01,"Tm_0":8.35850152,
                     "um_99":0.14836239,"vm_99":0.14007329,"Tm_99":7.49790942,
                     "um_253":0.13086694,"vm_253":0.12258011,"Tm_253":5.91231414}
    
    elif region == "Gulf_Stream_Ext":
        mean_dict = {"um":4.48291145e-02,"vm":-1.05211059e-02,"Tm":1.21081783e+01,"tau_u":3.82348062e-05,"tau_v":8.21162517e-06,
                   "t_ref":2.85715049e+2,"um_0":4.48291145e-02,"vm_0":-1.05211059e-02,"Tm_0":1.21081783e+01,
                     "um_99":2.93019213e-02,"vm_99":4.04511051e-03,"Tm_99":9.98328467e+00,
                     "um_253":2.39374764e-02,"vm_253":4.68866993e-03,"Tm_253":8.55389633e+00}
        
        std_dict = {"um":1.51401489e-01,"vm":1.43377056e-01,"Tm":9.03735246e+00,"tau_u":9.60773988e-05,"tau_v":8.42097158e-05,
                   "t_ref":9.21912876e+00,"um_0":1.51401489e-01,"vm_0":1.43377056e-01,"Tm_0":9.03735246e+00,
                     "um_99":0.13803744,"vm_99":0.12650987,"Tm_99":8.29932087,
                     "um_253":0.12300715,"vm_253":0.11049998,"Tm_253":7.68735387}    

    elif region == "Tropics_Ext":
        mean_dict = {"um":-6.56001477e-02,"vm":3.03974905e-02,"Tm":1.85474497e+01,"tau_u":-3.79044117e-05,"tau_v":-4.91180848e-06,
                   "t_ref":2.96640798e+02,"um_0":-6.56001477e-02,"vm_0":3.03974905e-02,"Tm_0":1.85474497e+01,
                     "um_99":4.71830777e-03,"vm_99":4.80014641e-04,"Tm_99":1.22588992e+01,
                     "um_253":-6.63711393e-03,"vm_253":1.01151822e-03,"Tm_253":8.36092632e+00}
        
        std_dict = {"um":1.95064212e-01,"vm":1.42985598e-01,"Tm":1.13369541e+01,"tau_u":4.90698542e-05,"tau_v":3.49303944e-05,
                   "t_ref":2.97622406e+00,"um_0":1.95064212e-01,"vm_0":1.42985598e-01,"Tm_0":1.13369541e+01,
                     "um_99":0.14478501,"vm_99":0.0861209,"Tm_99":9.17627311,
                     "um_253":0.0638834,"vm_253":0.05032483,"Tm_253":6.74403382}            

    elif region == "Quiescent_Ext":
        mean_dict = {"um":3.18046221e-02,"vm":1.31442399e-03,"Tm":1.61681938e+01,"tau_u":2.58048575e-05,"tau_v":-3.11680868e-06,
                   "t_ref":2.87996087e+02,"um_0":3.18046221e-02,"vm_0":1.31442399e-03,"Tm_0":1.61681938e+01,
                     "um_99":1.87243772e-02,"vm_99":2.10032583e-03,"Tm_99":1.41782428e+01,
                     "um_253":1.18352078e-02,"vm_253":1.63565767e-03,"Tm_253":1.10941377e+01}
        std_dict = {"um":1.04966172e-01,"vm":8.70280337e-02,"Tm":7.18070321e+00,"tau_u":1.25641199e-04,"tau_v":9.48015232e-05,
                   "t_ref":7.01619519e+00,"um_0":1.04966172e-01,"vm_0":8.70280337e-02,"Tm_0":7.18070321e+00,
                     "um_99":0.07786532,"vm_99":0.05987989,"Tm_99":6.86105509,
                     "um_253":0.07023565,"vm_253":0.05248215,"Tm_253":5.19177608}           
        
    if Lateral:
        mean_in = np.zeros(len(inputs+extra_in+inputs))
        std_in = np.zeros(len(inputs+extra_in+inputs))
    else:
        mean_in = np.zeros(len(inputs+extra_in))
        std_in = np.zeros(len(inputs+extra_in))      
        
    mean_out = np.zeros(len(outputs))
    std_out = np.zeros(len(outputs))
    
    for (i,j) in zip(range(len(inputs+extra_in)),inputs+extra_in):
        mean_in[i] = mean_dict[j]
        std_in[i] = std_dict[j]
    if Lateral:
        mean_in[-len(inputs):] = mean_in[:len(inputs)]
        std_in[-len(inputs):] = std_in[:len(inputs)]
        
    for (i,j) in zip(range(len(outputs)),outputs):
        mean_out[i] = mean_dict[j]
        std_out[i] = std_dict[j]
        
    std_dict = {'s_in':std_in,'s_out':std_out,'m_in':mean_in, 'm_out':mean_out}
    
    return std_dict


def get_train_test_ranges(N_samples, N_val, lag, hist, interval):
    s_train = lag*hist # 1*0=0
    e_train = s_train + N_samples*interval # 0 + 4000*1 = 4000
    e_test = e_train + interval*N_val # 4000 + 1*300 = 4300
    return s_train, e_train, e_test

def get_wet_mask(inputs, device="cpu"):
    wet = xr.zeros_like(inputs[0][0])
    # inputs[0][0,12,12] = np.nan
    for data in inputs:
        wet +=np.isnan(data[0])
    
    wet_nan = xr.where(wet!=0,np.nan,1).to_numpy()
    wet = np.isnan(xr.where(wet==0,np.nan,0))
    wet = np.nan_to_num(wet.to_numpy())
    wet = torch.from_numpy(wet).type(torch.float32).to(device=device)
    return wet, wet_nan