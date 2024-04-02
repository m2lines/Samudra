import torch
from utils.data_utils import data_CNN_Lateral, data_CNN_steps_Lateral, RecUnetDataset, RecUnetEvalDataset

if __name__ == "__main__":
    train_data = torch.load('/scratch/sd5313/M2Lines/emulator/Ocean_Emulator/.LOCAL/save_data/2024-03-31-save_data_gulfext2/data/train_data_cnn_steps_8_Gulf_Stream_Ext2_Test_in_um_vm_Tm_ext_tau_u_tau_v_t_ref__outum_vm_Tm_N_train_4000_Lateral_Data_025_no_smooth.pt', map_location='cpu')
    recdata = RecUnetDataset('/scratch/sd5313/M2Lines/emulator/Ocean_Emulator/.LOCAL/save_data_tensor/2024-03-31-save_data_tensor_recunet_with_wet/data/train_data_steps_0_Gulf_Stream_Ext2_Test_in_um_vm_Tm_ext_tau_u_tau_v_t_ref__outum_vm_Tm_N_train_4000_Lateral_Data_025_no_smooth.pt', 8, 2, 1, 3, 4, '/scratch/sd5313/M2Lines/emulator/Ocean_Emulator/.LOCAL/save_data_tensor/2024-03-31-save_data_tensor_recunet_with_wet/data/wet_steps_0_Gulf_Stream_Ext2_Test_in_um_vm_Tm_ext_tau_u_tau_v_t_ref__outum_vm_Tm_N_train_4000_Lateral_Data_025_no_smooth.pt')
    

    train_inputs = []
    train_outputs = []
    for data in train_data:
        inputs = []
        outputs = []
        for step in range(8):
            inputs.append(data[2*step].unsqueeze(0))
            outputs.append(data[2*step+1].unsqueeze(0))
        
        train_inputs.append(torch.cat(inputs, dim=0).unsqueeze(0).cpu())
        train_outputs.append(torch.cat(outputs, dim=0).unsqueeze(0).cpu())

    train_data_inputs = torch.cat(train_inputs, dim=0).cpu()
    train_data_outputs = torch.cat(train_outputs, dim=0).cpu()

    torch.cuda.empty_cache()

    print(train_data_inputs.shape)
    print(train_data_outputs.shape)

    # test_data = torch.load('/scratch/sd5313/M2Lines/emulator/Ocean_Emulator/data/test_data_cnn_steps_8_Gulf_Stream_Ext2_in_um_vm_Tm_ext_tau_u_tau_v_t_ref_N_samples_4000.pt')
    # recevaldata = RecUnetEvalDataset('/scratch/sd5313/M2Lines/emulator/Ocean_Emulator/data/test_data_steps_8_Gulf_Stream_Ext2_in_um_vm_Tm_ext_tau_u_tau_v_t_ref_N_samples_4000.pt', 8, 2, 1, 3, 4, '/scratch/sd5313/M2Lines/emulator/Ocean_Emulator/.LOCAL/save_data_tensor/2024-03-31-save_data_tensor_recunet_with_wet/data/wet_steps_0_Gulf_Stream_Ext2_Test_in_um_vm_Tm_ext_tau_u_tau_v_t_ref__outum_vm_Tm_N_train_4000_Lateral_Data_025_no_smooth.pt')

    # print(torch.sum(recevaldata[0][0][4,:3] - test_data[0][1][:]))
    # print(torch.sum(recevaldata[0][1][0,:3] - test_data[0][1][:]))
    # print(torch.sum(recevaldata[1][1][0,:3] - test_data[8][1][:]) )

    # # import pdb; pdb.set_trace()

    # recevaldata.set_device_and_convert_returns(3000, device="cpu")

    # print(torch.sum(recevaldata[:][1] - test_data[:][1].cpu()) )
    # print(torch.sum(recevaldata[:][0] - test_data[:][0].cpu()))


    # import pdb; pdb.set_trace()