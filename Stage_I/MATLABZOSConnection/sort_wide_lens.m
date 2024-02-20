% filter and save the wide lens: judge whether the wavefront fulfill requirements of wide lens
% this file should be placed in the "*\Zemax\ZOS-API Projects\MATLABZOSConnection" directory,
clear;
clc;
import ZOSAPI.*;

TheApplication = MATLABZOSConnection;
       
% Set up primary optical system
TheSystem = TheApplication.PrimarySystem;

%change it to your directory
FileDir = "E:\Code\LWNet\Data\lens_dataset\lens";
file_list   = dir(FileDir);
file_list = file_list(endsWith({file_list.name}, {'.zmx', '.ZMX'}));
SaveDir = "E:\Code\LWNet\Data\lens_dataset\wide_lens";
 
TheSystemData = TheSystem.SystemData;
file_index = 500;


while file_index<= size(file_list,1)
    
    wfdir = fullfile(SaveDir,'try_wavefront.csv');
    if exist(wfdir, 'file') == 2
        delete(wfdir);        
    end

	fprintf('Test file is %s.\n',file_list(file_index).name);
	testFile = fullfile(file_list(file_index).folder,file_list(file_index).name);
	TheSystem.LoadFile(testFile,false);
	TheSystemData.Fields.SetFieldType(ZOSAPI.SystemData.FieldType.Angle);

    %%%%%%%%%%%%%%%%%%% interpolate across angles %%%%%%%%%%%%%%%%%%%%%%%%%%%%
    Fnum_start = TheSystem.SystemData.Fields.NumberOfFields;
    TheSystemData.Fields.AddField(0,15,1); 

    %%%% try to extract the 15 degree wavefront,if failed skip
    Wavefront = TheSystem.Analyses.New_WavefrontMap();   
    WavefrontSettings =Wavefront.GetSettings();

    if isequal(WavefrontSettings, [])
        disp('wavefrontmap analyse failed.');
        disp('');
    else
        WavefrontSettings.Wavelength.SetWavelengthNumber(0);
        WavefrontSettings.Field.SetFieldNumber(Fnum_start+1);
        Field = TheSystemData.Fields.GetField(Fnum_start+1);
        wl = TheSystemData.Wavelengths.GetWavelength(1);
%         fprintf('Analysis wavelength is %.3f um and fields angle is %d degree.\n',wl.Wavelength,Field.Y);
        % extract the wavefront of 15 degree
        WavefrontSettings.Surface.UseImageSurface();
        WavefrontSettings.Sampling=ZOSAPI.Analysis.SampleSizes.S_256x256;
        Wavefront.ApplyAndWaitForCompletion();   
        WFResults = Wavefront.GetResults();
        tempstr_wf = fullfile(SaveDir,'try_wavefront.xls');
        WFResults.GetTextFile(tempstr_wf);
        Wavefront.Close();
        
        WFdata=xlsread(tempstr_wf);

        nonezeroN = sum(sum(WFdata~=0));
        PV = max(max(WFdata)) - min(min(WFdata));
        % judge whether the wavefront fulfill requirements to filter the wide lens
        if nonezeroN/65536 < 0.7 || PV < 0.1 || PV > 10
            fprintf('The value of PV is %.2f and the non zero ratio is  %.2f.\n\n',PV,nonezeroN/65536);
            file_index = file_index +1;
            continue
        else
            TheSystemData.Fields.RemoveField(Fnum_start+1);
            copyfile(testFile, SaveDir); 
            fprintf('\n   %s is saved.\n\n',file_list(file_index).name);
        end
    end

    %next file
    file_index = file_index +1;
end

