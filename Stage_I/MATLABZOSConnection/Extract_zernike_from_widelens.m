clear;
clc;
import ZOSAPI.*;

TheApplication = MATLABZOSConnection;
       
% Set up primary optical system
TheSystem = TheApplication.PrimarySystem;
sampleDir = TheApplication.SamplesDir;

% change it to your directory
FileDir = "E:\Code\LWNet\Data\lens_dataset\wide_lens";
file_list   = dir(FileDir); 
file_list = file_list(endsWith({file_list.name}, {'.zmx', '.ZMX'}));
SaveDir = "E:\Code\LWNet\Data\lens_dataset";
SaveZerDir = "E:\Code\LWNet\Data\lens_dataset\Lens_Zernike_Lib.xlsx";

TheSystemData = TheSystem.SystemData;

%change file_index and column if it is interrupted accidentally
file_index = 1;
column = 1;


while file_index <= size(file_list,1)
    close all;
    flag = false;
    
	fprintf('Test file is %s.\n',file_list(file_index).name);
	testFile = fullfile(file_list(file_index).folder,file_list(file_index).name);
	TheSystem.LoadFile(testFile,false);
	TheSystemData.Fields.SetFieldType(ZOSAPI.SystemData.FieldType.Angle);
    
    %%%%%%%%%%%%%%%%%%% interpolate across angles %%%%%%%%%%%%%%%%%%%%%%%%%%%%
    Fnum_start = TheSystem.SystemData.Fields.NumberOfFields;
    index = 1;
        for  j = 0:60
            angle = j*1;
            TheSystemData.Fields.AddField(0,angle,1);
            fldarray(index) = round(angle);
            index = index+1;
        end    
    Fnum_end = TheSystem.SystemData.Fields.NumberOfFields;

    %%%%%%%%%%%%%%% obtain the design wavelength range %%%%%%%%%%%%%%%%
    TheWavelengths = TheSystemData.Wavelengths;
    num_wl = TheWavelengths.NumberOfWavelengths;
    wl1 = TheWavelengths.GetWavelength(1).Wavelength;
    wl2 = TheWavelengths.GetWavelength(num_wl).Wavelength;
    wl_start = min(wl1,wl2);
    wl_end = max(wl1,wl2);

    %%%%%%%%%%%%%%%%%%% interpolate across wavelengths %%%%%%%%%%%%%%%%%%%%%%%%%%%%
    wlnum_start = num_wl;
    index =1;
        for  i = wl_start:0.02:wl_end
            TheWavelengths.AddWavelength(i,1);
            wlarray(index) = round(1000*i); 
            index = index+1;
        end 
        
    wlnum_end = TheWavelengths.NumberOfWavelengths;
    Wavefront = TheSystem.Analyses.New_WavefrontMap();     
    WavefrontSettings =Wavefront.GetSettings();
             
	for fldnum = (Fnum_start+1):Fnum_end
        
            if flag
                break;
            end
            
        for wlnum = (wlnum_start+1):wlnum_end 
            
            if isequal(WavefrontSettings, [])
                disp('wavefrontmap analyse failed.');
                disp('');
            else
            
                WavefrontSettings.Wavelength.SetWavelengthNumber(wlnum);
                WavefrontSettings.Field.SetFieldNumber(fldnum);
                % attention, set the analysis plane as image plane
                WavefrontSettings.Surface.UseImageSurface();
                WavefrontSettings.Sampling=ZOSAPI.Analysis.SampleSizes.S_256x256;
                Wavefront.ApplyAndWaitForCompletion();   
                WFResults = Wavefront.GetResults();
                tempstr = fullfile(SaveDir,'latest_wavefront.xls');
                WFResults.GetTextFile(tempstr);

                WFdata = xlsread(tempstr);
                nonezeroN = sum(sum(WFdata~=0));
                PV = max(max(WFdata)) - min(min(WFdata));
            
                if nonezeroN/65536 > 0.7 && PV >0.1 && PV<10
                    Zernike = TheSystem.Analyses.New_ZernikeStandardCoefficients();
                    ZernikeSettings =Zernike.GetSettings(); 
                    filename = strcat(file_list(file_index).name(1:end-4),'@',num2str(wlarray(wlnum-wlnum_start)),"nm_",num2str(fldarray(fldnum-Fnum_start)));
                    ZernikeSettings.Wavelength.SetWavelengthNumber(wlarray(wlnum-wlnum_start));
                    ZernikeSettings.Field.SetFieldNumber(fldarray(fldnum-Fnum_start));
                    Zernike.ApplyAndWaitForCompletion();   
                    ZNResults = Zernike.GetResults();
                    str = fullfile(SaveDir,"latest_zernike.csv");
                    ZNResults.GetTextFile(str);
                    Zernike.Close();
                    
                    try
                        ZNdata=xlsread(str,"B39:B75")';        
                        range1 = strcat('A',num2str(column));
                        range2 = strcat('B',num2str(column));
                        range3 = strcat('C',num2str(column));
                        xlswrite(SaveZerDir,ZNdata,'Sheet1',range1);
                        xlswrite(SaveZerDir,cellstr(file_list(file_index).name(1:end-4)),'Sheet2',range1);
                        xlswrite(SaveZerDir,cellstr(num2str(wlarray(wlnum-wlnum_start))),'Sheet2',range2);
                        xlswrite(SaveZerDir,cellstr(num2str(fldarray(fldnum-Fnum_start))),'Sheet2',range3);
                        column = column+1;
                        clear ZNdata filename;
                        delete(str);
                        delete(tempstr);
                    catch
                    end
                 
                else
                    flag = true;
                    break;                    
                end 
            end
        end
	end

    Wavefront.Close();
    file_index = file_index +1;
 
end   


    
    
