1. move the folder 'MATLABZOSConnection' to '**\Zemax\ZOS-API Projects' 
2. run 'sort_wide_lens.m' to filter and save wide angle lens.
3. run 'Extract_zernike_from_widelens' to extract zernike coefficients and save them to excel file.
4. Run `Sort_zernike_coefficient_save_PSF_mat.py` to generate .mat files from `Lens_Zernike_lib.xlsx` 
5. Run `dataset_generator_mat.py` to generate dataset file from .mat files,  train dataset and valid dataset are generated at the same time.
6. Run `main_wavefront_PSF_CircleMSE_0.1TV.py` to train supervised net by dataset, checkpoints will be saved in `Models` folders