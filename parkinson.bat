@echo off
setlocal enabledelayedexpansion

if "%1"=="" (
    echo Parkinson's MRI Detection - Usage
    echo ---------------------------------
    echo parkinson.bat setup    : Install dependencies and generate dataset
    echo parkinson.bat train    : Train the model
    echo parkinson.bat predict [path_to_mri.nii.gz] : Predict Parkinson's from MRI scan
    echo parkinson.bat stop     : Stop any running training processes
    goto :eof
)

if "%1"=="setup" (
    echo Setting up Parkinson's MRI Detection...
    
    echo Installing dependencies...
    pip install -r requirements.txt
    
    echo Creating directories...
    mkdir data\raw\simulated 2>nul
    mkdir data\processed 2>nul
    mkdir data\metadata 2>nul
    mkdir models\pretrained 2>nul
    mkdir training\results 2>nul
    mkdir visualizations 2>nul
    
    echo Generating synthetic dataset...
    python scripts/synthetic_data_generation.py --num_subjects 100 --pd_ratio 0.5 --contrast_enhance 2.0 --feature_strength 3.0
    
    echo Setup complete!
    goto :eof
)

if "%1"=="train" (
    echo Starting Parkinson's MRI Detection training...
    
    python training/train.py ^
      --data_dir data/raw/simulated ^
      --metadata data/metadata/simulated_metadata.json ^
      --model_dir models/pretrained ^
      --epochs 100 ^
      --batch_size 4 ^
      --lr 0.0002 ^
      --weight_decay 0.01 ^
      --dropout 0.3 ^
      --use_contrastive ^
      --use_regions ^
      --embed_dim 128 ^
      --depth 6 ^
      --num_heads 4 ^
      --input_size 64 ^
      --patch_size 8 ^
      --seed 123
    
    echo Training complete!
    goto :eof
)

if "%1"=="predict" (
    if "%2"=="" (
        echo Error: Please provide the path to an MRI scan.
        echo Usage: parkinson.bat predict [path_to_mri.nii.gz]
        goto :eof
    )
    
    echo Processing MRI scan: %2
    
    if not exist "models\pretrained\best_model.pt" (
        echo Warning: No best model found. Using the latest available model.
        for /f "delims=" %%i in ('dir /b /od "models\pretrained\pd_model_*.pt" 2^>nul') do set "latest_model=%%i"
        
        if not defined latest_model (
            echo Error: No trained models found. Please train a model first.
            goto :eof
        )
        
        echo Using model: !latest_model!
        python scripts/predict.py --mri_path "%2" --model_path "models\pretrained\!latest_model!" --input_size 64 --embed_dim 128 --depth 6 --num_heads 4 --patch_size 8
    ) else (
        python scripts/predict.py --mri_path "%2" --input_size 64 --embed_dim 128 --depth 6 --num_heads 4 --patch_size 8
    )
    
    echo The prediction result has been saved in the visualizations folder.
    goto :eof
)

if "%1"=="stop" (
    echo Stopping Parkinson's MRI Detection processes...
    
    for /f "tokens=2" %%i in ('tasklist /fi "imagename eq python.exe" /fo csv /nh') do (
        wmic process where "processid=%%i" get commandline | find "train.py" > nul
        if not errorlevel 1 (
            echo Stopping training process %%i...
            taskkill /PID %%i /F
        )
    )
    
    echo All training processes have been stopped.
    goto :eof
)

echo Unknown command: %1
echo Run "parkinson.bat" without arguments to see available commands. 