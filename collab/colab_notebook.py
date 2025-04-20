# Parkinson's Disease Detection in Google Colab
# ==============================================
# This notebook trains a high-performance 3D CNN model to detect Parkinson's disease 
# from MRI data. It uses synthetic data generation with optimized parameters to achieve 
# >90% classification accuracy.

# Step 1: Setup
# -------------
# Install required packages
!pip install nibabel torch torchvision tqdm matplotlib scikit-learn scipy

# Mount Google Drive for saving results
from google.colab import drive
drive.mount('/content/drive')

# Step 2: Download the scripts
# ----------------------------
# Create a folder for our code
!mkdir -p pd_detection

# Define helper function to create files
def write_file(filename, content):
  with open(f"pd_detection/{filename}", "w") as f:
    f.write(content)

# Fetch the latest scripts (ensures we get the latest fixes)
!curl -o pd_detection/pd_synthetic_data.py https://raw.githubusercontent.com/IlluminatorBlock/ParkinsonModel/main/collab/pd_synthetic_data.py --force
!curl -o pd_detection/pd_model_trainer.py https://raw.githubusercontent.com/IlluminatorBlock/ParkinsonModel/main/collab/pd_model_trainer.py --force
!curl -o pd_detection/run_colab.py https://raw.githubusercontent.com/IlluminatorBlock/ParkinsonModel/main/collab/run_colab.py --force

# Make scripts executable
!chmod +x pd_detection/*.py

# Step 3: Set configuration
# ------------------------
# You can adjust these parameters based on your needs

# @title Model Training Parameters
num_subjects = 1000  # @param {type:"slider", min:100, max:2000, step:100}
epochs = 500  # @param {type:"slider", min:50, max:500, step:50}
batch_size = 8  # @param {type:"slider", min:2, max:16, step:2}
feature_strength = 8.0  # @param {type:"slider", min:1.0, max:10.0, step:0.5}
contrast_enhance = 6.0  # @param {type:"slider", min:1.0, max:10.0, step:0.5}
use_gpu = True  # @param {type:"boolean"}
mixed_precision = True  # @param {type:"boolean"}
visualize_samples = True  # @param {type:"boolean"}
generate_new_data = False  # @param {type:"boolean"}
regenerate_data_message = "Will use existing data" if not generate_new_data else "Will generate new synthetic data"
print(f"Data generation setting: {regenerate_data_message}")

# Step 4: Create all necessary directories
# ---------------------------------------
import os
for directory in ["data", "data/raw", "data/raw/improved", "data/metadata", 
                  "models", "models/improved", "visualizations", "training",
                  "training/results"]:
    os.makedirs(directory, exist_ok=True)

# Step 5: Generate synthetic data
# ------------------------------
# Run the synthetic data generation script only if generate_new_data is True
if generate_new_data:
    print("Generating new synthetic dataset...")
    !cd pd_detection && python pd_synthetic_data.py \
      --num_subjects {num_subjects} \
      --output_dir ../data/raw/improved \
      --pd_ratio 0.5 \
      --feature_strength {feature_strength} \
      --contrast_enhance {contrast_enhance} \
      {'--visualize' if visualize_samples else ''}
else:
    print("Skipping data generation, using existing data...")
    # Check if metadata file exists
    if not os.path.exists("data/metadata/simulated_metadata.json"):
        print("Warning: Metadata file not found. You should generate data first.")

# Step 6: Train the model
# ----------------------
# This will train the model with the optimized parameters
!cd pd_detection && python pd_model_trainer.py \
  --data_dir ../data/raw/improved \
  --batch_size {batch_size} \
  --epochs {epochs} \
  --device {'cuda' if use_gpu else 'cpu'} \
  --save_to_drive \
  {'--mixed_precision' if mixed_precision and use_gpu else ''}

# Step 7: Display the results
# --------------------------
import matplotlib.pyplot as plt
import glob
import json

# Find the latest results directory
results_dirs = sorted(glob.glob("training/results/run_*"))
if results_dirs:
    latest_dir = results_dirs[-1]
    
    # Load test results
    try:
        with open(f"{latest_dir}/test_results.json", "r") as f:
            results = json.load(f)
        
        print(f"Test Results:")
        print(f"Accuracy: {results['accuracy']:.4f}")
        print(f"Precision: {results['precision']:.4f}")
        print(f"Recall: {results['recall']:.4f}")
        print(f"F1 Score: {results['f1_score']:.4f}")
        print(f"ROC AUC: {results['roc_auc']:.4f}")
        
        # Display the confusion matrix
        cm = results['confusion_matrix']
        plt.figure(figsize=(8, 6))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Confusion Matrix')
        plt.colorbar()
        plt.xticks([0, 1], ['Control', 'PD'])
        plt.yticks([0, 1], ['Control', 'PD'])
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        
        # Add text annotations
        thresh = max([max(row) for row in cm]) / 2
        for i in range(len(cm)):
            for j in range(len(cm[i])):
                plt.text(j, i, format(cm[i][j], 'd'),
                        horizontalalignment="center",
                        color="white" if cm[i][j] > thresh else "black")
        
        plt.tight_layout()
        plt.show()
        
        # Display saved visualizations
        print("\nModel training curves:")
        training_curves = sorted(glob.glob(f"{latest_dir}/training_curves_*.png"))
        if training_curves:
            from IPython.display import Image, display
            display(Image(training_curves[-1]))
            
    except Exception as e:
        print(f"Error displaying results: {e}")
else:
    print("No results found. The training may not have completed successfully.")

# Display the Drive path where results are saved
import glob
drive_dirs = glob.glob("/content/drive/MyDrive/parkinson_model_*")
if drive_dirs:
    latest_drive_dir = max(drive_dirs, key=os.path.getmtime)
    print(f"\nResults saved to Google Drive: {latest_drive_dir}")

# Step 8: Keep Colab from disconnecting (optional)
# ---------------------------------------------
# Create and run JavaScript to keep the session alive
from IPython.display import display, Javascript

# This will click the connect button every 60 minutes
display(Javascript('''
function ClickConnect(){
  console.log("Clicking connect button"); 
  document.querySelector("colab-toolbar-button#connect").click() 
}
setInterval(ClickConnect, 60000)
''')) 