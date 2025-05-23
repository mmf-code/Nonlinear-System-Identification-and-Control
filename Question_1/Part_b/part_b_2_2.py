import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt
import pathlib
import importlib.util 

def load_system_module(system_name="System2"):
    pyc_filename = f"{system_name}.cpython-38.pyc"
    pyc_path = pathlib.Path("./python") / pyc_filename
    if not pyc_path.exists():
        print(f"Error: {pyc_filename} not found at {pyc_path}")
        print("Please ensure the pyc_path variable is set correctly.")
        sys.exit()
    spec = importlib.util.spec_from_file_location(system_name.lower(), pyc_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, system_name)

# --- Configuration ---
STUDENT_ID = 35
# These MUST match the NARX orders used for dataset generation
N_u = 2  # Number of input terms (u(k), u(k-1))
N_y = 1  # Number of past output terms (y(k-1))
INPUT_FEATURES = N_u + N_y # Total number of features in X

# Dataset filenames
dataset_dir = pathlib.Path("Question_1/Part_b")
dataset_filename_X = dataset_dir / f"system2_narx_X_data_Nu{N_u}_Ny{N_y}_ID{STUDENT_ID}.npy"
dataset_filename_Y = dataset_dir / f"system2_narx_Y_data_Nu{N_u}_Ny{N_y}_ID{STUDENT_ID}.npy"


# NN Hyperparameters
HIDDEN_SIZE_1 = 32 # Number of neurons in the hidden layer
HIDDEN_SIZE_2 = 16 # For a second hidden layer, 
LEARNING_RATE = 1e-3
BATCH_SIZE = 64
NUM_EPOCHS = 400 # Monitor validation loss
VALIDATION_SPLIT = 0.2 # 20% for validation
RANDOM_SEED = 42 # For reproducible splits

# Set random seeds for reproducibility
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# --- 1. Load Dataset ---
print("--- 1. Loading Dataset ---")
try:
    X_data_np = np.load(dataset_filename_X)
    Y_data_np = np.load(dataset_filename_Y)
    print(f"Loaded X data shape: {X_data_np.shape}")
    print(f"Loaded Y data shape: {Y_data_np.shape}")
    assert X_data_np.shape[1] == INPUT_FEATURES, \
        f"Loaded X data has {X_data_np.shape[1]} features, expected {INPUT_FEATURES} based on N_u and N_y."
except FileNotFoundError:
    print(f"Error: Dataset files not found. Please run the generation script first.")
    print(f"Expected X: {dataset_filename_X}")
    print(f"Expected Y: {dataset_filename_Y}")
    sys.exit()

# --- 2. Data Preprocessing & Splitting ---
print("\n--- 2. Data Preprocessing & Splitting ---")

# Split into training and validation sets
X_train, X_val, Y_train, Y_val = train_test_split(
    X_data_np, Y_data_np, test_size=VALIDATION_SPLIT, random_state=RANDOM_SEED
)
print(f"Training set size: X_train={X_train.shape}, Y_train={Y_train.shape}")
print(f"Validation set size: X_val={X_val.shape}, Y_val={Y_val.shape}")

# Scaling (Standardization)
# Fit scaler ONLY on training data, then transform both train and val
scaler_X = StandardScaler()
X_train_scaled = scaler_X.fit_transform(X_train)
X_val_scaled = scaler_X.transform(X_val)

scaler_Y = StandardScaler() # Output scaling
Y_train_scaled = scaler_Y.fit_transform(Y_train)
Y_val_scaled = scaler_Y.transform(Y_val)

print("Data scaled using StandardScaler.")

# Convert to PyTorch Tensors
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
Y_train_tensor = torch.tensor(Y_train_scaled, dtype=torch.float32)
X_val_tensor = torch.tensor(X_val_scaled, dtype=torch.float32)
Y_val_tensor = torch.tensor(Y_val_scaled, dtype=torch.float32)

# Create DataLoaders
train_dataset = TensorDataset(X_train_tensor, Y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

val_dataset = TensorDataset(X_val_tensor, Y_val_tensor)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# --- 3. Define Neural Network Model ---
print("\n--- 3. Defining Neural Network Model ---")
class System2NN(nn.Module):
    def __init__(self, input_size=INPUT_FEATURES, hidden_size1=HIDDEN_SIZE_1, hidden_size2=HIDDEN_SIZE_2, output_size=1):
        super(System2NN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.relu2 = nn.ReLU()
        self.fc_out = nn.Linear(hidden_size2, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc_out(x)
        return x


model = System2NN()
print(model)

# --- 4. Training Setup ---
print("\n--- 4. Training Setup ---")
criterion = nn.MSELoss() # Mean Squared Error for regression
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# --- 5. Training Loop ---
print("\n--- 5. Training Neural Network ---")
train_losses = []
val_losses = []

# For Early Stopping
best_val_loss = float('inf')
patience_epochs = 15  # Number of epochs to wait for improvement before stopping
patience_counter = 0

for epoch in range(NUM_EPOCHS):
    model.train() # Set model to training mode
    epoch_train_loss = 0.0
    for batch_X, batch_Y in train_loader:
        optimizer.zero_grad()       # Clear gradients
        outputs = model(batch_X)    # Forward pass
        loss = criterion(outputs, batch_Y) # Calculate loss
        loss.backward()             # Backward pass
        optimizer.step()            # Update weights
        epoch_train_loss += loss.item() * batch_X.size(0)
    
    epoch_train_loss /= len(train_loader.dataset)
    train_losses.append(epoch_train_loss)

    # Validation
    model.eval() # Set model to evaluation mode
    epoch_val_loss = 0.0
    with torch.no_grad(): # No gradient calculation during validation
        for batch_X_val, batch_Y_val in val_loader:
            outputs_val = model(batch_X_val)
            loss_val = criterion(outputs_val, batch_Y_val)
            epoch_val_loss += loss_val.item() * batch_X_val.size(0)
            
    epoch_val_loss /= len(val_loader.dataset)
    val_losses.append(epoch_val_loss)

    print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Train Loss: {epoch_train_loss:.6f}, Val Loss: {epoch_val_loss:.6f}")

    # Early Stopping Check
    if epoch_val_loss < best_val_loss:
        best_val_loss = epoch_val_loss
        patience_counter = 0
        # Save the best model
        torch.save(model.state_dict(), f"best_model_system2_Nu{N_u}_Ny{N_y}_ID{STUDENT_ID}.pth")
        print(f"   Validation loss improved. Saved best model.")
    else:
        patience_counter += 1
        if patience_counter >= patience_epochs:
            print(f"   Early stopping triggered after {patience_epochs} epochs without improvement.")
            break
print("Training finished.")

# Load the best model for final evaluation
model.load_state_dict(torch.load(f"best_model_system2_Nu{N_u}_Ny{N_y}_ID{STUDENT_ID}.pth"))
model.eval()

# --- 6. Validation & Evaluation ---
print("\n--- 6. Validation & Evaluation ---")

# Plotting Loss Curves
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.title('Training and Validation Loss Curves')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.legend()
plt.grid(True)

# Save PNG
output_dir = pathlib.Path("Question_1/Part_b")
output_dir.mkdir(parents=True, exist_ok=True)
plt.savefig(output_dir / "q1b2_loss_curve.png", dpi=300, bbox_inches='tight')
plt.close()
print("Saved: q1b2_loss_curve.png")

# Predictions on Validation Set (for R^2 and scatter plot)
with torch.no_grad():
    Y_val_pred_scaled_tensor = model(X_val_tensor)

# Inverse transform predictions and actuals to original scale
Y_val_pred_np = scaler_Y.inverse_transform(Y_val_pred_scaled_tensor.numpy())
Y_val_actual_np = scaler_Y.inverse_transform(Y_val_tensor.numpy()) # This is just Y_val

mse_val = mean_squared_error(Y_val_actual_np, Y_val_pred_np)
r2_val = r2_score(Y_val_actual_np, Y_val_pred_np)

print(f"\nValidation Set Performance (original scale):")
print(f"  Mean Squared Error (MSE): {mse_val:.6f}")
print(f"  R-squared (R²) Score: {r2_val:.4f}")

# Scatter Plot: Actual vs. Predicted on Validation Set
plt.figure(figsize=(8, 8))
plt.scatter(Y_val_actual_np, Y_val_pred_np, alpha=0.5, label='Validation Data')
min_val = min(Y_val_actual_np.min(), Y_val_pred_np.min())
max_val = max(Y_val_actual_np.max(), Y_val_pred_np.max())
plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Ideal y=x line')
plt.title('Actual vs. Predicted Output (Validation Set)')
plt.xlabel('Actual Output y_val')
plt.ylabel('Predicted Output ŷ_val')
plt.legend()
plt.grid(True)
plt.axis('equal')

# Save PNG
plt.savefig(output_dir / "q1b2_scatter_val.png", dpi=300, bbox_inches='tight')
plt.close()
print("Saved: q1b2_scatter_val.png")


# --- 7. Time Series Prediction Test (Qualitative) ---
print("\n--- 7. Time Series Prediction Test ---")
print("Generating a new test sequence and simulating NN response...")

# Load System2 again for a clean simulation environment
System2_val = load_system_module("System2") # Renamed to avoid conflict
s_val = System2_val(STUDENT_ID)

# Generate a test input signal (e.g., a sinusoid or steps not seen in exact form during training)
test_signal_length = 200
# u_test_signal = np.sin(np.linspace(0, 10 * np.pi, test_signal_length)) * 1.0 # Example: Sinusoid
u_test_signal = np.zeros(test_signal_length)
u_test_signal[test_signal_length//4 : test_signal_length//2] = 1.0
u_test_signal[test_signal_length*3//4 : ] = -0.5 # Example: Multi-step input

s_val.reset()
model.eval() # Ensure model is in evaluation mode

y_nn_predictions = []
y_actual_system_outputs = []

# Initialize history for NARX input to the NN
u_history_nn = [0.0] * (N_u - 1) 
y_history_nn = [0.0] * N_y    

for k_test in range(test_signal_length):
    u_k_test = u_test_signal[k_test]
    
    # Prepare NN input features [u(k), u(k-1)..., y(k-1)...]
    current_u_features = [u_k_test]
    if N_u > 1: current_u_features.extend(u_history_nn[-(N_u-1):])
    
    current_y_features = []
    if N_y > 0: current_y_features.extend(y_history_nn[-N_y:])

    nn_input_list = current_u_features + current_y_features
    nn_input_np = np.array(nn_input_list).reshape(1, -1) # Reshape for single sample
    
    # Scale NN input using the *training data's* scaler_X
    nn_input_scaled_np = scaler_X.transform(nn_input_np)
    nn_input_tensor = torch.tensor(nn_input_scaled_np, dtype=torch.float32)
    
    # Get NN prediction
    with torch.no_grad():
        predicted_y_k_scaled_tensor = model(nn_input_tensor)
    
    # Inverse transform the prediction using the *training data's* scaler_Y
    predicted_y_k_np = scaler_Y.inverse_transform(predicted_y_k_scaled_tensor.numpy())
    predicted_y_k_val = predicted_y_k_np[0,0]
    y_nn_predictions.append(predicted_y_k_val)
    
    # Get actual system output for comparison
    actual_y_k_val = s_val.output(u_k_test) # Use the separate s_val instance
    y_actual_system_outputs.append(actual_y_k_val)
    
    # Update history for the *next* NN prediction
    # u_history uses actual u_k_test
    if N_u > 1:
        u_history_nn.append(u_k_test)
        if len(u_history_nn) > (N_u-1): u_history_nn.pop(0)
    # y_history uses the NN's *own prediction* for multi-step ahead simulation
    if N_y > 0:
        y_history_nn.append(predicted_y_k_val) # Use predicted y for next step's y(k-1)
        if len(y_history_nn) > N_y: y_history_nn.pop(0)

# Plot time series comparison
plt.figure(figsize=(12, 6))
plt.plot(y_actual_system_outputs, 'b-', label='Actual System2 Output')
plt.plot(y_nn_predictions, 'r--', label='NN Predicted Output')
plt.title(f'Time Series Prediction Comparison on Test Signal (Nu={N_u}, Ny={N_y})')
plt.xlabel('Time step k')
plt.ylabel('Output y(k)')
plt.legend()
plt.grid(True)

# Save PNG
plt.savefig(output_dir / "q1b2_nn_vs_actual_timeseries.png", dpi=300, bbox_inches='tight')
plt.close()
print("Saved: q1b2_nn_vs_actual_timeseries.png")


print("--- End of Q1.b.2 Script ---")