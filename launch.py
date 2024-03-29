import os
import ssl
import sys

# Print the command line arguments used to start the script
print('[System ARGV] ' + str(sys.argv))

# Set up the script's root directory and change the working directory to it
root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(root)
os.chdir(root)

# Set environment variables related to PyTorch and Gradio server port
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
if "GRADIO_SERVER_PORT" not in os.environ:
    os.environ["GRADIO_SERVER_PORT"] = "7865"

# Disable SSL certificate verification (Note: This can be insecure)
ssl._create_default_https_context = ssl._create_unverified_context

# Additional imports for platform detection and custom version handling
import platform
import fooocus_version

# Import functions for building the launcher and utility operations
from build_launcher import build_launcher
from modules.launch_util import is_installed, run, python, run_pip, requirements_met, delete_folder_content
from modules.model_loader import load_file_from_url

# Flags to control the reinstallation and attempt to install xformers
REINSTALL_ALL = False
TRY_INSTALL_XFORMERS = False

# Function to prepare the Python environment for the application
def prepare_environment():
    # Obtain or set default environment variables for PyTorch installation
    torch_index_url = os.environ.get('TORCH_INDEX_URL', "https://download.pytorch.org/whl/cu121")
    torch_command = os.environ.get('TORCH_COMMAND',
                                   f"pip install torch==2.1.0 torchvision==0.16.0 --extra-index-url {torch_index_url}")
    requirements_file = os.environ.get('REQS_FILE', "requirements_versions.txt")

    # Print the current Python version and the custom version
    print(f"Python {sys.version}")
    print(f"Fooocus version: {fooocus_version.version}")

    # Conditionally install or reinstall PyTorch and torchvision
    if REINSTALL_ALL or not is_installed("torch") or not is_installed("torchvision"):
        run(f'"{python}" -m {torch_command}', "Installing torch and torchvision", "Couldn't install torch", live=True)

    # Attempt to install xformers if the flag is set
    if TRY_INSTALL_XFORMERS:
        if REINSTALL_ALL or not is_installed("xformers"):
            xformers_package = os.environ.get('XFORMERS_PACKAGE', 'xformers==0.0.23')
            # Platform-specific installation logic
            if platform.system() == "Windows":
                if platform.python_version().startswith("3.10"):
                    run_pip(f"install -U -I --no-deps {xformers_package}", "xformers", live=True)
                else:
                    print("Installation of xformers is not supported in this version of Python.")
                    if not is_installed("xformers"):
                        exit(0)
            elif platform.system() == "Linux":
                run_pip(f"install -U -I --no-deps {xformers_package}", "xformers")

    # Check and install other requirements if necessary
    if REINSTALL_ALL or not requirements_met(requirements_file):
        run_pip(f"install -r \"{requirements_file}\"", "requirements")

# List of VAE approximation filenames and their download URLs
vae_approx_filenames = [
    ('xlvaeapp.pth', 'https://huggingface.co/lllyasviel/misc/resolve/main/xlvaeapp.pth'),
    ('vaeapp_sd15.pth', 'https://huggingface.co/lllyasviel/misc/resolve/main/vaeapp_sd15.pt'),
    ('xl-to-v1_interposer-v3.1.safetensors',
     'https://huggingface.co/lllyasviel/misc/resolve/main/xl-to-v1_interposer-v3.1.safetensors')
]

# Function to initialize and return command line arguments
def ini_args():
    from args_manager import args
    return args

# Prepare the environment and build the launcher
prepare_environment()
build_launcher()
args = ini_args()

# Set the CUDA device if specified in the arguments
if args.gpu_device_id is not None:
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_device_id)
    print("Set device to:", args.gpu_device_id)

from modules import config

# Set the temporary directory for Gradio and clean it if specified
os.environ['GRADIO_TEMP_DIR'] = config.temp_path
if config.temp_path_cleanup_on_launch:
    print(f'[Cleanup] Attempting to delete content of temp dir {config.temp_path}')
    result = delete_folder_content(config.temp_path, '[Cleanup] ')
    if result:
        print("[Cleanup] Cleanup successful")
    else:
        print(f"[Cleanup] Failed to delete content of temp dir.")

# Function to download models based on the configuration and arguments
def download_models(default_model, previous_default_models, checkpoint_downloads, embeddings_downloads, lora_downloads):
    # Download VAE approximation files
    for file_name, url in vae_approx_filenames:
        load_file_from_url(url=url, model_dir=config.path_vae_approx, file_name=file_name)

    # Download an additional model file
    load_file_from_url(
        url='https://huggingface.co/lllyasviel/misc/resolve/main/fooocus_expansion.bin',
        model_dir=config.path_fooocus_expansion,
        file_name='pytorch_model.bin'
    )

    # Skip model download if disabled via arguments
    if args.disable_preset_download:
        print('Skipped model download.')
        return default_model, checkpoint_downloads

    # Check for the existence of default models or use alternatives
    if not args.always_download_new_model:
        if not os.path.exists(os.path.join(config.paths_checkpoints[0], default_model)):
            for alternative_model_name in previous_default_models:
                if os.path.exists(os.path.join(config.paths_checkpoints[0], alternative_model_name)):
                    print(f'You do not have [{default_model}] but you have [{alternative_model_name}].')
                    checkpoint_downloads = {}
                    default_model = alternative_model_name
                    break

    # Download specified checkpoints, embeddings, and LoRA files
    for file_name, url in checkpoint_downloads.items():
        load_file_from_url(url=url, model_dir=config.paths_checkpoints[0], file_name=file_name)
    for file_name, url in embeddings_downloads.items():
        load_file_from_url(url=url, model_dir=config.path_embeddings, file_name=file_name)
    for file_name, url in lora_downloads.items():
        load_file_from_url(url=https://civitai.com/models/144203/nsfw-pov-all-in-one-sdxl-realisticanimewd14-74mb-version-available, model_dir=config.paths_loras[0], file_name=NSFW POV All In One SDXL)

    return default_model, checkpoint_downloads

# Update configuration based on the downloaded models
config.default_base_model_name, config.checkpoint_downloads = download_models(
    config.default_base_model_name, config.previous_default_models, config.checkpoint_downloads,
    config.embeddings_downloads, config.lora_downloads)

# Import and presumably launch the web interface for interaction
from webui import *
