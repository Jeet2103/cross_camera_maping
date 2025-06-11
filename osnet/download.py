import gdown
from logger_config.logger import get_logger

# Initialize logger
logger = get_logger('download.log')

def download_reid_model():
    """
    Downloads the OSNet re-identification model using gdown from Google Drive.
    The model is used in StrongSORT for feature embedding.
    """
    # === Google Drive model URL ===
    model_url = "https://drive.google.com/uc?id=1sSwXSUlj4_tHZequ_iZ8w_Jh0VaRQMqF"
    
    # === Destination path to save the model ===
    weights = "./osnet_x0_25_msmt17.pt"

    try:
        logger.info("Starting download of OSNet re-identification weights...")
        gdown.download(model_url, weights, quiet=False)
        logger.info(f"Model downloaded successfully to {weights}")
    except Exception as e:
        logger.error(f"Failed to download the model: {e}")

# === Entry Point ===
if __name__ == "__main__":
    logger.info("Download script initialized.")
    download_reid_model()
    logger.info("Download script completed.")
