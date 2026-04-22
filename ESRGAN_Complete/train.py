# Local
from ESRGAN_implemetation_tiny_face import Config, ESRGANTrainer, upscale_directory


# =============================================================================
# TRAINING LOOP
# =============================================================================

# For training
def run_training():
    """Configure and launch a standard ESRGAN training run.

    No inputs. No return value; checkpoints are written to disk.
    """
    config = Config()
    config.wider_face_path = "/Volumes/T7/Capstone/widerface"
    config.batch_size = 16
    config.num_epochs = 100
    config.lr = 2e-4
    
    # Either use standard training
    trainer = ESRGANTrainer(config)
    trainer.train()
    
    # Or use face-focused training
    # train_with_face_focus(config)

# =============================================================================
# INFERENCE / UPSCALING
# =============================================================================

# For upscaling
def run_upscaling():
    """Run inference on the configured input directory using a saved model.

    No inputs. No return value; upscaled images are written to disk.
    """
    model_path = "output/models/best_model_psnr.pth"
    input_dir = "test_images"
    output_dir = "upscaled_images"
    upscale_directory(model_path, input_dir, output_dir)

# Choose which function to run
if __name__ == "__main__":
    run_training()  # or run_upscaling()