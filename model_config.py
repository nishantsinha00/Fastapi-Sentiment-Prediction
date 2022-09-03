try:
    import os
except:
    print("Error importing packages!")
    
class CONFIG:
    batch_size = 32
    epochs = 100
    embedding_dim = 32
    checkpoint_path = "training_1/cp.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path)