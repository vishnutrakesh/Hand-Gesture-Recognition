import h5py

model_path = r"C:\Users\Vishnu\Documents\College\5th Sem\Projects\Hand Gesture Recognition\converted_keras\keras_model.h5"

with h5py.File(model_path, "r+") as f:
    model_config = f.attrs.get("model_config")

    if isinstance(model_config, bytes):
        model_config = model_config.decode("utf-8")

    if '"groups": 1,' in model_config:
        model_config = model_config.replace('"groups": 1,', '')
        f.attrs.modify("model_config", model_config)
        print("Model fixed successfully.")
    else:
        print("groups parameter not found.")