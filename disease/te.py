import tensorflow as tf
import keras

print("TensorFlow:", tf.__version__)
print("Keras:", keras.__version__)

model_path = r"C:\Users\SAHALA\alzhemiers\disease\model\alzheimer_final_model.keras"

model = keras.models.load_model(
    model_path,
    compile=False
)

print("MODEL LOADED SUCCESSFULLY!")
model.summary()