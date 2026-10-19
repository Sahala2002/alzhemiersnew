import tensorflow as tf

MODEL_PATH = r"C:\Users\SAHALA\alzhemiers\disease\model\alzheimer_fixed.keras"

print("=" * 80)
print("LOADING MODEL")
print("=" * 80)

model = tf.keras.models.load_model(
    MODEL_PATH,
    compile=False
)

print("\nMODEL INPUT:")
print(model.input)

print("\nMODEL OUTPUT:")
print(model.output)

print("\n" + "=" * 80)
print("FULL MODEL SUMMARY")
print("=" * 80)

model.summary(expand_nested=True)

print("\n" + "=" * 80)
print("TOP LEVEL LAYERS")
print("=" * 80)

for i, layer in enumerate(model.layers):

    print(
        f"{i}: "
        f"name={layer.name}, "
        f"type={type(layer).__name__}"
    )

print("\n" + "=" * 80)
print("NESTED MODELS")
print("=" * 80)

for layer in model.layers:

    if isinstance(layer, tf.keras.Model):

        print("\nNested model:")
        print("Name:", layer.name)
        print("Input:", layer.input)
        print("Output:", layer.output)

        print("\nNested layers:")

        for i, nested_layer in enumerate(layer.layers):

            print(
                f"  {i}: "
                f"name={nested_layer.name}, "
                f"type={type(nested_layer).__name__}"
            )

print("\n" + "=" * 80)
print("END")
print("=" * 80)