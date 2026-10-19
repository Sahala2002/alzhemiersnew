import tensorflow as tf

MODEL_PATH = r"C:\Users\SAHALA\alzhemiers\disease\model\alzheimer_fixed.keras"

print("=" * 80)
print("LOADING MODEL")
print("=" * 80)

model = tf.keras.models.load_model(
    MODEL_PATH,
    compile=False
)

print("\nMODEL TYPE:")
print(type(model).__name__)

print("\nMODEL NAME:")
print(model.name)

print("\nMODEL INPUTS:")
print(model.inputs)

print("\nMODEL OUTPUTS:")
print(model.outputs)

print("\n" + "=" * 80)
print("MODEL SUMMARY")
print("=" * 80)

model.summary(expand_nested=True)

print("\n" + "=" * 80)
print("TOP LEVEL LAYERS")
print("=" * 80)

for i, layer in enumerate(model.layers):

    print(
        f"{i}: "
        f"name={layer.name} | "
        f"type={type(layer).__name__}"
    )

print("\n" + "=" * 80)
print("DETAILED NESTED MODEL INFORMATION")
print("=" * 80)

for i, layer in enumerate(model.layers):

    if isinstance(layer, tf.keras.Model):

        print("\n----------------------------------------")
        print("NESTED MODEL")
        print("----------------------------------------")

        print("Index:", i)
        print("Name:", layer.name)
        print("Type:", type(layer).__name__)

        print("\nNested model inputs:")
        print(layer.inputs)

        print("\nNested model outputs:")
        print(layer.outputs)

        print("\nNested layers:")

        for j, nested_layer in enumerate(layer.layers):

            print(
                f"  {j}: "
                f"{nested_layer.name} | "
                f"{type(nested_layer).__name__}"
            )

print("\n" + "=" * 80)
print("CONVOLUTIONAL LAYERS")
print("=" * 80)

def print_conv_layers(layers, prefix=""):

    for layer in layers:

        if isinstance(layer, tf.keras.layers.Conv2D):

            print(
                prefix +
                f"{layer.name} | "
                f"output_shape={getattr(layer, 'output_shape', 'N/A')}"
            )

        if isinstance(layer, tf.keras.Model):

            print(
                prefix +
                f"[Nested model: {layer.name}]"
            )

            print_conv_layers(
                layer.layers,
                prefix + "    "
            )

print_conv_layers(model.layers)

print("\n" + "=" * 80)
print("END")
print("=" * 80)