import zipfile
import json
import os

input_model = r"C:\Users\SAHALA\alzhemiers\disease\model\alzheimer_final_model.keras"
output_model = r"C:\Users\SAHALA\alzhemiers\disease\model\alzheimer_fixed.keras"


def remove_quantization_config(obj):
    if isinstance(obj, dict):
        obj.pop("quantization_config", None)

        for key in list(obj.keys()):
            obj[key] = remove_quantization_config(obj[key])

    elif isinstance(obj, list):
        obj = [remove_quantization_config(item) for item in obj]

    return obj


print("Reading original model...")

with zipfile.ZipFile(input_model, "r") as zin:
    config = json.loads(zin.read("config.json"))

    config = remove_quantization_config(config)

    with zipfile.ZipFile(output_model, "w", zipfile.ZIP_DEFLATED) as zout:
        for item in zin.infolist():
            if item.filename == "config.json":
                zout.writestr(
                    "config.json",
                    json.dumps(config, indent=2)
                )
            else:
                zout.writestr(
                    item,
                    zin.read(item.filename)
                )

print("Fixed model created successfully!")
print(output_model)