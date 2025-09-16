from transformers import AutoModel

hub_model_name = "facebook/dinov2-base"
local_save_path = "./weights/dinov2-base"

model = AutoModel.from_pretrained(hub_model_name)
model.save_pretrained(local_save_path)

print("saved successfully!")